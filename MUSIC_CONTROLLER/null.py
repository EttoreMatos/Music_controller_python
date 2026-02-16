from collections import deque
from dataclasses import dataclass
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

import librosa
import numpy as np
import pygame
import serial
from mistralai import Mistral
import json

@dataclass(frozen=True)
class EffectConfig:
    FPS: int = 30
    NUM_LEDS: int = 6

    STEP_FRAMES_MIN: int = 2
    STEP_FRAMES_MAX: int = 9
    BPM_STEP_MULT: float = 0.50
    MIDDLE_SEQ_SWITCH_REPEATS: int = 4
    MIDDLE_SEQUENCES: tuple[tuple[tuple[int, ...], ...], ...] = (
        ((0,), (2,), (1,), (3,)),
        ((3,), (1,), (2,), (0,)),
        ((0, 2), (1, 3), (0, 2), (1, 3)),
    )

    LOW_BAND_MIN_HZ: float = 30.0
    LOW_BAND_MAX_HZ: float = 120.0
    KICK_GAIN: float = 18.0
    KICK_EVENT_THRESHOLD: float = 0.15
    KICK_RISE_MIN: float = 0.06
    KICK_WINDOW_MS: int = 100
    KICK_COOLDOWN_FRAMES: int = 2
    KICK_MULT_MIN: float = 2.0
    KICK_MULT_MAX: float = 3.0
    LOW_FAST_ALPHA: float = 0.45
    LOW_SLOW_ALPHA: float = 0.08

    MIDDLE_FADE_DECAY: float = 0.3

    EDGE_FADE_DECAY_BASE: float = 0.84
    EDGE_FADE_DECAY_HEAVY: float = 0.80
    TAIL_CUTOFF: float = 2.0

    ACTIVE_MIN_PWM: float = 55.0
    ACTIVE_MAX_PWM: float = 190.0
    BEAT_BOOST_PWM: float = 25.0
    MIDDLE_KICK_BRIGHT_GAIN: float = 120.0
    MIDDLE_STRONG_KICK_BRIGHT_GAIN: float = 65.0
    MIDDLE_KICK_WINDOW_BRIGHT_GAIN: float = 40.0
    MIDDLE_EDGE_SYNC_GAIN: float = 85.0
    HEAVY_BEAT_THRESHOLD: float = 0.58

    EDGE_BEAT_THRESHOLD: float = 0.20
    EDGE_ATTACK_THRESHOLD: float = 0.03
    EDGE_PULSE_GAIN: float = 160.0
    EDGE_HEAVY_BONUS_GAIN: float = 150.0
    EDGE_KICK_WINDOW_BOOST: float = 200.0
    EDGE_GATE_OPEN_THRESHOLD: float = 0.18
    EDGE_GATE_CLOSE_THRESHOLD: float = 0.10
    EDGE_GATE_MIN_LOW_DOMINANCE: float = 0.16
    EDGE_GATE_HOLD_FRAMES: int = 3
    EDGE_NOISE_FLOOR_PWM: float = 12.0
    EDGE_SYNC_MIN_PWM: float = 16.0
    EDGE_IDLE_DECAY: float = 0.68

    RMS_FAST_ALPHA: float = 0.30
    RMS_SLOW_ALPHA: float = 0.06
    FLUX_FAST_ALPHA: float = 0.35
    FLUX_SLOW_ALPHA: float = 0.07
    TRANSITION_THRESHOLD: float = 0.55

    STARTUP_RAMP_SECONDS: float = 2.0
    
    # Transições
    TRANSITION_EFFECT_DURATION: float = 1.0  # segundos

class MusicLEDController:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200, fps=None, config=None):
        self.port = port
        self.baudrate = baudrate
        self.config = config or EffectConfig()
        self.fps = self.config.FPS if fps is None else int(fps)
        self.ser = None
        self.playing = False
        self._temp_playback_files = set()
        
        self.mistral_api_key = os.environ.get('MISTRAL_API_KEY', 'vRY0PPQV1MX20P4m2NCynnLuvfDWHnMc')
        self.mistral_client = None
        if self.mistral_api_key:
            try:
                self.mistral_client = Mistral(api_key=self.mistral_api_key)
                print("✅ Mistral AI habilitado")
            except Exception as e:
                print(f"⚠️  Mistral AI não disponível: {e}")

    def get_transition_effect(self, from_type, to_type):
        """Define efeito visual para transição entre seções"""
        effects = {
            # Transições para DROP: efeitos intensos
            ("buildup", "drop"): "strobe_burst",
            ("pre_chorus", "drop"): "strobe_burst",
            ("verse", "drop"): "flash_all",
            
            # Transições para CHORUS: energéticas
            ("pre_chorus", "chorus"): "pulse_wave",
            ("verse", "chorus"): "strobe_burst",
            ("buildup", "chorus"): "strobe_burst",
            
            # Transições para BREAKDOWN: suaves
            ("chorus", "breakdown"): "fade_out_in",
            ("drop", "breakdown"): "fade_out_in",
            
            # Transições para BUILDUP: crescentes
            ("verse", "buildup"): "accelerate",
            ("breakdown", "buildup"): "accelerate",
            ("pre_chorus", "buildup"): "accelerate",
            
            # Outras transições
            ("intro", "verse"): "fade_in",
            ("intro", "drop"): "pulse_wave",
        }
        
        key = (from_type, to_type)
        return effects.get(key, "pulse_wave")  # padrão

    def apply_transition_effect(self, effect_type, progress, middle_levels, active_pwm):
        """
        Aplica efeito de transição
        progress: 0.0 a 1.0 (início ao fim do efeito)
        Retorna: níveis modificados dos 4 LEDs do meio
        """
        result = middle_levels.copy()
        
        if effect_type == "strobe_burst":
            # Estrobo rápido: todos piscam juntos 8x
            cycle = int(progress * 8)
            if cycle % 2 == 0:
                result[:] = active_pwm * 1.5  # Todos ligados
            else:
                result[:] = 0  # Todos desligados
                
        elif effect_type == "flash_all":
            # Flash intenso que decai
            flash_intensity = (1.0 - progress) ** 0.5
            result[:] = active_pwm * (1.0 + flash_intensity)
            
        elif effect_type == "pulse_wave":
            # Onda pulsante que passa pelos LEDs
            wave_pos = progress * 3  # 3 ciclos
            for i in range(4):
                phase = (wave_pos + i * 0.25) % 1.0
                pulse = np.sin(phase * np.pi * 2) ** 2
                result[i] = active_pwm * (0.5 + pulse * 0.5)
                
        elif effect_type == "fade_out_in":
            # Fade out completo, depois fade in
            if progress < 0.5:
                # Fade out
                result[:] = active_pwm * (1.0 - progress * 2)
            else:
                # Fade in
                result[:] = active_pwm * ((progress - 0.5) * 2)
                
        elif effect_type == "accelerate":
            # Perseguição que acelera
            speed = 1.0 + progress * 3  # Acelera 4x
            phase = (progress * speed * 4) % 1.0
            active_led = int(phase * 4)
            result[:] = 0
            result[active_led] = active_pwm * 1.2
            
        elif effect_type == "fade_in":
            # Fade in suave
            result[:] = active_pwm * progress
            
        return result

    def analyze_music_structure_with_ai(self, y, sr, tempo_bpm, duration):
        """Usa IA para identificar seções e transições da música"""
        if not self.mistral_client:
            print("⚠️  IA não disponível, usando análise padrão")
            return None
            
        print("🤖 Analisando estrutura da música com IA...")
        
        hop_length = 512
        window_size = int(sr * 0.5)
        hop = 0.25
        n_windows = int(duration / hop)
        
        features_timeline = []
        global_rms = float(np.percentile(np.sqrt(y*y), 95))
        for i in range(min(n_windows, 200)):
            start_sample = int(i * sr * hop)
            end_sample = min(start_sample + window_size, len(y))
            window = y[start_sample:end_sample]
            
            if len(window) < 100:
                continue
            rms = float(np.sqrt(np.mean(window ** 2)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(window)[0]))
            
            spec = np.abs(librosa.stft(window))
            spec_centroid = float(np.mean(librosa.feature.spectral_centroid(S=spec, sr=sr)))
            
            freqs = librosa.fft_frequencies(sr=sr)
            bass = float(np.mean(spec[freqs < 250]))
            mid = float(np.mean(spec[(freqs >= 250) & (freqs < 2000)]))
            high = float(np.mean(spec[freqs >= 2000]))
            
            energy = rms / (global_rms + 1e-9)
            low_mid_ratio = bass / (mid + 1e-6)
            brightness_norm = spec_centroid / 8000.0
            flux = float(np.mean(np.maximum(spec - np.roll(spec,1,axis=1),0)))

            features_timeline.append({
                "time": float(round(i * hop, 2)),
                "energy": float(round(energy, 4)),
                "low_mid_ratio": float(round(low_mid_ratio, 4)),
                "brightness": float(round(brightness_norm, 4)),
                "flux": float(round(flux,4)),
                "bass_energy": float(round(bass,4))
            })
        
        features_timeline = [
            {k: float(v) for k, v in frame.items()}
            for frame in features_timeline
        ]
        
        prompt = f"""Você é um analisador de estrutura musical para sistemas de iluminação em tempo real.

Seu trabalho NÃO é imaginar seções.

Você deve detectar:

- Drops reais
- Breakdowns reais
- Crescimentos reais
- Quedas reais de energia

Baseando-se APENAS nos dados numéricos.

REGRAS CRÍTICAS:

1. Um DROP só existe se:
   - energia sobe >40%
   - low_mid_ratio sobe >30%

2. Um BREAKDOWN só existe se:
   - energia cai >35%

3. Buildup só se:
   - energia cresce continuamente por ≥4 segundos

4. Chorus só se:
   - energia se mantém alta ≥6 segundos

5. NÃO invente seções.

6. Prefira poucas seções bem colocadas.

7. Transições devem ocorrer NO PRIMEIRO FRAME da mudança energética.

Tipos permitidos:
intro, verse, buildup, drop, breakdown, chorus, outro

Retorne apenas JSON.

Cada seção deve ter:

start
end
type
intensity (0–1)
speed_multiplier (0.6–1.8)
pattern

Patterns:

intro/outro → slow_chase
verse → alternating
buildup → build_accelerate
drop → strobe_sync
chorus → all_pulse
breakdown → pairs

Nunca use random_flash.
Use strobe apenas em drop ou chorus.

OBS: Use pouco o flash_all, pois é um pouco desconfortável e quando usar, deixe em um tempo curto.

Dados:

INFORMAÇÕES DA MÚSICA:
- BPM: {tempo_bpm:.1f}
- Duração: {duration:.1f} segundos
- Features ao longo do tempo (amostra a cada 0.25s):

{json.dumps(features_timeline[:480], indent=1)}
... (total de {len(features_timeline)} amostras)

TIPOS DE PADRÃO VISUAL (para os 4 LEDs centrais):
- slow_chase: perseguição lenta e suave
- fast_chase: perseguição rápida e intensa
- alternating: alternância simples
- pairs: acende em pares
- wave: efeito onda
- strobe_sync: sincronia estroboscópica com batida
- all_pulse: todos pulsam juntos
- build_accelerate: acelera progressivamente
- random_flash: flashes aleatórios
- center_out: do centro para fora

REGRAS:
1. Identifique mudanças na estrutura (energia, frequência, ritmo), ex: se a energia da música abaixou muito, significa que provavelmente terá uma transição.
2. Seções típicas: 8-32 segundos cada
3. Transições marcam mudanças claras (drops, breaks, etc)
4. Escolha padrão visual adequado para cada seção:
   - Intro/Outro: slow_chase, wave
   - Verse: alternating, slow_chase
   - Pre-chorus: fast_chase, build_accelerate
   - Chorus/Drop: strobe_sync, all_pulse, fast_chase
   - Breakdown: pairs, slow_chase
   - Buildup: build_accelerate
5. Intensidade: 0.0-1.0 (quão forte/brilhante)
6. Speed_multiplier: 0.5-2.0 (velocidade da sequência)

RETORNE JSON:
{{
  "sections": [
    {{
      "start": 0.0,
      "end": 15.5,
      "type": "intro",
      "pattern": "slow_chase",
      "intensity": 0.4,
      "speed_multiplier": 0.8,
      "description": "Introdução suave com melodia ambiente"
    }},
    {{
      "start": 15.5,
      "end": 30.0,
      "type": "verse",
      "pattern": "alternating",
      "intensity": 0.6,
      "speed_multiplier": 1.0,
      "description": "Primeiro verso com batida constante"
    }}
  ]
}}

Analise TODA a duração ({duration:.1f}s) e retorne APENAS o JSON válido.
"""

        try:
            response = self.mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            sections = result.get("sections", [])
            
            if len(sections) < 2:
                print("⚠️  IA retornou poucas seções, usando análise padrão")
                return None
            
            print(f"✅ IA identificou {len(sections)} seções:")
            for sec in sections:
                print(f"   {sec['start']:.1f}s-{sec['end']:.1f}s: {sec['type']} ({sec['pattern']})")
            
            return sections
            
        except Exception as e:
            print(f"⚠️  Erro na análise IA: {e}")
            return None

    def get_pattern_for_section(self, section_type, pattern_name):
        """Retorna sequência de LEDs baseada no tipo de padrão"""
        patterns = {
            "slow_chase": ((0,), (1,), (2,), (3,)),
            "fast_chase": ((0,), (1,), (2,), (3,)),
            "alternating": ((0,), (2,), (1,), (3,)),
            "pairs": ((0, 1), (2, 3)),
            "wave": ((0,), (0, 1), (1, 2), (2, 3), (3,)),
            "strobe_sync": ((0, 1, 2, 3),),
            "all_pulse": ((0, 1, 2, 3),),
            "build_accelerate": ((0,), (1,), (2,), (3,)),
            "random_flash": ((0,), (2,), (1,), (3,)),
            "center_out": ((1, 2), (0, 3)),
        }
        return patterns.get(pattern_name, ((0,), (1,), (2,), (3,)))

    def connect_arduino(self):
        """Conecta ao Arduino via serial."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            print(f"✅ Conectado ao Arduino na porta {self.port}")
            if self.ser.in_waiting:
                print(self.ser.readline().decode(errors="ignore").strip())
            return True
        except Exception as exc:
            print(f"✗ Erro ao conectar ao Arduino: {exc}")
            print("  Verifique a porta serial e se o Arduino está conectado")
            return False

    @staticmethod
    def _ema(prev, value, alpha):
        return alpha * value + (1.0 - alpha) * prev

    @staticmethod
    def _sequence_step_focus(step):
        if not step:
            return 0.0
        return float(sum(step)) / float(len(step))

    @staticmethod
    def _is_url(value):
        return isinstance(value, str) and (
            value.startswith("http://") or value.startswith("https://")
        )

    def _load_audio_data(self, audio_source, target_sr=22050):
        """Carrega áudio local com librosa ou stream URL com ffmpeg (sem download)."""
        if self._is_url(audio_source):
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                raise RuntimeError("ffmpeg não encontrado. Instale ffmpeg para processar stream do YouTube.")

            cmd = [
                ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", audio_source, "-vn", "-ac", "1", "-ar", str(target_sr),
                "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode != 0:
                err = result.stderr.decode(errors="ignore").strip()
                raise RuntimeError(f"falha ao decodificar stream com ffmpeg: {err}")

            pcm = np.frombuffer(result.stdout, dtype=np.int16)
            if pcm.size == 0:
                raise RuntimeError("stream sem áudio decodificado")

            y = pcm.astype(np.float32) / 32768.0
            duration = float(len(y)) / float(target_sr)
            return y, target_sr, duration

        y, sr = librosa.load(audio_source, sr=target_sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        return y, sr, duration

    def _create_temp_wav_from_audio(self, y, sr):
        """Cria WAV temporário mono PCM16 para playback estável."""
        fd, path = tempfile.mkstemp(prefix="ledsync_", suffix=".wav")
        os.close(fd)

        pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(path, "wb") as wavf:
            wavf.setnchannels(1)
            wavf.setsampwidth(2)
            wavf.setframerate(int(sr))
            wavf.writeframes(pcm.tobytes())

        self._temp_playback_files.add(path)
        return path

    def _cleanup_temp_playback_files(self):
        for path in list(self._temp_playback_files):
            try:
                os.remove(path)
            except OSError:
                pass
            finally:
                self._temp_playback_files.discard(path)

    def _play_with_ffplay(self, audio_source, missing_msg):
        ffplay_bin = shutil.which("ffplay")
        if not ffplay_bin:
            print(missing_msg)
            self.playing = False
            return

        cmd = [ffplay_bin, "-nodisp", "-autoexit", "-loglevel", "quiet", audio_source]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            while self.playing and proc.poll() is None:
                time.sleep(0.1)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    proc.kill()

        if proc.returncode not in (0, None):
            print("✗ Falha ao reproduzir áudio com ffplay.")
            self.playing = False

    def print_pattern_telemetry(self, led_patterns, dominant_leds):
        """Diagnóstico curto do padrão gerado."""
        num_leds = self.config.NUM_LEDS
        pwms = np.array([v for _, v in led_patterns], dtype=np.int32)
        dom = np.array(dominant_leds, dtype=np.int32)

        sat_245 = np.mean(pwms >= 245, axis=0) * 100.0
        dark_5 = np.mean(pwms <= 5, axis=0) * 100.0
        dom_share = [float(np.mean(dom == i) * 100.0) for i in range(num_leds)]

        print("\n📊 Telemetria do padrão")
        print("   Saturação >=245 (%): " + ", ".join(f"L{i+1}:{sat_245[i]:.1f}" for i in range(num_leds)))
        print("   Escuro <=5 (%): " + ", ".join(f"L{i+1}:{dark_5[i]:.1f}" for i in range(num_leds)))
        print("   Dominância (%): " + ", ".join(f"L{i+1}:{dom_share[i]:.1f}" for i in range(num_leds)))

    def process_audio_file(self, audio_file):
        """Processa arquivo de áudio e gera padrões PWM usando IA para transições."""
        cfg = self.config
        num_leds = cfg.NUM_LEDS

        source_label = audio_file
        if self._is_url(audio_file):
            source_label = "YouTube stream"
        print(f"\n📀 Carregando áudio: {source_label}")

        y, sr, duration = self._load_audio_data(audio_file, target_sr=22050)
        print(f"   Duração: {duration:.2f}s | Sample Rate: {sr} Hz")

        samples_per_frame = int(sr / self.fps)
        n_frames = max(1, int(duration * self.fps))
        bytes_per_frame = samples_per_frame * 2

        pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        pcm_bytes = pcm.tobytes()

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        try:
            tempo_bpm = float(np.asarray(tempo).item())
        except Exception:
            tempo_bpm = 120.0
        if not np.isfinite(tempo_bpm) or tempo_bpm <= 1.0:
            tempo_bpm = 120.0

        ai_sections = self.analyze_music_structure_with_ai(y, sr, tempo_bpm, duration)

        base_step_frames = int(
            np.clip(
                round((60.0 / tempo_bpm) * self.fps * cfg.BPM_STEP_MULT),
                cfg.STEP_FRAMES_MIN,
                cfg.STEP_FRAMES_MAX,
            )
        )
        base_speed = 1.0 / float(base_step_frames)
        
        if ai_sections:
            current_section_idx = 0
            current_section = ai_sections[0]
            current_pattern = self.get_pattern_for_section(
                current_section['type'],
                current_section['pattern']
            )
            intensity_mult = current_section.get('intensity', 1.0)
            speed_mult = current_section.get('speed_multiplier', 1.0)
        else:
            ai_sections = None
            current_section = None
            current_pattern = cfg.MIDDLE_SEQUENCES[0]
            intensity_mult = 1.0
            speed_mult = 1.0
            current_section_idx = 0

        print(f"   Processando {n_frames} frames ({self.fps} FPS)...")
        print(f"   BPM detectado: {tempo_bpm:.1f} | Base step: {base_step_frames} frames")

        led_patterns = []
        dominant_leds = []

        prev_spectrum = None
        ema_rms_fast = 0.0
        ema_rms_slow = 0.0
        ema_flux_fast = 0.0
        ema_flux_slow = 0.0

        energy_window = deque(maxlen=max(4, int(self.fps * 2)))
        low_band_window = deque(maxlen=max(4, int(self.fps * 1.2)))

        middle_leds = np.array([1, 2, 3, 4], dtype=np.int32)
        sequence_step_idx = 0
        active_step = current_pattern[sequence_step_idx]
        active_mid = int(round(self._sequence_step_focus(active_step)))
        phase_accumulator = 0.0

        middle_levels = np.zeros(4, dtype=np.float32)
        edge_levels = np.zeros(2, dtype=np.float32)
        prev_kick_strength = 0.0
        prev_low_norm = 0.0
        low_fast = 0.0
        low_slow = 0.0
        kick_window_left = 0
        kick_cooldown_left = 0
        edge_gate_open = False
        edge_gate_hold = 0

        startup_frames = max(1, int(cfg.STARTUP_RAMP_SECONDS * self.fps))
        kick_window_frames = max(1, int((cfg.KICK_WINDOW_MS / 1000.0) * self.fps))
        
        # ===== SISTEMA DE TRANSIÇÕES =====
        transition_active = False
        transition_effect = None
        transition_start_frame = 0
        transition_duration_frames = int(cfg.TRANSITION_EFFECT_DURATION * self.fps)
        prev_section_type = current_section['type'] if current_section else None
        drop_lock = 0
        for i in range(n_frames):
            current_time = i / self.fps
            
            # ===== VERIFICA SE TRANSIÇÃO TERMINOU =====
            if transition_active:
                frames_since_transition = i - transition_start_frame
                if frames_since_transition >= transition_duration_frames:
                    transition_active = False
                    transition_effect = None

            start_b = i * bytes_per_frame
            end_b = start_b + bytes_per_frame
            chunk = pcm_bytes[start_b:end_b]
            frame = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)

            if len(frame) < samples_per_frame:
                frame = np.pad(frame, (0, samples_per_frame - len(frame)))

            frame = frame / 32768.0

            # AGORA aplica janela
            frame = frame * np.hanning(len(frame))


            rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
            spectrum = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
            
            low_mask = (freqs >= cfg.LOW_BAND_MIN_HZ) & (freqs <= cfg.LOW_BAND_MAX_HZ)
            if np.any(low_mask):
                low_band_energy = float(np.mean(spectrum[low_mask]))
            else:
                low_band_energy = 0.0
                
            high_mask = (freqs >= 2000.0) & (freqs <= 8000.0)
            if np.any(high_mask):
                high_band_energy = float(np.mean(spectrum[high_mask]))
            else:
                high_band_energy = 0.0

            if prev_spectrum is None:
                flux = 0.0
            else:
                diff = spectrum - prev_spectrum
                flux = float(np.mean(np.maximum(diff, 0.0)))

            prev_spectrum = spectrum

            if i == 0:
                ema_rms_fast = rms
                ema_rms_slow = rms
                ema_flux_fast = flux
                ema_flux_slow = flux
                low_fast = low_band_energy
                low_slow = low_band_energy
            else:
                ema_rms_fast = self._ema(ema_rms_fast, rms, cfg.RMS_FAST_ALPHA)
                ema_rms_slow = self._ema(ema_rms_slow, rms, cfg.RMS_SLOW_ALPHA)
                ema_flux_fast = self._ema(ema_flux_fast, flux, cfg.FLUX_FAST_ALPHA)
                ema_flux_slow = self._ema(ema_flux_slow, flux, cfg.FLUX_SLOW_ALPHA)
                low_fast = self._ema(low_fast, low_band_energy, cfg.LOW_FAST_ALPHA)
                low_slow = self._ema(low_slow, low_band_energy, cfg.LOW_SLOW_ALPHA)

            beat_norm = float(np.clip((ema_rms_fast - ema_rms_slow) * 12.0, 0.0, 1.0))
            kick_strength = float(np.clip((low_fast - low_slow) * cfg.KICK_GAIN, 0.0, 1.0))
            if drop_lock > 0:
                drop_lock -= 1
            raw_drop = (
                kick_strength > 0.6 and
                beat_norm > 0.5 and
                ema_flux_fast > ema_flux_slow * 1.6
            )

            real_drop = raw_drop and drop_lock == 0

            if real_drop:
                drop_lock = int(self.fps * 0.8)

            # ===== TROCA DE SEÇÃO COM EFEITO DE TRANSIÇÃO =====
            if ai_sections and current_section:
                section_type = current_section["type"]

                if section_type == "drop":
                    should_advance = real_drop
                else:
                    should_advance = current_time >= current_section["end"]

                if should_advance:
                    current_section_idx += 1

                    if current_section_idx < len(ai_sections):
                        next_section = ai_sections[current_section_idx]

                        transition_effect = self.get_transition_effect(
                            current_section["type"],
                            next_section["type"]
                        )

                        transition_active = True
                        transition_start_frame = i

                        print(f"\n🎬 Transição: {current_section['type']} → {next_section['type']} ({transition_effect})")

                        current_section = next_section
                        current_pattern = self.get_pattern_for_section(
                            current_section["type"],
                            current_section["pattern"]
                        )

                        intensity_mult = current_section.get("intensity", 1.0)
                        speed_mult = current_section.get("speed_multiplier", 1.0)

                        sequence_step_idx = 0
                        active_step = current_pattern[sequence_step_idx]
                        active_mid = int(round(self._sequence_step_focus(active_step)))

            if kick_cooldown_left > 0:
                kick_cooldown_left -= 1
            if (
                kick_cooldown_left == 0
                and prev_kick_strength < cfg.KICK_EVENT_THRESHOLD
                and kick_strength >= cfg.KICK_EVENT_THRESHOLD
                and (kick_strength - prev_kick_strength) >= cfg.KICK_RISE_MIN
            ):
                kick_window_left = kick_window_frames
                kick_cooldown_left = cfg.KICK_COOLDOWN_FRAMES

            energy_window.append(rms)
            p90_energy = float(np.percentile(np.array(energy_window), 90))
            norm_energy = float(np.clip(rms / (p90_energy + 1e-9), 0.0, 1.0))
            
            low_band_window.append(low_band_energy)
            p95_low = float(np.percentile(np.array(low_band_window), 95))
            low_norm = float(np.clip(low_band_energy / (p95_low + 1e-9), 0.0, 1.0))

            startup_factor = float(np.clip((i + 1) / startup_frames, 0.0, 1.0))
            startup_gain = 0.45 + 0.55 * startup_factor

            if kick_window_left > 0:
                kick_window_left -= 1

            current_speed = base_speed * speed_mult
            phase_accumulator += current_speed
            
            while phase_accumulator >= 1.0:
                phase_accumulator -= 1.0
                sequence_step_idx = (sequence_step_idx + 1) % len(current_pattern)
                active_step = current_pattern[sequence_step_idx]
                active_mid = int(round(self._sequence_step_focus(active_step)))

            middle_levels *= cfg.MIDDLE_FADE_DECAY

            active_pwm = (
                cfg.ACTIVE_MIN_PWM
                + (cfg.ACTIVE_MAX_PWM - cfg.ACTIVE_MIN_PWM) * norm_energy * intensity_mult
                + cfg.BEAT_BOOST_PWM * beat_norm
            )
            
            if beat_norm > cfg.HEAVY_BEAT_THRESHOLD:
                active_pwm += 25.0 * ((beat_norm - cfg.HEAVY_BEAT_THRESHOLD) / (1.0 - cfg.HEAVY_BEAT_THRESHOLD))
            
            active_pwm += cfg.MIDDLE_KICK_BRIGHT_GAIN * kick_strength
            
            if kick_strength > 0.7:
                active_pwm += cfg.MIDDLE_STRONG_KICK_BRIGHT_GAIN * ((kick_strength - 0.7) / 0.3)
            
            if kick_window_left > 0:
                active_pwm += cfg.MIDDLE_KICK_WINDOW_BRIGHT_GAIN * (0.4 + 0.6 * kick_strength)
            
            active_pwm = float(np.clip(active_pwm * startup_gain, cfg.ACTIVE_MIN_PWM * 0.45, cfg.ACTIVE_MAX_PWM))

            # ===== APLICA EFEITO DE TRANSIÇÃO SE ATIVO =====
            if transition_active:
                frames_since_transition = i - transition_start_frame
                transition_progress = float(frames_since_transition) / float(transition_duration_frames)
                middle_levels = self.apply_transition_effect(
                    transition_effect,
                    transition_progress,
                    middle_levels,
                    active_pwm
                )
            else:
                # Comportamento normal da seção
                for mid_idx in active_step:
                    middle_levels[mid_idx] = max(middle_levels[mid_idx], active_pwm)

            # Pontas (mantém lógica original)
            low_ratio = low_band_energy / (high_band_energy + 1e-9)
            low_attack = max(0.0, low_norm - prev_low_norm)
            edge_drive = float(np.clip((max(kick_strength, low_norm) - cfg.EDGE_BEAT_THRESHOLD) / (1.0 - cfg.EDGE_BEAT_THRESHOLD), 0.0, 1.0))
            low_dominance = float(np.clip((low_ratio - 0.3) / 1.2, 0.0, 1.0))
            edge_drive *= (0.35 + 0.65 * low_dominance)
            edge_decay = cfg.EDGE_FADE_DECAY_BASE + (cfg.EDGE_FADE_DECAY_HEAVY - cfg.EDGE_FADE_DECAY_BASE) * edge_drive
            edge_levels *= edge_decay

            edge_gate_metric = max(kick_strength, low_attack * 1.6)
            if edge_gate_open:
                if edge_gate_metric >= cfg.EDGE_GATE_OPEN_THRESHOLD:
                    edge_gate_hold = cfg.EDGE_GATE_HOLD_FRAMES
                elif edge_gate_hold > 0:
                    edge_gate_hold -= 1
                elif edge_gate_metric < cfg.EDGE_GATE_CLOSE_THRESHOLD or low_dominance < (cfg.EDGE_GATE_MIN_LOW_DOMINANCE * 0.8):
                    edge_gate_open = False
            elif edge_gate_metric >= cfg.EDGE_GATE_OPEN_THRESHOLD and low_dominance >= cfg.EDGE_GATE_MIN_LOW_DOMINANCE:
                edge_gate_open = True
                edge_gate_hold = cfg.EDGE_GATE_HOLD_FRAMES

            if edge_gate_open:
                edge_pulse = (
                    cfg.EDGE_PULSE_GAIN * max(0.0, (low_attack * 1.8) - cfg.EDGE_ATTACK_THRESHOLD)
                    + cfg.EDGE_HEAVY_BONUS_GAIN * max(0.0, max(kick_strength, low_norm) - 0.55)
                )
                if kick_window_left > 0:
                    edge_pulse += cfg.EDGE_KICK_WINDOW_BOOST * (0.4 + 0.6 * max(kick_strength, low_norm))
                edge_pulse *= (0.25 + 0.75 * low_dominance)
                if edge_pulse < cfg.EDGE_NOISE_FLOOR_PWM:
                    edge_pulse = 0.0
            else:
                edge_levels *= cfg.EDGE_IDLE_DECAY
                edge_pulse = 0.0

            edge_pulse *= startup_gain
            if edge_pulse >= cfg.EDGE_SYNC_MIN_PWM:
                sync_strength = float(np.clip(edge_pulse / 255.0, 0.0, 1.0))
                for mid_idx in active_step:
                    middle_levels[mid_idx] = min(255.0, middle_levels[mid_idx] + cfg.MIDDLE_EDGE_SYNC_GAIN * sync_strength)
            if edge_pulse > 1.0:
                edge_levels[0] = max(edge_levels[0], edge_pulse)
                edge_levels[1] = max(edge_levels[1], edge_pulse)

            prev_kick_strength = kick_strength
            prev_low_norm = low_norm

            pwm = np.zeros(num_leds, dtype=np.float32)
            pwm[middle_leds] = middle_levels
            pwm[0] = edge_levels[0]
            pwm[5] = edge_levels[1]
            pwm = np.clip(pwm, 0.0, 255.0)
            pwm[pwm < cfg.TAIL_CUTOFF] = 0.0
            pwm_int = pwm.astype(np.int32)

            led_patterns.append((1, pwm_int))
            dominant_leds.append(int(middle_leds[active_mid]))

            if (i + 1) % 100 == 0 or i == n_frames - 1:
                progress = (i + 1) / n_frames * 100
                print(f"   Progresso: {progress:.1f}%", end="\r")

        playback_source = audio_file
        if self._is_url(audio_file):
            playback_source = self._create_temp_wav_from_audio(y, sr)
            print("   Playback local cache: pronto")

        print("\n✅ Processamento concluído!")
        self.print_pattern_telemetry(led_patterns, dominant_leds)
        return led_patterns, duration, playback_source

    def send_led_command(self, led_values, mode=1):
        """Envia comando PWM para Arduino (mode 1 por padrão)."""
        if self.ser and self.ser.is_open:
            num_leds = self.config.NUM_LEDS
            values = np.zeros(num_leds, dtype=np.int32)
            src = np.asarray(led_values, dtype=np.int32)
            n = min(num_leds, src.shape[0])
            values[:n] = src[:n]
            payload = ",".join(str(int(v)) for v in values)
            command = f"P,{int(mode)},{payload}\n"
            self.ser.write(command.encode())

    def play_audio_thread(self, audio_source):
        if self._is_url(audio_source):
            self._play_with_ffplay(audio_source, "✗ ffplay não encontrado para reproduzir stream (instale ffmpeg).")
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(audio_source)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy() and self.playing:
                time.sleep(0.1)
        except Exception as exc:
            print(f"⚠️  pygame falhou ao reproduzir arquivo ({exc}). Usando ffplay...")
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            self._play_with_ffplay(audio_source, "✗ ffplay não encontrado para fallback de reprodução local (instale ffmpeg).")

    def sync_and_play(self, audio_source):
        led_patterns, duration, playback_source = self.process_audio_file(audio_source)

        if not self.ser or not self.ser.is_open:
            print("✗ Arduino não conectado!")
            self._cleanup_temp_playback_files()
            return

        print("\n🎵 Iniciando reprodução sincronizada...\n")
        print("Pressione Ctrl+C para parar\n")

        self.playing = True
        audio_thread = threading.Thread(target=self.play_audio_thread, args=(playback_source,))
        audio_thread.start()

        time.sleep(0.1)
        frame_duration = 1.0 / self.fps

        try:
            start_time = time.time()
            for i, (mode, led_values) in enumerate(led_patterns):
                if not self.playing:
                    break

                self.send_led_command(led_values, mode=mode)

                expected_time = start_time + (i * frame_duration)
                now = time.time()
                sleep_time = expected_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if i % self.fps == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"⏱️  {elapsed:.1f}s / {duration:.1f}s | PWM: {list(map(int, led_values))}",
                        flush=True
                    )

            print("\n\n✅ Reprodução finalizada!")
        except KeyboardInterrupt:
            print("\n\n⏹️  Reprodução interrompida pelo usuário")
        finally:
            self.playing = False
            self.send_led_command([0] * self.config.NUM_LEDS, mode=1)
            audio_thread.join()
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            self._cleanup_temp_playback_files()

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.send_led_command([0] * self.config.NUM_LEDS, mode=1)
            self.ser.close()
            print("✅ Desconectado do Arduino")


def select_audio_file():
    """Abre o seletor de arquivo do PyQt5 e retorna o caminho escolhido."""
    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog
    except Exception:
        print("✗ PyQt5 não encontrado. Instale com: pip install pyqt5")
        return None

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True

    filters = "Arquivos de áudio (*.mp3 *.wav *.flac *.ogg *.m4a *.aac);;Todos os arquivos (*)"
    selected_file, _ = QFileDialog.getOpenFileName(None, "Selecione o arquivo de áudio", "", filters)

    if owns_app:
        app.quit()

    return selected_file or None


def select_audio_source():
    """Seleciona fonte de áudio: arquivo local ou busca no YouTube."""
    try:
        from PyQt5.QtWidgets import QApplication, QInputDialog
    except Exception:
        print("✗ PyQt5 não encontrado. Instale com: pip install pyqt5")
        return None

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True

    options = ["Arquivo local", "Buscar no YouTube (yt-dlp)"]
    choice, ok = QInputDialog.getItem(None, "Fonte do áudio", "Escolha de onde carregar o áudio:", options, 0, False)
    if not ok:
        if owns_app:
            app.quit()
        return None

    if choice == options[0]:
        selected_file = select_audio_file()
        if owns_app:
            app.quit()
        if not selected_file:
            return None
        return ("file", selected_file)

    query, ok = QInputDialog.getText(None, "Buscar no YouTube", "Digite o nome da música ou cole a URL:")
    if owns_app:
        app.quit()
    if not ok or not query.strip():
        return None
    return ("youtube", query.strip())


def resolve_youtube_stream(query_or_url):
    """Busca/resolve stream de áudio via API Python do yt-dlp (sem download)."""
    try:
        import yt_dlp
    except Exception as exc:
        raise RuntimeError("Biblioteca yt-dlp não encontrada. Instale com: pip install yt-dlp") from exc

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestaudio/best",
        "default_search": "ytsearch1",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query_or_url, download=False)

    if not info:
        raise RuntimeError("yt-dlp não retornou informações do vídeo")

    if "entries" in info and info["entries"]:
        info = next((entry for entry in info["entries"] if entry), None)
        if not info:
            raise RuntimeError("playlist vazia ou sem entradas válidas")

    stream_url = info.get("url")
    if not stream_url:
        formats = info.get("formats") or []
        audio_only = [
            f for f in formats
            if f.get("url") and f.get("acodec") not in (None, "none") and f.get("vcodec") in (None, "none")
        ]
        if audio_only:
            audio_only.sort(key=lambda f: (f.get("abr") or 0.0, f.get("tbr") or 0.0))
            stream_url = audio_only[-1]["url"]

    if not stream_url:
        raise RuntimeError("não foi possível resolver a URL de áudio do YouTube")

    title = info.get("title") or "YouTube"
    return stream_url, title


def main():
    print("=" * 60)
    print("     🎵 LED MUSIC SYNC 🤖 AI TRANSITIONS")
    print("=" * 60)

    arduino_port = "/dev/ttyACM0"
    selection = select_audio_source()
    if not selection:
        print("\n⚠️  Nenhum arquivo selecionado.")
        return

    source_type, source_value = selection
    if source_type == "youtube":
        print("\n🔗 Resolvendo stream do YouTube com yt-dlp...")
        try:
            audio_file, title = resolve_youtube_stream(source_value)
            print(f"✅ Stream pronto: {title}")
        except Exception as exc:
            print(f"✗ Falha ao resolver stream: {exc}")
            return
    else:
        audio_file = source_value

    controller = MusicLEDController(port=arduino_port, fps=30)

    if not controller.connect_arduino():
        print("\n⚠️  Configure a porta serial correta e tente novamente")
        return

    try:
        controller.sync_and_play(audio_file)
    except FileNotFoundError:
        print(f"\n✗ Arquivo não encontrado: {audio_file}")
    except Exception as exc:
        print(f"\n✗ Erro: {exc}")
    finally:
        controller.disconnect()

    print("\n" + "=" * 60)
    print("Obrigado por usar o LED Music Sync!")
    print("=" * 60)


if __name__ == "__main__":
    main()