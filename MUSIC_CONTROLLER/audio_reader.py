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

@dataclass(frozen=True)
class EffectConfig:
    FPS: int = 30
    NUM_LEDS: int = 6

    STEP_FRAMES_MIN: int = 2
    STEP_FRAMES_MAX: int = 9
    BPM_STEP_MULT: float = 0.50
    MIDDLE_SEQ_SWITCH_REPEATS: int = 4
    MIDDLE_SEQUENCES: tuple[tuple[tuple[int, ...], ...], ...] = (
        # Modos:
        ((0,), (2,), (1,), (3,)),  # alternada
        ((3,), (1,), (2,), (0,)),  # alternada invertida
        ((0, 2), (1, 3), (0, 2), (1, 3)),  # pares
        ((0, 1), (2, 3), (0, 1), (2, 3)),  # bordas
    )

    # Kick por baixa frequência (30-180 Hz), sem depender de beat_track
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

    # 4 LEDs do meio: fade quase nulo (mais seco)
    MIDDLE_FADE_DECAY: float = 0.3

    # 2 LEDs de ponta: fade mais alto, pulso por graves
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

    # Envelopes
    RMS_FAST_ALPHA: float = 0.30
    RMS_SLOW_ALPHA: float = 0.06
    FLUX_FAST_ALPHA: float = 0.35
    FLUX_SLOW_ALPHA: float = 0.07
    TRANSITION_THRESHOLD: float = 0.55

    # Evita pico exagerado nos primeiros segundos
    STARTUP_RAMP_SECONDS: float = 2.0

class MusicLEDController:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200, fps=None, config=None):
        self.port = port
        self.baudrate = baudrate
        self.config = config or EffectConfig()
        self.fps = self.config.FPS if fps is None else int(fps)
        self.ser = None
        self.playing = False
        self._temp_playback_files = set()

    def connect_arduino(self):
        """Conecta ao Arduino via serial."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            print(f"✓ Conectado ao Arduino na porta {self.port}")

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
    def _sequence_step_equals(step_a, step_b):
        return tuple(step_a) == tuple(step_b)

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
                raise RuntimeError(
                    "ffmpeg não encontrado. Instale ffmpeg para processar stream do YouTube."
                )

            cmd = [
                ffmpeg_bin,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                audio_source,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(target_sr),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "pipe:1",
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

        cmd = [
            ffplay_bin,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            audio_source,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
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
        print(
            "   Saturação >=245 (%): "
            + ", ".join(f"L{i+1}:{sat_245[i]:.1f}" for i in range(num_leds))
        )
        print(
            "   Escuro <=5 (%): "
            + ", ".join(f"L{i+1}:{dark_5[i]:.1f}" for i in range(num_leds))
        )
        print(
            "   Dominância (%): "
            + ", ".join(f"L{i+1}:{dom_share[i]:.1f}" for i in range(num_leds))
        )

    def process_audio_file(self, audio_file):
        """
        Processa arquivo de áudio e gera padrões PWM de 6 canais.

        Comportamento alvo:
        - 4 LEDs do meio (1..4): chaser claro com fade leve
        - 2 LEDs das pontas (0 e 5): sem passar, pulsando com bateria e fade alto
        """
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
        bytes_per_frame = samples_per_frame * 2  # int16 mono

        pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        pcm_bytes = pcm.tobytes()

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        try:
            tempo_bpm = float(np.asarray(tempo).item())
        except Exception:
            tempo_bpm = 120.0
        if not np.isfinite(tempo_bpm) or tempo_bpm <= 1.0:
            tempo_bpm = 120.0

        base_step_frames = int(
            np.clip(
                round((60.0 / tempo_bpm) * self.fps * cfg.BPM_STEP_MULT),
                cfg.STEP_FRAMES_MIN,
                cfg.STEP_FRAMES_MAX,
            )
        )
        base_speed = 1.0 / float(base_step_frames)

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
        middle_sequences = cfg.MIDDLE_SEQUENCES
        sequence_idx = 0
        current_sequence = middle_sequences[sequence_idx]
        sequence_step_idx = 0
        sequence_loop_count = 0
        active_step = current_sequence[sequence_step_idx]
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

        for i in range(n_frames):
            start_b = i * bytes_per_frame
            end_b = start_b + bytes_per_frame
            chunk = pcm_bytes[start_b:end_b]

            frame = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            if len(frame) < samples_per_frame:
                frame = np.pad(frame, (0, samples_per_frame - len(frame)))
            frame = frame / 32768.0

            # Features
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
            else:
                ema_rms_fast = self._ema(ema_rms_fast, rms, cfg.RMS_FAST_ALPHA)
                ema_rms_slow = self._ema(ema_rms_slow, rms, cfg.RMS_SLOW_ALPHA)
                ema_flux_fast = self._ema(ema_flux_fast, flux, cfg.FLUX_FAST_ALPHA)
                ema_flux_slow = self._ema(ema_flux_slow, flux, cfg.FLUX_SLOW_ALPHA)

            if i == 0:
                low_fast = low_band_energy
                low_slow = low_band_energy
            else:
                low_fast = self._ema(low_fast, low_band_energy, cfg.LOW_FAST_ALPHA)
                low_slow = self._ema(low_slow, low_band_energy, cfg.LOW_SLOW_ALPHA)

            beat_norm = float(np.clip((ema_rms_fast - ema_rms_slow) * 12.0, 0.0, 1.0))
            transition_norm = float(
                np.clip((ema_flux_fast - ema_flux_slow) * 16.0, 0.0, 1.0)
            )
            kick_strength = float(
                np.clip((low_fast - low_slow) * cfg.KICK_GAIN, 0.0, 1.0)
            )

            # Disparo do kick por cruzamento de limiar + cooldown
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

            # Rampa inicial para evitar fade exagerado no começo da música
            startup_factor = float(np.clip((i + 1) / startup_frames, 0.0, 1.0))
            startup_gain = 0.45 + 0.55 * startup_factor

            # Janela de kick segue ativa para brilho, sem acelerar sequência do meio.
            if kick_window_left > 0:
                kick_window_left -= 1

            # Movimento do meio fixo no BPM detectado.
            phase_accumulator += base_speed
            while phase_accumulator >= 1.0:
                phase_accumulator -= 1.0
                prev_active_mid = active_mid
                prev_active_step = active_step
                sequence_step_idx += 1

                if sequence_step_idx >= len(current_sequence):
                    sequence_step_idx = 0
                    sequence_loop_count += 1

                    if sequence_loop_count >= cfg.MIDDLE_SEQ_SWITCH_REPEATS:
                        sequence_idx = (sequence_idx + 1) % len(middle_sequences)
                        current_sequence = middle_sequences[sequence_idx]
                        sequence_loop_count = 0
                        candidate_step_idx = min(
                            range(len(current_sequence)),
                            key=lambda idx: abs(
                                self._sequence_step_focus(current_sequence[idx])
                                - prev_active_mid
                            ),
                        )
                        # Evita repetir exatamente o mesmo estado na troca (sensação de delay).
                        if (
                            len(current_sequence) > 1
                            and self._sequence_step_equals(
                                current_sequence[candidate_step_idx], prev_active_step
                            )
                        ):
                            candidate_step_idx = (candidate_step_idx + 1) % len(current_sequence)
                        sequence_step_idx = candidate_step_idx

                active_step = current_sequence[sequence_step_idx]
                active_mid = int(round(self._sequence_step_focus(active_step)))

            # Meio com fade quase nulo
            middle_levels *= cfg.MIDDLE_FADE_DECAY

            # Intensidade do LED ativo do meio
            active_pwm = (
                cfg.ACTIVE_MIN_PWM
                + (cfg.ACTIVE_MAX_PWM - cfg.ACTIVE_MIN_PWM) * norm_energy
                + cfg.BEAT_BOOST_PWM * beat_norm
            )
            if beat_norm > cfg.HEAVY_BEAT_THRESHOLD:
                active_pwm += 25.0 * (
                    beat_norm - cfg.HEAVY_BEAT_THRESHOLD
                ) / (1.0 - cfg.HEAVY_BEAT_THRESHOLD)
            active_pwm += cfg.MIDDLE_KICK_BRIGHT_GAIN * kick_strength
            if kick_strength > 0.7:
                active_pwm += cfg.MIDDLE_STRONG_KICK_BRIGHT_GAIN * (
                    (kick_strength - 0.7) / 0.3
                )
            if kick_window_left > 0:
                active_pwm += cfg.MIDDLE_KICK_WINDOW_BRIGHT_GAIN * (
                    0.4 + 0.6 * kick_strength
                )
            active_pwm = float(
                np.clip(
                    active_pwm * startup_gain, cfg.ACTIVE_MIN_PWM * 0.45, cfg.ACTIVE_MAX_PWM
                )
            )

            for mid_idx in active_step:
                middle_levels[mid_idx] = max(middle_levels[mid_idx], active_pwm)

            # Pontas: fade alto e pulso da bateria (sem passar)
            low_ratio = low_band_energy / (high_band_energy + 1e-9)
            low_attack = max(0.0, low_norm - prev_low_norm)
            edge_drive = float(
                np.clip(
                    (max(kick_strength, low_norm) - cfg.EDGE_BEAT_THRESHOLD)
                    / (1.0 - cfg.EDGE_BEAT_THRESHOLD),
                    0.0,
                    1.0,
                )
            )
            low_dominance = float(
                np.clip(
                    (low_ratio - 0.3) / 1.2,
                    0.0,
                    1.0,
                )
            )
            edge_drive *= (0.35 + 0.65 * low_dominance)
            edge_decay = (
                cfg.EDGE_FADE_DECAY_BASE
                + (cfg.EDGE_FADE_DECAY_HEAVY - cfg.EDGE_FADE_DECAY_BASE) * edge_drive
            )
            edge_levels *= edge_decay

            # Gate de ruído com histerese para evitar piscadas falsas nas pontas.
            edge_gate_metric = max(kick_strength, low_attack * 1.6)
            if edge_gate_open:
                if edge_gate_metric >= cfg.EDGE_GATE_OPEN_THRESHOLD:
                    edge_gate_hold = cfg.EDGE_GATE_HOLD_FRAMES
                elif edge_gate_hold > 0:
                    edge_gate_hold -= 1
                elif (
                    edge_gate_metric < cfg.EDGE_GATE_CLOSE_THRESHOLD
                    or low_dominance < (cfg.EDGE_GATE_MIN_LOW_DOMINANCE * 0.8)
                ):
                    edge_gate_open = False
            elif (
                edge_gate_metric >= cfg.EDGE_GATE_OPEN_THRESHOLD
                and low_dominance >= cfg.EDGE_GATE_MIN_LOW_DOMINANCE
            ):
                edge_gate_open = True
                edge_gate_hold = cfg.EDGE_GATE_HOLD_FRAMES

            if edge_gate_open:
                edge_pulse = (
                    cfg.EDGE_PULSE_GAIN * max(0.0, (low_attack * 1.8) - cfg.EDGE_ATTACK_THRESHOLD)
                    + cfg.EDGE_HEAVY_BONUS_GAIN
                    * max(0.0, max(kick_strength, low_norm) - 0.55)
                )
                if kick_window_left > 0:
                    edge_pulse += cfg.EDGE_KICK_WINDOW_BOOST * (
                        0.4 + 0.6 * max(kick_strength, low_norm)
                    )
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
                    middle_levels[mid_idx] = min(
                        255.0,
                        middle_levels[mid_idx] + cfg.MIDDLE_EDGE_SYNC_GAIN * sync_strength,
                    )
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

            # Mode 1: snap. Fade visual é controlado no Python.
            led_patterns.append((1, pwm_int))
            dominant_leds.append(int(middle_leds[active_mid]))

            if (i + 1) % 100 == 0 or i == n_frames - 1:
                progress = (i + 1) / n_frames * 100
                print(f"   Progresso: {progress:.1f}%", end="\r")

        playback_source = audio_file
        if self._is_url(audio_file):
            playback_source = self._create_temp_wav_from_audio(y, sr)
            print("   Playback local cache: pronto")

        print("\n✓ Processamento concluído!")
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
            self._play_with_ffplay(
                audio_source,
                "✗ ffplay não encontrado para reproduzir stream (instale ffmpeg).",
            )
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
            self._play_with_ffplay(
                audio_source,
                "✗ ffplay não encontrado para fallback de reprodução local (instale ffmpeg).",
            )

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
                        f"⏱️  {elapsed:.1f}s / {duration:.1f}s | PASSANDO-6 | PWM: {list(map(int, led_values))}",
                        end="\r",
                    )

            print("\n\n✓ Reprodução finalizada!")
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
            print("✓ Desconectado do Arduino")


def select_audio_file():
    """Abre o seletor de arquivo do PySide6 e retorna o caminho escolhido."""
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
    except Exception:
        print("✗ PySide6 não encontrado. Instale com: pip install pyside6")
        return None

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True

    filters = (
        "Arquivos de áudio (*.mp3 *.wav *.flac *.ogg *.m4a *.aac);;"
        "Todos os arquivos (*)"
    )
    selected_file, _ = QFileDialog.getOpenFileName(
        None,
        "Selecione o arquivo de áudio",
        "",
        filters,
    )

    if owns_app:
        app.quit()

    return selected_file or None


def select_audio_source():
    """Seleciona fonte de áudio: arquivo local ou busca no YouTube."""
    try:
        from PySide6.QtWidgets import QApplication, QInputDialog
    except Exception:
        print("✗ PySide6 não encontrado. Instale com: pip install pyside6")
        return None

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True

    options = ["Arquivo local", "Buscar no YouTube (yt-dlp)"]
    choice, ok = QInputDialog.getItem(
        None,
        "Fonte do áudio",
        "Escolha de onde carregar o áudio:",
        options,
        0,
        False,
    )
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

    query, ok = QInputDialog.getText(
        None,
        "Buscar no YouTube",
        "Digite o nome da música ou cole a URL:",
    )
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
        raise RuntimeError(
            "Biblioteca yt-dlp não encontrada. Instale com: pip install yt-dlp"
        ) from exc

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
            f
            for f in formats
            if f.get("url")
            and f.get("acodec") not in (None, "none")
            and f.get("vcodec") in (None, "none")
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
    print("     🎵 LED MUSIC SYNC🎵")
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
            print(f"✓ Stream pronto: {title}")
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
