from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
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


# ---------------------------------------------------------------------------
# Tipos de efeito de transição
# ---------------------------------------------------------------------------

class TransitionEffectType(Enum):
    SWEEP         = auto()   # esquerda→direita           – kick forte / drop
    REVERSE_SWEEP = auto()   # direita→esquerda           – entrada de seção
    STROBE        = auto()   # flash rápido all-on/off    – beat muito pesado
    CASCADE       = auto()   # centro→bordas              – som brilhante / hi
    GLITCH        = auto()   # flickering digital         – ruído / bitcrushed
    BREATHE       = auto()   # fade-in/fade-out coletivo  – breakdown / suave
    PING_PONG     = auto()   # vai e vem                  – beat médio
    SPLIT         = auto()   # bordas→centro (anti-cascade) – queda de graves
    SPARKLE       = auto()   # flashes aleatórios rápidos  – hi-hats / percussão alta
    BURST         = auto()   # todos acendem e decaem      – impacto súbito


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EffectConfig:
    FPS: int = 30
    NUM_LEDS: int = 6

    STEP_FRAMES_MIN: int = 2
    STEP_FRAMES_MAX: int = 9
    BPM_STEP_MULT: float = 0.50
    MIDDLE_SEQ_SWITCH_REPEATS: int = 4
    MIDDLE_SEQUENCES: tuple = (
        ((0,), (2,), (1,), (3,)),
        ((3,), (1,), (2,), (0,)),
        ((0, 2), (1, 3), (0, 2), (1, 3)),
        ((0, 1), (2, 3), (0, 1), (2, 3)),
    )

    LOW_BAND_MIN_HZ: float = 30.0
    LOW_BAND_MAX_HZ: float = 120.0
    KICK_GAIN: float = 18.0
    KICK_EVENT_THRESHOLD: float = 0.15
    KICK_RISE_MIN: float = 0.06
    KICK_WINDOW_MS: int = 100
    KICK_COOLDOWN_FRAMES: int = 2
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

    STARTUP_RAMP_SECONDS: float = 2.0

    # ------------------------------------------------------------------ #
    # Segmentação estrutural                                               #
    # ------------------------------------------------------------------ #

    # Janela de análise de segmento (segundos)
    SEG_HOP_SECONDS: float = 1.25

    # Número de segmentos alvo (ajustado automaticamente para músicas curtas)
    SEG_K_SEGMENTS: int = 40

    # Distância mínima entre transições (segundos)
    SEG_MIN_GAP_SECONDS: float = 2.0

    # Ignora fronteiras nos primeiros N segundos
    SEG_IGNORE_START_SECONDS: float = 2.5

    # Limiar mínimo de novidade para aceitar uma fronteira (0-1)
    SEG_NOVELTY_THRESHOLD: float = 0.18

    # ------------------------------------------------------------------ #
    # Features de segmento para escolha do efeito                         #
    # ------------------------------------------------------------------ #

    NOISE_FLATNESS_THRESHOLD: float = 0.55
    BRIGHT_CENTROID_THRESHOLD: float = 0.28

    # ------------------------------------------------------------------ #
    # Efeito de transição                                                  #
    # ------------------------------------------------------------------ #

    TRANS_MIN_DURATION_S: float = 0.45
    TRANS_MAX_DURATION_S: float = 1.80
    TRANS_BLEND_FRAMES: int = 3


# ---------------------------------------------------------------------------
# Helpers espectrais
# ---------------------------------------------------------------------------

def _spectral_flatness(spectrum: np.ndarray) -> float:
    s = np.maximum(spectrum, 1e-12)
    geo   = float(np.exp(np.mean(np.log(s))))
    arith = float(np.mean(s))
    return float(np.clip(geo / (arith + 1e-12), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Evento de transição pré-calculado (partitura)
# ---------------------------------------------------------------------------

@dataclass
class TransitionEvent:
    frame_start:     int
    frame_end:       int
    effect_type:     TransitionEffectType
    pwm_brightness:  float
    beat_norm_after: float = 0.0
    kick_mean_after: float = 0.0


# ---------------------------------------------------------------------------
# Controller principal
# ---------------------------------------------------------------------------

class MusicLEDController:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200, fps=None, config=None):
        self.port      = port
        self.baudrate  = baudrate
        self.config    = config or EffectConfig()
        self.fps       = self.config.FPS if fps is None else int(fps)
        self.ser       = None
        self.playing   = False
        self._temp_playback_files = set()

    # ------------------------------------------------------------------ #
    # Serial                                                               #
    # ------------------------------------------------------------------ #

    def connect_arduino(self):
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

    def send_led_command(self, led_values, mode=1):
        if self.ser and self.ser.is_open:
            num_leds = self.config.NUM_LEDS
            values   = np.zeros(num_leds, dtype=np.int32)
            src      = np.asarray(led_values, dtype=np.int32)
            n        = min(num_leds, src.shape[0])
            values[:n] = src[:n]
            payload  = ",".join(str(int(v)) for v in values)
            self.ser.write(f"P,{int(mode)},{payload}\n".encode())

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.send_led_command([0] * self.config.NUM_LEDS, mode=1)
            self.ser.close()
            print("✓ Desconectado do Arduino")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Carregamento de áudio                                                #
    # ------------------------------------------------------------------ #

    def _load_audio_data(self, audio_source, target_sr=22050):
        if self._is_url(audio_source):
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                raise RuntimeError("ffmpeg não encontrado.")
            cmd = [
                ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", audio_source, "-vn", "-ac", "1", "-ar", str(target_sr),
                "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode != 0:
                err = result.stderr.decode(errors="ignore").strip()
                raise RuntimeError(f"falha ao decodificar stream: {err}")
            pcm = np.frombuffer(result.stdout, dtype=np.int16)
            if pcm.size == 0:
                raise RuntimeError("stream sem áudio decodificado")
            y = pcm.astype(np.float32) / 32768.0
            return y, target_sr, float(len(y)) / float(target_sr)

        y, sr = librosa.load(audio_source, sr=target_sr, mono=True)
        return y, sr, librosa.get_duration(y=y, sr=sr)

    def _create_temp_wav_from_audio(self, y, sr):
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

    # ------------------------------------------------------------------ #
    # FASE 1 — Análise estrutural completa da música                       #
    # ------------------------------------------------------------------ #

    def _analyze_structure(self, y: np.ndarray, sr: int) -> list:
        """
        Analisa a música completa e retorna lista de TransitionEvent
        (a partitura de transições) com frames LED exatos.

        Estratégia:
        - HPSS para separar harmônico/percussivo
        - MFCC + cromatograma em janelas de SEG_HOP_SECONDS
        - Segmentação aglomerativa para encontrar fronteiras de seção
        - Curva de novidade para filtrar fronteiras fracas
        - Gap mínimo entre transições para evitar excesso
        - Features do "antes" e "depois" de cada fronteira para
          escolher o efeito e sua duração
        """
        cfg = self.config
        print("   [Análise] Separando harmônico/percussivo...")
        y_harm, y_perc = librosa.effects.hpss(y, margin=3.0)

        hop_samples = int(cfg.SEG_HOP_SECONDS * sr)

        print("   [Análise] Calculando features de segmento...")

        mfcc   = librosa.feature.mfcc(y=y_harm, sr=sr, n_mfcc=13, hop_length=hop_samples)
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_samples)

        rms_total = librosa.feature.rms(
            y=y, frame_length=hop_samples * 2, hop_length=hop_samples
        )[0]
        rms_perc = librosa.feature.rms(
            y=y_perc, frame_length=hop_samples * 2, hop_length=hop_samples
        )[0]

        spec_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop_samples
        )[0]
        spec_centroid = spec_centroid / (sr / 2.0)

        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_samples)[0]

        # Energia de baixas frequências por janela
        S         = np.abs(librosa.stft(y, hop_length=hop_samples))
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=(S.shape[0] - 1) * 2)
        low_mask  = (freq_bins >= cfg.LOW_BAND_MIN_HZ) & (freq_bins <= cfg.LOW_BAND_MAX_HZ)

        # Alinha comprimentos
        n_win = min(
            mfcc.shape[1], chroma.shape[1], len(rms_total), len(rms_perc),
            len(spec_centroid), len(flatness),
        )
        mfcc          = mfcc[:, :n_win]
        chroma        = chroma[:, :n_win]
        rms_total     = rms_total[:n_win]
        rms_perc      = rms_perc[:n_win]
        spec_centroid = spec_centroid[:n_win]
        flatness      = flatness[:n_win]

        S_trim = S[:, :n_win] if S.shape[1] >= n_win else np.pad(
            S, ((0, 0), (0, n_win - S.shape[1]))
        )
        low_energy_per_win = (
            np.mean(S_trim[low_mask, :], axis=0) if np.any(low_mask)
            else np.zeros(n_win)
        )
        low_energy_per_win = low_energy_per_win[:n_win]

        # Matriz de features para segmentação
        mfcc_norm   = librosa.util.normalize(mfcc, axis=1)
        chroma_norm = librosa.util.normalize(chroma, axis=0)
        feat_matrix = np.vstack([mfcc_norm, chroma_norm])  # (25, n_win)

        print("   [Análise] Detectando fronteiras de seção...")

        k = min(cfg.SEG_K_SEGMENTS, max(2, n_win // 4))
        try:
            bound_frames_seg = librosa.segment.agglomerative(feat_matrix, k=k)
        except Exception:
            bound_frames_seg = np.array([0, n_win - 1])

        bound_times = librosa.frames_to_time(
            bound_frames_seg, sr=sr, hop_length=hop_samples
        )

        # Curva de novidade simples (derivada da feature matrix)
        novelty_simple = np.zeros(n_win)
        for w in range(1, n_win):
            diff = feat_matrix[:, w] - feat_matrix[:, w - 1]
            novelty_simple[w] = float(np.linalg.norm(diff))
        novelty_simple = novelty_simple / (novelty_simple.max() + 1e-9)

        # Filtra fronteiras
        ignore_start = cfg.SEG_IGNORE_START_SECONDS
        min_gap      = cfg.SEG_MIN_GAP_SECONDS

        valid_boundaries = []
        last_accepted_time = -999.0

        for b_time in bound_times:
            if b_time < ignore_start:
                continue

            b_win = int(np.clip(int(b_time / cfg.SEG_HOP_SECONDS), 0, n_win - 1))
            w_lo  = max(0, b_win - 2)
            w_hi  = min(n_win - 1, b_win + 2)
            local_novelty = float(np.max(novelty_simple[w_lo:w_hi + 1]))

            if local_novelty < cfg.SEG_NOVELTY_THRESHOLD:
                continue

            if (b_time - last_accepted_time) < min_gap:
                continue

            valid_boundaries.append(b_time)
            last_accepted_time = b_time

        print(f"   [Análise] {len(valid_boundaries)} fronteiras válidas")

        # ---- Features antes/depois de cada fronteira -------------------- #
        p90_perc = float(np.percentile(rms_perc, 90)) + 1e-9
        p90_rms  = float(np.percentile(rms_total, 90)) + 1e-9
        p95_low  = float(np.percentile(low_energy_per_win, 95)) + 1e-9

        transitions = []
        wins_ctx = max(1, int(2.0 / cfg.SEG_HOP_SECONDS))

        for b_time in valid_boundaries:
            b_win = int(np.clip(int(b_time / cfg.SEG_HOP_SECONDS), 0, n_win - 1))

            w_bef_lo = max(0, b_win - wins_ctx)
            w_bef_hi = max(0, b_win - 1)
            w_aft_lo = min(n_win - 1, b_win)
            w_aft_hi = min(n_win - 1, b_win + wins_ctx)

            def wmean(arr, lo, hi):
                lo = int(np.clip(lo, 0, len(arr) - 1))
                hi = int(np.clip(hi, lo, len(arr) - 1))
                return float(np.mean(arr[lo:hi + 1]))

            rms_bef  = wmean(rms_total, w_bef_lo, w_bef_hi)
            rms_aft  = wmean(rms_total, w_aft_lo, w_aft_hi)
            perc_aft = wmean(rms_perc, w_aft_lo, w_aft_hi)
            flat_aft = wmean(flatness, w_aft_lo, w_aft_hi)
            cent_aft = wmean(spec_centroid, w_aft_lo, w_aft_hi)
            low_aft  = wmean(low_energy_per_win, w_aft_lo, w_aft_hi)

            beat_norm_after = float(np.clip(perc_aft / p90_perc, 0.0, 1.0))
            kick_mean_after = float(np.clip(low_aft / p95_low, 0.0, 1.0))
            energy_ratio    = rms_aft / (rms_bef + 1e-9)
            energy_norm_aft = float(np.clip(rms_aft / p90_rms, 0.0, 1.0))

            eff_type = self._pick_effect_type(
                beat_norm       = beat_norm_after,
                kick_strength   = kick_mean_after,
                flatness        = flat_aft,
                centroid_norm   = cent_aft,
                energy_norm     = energy_norm_aft,
                energy_dropping = energy_ratio < 0.70,
                energy_rising   = energy_ratio > 1.35,
                cfg             = cfg,
            )

            dur_s = cfg.TRANS_MIN_DURATION_S + (
                cfg.TRANS_MAX_DURATION_S - cfg.TRANS_MIN_DURATION_S
            ) * (1.0 - min(beat_norm_after, 1.0))
            dur_s = float(np.clip(dur_s, cfg.TRANS_MIN_DURATION_S, cfg.TRANS_MAX_DURATION_S))

            brightness = float(np.clip(
                cfg.ACTIVE_MIN_PWM
                + (cfg.ACTIVE_MAX_PWM - cfg.ACTIVE_MIN_PWM) * energy_norm_aft
                + 40.0 * beat_norm_after
                + 35.0 * kick_mean_after,
                90.0, 235.0,
            ))

            frame_start = int(b_time * self.fps)
            frame_end   = frame_start + max(2, int(dur_s * self.fps))

            transitions.append(TransitionEvent(
                frame_start     = frame_start,
                frame_end       = frame_end,
                effect_type     = eff_type,
                pwm_brightness  = brightness,
                beat_norm_after = beat_norm_after,
                kick_mean_after = kick_mean_after,
            ))

            print(
                f"     t={b_time:6.2f}s → {eff_type.name:<14}"
                f" beat={beat_norm_after:.2f} kick={kick_mean_after:.2f}"
                f" flat={flat_aft:.2f} dur={dur_s:.2f}s"
            )

        return transitions

    # ------------------------------------------------------------------ #
    # Escolha do efeito                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pick_effect_type(
        beat_norm: float,
        kick_strength: float,
        flatness: float,
        centroid_norm: float,
        energy_norm: float,
        energy_dropping: bool,
        energy_rising: bool,
        cfg: EffectConfig,
    ) -> TransitionEffectType:

        if flatness > cfg.NOISE_FLATNESS_THRESHOLD:
            return TransitionEffectType.GLITCH

        # Impacto súbito (energia sobe muito + kick alto)
        if energy_rising and kick_strength > 0.70:
            return TransitionEffectType.BURST

        if energy_rising and kick_strength > 0.55:
            return TransitionEffectType.STROBE

        if beat_norm > 0.78 and kick_strength > 0.50:
            return TransitionEffectType.SWEEP

        if kick_strength > 0.55 or (energy_rising and beat_norm > 0.45):
            return TransitionEffectType.REVERSE_SWEEP

        # Percussão alta / hi-hats sem muita energia de graves
        if centroid_norm > cfg.BRIGHT_CENTROID_THRESHOLD and kick_strength < 0.35:
            return TransitionEffectType.SPARKLE

        if centroid_norm > cfg.BRIGHT_CENTROID_THRESHOLD:
            return TransitionEffectType.CASCADE

        # Graves caindo → bordas para centro
        if energy_dropping and kick_strength > 0.25:
            return TransitionEffectType.SPLIT

        if energy_dropping or energy_norm < 0.28:
            return TransitionEffectType.BREATHE

        if beat_norm > 0.45:
            return TransitionEffectType.PING_PONG

        if beat_norm > 0.20:
            return TransitionEffectType.SWEEP

        return TransitionEffectType.REVERSE_SWEEP

    # ------------------------------------------------------------------ #
    # Renderização de um frame de efeito                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _render_transition_frame(
        effect_type: TransitionEffectType,
        progress: float,
        beat_norm: float,
        kick_strength: float,
        pwm_brightness: float,
    ) -> np.ndarray:
        """Retorna 4 floats (middle_levels[0..3])."""
        levels = np.zeros(4, dtype=np.float32)
        t = float(np.clip(progress, 0.0, 1.0))
        B = float(np.clip(pwm_brightness, 0.0, 255.0))

        if effect_type == TransitionEffectType.SWEEP:
            pos = t * 5.0
            for idx in range(4):
                levels[idx] = max(0.0, 1.0 - abs(idx - pos) * 0.80) * B

        elif effect_type == TransitionEffectType.REVERSE_SWEEP:
            pos = (1.0 - t) * 3.0
            for idx in range(4):
                levels[idx] = max(0.0, 1.0 - abs(idx - pos) * 0.80) * B

        elif effect_type == TransitionEffectType.STROBE:
            hz   = 8.0 + beat_norm * 10.0 + kick_strength * 4.0
            on   = (int(t * hz * 2.0) % 2) == 0
            fade = 1.0 if t < 0.75 else (1.0 - (t - 0.75) / 0.25)
            levels[:] = (B if on else 0.0) * fade

        elif effect_type == TransitionEffectType.CASCADE:
            c_in = float(np.clip(t * 3.0, 0.0, 1.0))
            e_in = float(np.clip(t * 3.0 - 0.7, 0.0, 1.0))
            fade = 1.0 if t < 0.55 else float(
                np.clip(1.0 - (t - 0.55) / 0.45, 0.0, 1.0)
            )
            levels[1] = c_in * B * fade
            levels[2] = c_in * B * fade
            levels[0] = e_in * B * fade
            levels[3] = e_in * B * fade

        elif effect_type == TransitionEffectType.GLITCH:
            seed    = int(t * 60.0)
            rng     = np.random.RandomState(seed + 7919)
            on_mask = rng.random(4) > 0.38
            amp     = rng.uniform(0.35, 1.0, 4)
            levels  = on_mask.astype(np.float32) * amp.astype(np.float32) * B
            if t > 0.70:
                levels *= (1.0 - (t - 0.70) / 0.30)

        elif effect_type == TransitionEffectType.BREATHE:
            levels[:] = float(np.sin(t * np.pi)) * B

        elif effect_type == TransitionEffectType.PING_PONG:
            raw = (t * 3.0) % 2.0
            pos = (raw if raw <= 1.0 else 2.0 - raw) * 3.0
            for idx in range(4):
                levels[idx] = max(0.0, 1.0 - abs(idx - pos) * 0.75) * B
            if t > 0.75:
                levels *= (1.0 - (t - 0.75) / 0.25)

        elif effect_type == TransitionEffectType.SPLIT:
            # Bordas (0,3) acendem primeiro, depois convergem para o centro (1,2)
            e_in = float(np.clip(t * 3.0, 0.0, 1.0))
            c_in = float(np.clip(t * 3.0 - 0.7, 0.0, 1.0))
            fade = 1.0 if t < 0.55 else float(
                np.clip(1.0 - (t - 0.55) / 0.45, 0.0, 1.0)
            )
            levels[0] = e_in * B * fade
            levels[3] = e_in * B * fade
            levels[1] = c_in * B * fade
            levels[2] = c_in * B * fade

        elif effect_type == TransitionEffectType.SPARKLE:
            # Flashes rápidos individuais semi-aleatórios — sensação de faísca
            hz   = 12.0 + beat_norm * 8.0
            tick = int(t * hz * 4.0)
            rng  = np.random.RandomState(tick + 3571)
            # Só 1 ou 2 leds acesos por vez, trocando rapidamente
            n_on = rng.choice([1, 1, 2])
            on_idx = rng.choice(4, size=n_on, replace=False)
            amp  = rng.uniform(0.6, 1.0, n_on)
            for k, idx in enumerate(on_idx):
                levels[idx] = amp[k] * B
            # Fade out suave no final
            if t > 0.65:
                levels *= (1.0 - (t - 0.65) / 0.35)

        elif effect_type == TransitionEffectType.BURST:
            # Todos acendem instantaneamente e decaem em curva exponencial
            decay = float(np.exp(-t * 5.5))
            levels[:] = decay * B

        return np.clip(levels, 0.0, 255.0)

    # ------------------------------------------------------------------ #
    # FASE 2 — Gera a partitura completa de LEDs                          #
    # ------------------------------------------------------------------ #

    def _generate_led_score(
        self,
        y: np.ndarray,
        sr: int,
        n_frames: int,
        transitions: list,
        tempo_bpm: float,
    ) -> tuple:
        """
        Gera todos os frames PWM de uma vez (a partitura).
        As transições são eventos agendados que sobrepõem a sequência normal.
        """
        cfg = self.config

        samples_per_frame = int(sr / self.fps)
        bytes_per_frame   = samples_per_frame * 2
        pcm       = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        pcm_bytes = pcm.tobytes()

        base_step_frames = int(np.clip(
            round((60.0 / tempo_bpm) * self.fps * cfg.BPM_STEP_MULT),
            cfg.STEP_FRAMES_MIN, cfg.STEP_FRAMES_MAX,
        ))
        base_speed = 1.0 / float(base_step_frames)

        # Índice na lista de transições
        trans_index  = 0
        n_trans      = len(transitions)
        active_trans = None   # TransitionEvent | None

        # Estado da sequência do meio
        middle_leds         = np.array([1, 2, 3, 4], dtype=np.int32)
        middle_sequences    = cfg.MIDDLE_SEQUENCES
        sequence_idx        = 0
        current_sequence    = middle_sequences[sequence_idx]
        sequence_step_idx   = 0
        sequence_loop_count = 0
        active_step         = current_sequence[sequence_step_idx]
        active_mid          = int(round(self._sequence_step_focus(active_step)))
        phase_accumulator   = 0.0

        middle_levels = np.zeros(4, dtype=np.float32)
        edge_levels   = np.zeros(2, dtype=np.float32)

        prev_spectrum      = None
        ema_rms_fast       = 0.0
        ema_rms_slow       = 0.0
        ema_flux_fast      = 0.0
        ema_flux_slow      = 0.0

        energy_window   = deque(maxlen=max(4, int(self.fps * 2)))
        low_band_window = deque(maxlen=max(4, int(self.fps * 1.2)))

        prev_kick_strength = 0.0
        prev_low_norm      = 0.0
        low_fast = 0.0
        low_slow = 0.0
        kick_window_left   = 0
        kick_cooldown_left = 0
        edge_gate_open = False
        edge_gate_hold = 0

        startup_frames     = max(1, int(cfg.STARTUP_RAMP_SECONDS * self.fps))
        kick_window_frames = max(1, int((cfg.KICK_WINDOW_MS / 1000.0) * self.fps))

        led_patterns  = []
        dominant_leds = []

        for i in range(n_frames):
            # ---- áudio deste frame ----------------------------------- #
            start_b = i * bytes_per_frame
            end_b   = start_b + bytes_per_frame
            chunk   = pcm_bytes[start_b:end_b]

            frame = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            if len(frame) < samples_per_frame:
                frame = np.pad(frame, (0, samples_per_frame - len(frame)))
            frame = frame / 32768.0

            rms      = float(np.sqrt(np.mean(frame * frame) + 1e-12))
            spectrum = np.abs(np.fft.rfft(frame))
            freqs    = np.fft.rfftfreq(len(frame), 1.0 / sr)

            low_mask  = (freqs >= cfg.LOW_BAND_MIN_HZ) & (freqs <= cfg.LOW_BAND_MAX_HZ)
            high_mask = (freqs >= 2000.0) & (freqs <= 8000.0)
            low_band_energy  = float(np.mean(spectrum[low_mask]))  if np.any(low_mask)  else 0.0
            high_band_energy = float(np.mean(spectrum[high_mask])) if np.any(high_mask) else 0.0

            flux = 0.0 if prev_spectrum is None else float(
                np.mean(np.maximum(spectrum - prev_spectrum, 0.0))
            )
            prev_spectrum = spectrum

            if i == 0:
                ema_rms_fast = rms;   ema_rms_slow  = rms
                ema_flux_fast = flux; ema_flux_slow  = flux
                low_fast = low_band_energy; low_slow = low_band_energy
            else:
                ema_rms_fast  = self._ema(ema_rms_fast,  rms,  cfg.RMS_FAST_ALPHA)
                ema_rms_slow  = self._ema(ema_rms_slow,  rms,  cfg.RMS_SLOW_ALPHA)
                ema_flux_fast = self._ema(ema_flux_fast, flux, cfg.FLUX_FAST_ALPHA)
                ema_flux_slow = self._ema(ema_flux_slow, flux, cfg.FLUX_SLOW_ALPHA)
                low_fast = self._ema(low_fast, low_band_energy, cfg.LOW_FAST_ALPHA)
                low_slow = self._ema(low_slow, low_band_energy, cfg.LOW_SLOW_ALPHA)

            beat_norm     = float(np.clip((ema_rms_fast - ema_rms_slow) * 12.0, 0.0, 1.0))
            kick_strength = float(np.clip((low_fast - low_slow) * cfg.KICK_GAIN, 0.0, 1.0))

            if kick_cooldown_left > 0:
                kick_cooldown_left -= 1
            if (
                kick_cooldown_left == 0
                and prev_kick_strength < cfg.KICK_EVENT_THRESHOLD
                and kick_strength >= cfg.KICK_EVENT_THRESHOLD
                and (kick_strength - prev_kick_strength) >= cfg.KICK_RISE_MIN
            ):
                kick_window_left   = kick_window_frames
                kick_cooldown_left = cfg.KICK_COOLDOWN_FRAMES

            energy_window.append(rms)
            p90_energy  = float(np.percentile(np.array(energy_window), 90))
            norm_energy = float(np.clip(rms / (p90_energy + 1e-9), 0.0, 1.0))

            low_band_window.append(low_band_energy)
            p95_low  = float(np.percentile(np.array(low_band_window), 95))
            low_norm = float(np.clip(low_band_energy / (p95_low + 1e-9), 0.0, 1.0))

            startup_factor = float(np.clip((i + 1) / startup_frames, 0.0, 1.0))
            startup_gain   = 0.45 + 0.55 * startup_factor

            if kick_window_left > 0:
                kick_window_left -= 1

            # ---- avança sequência do meio (sempre, mesmo em transição) #
            phase_accumulator += base_speed
            while phase_accumulator >= 1.0:
                phase_accumulator -= 1.0
                prev_active_mid  = active_mid
                prev_active_step = active_step
                sequence_step_idx += 1

                if sequence_step_idx >= len(current_sequence):
                    sequence_step_idx   = 0
                    sequence_loop_count += 1

                    if sequence_loop_count >= cfg.MIDDLE_SEQ_SWITCH_REPEATS:
                        sequence_idx        = (sequence_idx + 1) % len(middle_sequences)
                        current_sequence    = middle_sequences[sequence_idx]
                        sequence_loop_count = 0
                        candidate = min(
                            range(len(current_sequence)),
                            key=lambda idx: abs(
                                self._sequence_step_focus(current_sequence[idx])
                                - prev_active_mid
                            ),
                        )
                        if (
                            len(current_sequence) > 1
                            and self._sequence_step_equals(
                                current_sequence[candidate], prev_active_step
                            )
                        ):
                            candidate = (candidate + 1) % len(current_sequence)
                        sequence_step_idx = candidate

                active_step = current_sequence[sequence_step_idx]
                active_mid  = int(round(self._sequence_step_focus(active_step)))

            # =========================================================== #
            # PARTITURA: verifica início de transição agendada             #
            # =========================================================== #
            if active_trans is None and trans_index < n_trans:
                if i >= transitions[trans_index].frame_start:
                    active_trans = transitions[trans_index]
                    trans_index += 1

            # ---- LEDs do meio ---------------------------------------- #
            if active_trans is not None:
                local_frame  = i - active_trans.frame_start
                total_frames = active_trans.frame_end - active_trans.frame_start
                progress     = float(np.clip(
                    local_frame / max(1, total_frames - 1), 0.0, 1.0
                ))

                effect_levels = self._render_transition_frame(
                    effect_type    = active_trans.effect_type,
                    progress       = progress,
                    beat_norm      = active_trans.beat_norm_after,
                    kick_strength  = active_trans.kick_mean_after,
                    pwm_brightness = active_trans.pwm_brightness * startup_gain,
                )

                bf = cfg.TRANS_BLEND_FRAMES
                if local_frame < bf:
                    alpha = float(local_frame) / float(bf)
                elif local_frame > total_frames - bf:
                    alpha = float(total_frames - local_frame) / float(max(1, bf))
                else:
                    alpha = 1.0
                alpha = float(np.clip(alpha, 0.0, 1.0))

                middle_levels *= cfg.MIDDLE_FADE_DECAY
                middle_levels  = middle_levels * (1.0 - alpha) + effect_levels * alpha

                if i >= active_trans.frame_end:
                    active_trans  = None
                    middle_levels *= 0.55

            else:
                # ---- sequência normal -------------------------------- #
                middle_levels *= cfg.MIDDLE_FADE_DECAY

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
                active_pwm = float(np.clip(
                    active_pwm * startup_gain,
                    cfg.ACTIVE_MIN_PWM * 0.45,
                    cfg.ACTIVE_MAX_PWM,
                ))
                for mid_idx in active_step:
                    middle_levels[mid_idx] = max(middle_levels[mid_idx], active_pwm)

            # =========================================================== #
            # Pontas (inalteradas — já estão ótimas)                       #
            # =========================================================== #
            low_ratio    = low_band_energy / (high_band_energy + 1e-9)
            low_attack   = max(0.0, low_norm - prev_low_norm)
            edge_drive   = float(np.clip(
                (max(kick_strength, low_norm) - cfg.EDGE_BEAT_THRESHOLD)
                / (1.0 - cfg.EDGE_BEAT_THRESHOLD), 0.0, 1.0,
            ))
            low_dominance = float(np.clip((low_ratio - 0.3) / 1.2, 0.0, 1.0))
            edge_drive   *= (0.35 + 0.65 * low_dominance)
            edge_decay    = (
                cfg.EDGE_FADE_DECAY_BASE
                + (cfg.EDGE_FADE_DECAY_HEAVY - cfg.EDGE_FADE_DECAY_BASE) * edge_drive
            )
            edge_levels  *= edge_decay

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
                    cfg.EDGE_PULSE_GAIN
                    * max(0.0, (low_attack * 1.8) - cfg.EDGE_ATTACK_THRESHOLD)
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
                edge_pulse   = 0.0

            edge_pulse *= startup_gain
            if edge_pulse >= cfg.EDGE_SYNC_MIN_PWM:
                sync_strength = float(np.clip(edge_pulse / 255.0, 0.0, 1.0))
                if active_trans is None:
                    for mid_idx in active_step:
                        middle_levels[mid_idx] = min(
                            255.0,
                            middle_levels[mid_idx]
                            + cfg.MIDDLE_EDGE_SYNC_GAIN * sync_strength,
                        )
            if edge_pulse > 1.0:
                edge_levels[0] = max(edge_levels[0], edge_pulse)
                edge_levels[1] = max(edge_levels[1], edge_pulse)

            prev_kick_strength = kick_strength
            prev_low_norm      = low_norm

            # ---- monta frame ----------------------------------------- #
            pwm = np.zeros(cfg.NUM_LEDS, dtype=np.float32)
            pwm[middle_leds] = middle_levels
            pwm[0] = edge_levels[0]
            pwm[5] = edge_levels[1]
            pwm = np.clip(pwm, 0.0, 255.0)
            pwm[pwm < cfg.TAIL_CUTOFF] = 0.0
            pwm_int = pwm.astype(np.int32)

            led_patterns.append((1, pwm_int))
            dominant_leds.append(int(middle_leds[active_mid]))

            if (i + 1) % 150 == 0 or i == n_frames - 1:
                pct = (i + 1) / n_frames * 100.0
                print(f"   [Score] {pct:.1f}%", end="\r")

        print()
        return led_patterns, dominant_leds

    # ------------------------------------------------------------------ #
    # Ponto de entrada público                                             #
    # ------------------------------------------------------------------ #

    def process_audio_file(self, audio_file):
        cfg = self.config

        source_label = audio_file if not self._is_url(audio_file) else "YouTube stream"
        print(f"\n📀 Carregando áudio: {source_label}")

        y, sr, duration = self._load_audio_data(audio_file, target_sr=22050)
        print(f"   Duração: {duration:.2f}s | Sample Rate: {sr} Hz")

        n_frames = max(1, int(duration * self.fps))

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        try:
            tempo_bpm = float(np.asarray(tempo).item())
        except Exception:
            tempo_bpm = 120.0
        if not np.isfinite(tempo_bpm) or tempo_bpm <= 1.0:
            tempo_bpm = 120.0

        print(f"   BPM detectado: {tempo_bpm:.1f}")

        print("\n🔬 Fase 1 — Análise estrutural...")
        transitions = self._analyze_structure(y, sr)

        print(f"\n🎼 Fase 2 — Gerando partitura ({n_frames} frames @ {self.fps} FPS)...")
        led_patterns, dominant_leds = self._generate_led_score(
            y, sr, n_frames, transitions, tempo_bpm
        )
        print("   ✓ Partitura gerada!")

        # Telemetria rápida
        pwms = np.array([v for _, v in led_patterns], dtype=np.int32)
        sat  = np.mean(pwms >= 245, axis=0) * 100.0
        dark = np.mean(pwms <= 5,   axis=0) * 100.0
        num_leds = cfg.NUM_LEDS
        print("\n📊 Telemetria")
        print("   Saturação >=245 (%): " +
              ", ".join(f"L{i+1}:{sat[i]:.1f}" for i in range(num_leds)))
        print("   Escuro <=5 (%): " +
              ", ".join(f"L{i+1}:{dark[i]:.1f}" for i in range(num_leds)))

        playback_source = audio_file
        if self._is_url(audio_file):
            playback_source = self._create_temp_wav_from_audio(y, sr)

        return led_patterns, duration, playback_source

    # ------------------------------------------------------------------ #
    # Playback                                                             #
    # ------------------------------------------------------------------ #

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

    def play_audio_thread(self, audio_source):
        if self._is_url(audio_source):
            self._play_with_ffplay(
                audio_source, "✗ ffplay não encontrado para reproduzir stream."
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
            print(f"⚠️  pygame falhou ({exc}). Usando ffplay...")
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            self._play_with_ffplay(audio_source, "✗ ffplay não encontrado.")

    def sync_and_play(self, audio_source):
        led_patterns, duration, playback_source = self.process_audio_file(audio_source)

        if not self.ser or not self.ser.is_open:
            print("✗ Arduino não conectado!")
            self._cleanup_temp_playback_files()
            return

        print("\n🎵 Iniciando reprodução sincronizada...\n")
        print("Pressione Ctrl+C para parar\n")

        self.playing = True
        audio_thread = threading.Thread(
            target=self.play_audio_thread, args=(playback_source,)
        )
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
                sleep_time    = expected_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if i % self.fps == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"⏱️  {elapsed:.1f}s / {duration:.1f}s"
                        f" | PWM: {list(map(int, led_values))}",
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


# ---------------------------------------------------------------------------
# UI / entrada
# ---------------------------------------------------------------------------

def select_audio_file():
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
    except Exception:
        print("✗ PySide6 não encontrado. Instale com: pip install pyside6")
        return None
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)
    filters = (
        "Arquivos de áudio (*.mp3 *.wav *.flac *.ogg *.m4a *.aac);;"
        "Todos os arquivos (*)"
    )
    selected_file, _ = QFileDialog.getOpenFileName(
        None, "Selecione o arquivo de áudio", "", filters,
    )
    if owns_app:
        app.quit()
    return selected_file or None


def select_audio_source():
    try:
        from PySide6.QtWidgets import QApplication, QInputDialog
    except Exception:
        print("✗ PySide6 não encontrado. Instale com: pip install pyside6")
        return None
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)
    options = ["Arquivo local", "Buscar no YouTube (yt-dlp)"]
    choice, ok = QInputDialog.getItem(
        None, "Fonte do áudio", "Escolha de onde carregar o áudio:", options, 0, False,
    )
    if not ok:
        if owns_app:
            app.quit()
        return None
    if choice == options[0]:
        selected_file = select_audio_file()
        if owns_app:
            app.quit()
        return ("file", selected_file) if selected_file else None
    query, ok = QInputDialog.getText(
        None, "Buscar no YouTube", "Digite o nome da música ou cole a URL:",
    )
    if owns_app:
        app.quit()
    if not ok or not query.strip():
        return None
    return ("youtube", query.strip())


def resolve_youtube_stream(query_or_url):
    try:
        import yt_dlp
    except Exception as exc:
        raise RuntimeError(
            "Biblioteca yt-dlp não encontrada. Instale com: pip install yt-dlp"
        ) from exc
    ydl_opts = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "format": "bestaudio/best", "default_search": "ytsearch1",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query_or_url, download=False)
    if not info:
        raise RuntimeError("yt-dlp não retornou informações do vídeo")
    if "entries" in info and info["entries"]:
        info = next((e for e in info["entries"] if e), None)
        if not info:
            raise RuntimeError("playlist vazia")
    stream_url = info.get("url")
    if not stream_url:
        formats    = info.get("formats") or []
        audio_only = [
            f for f in formats
            if f.get("url")
            and f.get("acodec") not in (None, "none")
            and f.get("vcodec") in (None, "none")
        ]
        if audio_only:
            audio_only.sort(key=lambda f: (f.get("abr") or 0.0, f.get("tbr") or 0.0))
            stream_url = audio_only[-1]["url"]
    if not stream_url:
        raise RuntimeError("não foi possível resolver a URL de áudio do YouTube")
    return stream_url, info.get("title") or "YouTube"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("     🎵 LED MUSIC SYNC 🎵")
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