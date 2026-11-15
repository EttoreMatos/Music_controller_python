"""Motor de partitura de LEDs (audio_reader6).

Transforma um áudio numa partitura de PWM para 6 LEDs (2 pontas + 4 do meio),
totalmente offline:

  * pontas  — kick/graves (gate + envelope), inalteradas desde a v6;
  * meio    — sequências ancoradas no grid de batidas e guiadas pela música
              (ataques por banda, compasso, seções, drops);
  * transições — efeitos nas viradas de seção, refinados e alinhados ao compasso.

Uso (a interface é `visionaudio7_3.py`):

    engine = LedScoreEngine(fps=30, config=EffectConfig())
    result = engine.compute_score(y, sr, progress_cb=lambda frac, label: ...)

Também expõe o acesso a áudio remoto (YouTube) sem HTTP 403:
`resolve_youtube_stream_info`, `load_remote_audio`, `make_playback_wav`.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
import glob
import math
import os
import random
import shutil
import subprocess
import tempfile
from typing import Optional
import urllib.error
import urllib.request

import librosa
import numpy as np
from scipy import ndimage as ndi


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
    WAVE          = auto()   # onda senoidal correndo      – groove contínuo
    ALTERNATE     = auto()   # pares alternando            – batida marcada
    RIPPLE        = auto()   # pulso centro→bordas         – transição musical


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EffectConfig:
    FPS: int = 30
    NUM_LEDS: int = 6

    STEP_FRAMES_MIN: int = 2
    # FIX #3: era 9 — clipava músicas lentas (64 BPM = 13.9 frames → forçava 9).
    # 24 frames = 0.8s, suporta tempos a partir de ~37 BPM sem distorção.
    STEP_FRAMES_MAX: int = 24
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

    MIDDLE_FADE_DECAY: float = 0.6

    EDGE_FADE_DECAY_BASE: float = 0.80
    EDGE_FADE_DECAY_HEAVY: float = 0.70
    TAIL_CUTOFF: float = 2.0

    # FIX #13: era 55 — base muito escuro para músicas de baixa energia.
    ACTIVE_MIN_PWM: float = 68.0
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
    # FIX #10: era 0.68 → a 30fps: 0.68^5 = 14% — pontas somem em ~5 frames no idle.
    EDGE_IDLE_DECAY: float = 0.78


    STARTUP_RAMP_SECONDS: float = 2.0

    # ------------------------------------------------------------------ #
    # Segmentação estrutural                                               #
    # ------------------------------------------------------------------ #

    SEG_HOP_SECONDS: float = 0.85
    SEG_K_SEGMENTS: int = 56
    SEG_MIN_GAP_SECONDS: float = 1.4
    SEG_IGNORE_START_SECONDS: float = 2.5
    SEG_NOVELTY_THRESHOLD: float = 0.16
    # Distância cosseno mínima entre "antes" e "depois" da fronteira.
    # Filtra picos locais de novidade que não chegam a trocar seção.
    SEG_STRUCT_CHANGE_THRESHOLD: float = 0.10
    # Quanto tempo à frente (após a fronteira) medimos a persistência da mudança.
    # Usado para diagnóstico/telemetria; não bloqueia transições.
    SEG_PERSISTENCE_AHEAD_SECONDS: float = 3.0
    # Referência para marcar "mudança curta" no log.
    SEG_MIN_PERSIST_DELTA: float = 0.06
    # Referência de persistência relativa para marcar "mudança curta" no log.
    SEG_PERSISTENCE_MIN_RATIO: float = 0.50
    # Soma mínima de beat_norm + kick_norm depois da fronteira.
    # Rejeita transições em silêncio ou energia irrisória.
    # FIX #12: era 0.08 — ligeiramente relaxado; 0.10 filtra mais trechos de silêncio.
    SEG_MIN_ENERGY_THRESHOLD: float = 0.10

    # ------------------------------------------------------------------ #
    # Features de segmento para escolha do efeito                         #
    # ------------------------------------------------------------------ #

    NOISE_FLATNESS_THRESHOLD: float = 0.55
    # FIX #9: era 0.28 — raramente atingido, privando SPARKLE/CASCADE de aparecer.
    BRIGHT_CENTROID_THRESHOLD: float = 0.22

    # ------------------------------------------------------------------ #
    # Efeito de transição                                                  #
    # ------------------------------------------------------------------ #

    TRANS_MIN_DURATION_S: float = 0.45
    TRANS_MAX_DURATION_S: float = 1.80
    # FIX #11: era 3 frames (100ms) — entrada/saída de efeito muito abrupta. 5 = ~167ms.
    TRANS_BLEND_FRAMES: int = 5

    # ------------------------------------------------------------------ #
    # Ressincronização de BPM pós-transição                               #
    # ------------------------------------------------------------------ #

    # Fração de correção aplicada ao phase_accumulator após cada transição.
    # 1.0 = snap instantâneo ao grid de batidas; 0.0 = sem correção.
    BPM_RESYNC_STRENGTH: float = 0.85
    # Correção leve e contínua para manter troca de sequência no tempo.
    BPM_CONTINUOUS_SYNC_STRENGTH: float = 0.06

    # ------------------------------------------------------------------ #
    # LEDs do meio: sequências guiadas pela música                         #
    # ------------------------------------------------------------------ #

    # Sensibilidade dos ataques (0..1 sobre o fluxo espectral normalizado
    # localmente). Menor = mais ataques detectados; maior = só os fortes.
    MID_HIT_THRESHOLD: float = 0.38
    # O quanto o brilho de cada passo acompanha o ataque real naquele instante
    # (0 = todos os passos iguais; 1 = passo sem ataque quase apaga).
    MID_SUPPORT_DEPTH: float = 0.38
    # Fração do brilho do passo mantida até o próximo (menor = mais pulsante).
    MID_STEP_FLOOR: float = 0.78
    # Trava anti-saturação: média máxima de LEDs do meio acesos ao mesmo tempo.
    MID_MAX_LIT_AVG: float = 1.8
    # Duração mínima perceptível de um passo (frames). Evita "borrão".
    MID_MIN_STEP_FRAMES: float = 4.0
    # Ajusta passos/batida por compasso conforme a densidade rítmica (com histerese).
    MID_AUTO_SUBDIVISION: bool = True
    MID_DENSITY_SPARSE_EPB: float = 0.80
    MID_DENSITY_DENSE_EPB: float = 3.00
    # Frase mínima (em compassos) antes de trocar o padrão do sequenciador.
    MID_MIN_BARS_PER_PATTERN: int = 2
    # Além de MIDDLE_SEQUENCES, usa ping-pong / fora→dentro / dentro→fora.
    MID_EXTRA_PATTERNS: bool = False
    # Drops / buildups (lookahead offline).
    DROP_DETECT_ENABLED: bool = True
    DROP_ENERGY_JUMP: float = 0.22
    DROP_MIN_GAP_SECONDS: float = 8.0
    BUILD_MAX_SECONDS: float = 10.0
    # Semente do sorteio de padrões (mesma música → mesma partitura).
    MID_SEED: int = 7

    # Precisão das transições.
    TRANS_REFINE_ENABLED: bool = True
    TRANS_REFINE_SEARCH_S: float = 0.9
    TRANS_REFINE_MIN_RATIO: float = 2.0
    TRANS_PREFER_DOWNBEAT: bool = True
    # O quanto o brilho do efeito acompanha a energia ao vivo (0 = fixo).
    TRANS_LIVE_GAIN: float = 0.35


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
    refined:         bool  = False

    def __repr__(self) -> str:
        fps   = 30
        t_s   = self.frame_start / fps
        dur_s = (self.frame_end - self.frame_start) / fps
        return (
            f"TransitionEvent(t={t_s:.2f}s, dur={dur_s:.2f}s, "
            f"effect={self.effect_type.name}, brightness={self.pwm_brightness:.0f})"
        )


# ---------------------------------------------------------------------------
# Motor adaptativo dos LEDs do meio — ANÁLISE (offline, só numpy/scipy)
# ---------------------------------------------------------------------------
#
# Como a partitura é gerada offline, a análise pode olhar para o FUTURO da
# música (lookahead): detecta drops antes de acontecerem, sabe se um trecho vai
# ser tonal ou percussivo, alinha o relógio dos passos ao grid de batidas real
# e normaliza tudo localmente (AGC), então músicas baixas e altas usam toda a
# faixa dinâmica dos LEDs.
#
# Mapa de bandas (LED do meio k  ←  banda k+1; a banda 0 é kick/sub e fica com
# as pontas e com o brilho do passo):
#
#     banda 0  30–150 Hz    kick / sub          (pontas + acento)
#     banda 1  150–400 Hz   grave-médio         → LED 1 (esquerda)
#     banda 2  400–1500 Hz  médio / voz / caixa → LED 2
#     banda 3  1.5–5 kHz    presença / ataque   → LED 3
#     banda 4  5–10.5 kHz   brilho / hi-hat     → LED 4 (direita)

MID_BAND_EDGES_HZ = (30.0, 150.0, 400.0, 1500.0, 5000.0, 10500.0)
MID_NUM_BANDS = 5


@dataclass
class MiddleAnalysis:
    n: int
    fps: int
    band_onset: np.ndarray      # (5, n) força do onset (0 fora dos picos)
    support: np.ndarray         # (n,) 0..1 ataque contínuo: o que acontece AGORA
    sustain: np.ndarray         # (n,) 0..1 quão sustentada (vs. percussiva) é a música
    energy: np.ndarray          # (n,) 0..1  (global + local)
    energy_slow: np.ndarray     # (n,) 0..1  (~3 s)
    gate: np.ndarray            # (n,) 0..1  (0 = silêncio)
    density: np.ndarray         # (n,) momentos com ataque por segundo
    perc_ratio: np.ndarray      # (n,) 0..1 (percussivo vs. harmônico)
    novelty: np.ndarray         # (n,) mudança de textura (p/ refinar transições)
    beat_pos: np.ndarray        # (n,) posição em batidas (contínua)
    period: np.ndarray          # (n,) frames por batida (local)
    beat_conf: np.ndarray       # (n,) 0..1 confiança do pulso
    beat_times: np.ndarray      # frames (float) de todas as batidas do grid
    beat_origin: int            # índice de batida correspondente a pos=0
    bar_offsets: np.ndarray     # offset de compasso por batida inteira
    downbeat_conf: float
    build_progress: np.ndarray  # (n,) 0..1 rumo ao próximo drop
    drop_frames: list
    info: dict

    def bar_offset_at(self, pos: float) -> int:
        k = int(math.floor(pos)) + int(self.beat_origin)
        last = len(self.bar_offsets) - 1
        k = 0 if k < 0 else (last if k > last else k)
        return int(self.bar_offsets[k])

    def is_downbeat_index(self, k: int) -> bool:
        k = min(max(int(k), 0), len(self.bar_offsets) - 1)
        return ((k - int(self.beat_origin)) - int(self.bar_offsets[k])) % 4 == 0


class MiddleAnalyzer:
    N_FFT = 1024
    SUB_PER_GROUP = (3, 4, 6, 7, 5)
    GROUP_MIN_DIST = (4, 4, 4, 3, 2)   # distância mínima entre onsets (frames)

    def __init__(self, cfg, fps: int):
        self.cfg = cfg
        self.fps = int(fps)

    # ------------------------------------------------------------------ #
    # utilidades                                                           #
    # ------------------------------------------------------------------ #

    def _sec(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.fps)))

    @staticmethod
    def _uf(x, size):
        size = max(1, int(size))
        return ndi.uniform_filter1d(np.asarray(x, dtype=np.float64), size=size, mode="nearest")

    def _subband_layout(self, sr):
        n_bins = self.N_FFT // 2 + 1
        freqs = np.fft.rfftfreq(self.N_FFT, 1.0 / sr)
        nyq = sr / 2.0
        rows, groups = [], []
        for g in range(MID_NUM_BANDS):
            lo = MID_BAND_EDGES_HZ[g]
            hi = min(MID_BAND_EDGES_HZ[g + 1], nyq * 0.98)
            if hi <= lo * 1.05:
                hi = lo * 1.25
            k = self.SUB_PER_GROUP[g]
            edges = np.geomspace(lo, hi, k + 1)
            for s in range(k):
                sel = np.where((freqs >= edges[s]) & (freqs < edges[s + 1]))[0]
                if sel.size == 0:
                    centre = 0.5 * (edges[s] + edges[s + 1])
                    sel = np.array([int(np.argmin(np.abs(freqs - centre)))])
                row = np.zeros(n_bins, dtype=np.float32)
                row[sel] = 1.0 / sel.size
                rows.append(row)
                groups.append(g)
        return np.stack(rows), np.asarray(groups, dtype=np.int32)

    def _spectrogram(self, y, W, n, spf):
        """Retorna S_sub (n_sub, n) em amplitude e rms (n,) por frame de LED."""
        n_fft = self.N_FFT
        total = n * spf
        m = min(len(y), total)

        padded = np.zeros(total + n_fft, dtype=np.float32)
        padded[n_fft // 2: n_fft // 2 + m] = np.asarray(y[:m], dtype=np.float32)

        chunk_audio = np.zeros(total, dtype=np.float32)
        chunk_audio[:m] = np.asarray(y[:m], dtype=np.float32)
        rms = np.sqrt(np.mean(chunk_audio.reshape(n, spf) ** 2, axis=1) + 1e-12)

        win = np.hanning(n_fft).astype(np.float32)
        norm = 2.0 / float(win.sum())
        offsets = np.arange(n_fft)[None, :]
        starts_all = np.arange(n) * spf + spf // 2

        S = np.empty((W.shape[0], n), dtype=np.float32)
        CH = 2048
        for c0 in range(0, n, CH):
            c1 = min(n, c0 + CH)
            idx = starts_all[c0:c1, None] + offsets
            frames = padded[idx] * win
            mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) * norm
            S[:, c0:c1] = np.sqrt(W @ (mag.T ** 2))
        return S, rms.astype(np.float64)

    @staticmethod
    def _pick_peaks(x, thr, min_dist):
        n = len(x)
        out = np.zeros(n, dtype=np.float32)
        if n < 3:
            return out
        mx = ndi.maximum_filter1d(x, size=5, mode="nearest")
        cand = np.where((x >= mx) & (x >= thr))[0]
        kept = []
        for c in cand:
            if not kept or c - kept[-1] >= min_dist:
                kept.append(int(c))
            elif x[c] > x[kept[-1]]:
                kept[-1] = int(c)
        for c in kept:
            out[c] = x[c]
        return out

    def _local_norm(self, flux, win, scale_ref):
        med = ndi.percentile_filter(flux, 50, size=win, mode="nearest")
        hi = ndi.percentile_filter(flux, 95, size=win, mode="nearest")
        g_med = float(np.median(flux))
        g_hi = float(np.percentile(flux, 95))
        floor = max(0.30 * (g_hi - g_med), 0.12 * scale_ref, 1e-9)
        span = np.maximum(hi - med, floor)
        return np.clip((flux - med) / span, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # grid de batidas                                                      #
    # ------------------------------------------------------------------ #

    def _beat_grid(self, beat_frames, n, tempo_bpm, onset_ref):
        fps = self.fps
        period_default = float(np.clip(fps * 60.0 / max(float(tempo_bpm), 1.0), 6.0, 2.0 * fps))

        bf = np.asarray(beat_frames, dtype=np.float64).ravel()
        bf = bf[np.isfinite(bf)]
        bf = np.unique(bf[(bf >= 0) & (bf <= n + fps)])

        used_synthetic = False
        if len(bf) >= 4:
            med = float(np.median(np.diff(bf)))
            if med < 4.0 or med > 3.0 * fps:
                bf = np.array([])
        if len(bf) < 4:
            # Sem batidas confiáveis: ancora no maior onset do começo e segue o BPM.
            used_synthetic = True
            head = onset_ref[: max(2, int(period_default * 2))]
            anchor = float(np.argmax(head)) if len(head) else 0.0
            bf = np.arange(anchor, n + period_default, period_default)

        med = float(np.median(np.diff(bf))) if len(bf) > 1 else period_default

        # remove duplicadas muito próximas e preenche lacunas do beat tracker
        cleaned = [float(bf[0])]
        for b in bf[1:]:
            gap = float(b) - cleaned[-1]
            if gap < 0.55 * med:
                continue
            if gap > 1.6 * med:
                k = int(round(gap / med))
                base = cleaned[-1]
                for j in range(1, k):
                    cleaned.append(base + gap * j / k)
            cleaned.append(float(b))
        bt = np.asarray(cleaned, dtype=np.float64)

        # refinamento: puxa cada batida para o pico de onset mais próximo (±2 frames)
        if not used_synthetic and len(onset_ref) > 5:
            refined = bt.copy()
            for k, b in enumerate(bt):
                c = int(round(b))
                lo, hi = max(0, c - 2), min(len(onset_ref) - 1, c + 2)
                if hi <= lo:
                    continue
                local = onset_ref[lo:hi + 1]
                peak = lo + int(np.argmax(local))
                if peak != c and onset_ref[peak] >= 1.25 * onset_ref[min(max(c, 0), len(onset_ref) - 1)] + 1e-9:
                    refined[k] = float(peak)
            for k in range(1, len(refined)):
                if refined[k] < refined[k - 1] + 2.0:
                    refined[k] = bt[k] if bt[k] >= refined[k - 1] + 2.0 else refined[k - 1] + 2.0
            bt = refined

        local_med = float(np.median(np.diff(bt[: min(len(bt), 9)]))) if len(bt) > 2 else med
        local_med = max(local_med, 3.0)
        pre = []
        b = bt[0] - local_med
        while b > -2.0 * local_med and len(pre) < 4096:
            pre.append(b)
            b -= local_med
        tail_med = float(np.median(np.diff(bt[-min(len(bt), 9):]))) if len(bt) > 2 else med
        tail_med = max(tail_med, 3.0)
        post = []
        b = bt[-1] + tail_med
        while b < n + 2.0 * tail_med and len(post) < 4096:
            post.append(b)
            b += tail_med
        all_b = np.concatenate([np.asarray(pre[::-1], dtype=np.float64), bt, np.asarray(post, dtype=np.float64)])
        n_pre = len(pre)

        idx = np.arange(len(all_b), dtype=np.float64) - n_pre
        t = np.arange(n, dtype=np.float64)
        beat_pos = np.interp(t, all_b, idx)

        ibi = np.diff(all_b)
        ibi_s = ndi.median_filter(ibi, size=5, mode="nearest") if len(ibi) >= 5 else ibi
        mid = 0.5 * (all_b[:-1] + all_b[1:])
        period = np.clip(np.interp(t, mid, ibi_s), 3.0, 4.0 * fps)

        return beat_pos, period, all_b, n_pre, bt, used_synthetic

    def _beat_confidence(self, bt, onset_ref, n):
        """Evidência de pulso: onset na batida vs. no meio entre batidas."""
        if len(bt) < 4:
            return np.zeros(n)
        L = len(onset_ref)
        ons, offs, centres = [], [], []
        for k in range(len(bt) - 1):
            b, nb = bt[k], bt[k + 1]
            c = int(round(b))
            m = int(round(0.5 * (b + nb)))
            if c < 1 or m >= L - 1 or c >= L - 1:
                continue
            ons.append(float(np.max(onset_ref[c - 1:c + 2])))
            offs.append(float(np.max(onset_ref[m - 1:m + 2])))
            centres.append(b)
        if len(ons) < 4:
            return np.zeros(n)
        ons, offs = np.asarray(ons), np.asarray(offs)
        r = ons / (ons + offs + 1e-6)
        r = self._uf(r, 8)
        conf_b = np.clip((r - 0.5) / 0.28, 0.0, 1.0)
        return np.interp(np.arange(n), np.asarray(centres), conf_b)

    def _harmonic_change_at_beats(self, S, sub_group, all_b, n):
        """Mudança do conteúdo grave/médio-grave (até 1,5 kHz) logo antes vs. depois
        de cada batida. Acordes e notas de baixo costumam trocar no tempo 1."""
        sel = sub_group <= 2
        L = np.log1p(S[sel].astype(np.float64) / (0.02 * np.percentile(S[sel], 95, axis=1, keepdims=True) + 1e-9))
        out = np.zeros(len(all_b))
        for k, b in enumerate(all_b):
            c = int(round(b))
            pl = float(np.median(np.diff(all_b[max(0, k - 2):k + 3]))) if len(all_b) > 2 else 15.0
            w = max(3, int(round(0.55 * pl)))
            a0, a1 = c - w, c - 1
            b0, b1 = c + 1, c + w
            if a0 < 0 or b1 >= n:
                continue
            out[k] = float(np.linalg.norm(L[:, b0:b1 + 1].mean(1) - L[:, a0:a1 + 1].mean(1)))
        mx = np.percentile(out, 95) + 1e-9
        return np.clip(out / mx, 0.0, 1.5)

    def _downbeats(self, all_b, n_pre, acc_curve, harm_change, boundary_frames, boundary_w, n):
        """Fase do compasso (0..3 por batida). A fase quase nunca muda numa música,
        então a decisão é GLOBAL (muitas amostras); só trocamos localmente com
        evidência forte (1,5x) em janelas longas de 16 compassos."""
        M = len(all_b)
        L = len(acc_curve)
        acc = np.zeros(M)
        acc_mx = ndi.maximum_filter1d(acc_curve, size=3, mode="nearest")
        for k in range(M):
            c = int(round(all_b[k]))
            if 0 <= c < L:
                acc[k] = 0.40 * float(acc_mx[c]) + 0.60 * float(harm_change[k])

        valid = np.array([0 <= int(round(b)) < L for b in all_b])
        idx = np.arange(M)
        phase = (idx - n_pre)
        scores = np.array([float(acc[valid & (phase % 4 == o)].sum()) for o in range(4)])
        # votos esparsos: viradas de seção e drops quase sempre caem no tempo 1
        votes = np.zeros(4)
        for f, w in zip(boundary_frames, boundary_w):
            kb = int(np.argmin(np.abs(np.asarray(all_b) - f)))
            pl = float(np.median(np.diff(all_b[max(0, kb - 2):kb + 3]))) if M > 2 else 15.0
            if abs(all_b[kb] - f) <= 0.6 * pl:
                votes[(kb - n_pre) % 4] += w
        if votes.sum() > 0 and scores.sum() > 0 and len(boundary_frames) >= 2:
            scores = 0.35 * scores / scores.sum() + 0.65 * votes / votes.sum()
        glob = int(np.argmax(scores))
        srt = np.sort(scores)[::-1]
        conf = float((srt[0] - srt[1]) / (srt[0] + 1e-9)) if srt[0] > 0 else 0.0

        offsets = np.full(M, glob, dtype=np.int32)
        chunk, stride = 64, 32
        cur = glob
        for s in range(0, max(M - chunk // 2, 1), stride):
            e = min(M, s + chunk)
            if e - s < chunk // 2:
                break
            sl = slice(s, e)
            sc = np.array([float(acc[sl][(phase[sl] - o) % 4 == 0].sum()) for o in range(4)])
            best = int(np.argmax(sc))
            if best != cur and sc[best] > 1.5 * sc[cur]:
                cur = best
            elif best == glob and cur != glob and sc[glob] > 1.15 * sc[cur]:
                cur = glob
            offsets[s:min(M, s + stride)] = cur
            if e >= M:
                offsets[s:] = cur
        return offsets, conf

    # ------------------------------------------------------------------ #
    # drops / buildups                                                     #
    # ------------------------------------------------------------------ #

    def _detect_drops(self, energy, e_slow, bass_db, rms_s, all_b, n_pre, bar_offsets, period, n):
        """Drops com várias pistas (lookahead): salto de energia, retorno do grave
        (em dB, que não satura como a energia normalizada) e micro-silêncio antes
        do impacto. O buildup só existe quando há SUBIDA real antes do drop."""
        cfg, fps = self.cfg, self.fps
        drops = []
        build = np.zeros(n, dtype=np.float64)
        if not cfg.DROP_DETECT_ENABLED or n < 6 * fps:
            return drops, build

        e_fast = self._uf(energy, self._sec(0.5))
        w = self._sec(1.5)
        t = np.arange(n)
        a0, a1 = np.clip(t - w, 0, n), t
        b0, b1 = t, np.clip(t + w, 0, n)

        def win_mean(arr):
            cs = np.concatenate([[0.0], np.cumsum(arr)])
            return (cs[a1] - cs[a0]) / np.maximum(a1 - a0, 1), (cs[b1] - cs[b0]) / np.maximum(b1 - b0, 1)

        e_before, e_after = win_mean(e_fast)
        b_before, b_after = win_mean(bass_db)
        jump_e = e_after - e_before
        jump_b = np.clip((b_after - b_before) / 12.0, -1.0, 1.5)     # 12 dB de grave = 1.0
        # o grave "voltou" quando fica perto do seu nível máximo na faixa toda
        bass_ref = float(np.percentile(bass_db, 90))
        bass_return = np.clip(1.0 + (b_after - bass_ref) / 12.0, 0.0, 1.0)

        # micro-silêncio: o trecho logo antes do frame é bem mais baixo que o anterior
        gap_s = self._sec(0.5)
        cs_r = np.concatenate([[0.0], np.cumsum(rms_s)])
        near = (cs_r[t] - cs_r[np.clip(t - gap_s, 0, n)]) / np.maximum(t - np.clip(t - gap_s, 0, n), 1)
        far_lo, far_hi = np.clip(t - self._sec(2.0), 0, n), np.clip(t - gap_s, 0, n)
        far = (cs_r[far_hi] - cs_r[far_lo]) / np.maximum(far_hi - far_lo, 1)
        gap = np.clip(1.0 - near / (0.45 * far + 1e-9), 0.0, 1.0)

        # um drop é o FIM da subida: se a energia continua subindo depois, é só
        # meio de buildup / crescendo e não deve ser marcado.
        nb0, nb1 = np.clip(t + w, 0, n), np.clip(t + 2 * w, 0, n)
        cs_e = np.concatenate([[0.0], np.cumsum(e_fast)])
        e_next = np.where(nb1 > nb0, (cs_e[nb1] - cs_e[nb0]) / np.maximum(nb1 - nb0, 1), e_after)
        still_rising = np.maximum(e_next - e_after - 0.03, 0.0)

        score = (
            np.minimum(np.maximum(jump_e, 0.0), 0.6)
            + 0.45 * np.clip(jump_b, 0.0, 1.0) * bass_return
            + 0.20 * gap
            - 1.5 * still_rising
        )
        thr = float(cfg.DROP_ENERGY_JUMP)
        mx = ndi.maximum_filter1d(score, size=2 * w + 1, mode="nearest")
        evidence = (jump_e >= 0.10) | (jump_b >= 0.45)
        cand = np.where((score >= mx) & (score >= thr) & evidence & (e_after >= 0.45))[0]

        min_gap = self._sec(cfg.DROP_MIN_GAP_SECONDS)
        kept = []
        for c in cand:
            if c < self._sec(2.0) or c > n - self._sec(1.0):
                continue
            if not kept or c - kept[-1] >= min_gap:
                kept.append(int(c))
            elif score[c] > score[kept[-1]]:
                kept[-1] = int(c)

        for c in kept:
            # refina: maior subida local do grave+energia em ±0,7 s
            lo, hi = max(3, c - self._sec(0.7)), min(n - 4, c + self._sec(0.7))
            rise = (e_fast[lo + 3:hi + 3] - e_fast[lo - 3:hi - 3]) + 0.05 * (bass_db[lo + 3:hi + 3] - bass_db[lo - 3:hi - 3])
            f = lo + int(np.argmax(rise)) if len(rise) else c
            # encaixa na batida mais próxima, preferindo o tempo 1 do compasso
            k0 = int(np.searchsorted(all_b, f))
            best_k, best_cost = None, 1e9
            pl = float(period[min(max(int(f), 0), n - 1)])
            for k in range(max(0, k0 - 2), min(len(all_b), k0 + 2)):
                d = abs(all_b[k] - f)
                if d > 0.75 * pl:
                    continue
                is_down = ((k - n_pre) - int(bar_offsets[min(k, len(bar_offsets) - 1)])) % 4 == 0
                cost = d - (0.5 * pl if is_down else 0.0)
                if cost < best_cost:
                    best_cost, best_k = cost, k
            if best_k is not None:
                f = int(round(all_b[best_k]))
            drops.append(int(np.clip(f, 0, n - 1)))
        drops = sorted(set(drops))

        # buildup: varre PARA TRÁS a partir do drop enquanto a energia vinha subindo
        e_s = self._uf(energy, self._sec(1.0))
        step = self._sec(0.5)
        prev_drop = -10 ** 9
        for f in drops:
            limit = max(prev_drop + self._sec(2.0), f - self._sec(cfg.BUILD_MAX_SECONDS), 0)
            probe = f - self._sec(0.6)            # ignora o micro-silêncio colado no drop
            start = probe
            while start - step >= limit and e_s[start - step] <= e_s[start] + 0.02:
                start -= step
            risen = float(e_s[max(probe, 0)] - e_s[max(start, 0)])
            if (f - start) >= self._sec(2.0) and risen >= 0.08:
                seg_t = np.arange(start, f)
                build[start:f] = np.maximum(build[start:f], ((seg_t - start) / float(f - start)) ** 1.15)
            prev_drop = f
        return drops, build

    # ------------------------------------------------------------------ #
    # ponto de entrada                                                     #
    # ------------------------------------------------------------------ #

    def _novelty(self, rms_s, S, sub_group, density, perc_ratio, n):
        """Mudança de textura entre os 0,6 s antes e depois de cada frame
        (energia + 5 bandas + densidade + perc/harm). Pico = ponto exato da virada."""
        feats = [20.0 * np.log10(rms_s + 1e-5)]
        for g in range(MID_NUM_BANDS):
            amp = np.sqrt(np.mean(S[sub_group == g].astype(np.float64) ** 2, axis=0))
            feats.append(np.maximum(20.0 * np.log10(self._uf(amp, 3) + 1e-5), -80.0))
        feats.append(self._uf(density, self._sec(1.0)))
        feats.append(perc_ratio)
        F = np.vstack(feats)
        med = np.median(F, axis=1, keepdims=True)
        mad = np.median(np.abs(F - med), axis=1, keepdims=True) * 1.4826 + 1e-6
        Z = np.clip((F - med) / mad, -6.0, 6.0)
        w = self._sec(0.6)
        cs = np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(Z, axis=1)], axis=1)
        t = np.arange(n)
        a0, a1 = np.clip(t - w, 0, n), t
        b0, b1 = t, np.clip(t + w, 0, n)
        before = (cs[:, a1] - cs[:, a0]) / np.maximum(a1 - a0, 1)
        after = (cs[:, b1] - cs[:, b0]) / np.maximum(b1 - b0, 1)
        diff = after - before
        # entradas (algo novo começa) pesam mais que quedas (decaimento de prato,
        # fim de fill, cauda de reverb): essas não são viradas de seção.
        diff = np.where(diff > 0.0, diff, 0.55 * diff)
        return np.linalg.norm(diff, axis=0)

    def analyze(self, y, sr, n_frames, tempo_bpm, beat_frames) -> MiddleAnalysis:
        cfg, fps = self.cfg, self.fps
        n = max(1, int(n_frames))
        spf = max(1, int(sr / fps))

        W, sub_group = self._subband_layout(sr)
        S, rms = self._spectrogram(y, W, n, spf)
        n_sub = S.shape[0]

        # ---- HPSS leve (no domínio das sub-bandas) --------------------- #
        t_k = max(3, self._sec(0.5) | 1)
        Hh = ndi.median_filter(S, size=(1, t_k), mode="nearest")
        Pp = ndi.median_filter(S, size=(5, 1), mode="nearest")
        mask_p = (Pp ** 2) / (Hh ** 2 + Pp ** 2 + 1e-12)
        Sp = S * mask_p
        Sh = S - Sp
        Ep = np.sum(Sp.astype(np.float64) ** 2, axis=0)
        Eh = np.sum(Sh.astype(np.float64) ** 2, axis=0)
        perc_ratio = self._uf(Ep / (Ep + Eh + 1e-12), self._sec(1.5))

        # ---- fluxo espectral por banda → onsets ------------------------- #
        S_on = Sp + 0.4 * Sh
        ref = 0.02 * np.percentile(S_on, 95, axis=1) + 1e-9
        L = np.log1p(S_on / ref[:, None])
        d = np.maximum(L[:, 1:] - L[:, :-1], 0.0)
        flux_sub = np.concatenate([np.zeros((n_sub, 1), dtype=np.float32), d], axis=1)
        flux_g = np.zeros((MID_NUM_BANDS, n), dtype=np.float64)
        for g in range(MID_NUM_BANDS):
            flux_g[g] = np.mean(flux_sub[sub_group == g], axis=0)
        scale_ref = float(max(np.percentile(flux_g[g], 95) for g in range(MID_NUM_BANDS)))
        win_norm = self._sec(4.0)
        norm_g = np.zeros_like(flux_g)
        onset = np.zeros((MID_NUM_BANDS, n), dtype=np.float32)
        for g in range(MID_NUM_BANDS):
            norm_g[g] = self._local_norm(flux_g[g], win_norm, scale_ref)
        kick_curve = np.clip(norm_g[0] - 0.45 * np.maximum(norm_g[3], norm_g[4]), 0.0, 1.0)
        for g in range(MID_NUM_BANDS):
            curve = kick_curve if g == 0 else norm_g[g]
            onset[g] = self._pick_peaks(curve, float(cfg.MID_HIT_THRESHOLD), self.GROUP_MIN_DIST[g])

        ev = (onset > 0).any(axis=0).astype(np.float64)   # momentos distintos com ataque
        density = self._uf(ev, self._sec(2.0)) * fps

        # ataque contínuo: quanto "está acontecendo" em cada frame (kick pesa mais)
        weights = (1.0, 0.7, 0.9, 0.8, 0.5)
        supp = np.max([weights[g] * norm_g[g] for g in range(MID_NUM_BANDS)], axis=0)
        support = np.clip(ndi.maximum_filter1d(supp, size=3, mode="nearest"), 0.0, 1.0)

        perc_n = np.clip((perc_ratio - 0.30) / 0.45, 0.0, 1.0)
        sustain = np.clip(1.0 - perc_n, 0.0, 1.0)

        # ---- energia / silêncio ---------------------------------------- #
        rms_s = self._uf(rms, self._sec(0.4))
        db = 20.0 * np.log10(rms_s + 1e-5)
        lo_db, hi_db = np.percentile(db, [8, 96])
        if hi_db - lo_db < 6.0:
            hi_db = lo_db + 6.0
        e_global = np.clip((db - lo_db) / (hi_db - lo_db), 0.0, 1.0)
        p90_loc = ndi.percentile_filter(rms_s, 90, size=self._sec(2.0), mode="nearest")
        e_local = np.clip(rms_s / (p90_loc + 1e-9), 0.0, 1.0)
        energy = np.clip(0.5 * e_global + 0.5 * e_local, 0.0, 1.0)
        energy_slow = np.clip(self._uf(e_global, self._sec(3.0)), 0.0, 1.0)
        silence = (rms_s < 10.0 ** (-56.0 / 20.0)).astype(np.float64)
        gate = np.clip(1.0 - self._uf(silence, 5) * 1.5, 0.0, 1.0)

        # ---- batidas / compasso ----------------------------------------- #
        onset_ref = (norm_g[0] + norm_g[1] + norm_g[2]) / 3.0
        beat_pos, period, all_b, n_pre, bt, synthetic = self._beat_grid(
            beat_frames, n, tempo_bpm, onset_ref
        )
        beat_conf = self._beat_confidence(bt, onset_ref, n)
        acc_curve = 0.55 * norm_g[0] + 0.25 * norm_g[1] + 0.20 * norm_g[2]
        bass_amp = np.sqrt(np.mean(S[sub_group == 0].astype(np.float64) ** 2, axis=0))
        bass_db = self._uf(20.0 * np.log10(bass_amp + 1e-5), self._sec(0.4))
        harm_change = self._harmonic_change_at_beats(S, sub_group, all_b, n)
        novelty = self._novelty(rms_s, S, sub_group, density, perc_ratio, n)
        nov_n = novelty / (float(np.max(novelty)) + 1e-9)
        nov_peaks = self._pick_peaks(nov_n, 0.30, self._sec(2.0))
        b_frames = np.where(nov_peaks > 0)[0]
        b_w = nov_peaks[b_frames]
        bar_offsets, db_conf = self._downbeats(all_b, n_pre, acc_curve, harm_change, b_frames, b_w, n)

        # ---- drops / buildups ------------------------------------------ #
        drops, build = self._detect_drops(
            energy, energy_slow, bass_db, rms_s, all_b, n_pre, bar_offsets, period, n
        )

        info = {
            "synthetic_grid": bool(synthetic),
            "mean_beat_conf": float(np.mean(beat_conf)),
            "mean_density": float(np.mean(density)),
            "mean_perc_ratio": float(np.mean(perc_ratio)),
            "onsets": [int(np.count_nonzero(onset[g])) for g in range(MID_NUM_BANDS)],
            "n_beats": int(len(bt)),
        }
        return MiddleAnalysis(
            n=n, fps=fps, band_onset=onset, support=support, sustain=sustain,
            energy=energy, energy_slow=energy_slow, gate=gate, density=density,
            perc_ratio=perc_ratio, novelty=novelty, beat_pos=beat_pos, period=period,
            beat_conf=beat_conf, beat_times=np.asarray(all_b, dtype=np.float64),
            beat_origin=int(n_pre), bar_offsets=bar_offsets,
            downbeat_conf=float(db_conf), build_progress=build,
            drop_frames=drops, info=info,
        )


# ---------------------------------------------------------------------------
# Motor dos LEDs do meio — RENDERIZAÇÃO (frame a frame)
# ---------------------------------------------------------------------------
#
# A SEQUÊNCIA (MIDDLE_SEQUENCES) é sempre o esqueleto visual. A música entra em:
#
#   TEMPO       passo travado no grid de batidas (PLL); subdivisão por compasso
#               com histerese e piso perceptivo de duração de passo.
#   INTENSIDADE cada passo acende conforme o ATAQUE REAL naquele instante
#               (support): passo em cima de kick/caixa brilha, passo em cima de
#               nada, não. A posição métrica (tempo 1) dá acento.
#   FADE        a cauda depende de quão sustentada é a música (staccato = corta
#               rápido, pad/sustain = escorre) e da duração do passo.
#   FRASE       o padrão só troca em início de compasso, e SEMPRE no início de
#               cada seção detectada (transição): a sequência "vira de página"
#               junto com a música.
#   DROP        micro-apagão + flash curto no tempo 1 do drop.
#   GOVERNOR    trava anti-saturação: a média de LEDs acesos nunca passa de
#               MID_MAX_LIT_AVG, qualquer que seja a música.

# Padrões extras: só entram se cfg.MID_EXTRA_PATTERNS=True.
_MID_EXTRA_PATTERNS = (
    ((0,), (1,), (2,), (3,), (2,), (1,)),                # ping-pong
    ((0, 3), (1, 2)),                                    # fora → dentro
    ((1, 2), (0, 3)),                                    # dentro → fora
)


class MiddleRenderer:
    KICK_DECAY = 0.72
    BEAT_DECAY = 0.80
    DROP_DARK_FRAMES = 2
    FLASH_DECAY = 0.86

    def __init__(self, cfg, fps: int, ma: MiddleAnalysis, speed_mult: float, n_frames: int,
                 section_frames=()):
        self.cfg = cfg
        self.fps = int(fps)
        self.ma = ma
        self.rng = random.Random(int(cfg.MID_SEED) * 100003 + int(n_frames))

        self._period = ma.period.tolist()
        self._beat_pos = ma.beat_pos.tolist()
        self._energy = ma.energy.tolist()
        self._gate = ma.gate.tolist()
        self._support = ma.support.tolist()
        self._sustain = ma.sustain.tolist()
        self._build = ma.build_progress.tolist()
        self._onset = [ma.band_onset[g].tolist() for g in range(MID_NUM_BANDS)]
        self._drops = list(ma.drop_frames)
        self._drop_idx = 0
        self._sections = sorted(int(f) for f in section_frames)
        self._section_idx = 0

        seen, lib = set(), []
        base = tuple(cfg.MIDDLE_SEQUENCES)
        if cfg.MID_EXTRA_PATTERNS:
            base = base + _MID_EXTRA_PATTERNS
        for p in base:
            key = tuple(tuple(step) for step in p)
            if key not in seen:
                seen.add(key)
                lib.append(key)
        self.library = lib

        self.spb_base = (1.0 / max(float(cfg.BPM_STEP_MULT), 1e-3)) * float(speed_mult)

        coverage = 0.6 if ma.info.get("synthetic_grid") else 1.0
        self.lock = float(np.clip(cfg.BPM_CONTINUOUS_SYNC_STRENGTH * (0.65 + 0.90 * coverage) * 2.5, 0.04, 0.30))
        self.resync = float(np.clip(cfg.BPM_RESYNC_STRENGTH * (0.70 + 0.35 * min(coverage, 1.0)), 0.35, 0.98))

        self.pos = float(self._beat_pos[0])
        self.prev_pos = self.pos
        self.bar = None
        self.spb = self.spb_base
        self._factor = 1.0
        self._factor_votes = 0
        self.sub_phase = 0.0
        self.pattern = None
        self.step_idx = 0
        self.loop_count = 0
        self.cur_step = (0,)
        self.triggered = False
        self.accent = 1.0
        self.bars_since_switch = 0
        self._force_switch = False

        self.chase = [0.0] * 4
        self.kick_env = 0.0
        self.kick_win = 0
        self.kick_window_frames = max(1, int((cfg.KICK_WINDOW_MS / 1000.0) * self.fps))
        self.beat_env = 0.0
        self.flash = 0.0
        self.lit_ema = 1.5

        self.stats = {"bars": 0, "pattern_switches": 0, "section_switches": 0,
                      "spb_hist": {}, "drops_fired": 0, "build_frames": 0,
                      "governor_frames": 0, "steps": 0}

    # ------------------------------------------------------------------ #
    # subdivisão / padrões                                                 #
    # ------------------------------------------------------------------ #

    def _clamp_spb(self, spb: float, period: float) -> float:
        cfg = self.cfg
        lo_frames = max(2.0, float(cfg.STEP_FRAMES_MIN), float(cfg.MID_MIN_STEP_FRAMES))
        hi_frames = max(lo_frames + 1.0, float(cfg.STEP_FRAMES_MAX))
        spb = float(min(16.0, max(0.25, spb)))
        while period / spb < lo_frames and spb > 0.25:
            spb *= 0.5
        while period / spb > hi_frames and spb < 16.0:
            spb *= 2.0
        return spb

    def _pick_pattern(self, energy: float, hard: bool = False):
        steps_bar = 4.0 * self.spb
        cands, weights = [], []
        for p in self.library:
            if p == self.pattern and len(self.library) > 1:
                continue
            L = len(p)
            heat = sum(len(s) for s in p) / (4.0 * L)
            target = float(np.clip(0.15 + 1.7 * heat, 0.25, 0.95))
            e = 0.95 if hard else energy
            w = math.exp(-((e - target) / 0.40) ** 2) + 0.06
            ratio = steps_bar / L
            aligned = (ratio >= 1.0 and abs(ratio - round(ratio)) < 1e-6) or \
                      (ratio < 1.0 and abs(1.0 / ratio - round(1.0 / ratio)) < 1e-6)
            if not aligned:
                w *= 0.35
            if hard and heat < 0.5:
                w *= 0.15
            cands.append(p)
            weights.append(w)
        if not cands:
            return self.pattern or self.library[0]
        r = self.rng.random() * sum(weights)
        acc = 0.0
        for c, w in zip(cands, weights):
            acc += w
            if r <= acc:
                return c
        return cands[-1]

    def _set_pattern(self, pattern):
        self.pattern = pattern
        self.step_idx = 0
        self.loop_count = 0
        self.cur_step = pattern[0]
        self.triggered = True
        self.accent = 1.1
        self.bars_since_switch = 0
        self.stats["pattern_switches"] += 1

    def _align_to_beat(self):
        """Alinha a subfase dos passos à batida mais próxima (início de frase)."""
        self.sub_phase = ((self.pos - round(self.pos)) * self.spb) % 1.0

    def _advance_step(self):
        self.step_idx += 1
        if self.step_idx >= len(self.pattern):
            self.step_idx = 0
            self.loop_count += 1
        self.cur_step = self.pattern[self.step_idx]
        self.triggered = True
        self.stats["steps"] += 1
        fp = self.pos % 1.0
        on_beat = fp < 0.14 or fp > 0.86
        if on_beat:
            off = self.ma.bar_offset_at(self.pos)
            beat_in_bar = (int(math.floor(self.pos + 0.14)) - off) % 4
            emphasis = min(1.0, 0.4 + 2.0 * self.ma.downbeat_conf)
            self.accent = 1.0 + (0.12 * emphasis if beat_in_bar == 0 else 0.0)
        else:
            self.accent = 0.84

    def _conf_at(self, i: int) -> float:
        return float(self.ma.beat_conf[min(max(i, 0), self.ma.n - 1)])

    def _on_new_bar(self, i: int):
        cfg, ma = self.cfg, self.ma
        n = ma.n
        period = max(2.0, self._period[i])
        j = min(n, i + max(1, int(4 * period)))
        if j > i:
            e_bar = float(np.mean(ma.energy[i:j]))
            dens = float(np.mean(ma.density[i:j]))
        else:
            e_bar, dens = self._energy[i], 0.0
        epb = dens * period / self.fps

        # subdivisão com histerese: só muda após 2 compassos consecutivos pedindo
        target = 1.0
        if cfg.MID_AUTO_SUBDIVISION:
            if epb < cfg.MID_DENSITY_SPARSE_EPB:
                target = 0.5
            elif epb > cfg.MID_DENSITY_DENSE_EPB and self._conf_at(i) > 0.35:
                target = 2.0
        if self.pattern is None:
            self._factor, self._factor_votes = target, 0
        elif target == self._factor:
            self._factor_votes = 0
        else:
            self._factor_votes += 1
            if self._factor_votes >= 2:
                self._factor, self._factor_votes = target, 0

        spb = self.spb_base * self._factor
        p = self._build[i]
        if p > 0.02:
            spb *= 2.0 ** min(2, int(p * 2.6))
            self.stats["build_frames"] += int(4 * period)
        spb = self._clamp_spb(spb, period)
        changed = abs(spb - self.spb) > 1e-9
        self.spb = spb

        self.bars_since_switch += 1
        switch = (
            self.pattern is None
            or self._force_switch
            or (
                self.loop_count >= max(1, int(cfg.MIDDLE_SEQ_SWITCH_REPEATS))
                and self.bars_since_switch >= max(1, int(cfg.MID_MIN_BARS_PER_PATTERN))
            )
        )
        if switch:
            self._set_pattern(self._pick_pattern(e_bar, hard=False))
            self._force_switch = False
        if changed or switch:
            off = ma.bar_offset_at(self.pos)
            bar_start = self.bar * 4.0 + off
            self.sub_phase = ((self.pos - bar_start) * self.spb) % 1.0

        self.stats["bars"] += 1
        key = f"{self.spb:g}"
        self.stats["spb_hist"][key] = self.stats["spb_hist"].get(key, 0) + 1

    # ------------------------------------------------------------------ #
    # frame                                                                #
    # ------------------------------------------------------------------ #

    def frame(self, i: int, startup_gain: float, edge_pulse: float,
              trans_active: bool, trans_just_ended: bool):
        cfg, ma = self.cfg, self.ma
        i = min(max(i, 0), ma.n - 1)
        MAX, MIN = float(cfg.ACTIVE_MAX_PWM), float(cfg.ACTIVE_MIN_PWM)
        period = max(2.0, self._period[i])

        # ---- 1. relógio (PLL no grid de batidas) -------------------------- #
        self.prev_pos = self.pos
        self.pos += 1.0 / period
        err = self._beat_pos[i] - self.pos
        if trans_just_ended:
            self.pos += self.resync * err
        elif not trans_active:
            self.pos += (self.resync if abs(err) > 0.30 else self.lock) * err
        if self.pos < self.prev_pos:
            self.pos = self.prev_pos
        dpos = self.pos - self.prev_pos

        # ---- 2. passos ---------------------------------------------------- #
        self.triggered = False
        if self.pattern is not None:
            self.sub_phase += dpos * self.spb
            while self.sub_phase >= 1.0:
                self.sub_phase -= 1.0
                self._advance_step()

        # ---- 3. compasso -------------------------------------------------- #
        off = ma.bar_offset_at(self.pos)
        bar = int(math.floor((self.pos - off) / 4.0))
        if bar != self.bar:
            self.bar = bar
            self._on_new_bar(i)

        # ---- 4. nova SEÇÃO: a sequência vira de página junto com a música -- #
        while self._section_idx < len(self._sections) and i >= self._sections[self._section_idx]:
            self._section_idx += 1
            if self.pattern is not None:
                self._set_pattern(self._pick_pattern(self._energy[i], hard=False))
                self._align_to_beat()
                self.stats["section_switches"] += 1

        # ---- 5. drop ------------------------------------------------------ #
        dark = False
        if self._drop_idx < len(self._drops):
            f = self._drops[self._drop_idx]
            if f - self.DROP_DARK_FRAMES <= i < f:
                dark = True
            elif i >= f:
                self._drop_idx += 1
                self.stats["drops_fired"] += 1
                self.flash = min(245.0, MAX * 1.25)
                if self.pattern is not None:
                    self._set_pattern(self._pick_pattern(0.95, hard=True))
                    self._align_to_beat()
        if dark:
            self.chase = [0.0] * 4
            self.kick_env = 0.0
            self.flash = 0.0
            return [0.0] * 4, 0

        # ---- 6. features do frame ---------------------------------------- #
        energy = self._energy[i]
        gate = self._gate[i]
        kick_hit = self._onset[0][i]
        self.kick_env *= self.KICK_DECAY
        if kick_hit > 0.0:
            self.kick_env = max(self.kick_env, kick_hit)
            self.kick_win = self.kick_window_frames
        bo = max(self._onset[1][i], self._onset[2][i], self._onset[3][i])
        self.beat_env = max(self.beat_env * self.BEAT_DECAY, bo)
        build_p = self._build[i]

        # ---- 7. brilho do passo (alavancas originais + ataque real) ------- #
        nrg = energy ** 0.85
        base = MIN + (MAX - MIN) * nrg + cfg.BEAT_BOOST_PWM * self.beat_env
        if self.beat_env > cfg.HEAVY_BEAT_THRESHOLD:
            base += 25.0 * (self.beat_env - cfg.HEAVY_BEAT_THRESHOLD) / max(1e-6, 1.0 - cfg.HEAVY_BEAT_THRESHOLD)
        ks = self.kick_env
        base += cfg.MIDDLE_KICK_BRIGHT_GAIN * ks
        if ks > 0.7:
            base += cfg.MIDDLE_STRONG_KICK_BRIGHT_GAIN * ((ks - 0.7) / 0.3)
        if self.kick_win > 0:
            base += cfg.MIDDLE_KICK_WINDOW_BRIGHT_GAIN * (0.4 + 0.6 * ks)
        if build_p > 0.0:
            base *= 0.6 + 0.4 * build_p
        active_pwm = float(np.clip(base * startup_gain, MIN * 0.45, MAX))

        # o passo acompanha o ataque que REALMENTE existe nesse instante
        depth = float(cfg.MID_SUPPORT_DEPTH)
        lo_i, hi_i = max(0, i - 1), min(ma.n - 1, i + 1)
        support = max(self._support[lo_i], self._support[i], self._support[hi_i])
        step_gain = (1.0 - depth) + depth * support

        # ---- 8. chase: pulso + hold + cauda guiada pela música ------------ #
        step_frames = max(1.0, period / self.spb)
        sustain = self._sustain[i]
        fade_base = float(np.clip(cfg.MIDDLE_FADE_DECAY * (0.80 + 0.45 * sustain), 0.20, 0.93))
        tail = fade_base ** (8.0 / step_frames)
        pulse_decay = 0.5 ** (1.0 / max(0.35 * step_frames, 1.0))
        floor_v = active_pwm * cfg.MID_STEP_FLOOR
        lit = self.cur_step
        for k in range(4):
            if k in lit:
                if self.triggered:
                    v = max(self.chase[k], active_pwm * self.accent * step_gain)
                else:
                    v = self.chase[k] * pulse_decay
                if kick_hit > 0.0:
                    v = max(v, active_pwm)
                self.chase[k] = min(255.0, max(v, floor_v * (0.80 + 0.20 * step_gain)))
            else:
                self.chase[k] *= tail
        if edge_pulse >= cfg.EDGE_SYNC_MIN_PWM:
            ss = float(np.clip(edge_pulse / 255.0, 0.0, 1.0))
            for k in lit:
                self.chase[k] = min(255.0, self.chase[k] + cfg.MIDDLE_EDGE_SYNC_GAIN * ss)
        out = list(self.chase)

        # ---- 9. drop: flash curto ----------------------------------------- #
        if self.flash > 1.0:
            shape = (0.85, 1.0, 1.0, 0.85)
            for k in range(4):
                out[k] = max(out[k], self.flash * shape[k])
            self.flash *= self.FLASH_DECAY
        else:
            self.flash = 0.0

        # ---- 10. governor anti-saturação ---------------------------------- #
        n_lit = sum(1 for v in out if v >= 0.50 * MAX)
        self.lit_ema += 0.05 * (n_lit - self.lit_ema)
        cap = float(cfg.MID_MAX_LIT_AVG)
        if self.lit_ema > cap and self.flash <= 1.0:
            g = max(0.30, cap / self.lit_ema)
            for k in range(4):
                if k not in lit:
                    out[k] *= g
                    self.chase[k] *= g
            if g < 0.90:
                self.stats["governor_frames"] += 1

        if self.kick_win > 0:
            self.kick_win -= 1

        levels = [min(255.0, max(0.0, out[k] * gate)) for k in range(4)]
        active = int(np.argmax(levels))
        return levels, active

    def summary(self) -> dict:
        s = dict(self.stats)
        s["pattern_library"] = len(self.library)
        return s


# ---------------------------------------------------------------------------
# Precisão das transições (refino fino + encaixe em compasso)
# ---------------------------------------------------------------------------
#
# O detector estrutural trabalha em janelas de SEG_HOP_SECONDS (≈0,85 s), então
# o instante que ele devolve já nasce com centenas de ms de erro; depois o
# encaixe antigo caía na batida mais próxima QUALQUER. Aqui:
#   1) acha o ponto exato da virada: pico da "novidade fina" (energia, 5 bandas,
#      densidade e perc/harm: 0,6 s antes vs. depois) numa janela de ±0,9 s;
#   2) encaixa na batida certa, preferindo o TEMPO 1 do compasso (seções
#      começam em início de frase);
#   3) alinha a duração ao grid (fim em batida, de preferência em início de
#      compasso) sem invadir a transição seguinte.
# Modifica os TransitionEvent no lugar (a GUI enxerga o resultado) e marca
# `refined=True` para nunca refinar duas vezes.

def refine_transitions(cfg, fps: int, ma: "MiddleAnalysis", transitions: list, n_frames: int) -> dict:
    stats = {"refined": 0, "kept": 0, "to_downbeat": 0, "shifts": []}
    if not transitions or not cfg.TRANS_REFINE_ENABLED:
        return stats

    transitions.sort(key=lambda t: t.frame_start)
    nov = ma.novelty
    nov_med = float(np.median(nov)) + 1e-9
    bt = ma.beat_times
    have_grid = len(bt) >= 3
    S = max(2, int(round(cfg.TRANS_REFINE_SEARCH_S * fps)))
    min_frames = max(2, int(cfg.TRANS_MIN_DURATION_S * fps))
    max_frames = max(min_frames + 1, int(cfg.TRANS_MAX_DURATION_S * fps))
    last = n_frames - 1

    def period_at(f):
        return float(ma.period[min(max(int(f), 0), ma.n - 1)])

    def best_beat(f, reach, prefer_down, bonus):
        """índice da batida que minimiza distância (em batidas) com bônus de tempo 1."""
        k0 = int(np.searchsorted(bt, f))
        pl = period_at(f)
        best_k, best_cost = None, 1e9
        for k in range(max(0, k0 - 3), min(len(bt), k0 + 3)):
            d = abs(bt[k] - f) / pl
            if d > reach:
                continue
            cost = d - (bonus if (prefer_down and ma.is_downbeat_index(k)) else 0.0)
            if cost < best_cost:
                best_cost, best_k = cost, k
        return best_k

    prev_end = -1
    for t in transitions:
        if getattr(t, "refined", False):
            prev_end = max(prev_end, t.frame_end)
            stats["kept"] += 1
            continue

        dur0 = max(2, int(t.frame_end - t.frame_start))
        c = int(min(max(t.frame_start, 0), last))
        f = float(c)

        # 1) ponto exato da virada (peso por proximidade ao candidato original)
        lo, hi = max(1, c - S), min(len(nov) - 2, c + S)
        if hi > lo:
            idx = np.arange(lo, hi + 1)
            score = nov[lo:hi + 1] * (1.0 - 0.35 * np.abs(idx - c) / float(S))
            k = int(np.argmax(score))
            if nov[lo + k] >= cfg.TRANS_REFINE_MIN_RATIO * nov_med:
                f = float(lo + k)

        # 2) encaixe no grid, preferindo início de compasso
        start = int(round(f))
        if have_grid:
            kb = best_beat(f, 0.75, cfg.TRANS_PREFER_DOWNBEAT, 0.6)
            if kb is not None:
                start = int(round(bt[kb]))
                if ma.is_downbeat_index(kb):
                    stats["to_downbeat"] += 1
        start = int(min(max(start, 0), last))
        if start <= prev_end:
            if have_grid:
                kk = int(np.searchsorted(bt, prev_end + 1))
                start = int(round(bt[min(kk, len(bt) - 1)]))
            else:
                start = prev_end + 1
            start = int(min(start, last))

        # 3) duração alinhada ao grid
        end = start + dur0
        if have_grid:
            want = start + dur0
            pl = period_at(start)
            best_e, best_cost = None, 1e9
            k0 = int(np.searchsorted(bt, start + min_frames))
            for k in range(max(0, k0 - 1), min(len(bt), k0 + 40)):
                b = int(round(bt[k]))
                if b < start + min_frames:
                    continue
                if b > start + max_frames:
                    break
                cost = abs(b - want) / pl - (0.35 if ma.is_downbeat_index(k) else 0.0)
                if cost < best_cost:
                    best_cost, best_e = cost, b
            if best_e is not None:
                end = best_e
        end = int(min(max(end, start + 2), last))

        stats["shifts"].append(start - int(t.frame_start))
        t.frame_start, t.frame_end, t.refined = int(start), int(end), True
        prev_end = int(end)
        stats["refined"] += 1

    # nenhuma transição invade a seguinte
    for a, b in zip(transitions[:-1], transitions[1:]):
        if a.frame_end > b.frame_start:
            a.frame_end = max(a.frame_start + 2, b.frame_start)
            if a.frame_end > b.frame_start:
                b.frame_start = min(a.frame_end, last)
                b.frame_end = max(b.frame_end, b.frame_start + 2)
    return stats


# ---------------------------------------------------------------------------
# Áudio remoto (YouTube) — corrige o HTTP 403 ao alimentar o ffmpeg
# ---------------------------------------------------------------------------
#
# Por que dava 403: a URL "googlevideo" que o yt-dlp devolve é assinada e só
# funciona com (a) os mesmos headers que o yt-dlp associou ao formato,
# (b) download em blocos com Range, e (c) formatos que não exijam PO-token /
# desafio JS. Passar essa URL crua para `ffmpeg -i <url>` quebra em (a), (b)
# e (c). Aqui a ordem é invertida: o PRÓPRIO yt-dlp baixa o áudio (ele cuida de
# tudo isso) e o ffmpeg só decodifica um ARQUIVO LOCAL. Se o yt-dlp falhar,
# há fallbacks em camadas (download em blocos com os headers corretos e, por
# último, ffmpeg direto na URL com headers + reconexão).
#
# Variáveis de ambiente opcionais:
#   VISIONAUDIO_YT_COOKIES_BROWSER=firefox|chrome|...   (cookies do navegador)
#   VISIONAUDIO_YT_COOKIES_FILE=/caminho/cookies.txt
#   VISIONAUDIO_YT_REMOTE_COMPONENTS=1   (permite o yt-dlp baixar o solver JS
#                                          do GitHub; prefira instalar
#                                          `pip install -U "yt-dlp[default]"`)

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

# Clientes do YouTube tentados em ordem quando o padrão falha com 403.
_YT_CLIENT_LADDER = (None, ("tv",), ("android_vr",), ("web_safari",), ("mweb",))

YT_HELP_HINT = (
    "Dicas: (1) atualize o yt-dlp:  pip install -U \"yt-dlp[default]\"  "
    "(2) instale um runtime JavaScript no PATH (deno ou node) — o YouTube "
    "exige isso para liberar os formatos de áudio; "
    "(3) se persistir, exporte VISIONAUDIO_YT_COOKIES_BROWSER=firefox "
    "(ou chrome) para usar os cookies do navegador."
)


@dataclass
class RemoteStream:
    audio_url: str
    title: str
    video_url: Optional[str] = None
    page_url: Optional[str] = None
    audio_headers: dict = field(default_factory=dict)
    video_headers: dict = field(default_factory=dict)
    duration_s: Optional[float] = None


@dataclass
class RemoteAudio:
    y: np.ndarray
    sr: int
    local_path: Optional[str]     # arquivo baixado (None se veio direto por pipe)
    work_dir: Optional[str]       # pasta temporária (quem chamou deve apagar)


_REMOTE_STREAMS: dict = {}


def register_stream(stream: RemoteStream):
    for u in (stream.audio_url, stream.video_url):
        if u:
            _REMOTE_STREAMS[u] = stream


def lookup_stream(url) -> Optional[RemoteStream]:
    return _REMOTE_STREAMS.get(url)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def available_js_runtimes() -> list:
    return [n for n, exe in (("deno", "deno"), ("node", "node"), ("bun", "bun"), ("quickjs", "qjs"))
            if shutil.which(exe)]


def _ytdlp_options(player_clients=None) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "default_search": "ytsearch1",
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,
        "socket_timeout": 20,
        # o googlevideo derruba downloads "inteiros"; o yt-dlp baixa em blocos
        "http_chunk_size": 10 * 1024 * 1024,
    }
    runtimes = {name: {} for name in available_js_runtimes()}
    if runtimes:
        opts["js_runtimes"] = runtimes
    if _env_flag("VISIONAUDIO_YT_REMOTE_COMPONENTS"):
        opts["remote_components"] = ["ejs:github"]
    browser = os.environ.get("VISIONAUDIO_YT_COOKIES_BROWSER", "").strip()
    if browser:
        opts["cookiesfrombrowser"] = (browser,)
    cookie_file = os.environ.get("VISIONAUDIO_YT_COOKIES_FILE", "").strip()
    if cookie_file and os.path.isfile(cookie_file):
        opts["cookiefile"] = cookie_file
    if player_clients:
        opts["extractor_args"] = {"youtube": {"player_client": list(player_clients)}}
    return opts


def _short_err(exc: BaseException, limit: int = 220) -> str:
    msg = " ".join(str(exc).split())
    msg = msg.replace("ERROR: ", "")
    return msg if len(msg) <= limit else msg[: limit - 1] + "…"


def _import_ytdlp():
    try:
        import yt_dlp
        return yt_dlp
    except Exception as exc:
        raise RuntimeError("Biblioteca yt-dlp não encontrada. Instale com: pip install -U yt-dlp") from exc


def _pick_info(info):
    if info and "entries" in info and info["entries"]:
        info = next((e for e in info["entries"] if e), None)
    return info


def resolve_youtube_stream_info(query_or_url: str) -> RemoteStream:
    """Resolve título/URLs/headers sem baixar. Tenta vários clientes do YouTube."""
    yt_dlp = _import_ytdlp()
    errors = []
    for clients in _YT_CLIENT_LADDER:
        opts = _ytdlp_options(clients)
        opts["format"] = "bestaudio/best"
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = _pick_info(ydl.extract_info(query_or_url, download=False))
        except Exception as exc:
            errors.append(f"{'/'.join(clients) if clients else 'padrão'}: {_short_err(exc)}")
            continue
        if not info:
            continue

        formats = info.get("formats") or []
        stream_url = info.get("url")
        audio_headers = dict(info.get("http_headers") or {})
        if not stream_url:
            audio_only = [
                f for f in formats
                if f.get("url") and f.get("acodec") not in (None, "none")
                and f.get("vcodec") in (None, "none")
            ]
            if audio_only:
                audio_only.sort(key=lambda f: (f.get("abr") or 0.0, f.get("tbr") or 0.0))
                stream_url = audio_only[-1]["url"]
                audio_headers = dict(audio_only[-1].get("http_headers") or audio_headers)
        if not stream_url:
            errors.append(f"{'/'.join(clients) if clients else 'padrão'}: sem URL de áudio")
            continue

        av = [f for f in formats if f.get("url")
              and f.get("acodec") not in (None, "none") and f.get("vcodec") not in (None, "none")]
        vid = [f for f in formats if f.get("url") and f.get("vcodec") not in (None, "none")]
        pool = av or vid
        video_url, video_headers = None, {}
        if pool:
            pool.sort(key=lambda f: (f.get("height") or 0.0, f.get("fps") or 0.0,
                                     1 if f.get("acodec") not in (None, "none") else 0,
                                     f.get("tbr") or 0.0))
            video_url = pool[-1]["url"]
            video_headers = dict(pool[-1].get("http_headers") or {})

        stream = RemoteStream(
            audio_url=stream_url,
            title=info.get("title") or "YouTube",
            video_url=video_url or stream_url,
            page_url=info.get("webpage_url") or info.get("original_url")
            or (query_or_url if _is_url_str(query_or_url) else None),
            audio_headers=audio_headers,
            video_headers=video_headers or audio_headers,
            duration_s=float(info["duration"]) if info.get("duration") else None,
        )
        register_stream(stream)
        return stream
    raise RuntimeError("yt-dlp não conseguiu resolver o vídeo. " + " | ".join(errors) + "\n" + YT_HELP_HINT)


def _is_url_str(value) -> bool:
    return isinstance(value, str) and (value.startswith("http://") or value.startswith("https://"))


def _purge_partials(dest_dir: str):
    for p in glob.glob(os.path.join(dest_dir, "*")):
        try:
            os.remove(p)
        except OSError:
            pass


def _find_downloaded(info, dest_dir: str) -> Optional[str]:
    info = _pick_info(info) or {}
    for d in (info.get("requested_downloads") or []):
        fp = d.get("filepath") or d.get("_filename")
        if fp and os.path.isfile(fp) and os.path.getsize(fp) > 0:
            return fp
    cands = [p for p in glob.glob(os.path.join(dest_dir, "audio.*"))
             if not p.endswith((".part", ".ytdl", ".temp")) and os.path.getsize(p) > 0]
    return max(cands, key=os.path.getsize) if cands else None


def download_youtube_audio(target: str, dest_dir: str, log=None) -> str:
    """O yt-dlp baixa o áudio (headers, chunks, cookies e desafios JS por conta dele)."""
    log = log or (lambda m: None)
    yt_dlp = _import_ytdlp()
    errors = []
    for clients in _YT_CLIENT_LADDER:
        label = "/".join(clients) if clients else "padrão"
        opts = _ytdlp_options(clients)
        opts.update({
            "format": "bestaudio/best",
            "outtmpl": os.path.join(dest_dir, "audio.%(ext)s"),
            "overwrites": True,
            "noprogress": True,
        })
        try:
            log(f"yt-dlp baixando áudio (cliente: {label})...")
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(target, download=True)
            path = _find_downloaded(info, dest_dir)
            if path:
                return path
            errors.append(f"{label}: download terminou sem arquivo")
        except Exception as exc:
            errors.append(f"{label}: {_short_err(exc)}")
        _purge_partials(dest_dir)
    raise RuntimeError(" | ".join(errors))


def download_with_ranges(url: str, headers: dict, dest_path: str,
                         chunk: int = 10 * 1024 * 1024, timeout: float = 25.0) -> str:
    """Fallback: baixa em blocos com Range e os headers que o yt-dlp associou à URL."""
    hdrs = {"User-Agent": BROWSER_UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}
    hdrs.update({k: v for k, v in (headers or {}).items() if k.lower() != "range"})
    pos, total = 0, None
    with open(dest_path, "wb") as fh:
        while True:
            req = urllib.request.Request(url, headers={**hdrs, "Range": f"bytes={pos}-{pos + chunk - 1}"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read()
                    status = resp.status
                    content_range = resp.headers.get("Content-Range", "")
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and pos > 0:
                    break
                raise RuntimeError(f"HTTP {exc.code} ao baixar bloco em {pos} bytes") from exc
            if not data:
                break
            fh.write(data)
            pos += len(data)
            if status == 200:           # servidor ignorou Range e mandou tudo
                break
            tail = content_range.rsplit("/", 1)[-1]
            if tail.isdigit():
                total = int(tail)
            if (total is not None and pos >= total) or (total is None and len(data) < chunk):
                break
    if pos <= 0:
        raise RuntimeError("download em blocos retornou 0 bytes")
    return dest_path


def _ffmpeg_http_args(headers: Optional[dict]) -> list:
    headers = dict(headers or {})
    ua = BROWSER_UA
    for k in list(headers):
        if k.lower() == "user-agent":
            ua = headers.pop(k)
    args = ["-user_agent", ua]
    if headers:
        args += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    args += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    return args


def ffmpeg_decode_mono_pcm(source: str, target_sr: int, headers: Optional[dict] = None) -> np.ndarray:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg não encontrado.")
    cmd = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if _is_url_str(source):
        cmd += _ffmpeg_http_args(headers)
    cmd += ["-i", source, "-vn", "-ac", "1", "-ar", str(int(target_sr)),
            "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        err = result.stderr.decode(errors="ignore").strip()
        raise RuntimeError(f"falha ao decodificar: {err[-300:]}")
    pcm = np.frombuffer(result.stdout, dtype=np.int16)
    if pcm.size == 0:
        raise RuntimeError("áudio decodificado vazio")
    return pcm.astype(np.float32) / 32768.0


def make_playback_wav(local_audio_path: str, dest_path: str) -> str:
    """WAV 44,1 kHz estéreo para tocar (melhor que o WAV mono 22 kHz da análise)."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg não encontrado.")
    cmd = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", local_audio_path, "-vn", "-ac", "2", "-ar", "44100",
           "-acodec", "pcm_s16le", dest_path]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0 or not os.path.isfile(dest_path) or os.path.getsize(dest_path) < 1024:
        raise RuntimeError(result.stderr.decode(errors="ignore").strip()[-200:] or "wav vazio")
    return dest_path


def load_remote_audio(source: str, target_sr: int = 22050, work_dir: Optional[str] = None,
                      log=None) -> RemoteAudio:
    """Obtém o áudio de uma URL remota sem tomar 403.

    Ordem: (1) yt-dlp baixa → ffmpeg decodifica arquivo local;
           (2) download em blocos com headers corretos → ffmpeg local;
           (3) ffmpeg direto na URL com headers + reconexão.
    """
    log = log or (lambda m: None)
    own_dir = work_dir is None
    if own_dir:
        work_dir = tempfile.mkdtemp(prefix="ledsync_remote_")
    stream = lookup_stream(source)
    errors = []

    attempts = []
    if stream and stream.page_url:
        attempts.append(("yt-dlp", lambda: download_youtube_audio(stream.page_url, work_dir, log)))
    attempts.append((
        "download em blocos",
        lambda: download_with_ranges(
            source, stream.audio_headers if stream else {}, os.path.join(work_dir, "audio_direct.bin")
        ),
    ))
    for name, fn in attempts:
        try:
            local = fn()
            y = ffmpeg_decode_mono_pcm(local, target_sr)
            return RemoteAudio(y=y, sr=target_sr, local_path=local, work_dir=work_dir)
        except Exception as exc:
            errors.append(f"{name}: {_short_err(exc)}")
            log(f"{name} falhou: {_short_err(exc, 120)}")
            if name == "yt-dlp":
                _purge_partials(work_dir)

    try:
        log("tentando ffmpeg direto na URL (com headers)...")
        y = ffmpeg_decode_mono_pcm(source, target_sr, headers=stream.audio_headers if stream else None)
        return RemoteAudio(y=y, sr=target_sr, local_path=None, work_dir=work_dir)
    except Exception as exc:
        errors.append(f"ffmpeg direto: {_short_err(exc)}")

    if own_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    raise RuntimeError("Não foi possível obter o áudio remoto.\n  - " + "\n  - ".join(errors) + "\n" + YT_HELP_HINT)


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    """Partitura completa de uma faixa."""
    led_patterns: list          # [(modo, np.ndarray[6] PWM)] por frame
    dominant_leds: list         # LED dominante (1..4) por frame
    transitions: list           # list[TransitionEvent] (já refinadas)
    tempo_bpm: float
    beat_frames: list           # batidas, em frames
    n_frames: int
    report: dict                # métricas do motor (para a interface)
    report_lines: list          # o mesmo, em texto


class LedScoreEngine:
    """Transforma áudio em uma partitura de PWM (6 LEDs) — 100% offline."""

    def __init__(self, fps=None, config=None, middle_bpm_multiplier=1.0, log_callback=None):
        self.config = config or EffectConfig()
        self.fps = self.config.FPS if fps is None else int(fps)
        try:
            speed_mult = float(middle_bpm_multiplier)
        except Exception:
            speed_mult = 1.0
        if not np.isfinite(speed_mult) or speed_mult <= 0.0:
            speed_mult = 1.0
        self.middle_bpm_multiplier = speed_mult
        self.log_callback = log_callback
        self.report = {}
        self.report_lines = []
        self._progress_cb = None

    # ------------------------------------------------------------------ #
    # Utilidades                                                           #
    # ------------------------------------------------------------------ #

    def _note(self, message):
        self.report_lines.append(message)
        if self.log_callback:
            self.log_callback(message)

    def _progress(self, fraction, label):
        if self._progress_cb:
            self._progress_cb(float(min(1.0, max(0.0, fraction))), label)

    @staticmethod
    def _ema(prev, value, alpha):
        return alpha * value + (1.0 - alpha) * prev

    @staticmethod
    def _window_mean(arr, lo, hi):
        lo = int(np.clip(lo, 0, len(arr) - 1))
        hi = int(np.clip(hi, lo, len(arr) - 1))
        return float(np.mean(arr[lo:hi + 1]))

    @staticmethod
    def _cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom <= 1e-12:
            return 0.0
        cos_sim = float(np.dot(vec_a, vec_b) / denom)
        return float(np.clip(1.0 - cos_sim, 0.0, 2.0))

    @staticmethod
    def _snap_to_beats(
        transitions: list,
        beat_frames: np.ndarray,
        tempo_bpm: float,
        fps: int,
        cfg,
        n_frames: int,
    ) -> list:
        """
        Snapa frame_start de cada transição para o beat real mais próximo,
        e frame_end para o beat que minimiza o desvio da duração desejada.
        Garante que nenhuma transição sobreponha a próxima.
        """
        if not transitions:
            return []

        ordered = sorted(transitions, key=lambda t: t.frame_start)
        if len(beat_frames) < 2:
            return ordered

        bf = np.sort(beat_frames.astype(np.int32))
        beat_period = fps * 60.0 / tempo_bpm   # frames por beat (média)

        min_frames = max(2, int(cfg.TRANS_MIN_DURATION_S * fps))
        max_frames = int(cfg.TRANS_MAX_DURATION_S * fps)

        def nearest_beat(frame: int) -> int:
            idx = int(np.searchsorted(bf, frame))
            # compara vizinho esquerdo e direito
            candidates = [idx - 1, idx, idx + 1]
            best = min(
                (c for c in candidates if 0 <= c < len(bf)),
                key=lambda c: abs(bf[c] - frame),
            )
            return int(bf[best])

        def beat_after(frame: int, n: int) -> int:
            """Retorna o n-ésimo beat a partir de frame (inclusive)."""
            idx = int(np.searchsorted(bf, frame))
            idx = min(idx + n, len(bf) - 1)
            return int(bf[idx])

        def snap_end(start: int, original_dur: int) -> int:
            """Snapa o end ao beat mais próximo da duração desejada, respeitando limites."""
            n_beats = max(1, round(original_dur / beat_period))
            end = beat_after(start, n_beats)
            dur = end - start
            if dur < min_frames:
                end = start + min_frames
            elif dur > max_frames:
                end = start + max_frames
            return min(end, n_frames - 1)

        snapped = []
        for t in ordered:
            s = nearest_beat(t.frame_start)
            e = snap_end(s, t.frame_end - t.frame_start)
            snapped.append(TransitionEvent(
                frame_start     = s,
                frame_end       = e,
                effect_type     = t.effect_type,
                pwm_brightness  = t.pwm_brightness,
                beat_norm_after = t.beat_norm_after,
                kick_mean_after = t.kick_mean_after,
            ))

        # Remove sobreposições — FIX #6: re-snapa o end após resolução de colisão
        # (no original, new_end = new_start + dur usava a duração antiga não-beatizada).
        cleaned = [snapped[0]]
        for t in snapped[1:]:
            prev_end = cleaned[-1].frame_end
            if t.frame_start < prev_end:
                new_start = nearest_beat(prev_end)
                if new_start <= prev_end:
                    new_start = beat_after(prev_end, 1)
                # FIX #6: re-snapa duração ao beat grid após deslocar o start
                new_end = snap_end(new_start, t.frame_end - t.frame_start)
                cleaned.append(TransitionEvent(
                    frame_start     = new_start,
                    frame_end       = new_end,
                    effect_type     = t.effect_type,
                    pwm_brightness  = t.pwm_brightness,
                    beat_norm_after = t.beat_norm_after,
                    kick_mean_after = t.kick_mean_after,
                ))
            else:
                cleaned.append(t)

        return cleaned

    # ------------------------------------------------------------------ #
    # Carregamento de áudio                                                #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Ponto de entrada                                                     #
    # ------------------------------------------------------------------ #

    def compute_score(self, y, sr, progress_cb=None):
        """Beats → estrutura → transições → partitura. `progress_cb(fração, rótulo)`."""
        self._progress_cb = progress_cb
        self.report_lines = []
        self.report = {}
        n_frames = max(1, int(len(y) / float(sr) * self.fps))

        self._progress(0.02, "Calculando BPM...")
        tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr)
        try:
            tempo_bpm = float(np.asarray(tempo).reshape(-1)[0])
        except Exception:
            tempo_bpm = 120.0
        if not math.isfinite(tempo_bpm) or tempo_bpm <= 1.0:
            tempo_bpm = 120.0
        beat_frames = (librosa.frames_to_time(beat_times, sr=sr) * self.fps).astype(np.int32)

        self._progress(0.10, "Detectando estrutura e transições...")
        transitions = self._analyze_structure(y, sr)
        transitions = self._snap_to_beats(
            transitions, beat_frames, tempo_bpm, self.fps, self.config, n_frames
        )

        led_patterns, dominant_leds = self._generate_led_score(
            y, sr, n_frames, transitions, tempo_bpm, beat_frames
        )
        self._progress_cb = None
        return ScoreResult(
            led_patterns=led_patterns,
            dominant_leds=dominant_leds,
            transitions=transitions,
            tempo_bpm=tempo_bpm,
            beat_frames=[int(b) for b in beat_frames],
            n_frames=n_frames,
            report=dict(self.report),
            report_lines=list(self.report_lines),
        )

    # ------------------------------------------------------------------ #
    # FASE 1 — Análise estrutural completa da música                       #
    # ------------------------------------------------------------------ #

    def _analyze_structure(self, y: np.ndarray, sr: int) -> list:
        cfg = self.config
        self._note("Separando harmônico/percussivo (HPSS)...")
        y_harm, y_perc = librosa.effects.hpss(y, margin=3.0)

        hop_samples = int(cfg.SEG_HOP_SECONDS * sr)

        self._note("Calculando features espectrais por janela...")

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

        S         = np.abs(librosa.stft(y, hop_length=hop_samples))
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=(S.shape[0] - 1) * 2)
        low_mask  = (freq_bins >= cfg.LOW_BAND_MIN_HZ) & (freq_bins <= cfg.LOW_BAND_MAX_HZ)

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

        mfcc_norm   = librosa.util.normalize(mfcc, axis=1)
        chroma_norm = librosa.util.normalize(chroma, axis=0)
        feat_matrix = np.vstack([mfcc_norm, chroma_norm])  # (25, n_win)

        self._note("Segmentação aglomerativa...")

        k = min(cfg.SEG_K_SEGMENTS, max(2, n_win // 4))
        try:
            bound_frames_seg = librosa.segment.agglomerative(feat_matrix, k=k)
        except Exception:
            bound_frames_seg = np.array([0, n_win - 1])

        bound_times = librosa.frames_to_time(
            bound_frames_seg, sr=sr, hop_length=hop_samples
        )

        novelty_simple = np.zeros(n_win)
        for w in range(1, n_win):
            diff = feat_matrix[:, w] - feat_matrix[:, w - 1]
            novelty_simple[w] = float(np.linalg.norm(diff))
        novelty_simple = novelty_simple / (novelty_simple.max() + 1e-9)

        ignore_start = cfg.SEG_IGNORE_START_SECONDS
        min_gap      = cfg.SEG_MIN_GAP_SECONDS
        wins_ctx     = max(1, int(np.ceil(2.0 / cfg.SEG_HOP_SECONDS)))
        persist_wins = max(
            1, int(np.ceil(cfg.SEG_PERSISTENCE_AHEAD_SECONDS / cfg.SEG_HOP_SECONDS))
        )

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

            w_bef_lo = max(0, b_win - wins_ctx)
            w_bef_hi = max(0, b_win - 1)
            w_aft_lo = min(n_win - 1, b_win)
            w_aft_hi = min(n_win - 1, b_win + wins_ctx)
            w_far_lo = min(n_win - 1, b_win + persist_wins)
            w_far_hi = min(n_win - 1, w_far_lo + wins_ctx)

            vec_bef = np.mean(feat_matrix[:, w_bef_lo:w_bef_hi + 1], axis=1)
            vec_aft = np.mean(feat_matrix[:, w_aft_lo:w_aft_hi + 1], axis=1)
            vec_far = np.mean(feat_matrix[:, w_far_lo:w_far_hi + 1], axis=1)

            struct_delta = self._cosine_distance(vec_bef, vec_aft)
            persist_delta = self._cosine_distance(vec_bef, vec_far)
            persist_ratio = persist_delta / (struct_delta + 1e-9)

            if struct_delta < cfg.SEG_STRUCT_CHANGE_THRESHOLD:
                continue

            is_short_change = (
                persist_delta < cfg.SEG_MIN_PERSIST_DELTA
                or persist_ratio < cfg.SEG_PERSISTENCE_MIN_RATIO
            )

            valid_boundaries.append(
                (b_time, struct_delta, persist_delta, persist_ratio, is_short_change)
            )
            last_accepted_time = b_time

        self._note(f"{len(valid_boundaries)} fronteiras de seção detectadas")

        p90_perc = float(np.percentile(rms_perc, 90)) + 1e-9
        p90_rms  = float(np.percentile(rms_total, 90)) + 1e-9
        p95_low  = float(np.percentile(low_energy_per_win, 95)) + 1e-9

        transitions = []
        wm = self._window_mean

        for (
            b_time,
            struct_delta,
            persist_delta,
            persist_ratio,
            is_short_change,
        ) in valid_boundaries:
            b_win = int(np.clip(int(b_time / cfg.SEG_HOP_SECONDS), 0, n_win - 1))

            w_bef_lo = max(0, b_win - wins_ctx)
            w_bef_hi = max(0, b_win - 1)
            w_aft_lo = min(n_win - 1, b_win)
            w_aft_hi = min(n_win - 1, b_win + wins_ctx)

            rms_bef  = wm(rms_total,          w_bef_lo, w_bef_hi)
            rms_aft  = wm(rms_total,          w_aft_lo, w_aft_hi)
            perc_aft = wm(rms_perc,           w_aft_lo, w_aft_hi)
            flat_aft = wm(flatness,           w_aft_lo, w_aft_hi)
            cent_aft = wm(spec_centroid,      w_aft_lo, w_aft_hi)
            low_aft  = wm(low_energy_per_win, w_aft_lo, w_aft_hi)

            beat_norm_after = float(np.clip(perc_aft / p90_perc, 0.0, 1.0))
            kick_mean_after = float(np.clip(low_aft / p95_low, 0.0, 1.0))
            energy_ratio    = rms_aft / (rms_bef + 1e-9)
            energy_norm_aft = float(np.clip(rms_aft / p90_rms, 0.0, 1.0))

            # Rejeita fronteiras onde o trecho seguinte tem energia irrisória
            if beat_norm_after + kick_mean_after < cfg.SEG_MIN_ENERGY_THRESHOLD:
                self._note(
                    f"t={b_time:6.2f}s  ignorada (silêncio)"
                    f"  beat={beat_norm_after:.2f}  kick={kick_mean_after:.2f}"
                    f"  Δ={struct_delta:.2f}  Δp={persist_delta:.2f}"
                )
                continue

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

            self._note(
                f"t={b_time:6.2f}s  {eff_type.name:<14}"
                f"  beat={beat_norm_after:.2f}  kick={kick_mean_after:.2f}"
                f"  Δ={struct_delta:.2f}  Δp={persist_delta:.2f}  pers={persist_ratio:.2f}"
                f"{'  curta' if is_short_change else ''}"
                f"  flat={flat_aft:.2f}  dur={dur_s:.2f}s"
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

        if flatness > (cfg.NOISE_FLATNESS_THRESHOLD + 0.08) and beat_norm > 0.30:
            return TransitionEffectType.ALTERNATE

        if flatness > cfg.NOISE_FLATNESS_THRESHOLD:
            return TransitionEffectType.GLITCH

        # Impacto súbito (energia sobe muito + kick alto)
        if energy_rising and kick_strength > 0.70:
            return TransitionEffectType.BURST

        if energy_rising and kick_strength > 0.55:
            return TransitionEffectType.STROBE

        if beat_norm > 0.78 and kick_strength > 0.50:
            return TransitionEffectType.SWEEP

        if beat_norm > 0.72 and 0.30 <= kick_strength <= 0.55:
            return TransitionEffectType.WAVE

        if kick_strength > 0.55 or (energy_rising and beat_norm > 0.45):
            return TransitionEffectType.REVERSE_SWEEP

        # Percussão alta / hi-hats sem muita energia de graves
        if (
            centroid_norm > cfg.BRIGHT_CENTROID_THRESHOLD
            and kick_strength < 0.35
            and beat_norm > 0.35
        ):
            return TransitionEffectType.RIPPLE

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
            if centroid_norm > (cfg.BRIGHT_CENTROID_THRESHOLD * 0.90):
                return TransitionEffectType.ALTERNATE
            return TransitionEffectType.PING_PONG

        if beat_norm > 0.20:
            return TransitionEffectType.WAVE

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
            # FIX #1: era t * 5.0 — sweep saía do array em t=0.6 (pos=3);
            # os últimos 40% da animação ficavam apagados. t * 3.0 percorre os 4 LEDs.
            pos = t * 3.0
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
            rng     = np.random.default_rng(int(t * 60.0) + 7919)
            on_mask = rng.random(4) > 0.38
            amp     = rng.uniform(0.35, 1.0, 4)
            levels  = on_mask.astype(np.float32) * amp.astype(np.float32) * B
            if t > 0.70:
                levels *= (1.0 - (t - 0.70) / 0.30)

        elif effect_type == TransitionEffectType.BREATHE:
            # FIX #14: era todos os 4 LEDs idênticos — sem variação espacial.
            # LEDs externos (0,3) ligeiramente defasados dos internos (1,2).
            base   = float(np.sin(t * np.pi))
            offset = float(np.sin(t * np.pi + 0.4))
            levels[0] = base   * B
            levels[3] = base   * B
            levels[1] = offset * B
            levels[2] = offset * B

        elif effect_type == TransitionEffectType.WAVE:
            cycles = 1.2 + beat_norm * 1.6
            for idx in range(4):
                phase = t * (2.0 * np.pi * cycles) - idx * 0.90
                wave = 0.5 + 0.5 * np.sin(phase)
                levels[idx] = (0.28 + 0.72 * wave) * B
            if t > 0.82:
                levels *= (1.0 - (t - 0.82) / 0.18)

        elif effect_type == TransitionEffectType.ALTERNATE:
            hz    = 3.5 + beat_norm * 6.0 + kick_strength * 2.5
            phase = (int(t * hz * 2.0) % 2) == 0
            levels[:] = 0.22 * B
            if phase:
                levels[0] = B; levels[2] = B
                levels[1] = 0.32 * B; levels[3] = 0.32 * B
            else:
                levels[1] = B; levels[3] = B
                levels[0] = 0.32 * B; levels[2] = 0.32 * B
            if t > 0.78:
                levels *= (1.0 - (t - 0.78) / 0.22)

        elif effect_type == TransitionEffectType.RIPPLE:
            center_peak = float(np.exp(-((t - 0.22) / 0.20) ** 2))
            edge_peak   = float(np.exp(-((t - 0.56) / 0.22) ** 2))
            tail        = float(np.exp(-((t - 0.84) / 0.16) ** 2))
            levels[1] = B * (0.80 * center_peak + 0.20 * tail)
            levels[2] = B * (0.80 * center_peak + 0.20 * tail)
            levels[0] = B * (0.85 * edge_peak + 0.15 * tail)
            levels[3] = B * (0.85 * edge_peak + 0.15 * tail)

        elif effect_type == TransitionEffectType.PING_PONG:
            raw = (t * 3.0) % 2.0
            pos = (raw if raw <= 1.0 else 2.0 - raw) * 3.0
            for idx in range(4):
                levels[idx] = max(0.0, 1.0 - abs(idx - pos) * 0.75) * B
            if t > 0.75:
                levels *= (1.0 - (t - 0.75) / 0.25)

        elif effect_type == TransitionEffectType.SPLIT:
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
            hz     = 12.0 + beat_norm * 8.0
            tick   = int(t * hz * 4.0)
            rng    = np.random.default_rng(tick + 3571)
            n_on   = rng.choice([1, 1, 2])
            on_idx = rng.choice(4, size=n_on, replace=False)
            amp    = rng.uniform(0.6, 1.0, n_on)
            for k, idx in enumerate(on_idx):
                levels[idx] = amp[k] * B
            if t > 0.65:
                levels *= (1.0 - (t - 0.65) / 0.35)

        elif effect_type == TransitionEffectType.BURST:
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
        beat_frames: np.ndarray,
    ) -> tuple:
        cfg = self.config

        samples_per_frame = int(sr / self.fps)
        bytes_per_frame   = samples_per_frame * 2
        pcm       = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        pcm_bytes = pcm.tobytes()

        # FIX #4: freqs/low_mask/high_mask são constantes — pré-computar uma vez.
        freqs     = np.fft.rfftfreq(samples_per_frame, 1.0 / sr)
        low_mask  = (freqs >= cfg.LOW_BAND_MIN_HZ) & (freqs <= cfg.LOW_BAND_MAX_HZ)
        high_mask = (freqs >= 2000.0)              & (freqs <= 8000.0)
        _has_low  = bool(np.any(low_mask))
        _has_high = bool(np.any(high_mask))

        # ---- LEDs do meio: análise offline + renderizador adaptativo ---- #
        self._progress(0.42, "Analisando ataques, compasso, seções e drops...")
        mid_analysis = MiddleAnalyzer(cfg, self.fps).analyze(
            y, sr, n_frames, tempo_bpm, beat_frames
        )
        # precisão das transições: ponto exato da virada + encaixe no compasso
        refine_stats = refine_transitions(cfg, self.fps, mid_analysis, transitions, n_frames)
        mid_renderer = MiddleRenderer(
            cfg, self.fps, mid_analysis, self.middle_bpm_multiplier, n_frames,
            section_frames=[t.frame_start for t in transitions],
        )
        trans_energy_ref = 1.0
        middle_leds = np.array([1, 2, 3, 4], dtype=np.int32)

        trans_index  = 0
        n_trans      = len(transitions)
        active_trans = None   # TransitionEvent | None
        trans_just_ended = False

        # ---- estado das PONTAS (inalterado) ----------------------------- #
        edge_levels   = np.zeros(2, dtype=np.float32)

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

        # FIX #5: cache do percentil (atualizado a cada 10 frames)
        p95_low    = 1e-9

        for i in range(n_frames):
            # ---- áudio deste frame ----------------------------------- #
            start_b = i * bytes_per_frame
            end_b   = start_b + bytes_per_frame
            chunk   = pcm_bytes[start_b:end_b]

            frame = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            if len(frame) < samples_per_frame:
                frame = np.pad(frame, (0, samples_per_frame - len(frame)))
            frame = frame / 32768.0

            spectrum = np.abs(np.fft.rfft(frame))
            low_band_energy  = float(np.mean(spectrum[low_mask]))  if _has_low  else 0.0
            high_band_energy = float(np.mean(spectrum[high_mask])) if _has_high else 0.0

            if i == 0:
                low_fast = low_band_energy; low_slow = low_band_energy
            else:
                low_fast = self._ema(low_fast, low_band_energy, cfg.LOW_FAST_ALPHA)
                low_slow = self._ema(low_slow, low_band_energy, cfg.LOW_SLOW_ALPHA)

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

            low_band_window.append(low_band_energy)

            if i % 10 == 0:
                p95_low = float(np.percentile(np.array(low_band_window), 95)) + 1e-9
            low_norm    = float(np.clip(low_band_energy / p95_low, 0.0, 1.0))

            startup_factor = float(np.clip((i + 1) / startup_frames, 0.0, 1.0))
            startup_gain   = 0.45 + 0.55 * startup_factor

            if kick_window_left > 0:
                kick_window_left -= 1

            # ---- pontas (lógica inalterada) -------------------------- #
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
            if edge_pulse > 1.0:
                edge_levels[0] = max(edge_levels[0], edge_pulse)
                edge_levels[1] = max(edge_levels[1], edge_pulse)

            prev_kick_strength = kick_strength
            prev_low_norm      = low_norm

            # ---- transição agendada ----------------------------------- #
            if active_trans is None and trans_index < n_trans:
                if i >= transitions[trans_index].frame_start:
                    active_trans = transitions[trans_index]
                    trans_index += 1
                    span = mid_analysis.energy[
                        active_trans.frame_start:max(active_trans.frame_start + 1, active_trans.frame_end)
                    ]
                    trans_energy_ref = float(np.mean(span)) if len(span) else 1.0

            # ---- LEDs do meio ---------------------------------------- #
            # O motor roda SEMPRE (mesmo durante transições): assim o relógio
            # dos passos, o estado dos padrões e o AGC continuam coerentes e
            # a saída da transição não dá "salto".
            under_levels, active_mid = mid_renderer.frame(
                i,
                startup_gain     = startup_gain,
                edge_pulse       = edge_pulse,
                trans_active     = active_trans is not None,
                trans_just_ended = trans_just_ended,
            )
            trans_just_ended = False
            middle_levels = np.asarray(under_levels, dtype=np.float32)

            if active_trans is not None:
                local_frame  = i - active_trans.frame_start
                total_frames = active_trans.frame_end - active_trans.frame_start
                progress     = float(np.clip(
                    local_frame / max(1, total_frames - 1), 0.0, 1.0
                ))

                # o efeito respira com a energia ao vivo da música
                live_ratio = float(np.clip(
                    mid_analysis.energy[min(i, n_frames - 1)] / (trans_energy_ref + 1e-6), 0.5, 1.3
                ))
                live_gain = (1.0 - cfg.TRANS_LIVE_GAIN) + cfg.TRANS_LIVE_GAIN * live_ratio

                effect_levels = self._render_transition_frame(
                    effect_type    = active_trans.effect_type,
                    progress       = progress,
                    beat_norm      = active_trans.beat_norm_after,
                    kick_strength  = active_trans.kick_mean_after,
                    pwm_brightness = min(235.0, active_trans.pwm_brightness * startup_gain * live_gain),
                )

                # rampa de entrada/saída em frações de batida (não em frames fixos)
                beat_frames_now = float(mid_analysis.period[min(i, n_frames - 1)])
                bf = max(2, min(int(cfg.TRANS_BLEND_FRAMES), int(round(0.5 * beat_frames_now))))
                if local_frame < bf:
                    alpha = float(local_frame) / float(bf)
                elif local_frame > total_frames - bf:
                    alpha = float(total_frames - local_frame) / float(max(1, bf))
                else:
                    alpha = 1.0
                alpha = float(np.clip(alpha, 0.0, 1.0))

                middle_levels = middle_levels * (1.0 - alpha) + effect_levels * alpha
                active_mid = int(np.argmax(middle_levels))

                if i >= active_trans.frame_end:
                    active_trans     = None
                    trans_just_ended = True

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
                self._progress(0.55 + 0.45 * (i + 1) / n_frames, "Gerando LEDs...")

        self._build_report(mid_analysis, mid_renderer, n_frames, refine_stats)
        return led_patterns, dominant_leds

    def _build_report(self, ma, rd, n_frames, refine):
        info, st = ma.info, rd.summary()
        total = max(1, n_frames)
        names = ("kick", "grave", "médio", "presença", "brilho")
        shifts_ms = [abs(s) * 1000.0 / self.fps for s in refine.get("shifts", [])]
        drops_s = [round(f / self.fps, 2) for f in ma.drop_frames]
        self.report = {
            "n_beats": info["n_beats"],
            "synthetic_grid": info["synthetic_grid"],
            "beat_confidence": info["mean_beat_conf"],
            "downbeat_confidence": ma.downbeat_conf,
            "attacks": {names[g]: info["onsets"][g] for g in range(MID_NUM_BANDS)},
            "bars": st["bars"],
            "steps_per_beat": dict(st["spb_hist"]),
            "pattern_switches": st["pattern_switches"],
            "section_switches": st["section_switches"],
            "governor_pct": 100.0 * st["governor_frames"] / total,
            "transitions_refined": refine.get("refined", 0),
            "transitions_on_downbeat": refine.get("to_downbeat", 0),
            "refine_mean_shift_ms": float(np.mean(shifts_ms)) if shifts_ms else 0.0,
            "refine_max_shift_ms": float(np.max(shifts_ms)) if shifts_ms else 0.0,
            "drops_s": drops_s,
        }
        r = self.report
        grid = "sintético (BPM)" if r["synthetic_grid"] else f"{r['n_beats']} batidas rastreadas"
        spb = "  ".join(f"{k}x:{v}" for k, v in sorted(r["steps_per_beat"].items(), key=lambda kv: float(kv[0])))
        self._note("Motor dos LEDs do meio:")
        self._note("  ataques por banda → " + "  ".join(f"{k}:{v}" for k, v in r["attacks"].items()))
        self._note(f"  grid: {grid}  |  confiança de pulso {r['beat_confidence']:.2f}"
                   f"  |  confiança de compasso {r['downbeat_confidence']:.2f}")
        self._note(f"  compassos: {r['bars']}  |  passos/batida → {spb}")
        self._note(f"  sequências: {r['pattern_switches']} trocas ({r['section_switches']} em início de seção)"
                   f"  |  trava anti-saturação: {r['governor_pct']:.1f}% do tempo")
        if shifts_ms:
            self._note(f"  transições refinadas: {r['transitions_refined']}  |  {r['transitions_on_downbeat']} no tempo 1"
                       f"  |  ajuste médio {r['refine_mean_shift_ms']:.0f} ms (máx {r['refine_max_shift_ms']:.0f} ms)")
        self._note("  drops detectados: " + (", ".join(f"{d:.1f}s" for d in drops_s) if drops_s else "nenhum"))


if __name__ == "__main__":
    print("audio_reader6 é o motor de partituras. Abra a interface:  python3 visionaudio7_2.py")
