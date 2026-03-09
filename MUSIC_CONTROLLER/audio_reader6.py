from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional
import wave

import librosa
import numpy as np
import pygame
import serial


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

class C:
    RESET    = "\033[0m";  BOLD     = "\033[1m";  DIM      = "\033[2m"
    RED      = "\033[31m"; GREEN    = "\033[32m"; YELLOW   = "\033[33m"
    BLUE     = "\033[34m"; MAGENTA  = "\033[35m"; CYAN     = "\033[36m"
    BRED     = "\033[91m"; BGREEN   = "\033[92m"; BYELLOW  = "\033[93m"
    BBLUE    = "\033[94m"; BMAGENTA = "\033[95m"; BCYAN    = "\033[96m"
    BWHITE   = "\033[97m"


def _c(text, *codes):
    return "".join(codes) + str(text) + C.RESET


def banner():
    lines = [
        "  ██╗     ███████╗██████╗     ███████╗██╗   ██╗███╗   ██╗ ██████╗ ",
        "  ██║     ██╔════╝██╔══██╗    ██╔════╝╚██╗ ██╔╝████╗  ██║██╔════╝ ",
        "  ██║     █████╗  ██║  ██║    ███████╗ ╚████╔╝ ██╔██╗ ██║██║      ",
        "  ██║     ██╔══╝  ██║  ██║    ╚════██║  ╚██╔╝  ██║╚██╗██║██║      ",
        "  ███████╗███████╗██████╔╝    ███████║   ██║   ██║ ╚████║╚██████╗ ",
        "  ╚══════╝╚══════╝╚═════╝     ╚══════╝   ╚═╝   ╚═╝  ╚═══╝ ╚═════╝ ",
    ]
    width = len(lines[0])
    sep = _c("─" * (width + 4), C.DIM, C.CYAN)
    print()
    print(sep)
    for line in lines:
        print(_c("  " + line, C.BMAGENTA, C.BOLD))
    print(_c("  " + " " * 22 + "🎵  Music → LED  🎵", C.BYELLOW, C.BOLD))
    print(sep)
    print()


def log_ok(msg):   print(_c("  ✓ ", C.BGREEN,   C.BOLD) + _c(msg, C.BWHITE))
def log_err(msg):  print(_c("  ✗ ", C.BRED,     C.BOLD) + _c(msg, C.BWHITE))
def log_warn(msg): print(_c("  ⚠  ", C.BYELLOW, C.BOLD) + _c(msg, C.BWHITE))
def log_step(msg): print(_c("\n  ▶ ", C.BBLUE,  C.BOLD) + _c(msg, C.BWHITE, C.BOLD))
def log_sub(msg):  print(_c("    │ ", C.DIM, C.BLUE) + _c(msg, C.BWHITE))


def section_line():
    print(_c("  " + "─" * 64, C.DIM, C.CYAN))


def progress_bar(current, total, width=30, label=""):
    pct    = current / max(total, 1)
    filled = int(pct * width)
    bar    = _c("█" * filled, C.BMAGENTA) + _c("░" * (width - filled), C.DIM)
    pct_s  = _c(f"{pct*100:5.1f}%", C.BYELLOW, C.BOLD)
    print(f"    │ {bar} {pct_s}  {_c(label, C.DIM)}", end="\r")


SEEK_STEP_S = 10


class KeyReader:
    """Leitura de teclas sem bloqueio; reconhece setas (sequências de escape)."""

    def __init__(self):
        self.enabled = False
        self._is_win = os.name == "nt"
        self._fd: Optional[int] = None
        self._old_attrs = None

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        if self._is_win:
            try:
                import msvcrt  # noqa: F401
                self.enabled = True
            except Exception:
                pass
            return self
        try:
            import termios

            self._fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            attrs = termios.tcgetattr(self._fd)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSADRAIN, attrs)
            self.enabled = True
        except Exception:
            pass
        return self

    def __exit__(self, *_):
        if self._is_win or self._fd is None or self._old_attrs is None:
            return
        try:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
        except Exception:
            pass

    def read_key(self) -> Optional[str]:
        if not self.enabled:
            return None

        if self._is_win:
            import msvcrt

            if not msvcrt.kbhit():
                return None
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                ch2 = msvcrt.getwch()
                return {"K": "LEFT", "M": "RIGHT", "H": "UP", "P": "DOWN"}.get(ch2)
            return ch

        import select

        if self._fd is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        try:
            ch = os.read(self._fd, 1).decode("utf-8", errors="ignore")
        except Exception:
            return None

        if ch == "\x1b":
            r2, _, _ = select.select([sys.stdin], [], [], 0.04)
            if r2:
                try:
                    rest = os.read(self._fd, 4).decode("utf-8", errors="ignore")
                    if rest.startswith("["):
                        return {
                            "A": "UP",
                            "B": "DOWN",
                            "C": "RIGHT",
                            "D": "LEFT",
                        }.get(rest[1:2], "ESC")
                except Exception:
                    pass
            return "ESC"
        return ch


def _fit_text(text: str, width: int) -> str:
    s = str(text).replace("\n", " ")
    if width <= 0:
        return ""
    if len(s) > width:
        return s[: width - 3] + "..." if width > 3 else s[:width]
    return s.ljust(width)


def _fmt_dur(s: Optional[float]) -> str:
    if s is None:
        return "--:--"
    m, sec = divmod(int(max(0, s)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


class PlaybackBox:
    """HUD de reprodução no mesmo estilo do youtube_player."""

    def __init__(
        self,
        title: str,
        player_name: str,
        duration_s: Optional[float],
        queue_idx: int = 0,
        queue_total: int = 1,
    ):
        self.title = title
        self.player_name = player_name
        self.duration_s = duration_s
        self.queue_idx = queue_idx
        self.queue_total = queue_total
        self.enabled = sys.stdout.isatty()
        self._line_count = 0

    def _progress_bar(self, elapsed: float, inner: int, frame_idx: int) -> str:
        el_str = _fmt_dur(elapsed)
        tot_str = _fmt_dur(self.duration_s)
        prefix = f" {el_str} "
        suffix = f" {tot_str}"
        space = max(6, inner - len(prefix) - len(suffix) - 4)
        if self.duration_s and self.duration_s > 0:
            ratio = min(1.0, max(0.0, elapsed / self.duration_s))
            filled = int(space * ratio)
            bar = "▓" * filled + "░" * (space - filled)
        else:
            cur = frame_idx % space
            bar = "".join("▓" if i == cur else "░" for i in range(space))
        return f"{prefix}[{bar}]{suffix}"

    def draw(
        self,
        *,
        state: str,
        frame_idx: int,
        elapsed_s: float,
        paused: bool,
        supports_seek: bool = True,
    ):
        if not self.enabled:
            return

        cols = shutil.get_terminal_size((92, 20)).columns
        width = min(max(cols, 72), 132)
        inner = width - 2

        ctrl_parts = ["p/⎵ pause"]
        if supports_seek:
            ctrl_parts.append("[ ] seek±10s")
            ctrl_parts.append("←/→ seek±10s")
        ctrl_parts.append("q sair")
        ctrl_hint = "  |  ".join(ctrl_parts) + f"  —  {self.player_name}"

        state_color = C.BGREEN if state == "TOCANDO" else C.BYELLOW
        queue_str = f"[{self.queue_idx + 1}/{self.queue_total}]"

        sep = _c("+" + "─" * inner + "+", C.BCYAN, C.BOLD)
        out = [
            sep,
            _c("|" + _fit_text(f"  {state}  {queue_str}", inner) + "|", state_color, C.BOLD),
            _c("|" + _fit_text(f"  Faixa: {self.title}", inner) + "|", C.BWHITE),
            _c(
                "|" + _fit_text(
                    f"  Progresso: {self._progress_bar(elapsed_s, inner, frame_idx)}",
                    inner,
                ) + "|",
                C.BCYAN,
            ),
            _c("|" + _fit_text(f"  {ctrl_hint}", inner) + "|", C.DIM),
            sep,
        ]

        if self._line_count:
            sys.stdout.write(f"\033[{self._line_count}F")
        for line in out:
            sys.stdout.write("\r\033[2K" + line + "\n")
        sys.stdout.flush()
        self._line_count = len(out)


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


_EFFECT_ICONS = {
    TransitionEffectType.SWEEP:         ("→→→", C.BBLUE),
    TransitionEffectType.REVERSE_SWEEP: ("←←←", C.BCYAN),
    TransitionEffectType.STROBE:        ("◆◆◆", C.BRED),
    TransitionEffectType.CASCADE:       ("◈◈◈", C.BYELLOW),
    TransitionEffectType.GLITCH:        ("▒▒▒", C.BMAGENTA),
    TransitionEffectType.BREATHE:       ("~~~", C.BGREEN),
    TransitionEffectType.PING_PONG:     ("↔↔↔", C.BCYAN),
    TransitionEffectType.SPLIT:         ("◄►◄", C.BYELLOW),
    TransitionEffectType.SPARKLE:       ("✦✦✦", C.BWHITE),
    TransitionEffectType.BURST:         ("❋❋❋", C.BRED),
}


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

    RMS_FAST_ALPHA: float = 0.30
    RMS_SLOW_ALPHA: float = 0.06
    FLUX_FAST_ALPHA: float = 0.35
    FLUX_SLOW_ALPHA: float = 0.07

    STARTUP_RAMP_SECONDS: float = 2.0

    # ------------------------------------------------------------------ #
    # Segmentação estrutural                                               #
    # ------------------------------------------------------------------ #

    SEG_HOP_SECONDS: float = 1.25
    SEG_K_SEGMENTS: int = 40
    SEG_MIN_GAP_SECONDS: float = 2.0
    SEG_IGNORE_START_SECONDS: float = 2.5
    SEG_NOVELTY_THRESHOLD: float = 0.18
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

    # FIX #7: frames de lag acumulado antes de pular um frame visual.
    PLAYBACK_MAX_DRIFT_FRAMES: int = 2


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

    def __repr__(self) -> str:
        fps   = 30
        t_s   = self.frame_start / fps
        dur_s = (self.frame_end - self.frame_start) / fps
        return (
            f"TransitionEvent(t={t_s:.2f}s, dur={dur_s:.2f}s, "
            f"effect={self.effect_type.name}, brightness={self.pwm_brightness:.0f})"
        )


# ---------------------------------------------------------------------------
# Sessão de reprodução (áudio)
# ---------------------------------------------------------------------------

class PlaybackSession:
    def __init__(
        self,
        controller,
        source: str,
        duration_s: Optional[float],
        youtube_video_source: Optional[str] = None,
    ):
        self.controller = controller
        self.source = source
        self.duration_s = duration_s
        self.youtube_video_source = youtube_video_source
        self.backend = "none"
        self.player_name = "none"
        self.proc: Optional[subprocess.Popen] = None
        self.supports_pause = False
        self.supports_seek = False
        self.paused = False
        self._base_pos_s = 0.0
        self._started_at = 0.0
        self._paused_acc = 0.0
        self._pause_ts: Optional[float] = None

    def _reset_timing(self, base_pos_s: float = 0.0):
        self._base_pos_s = max(0.0, float(base_pos_s))
        self._started_at = time.monotonic()
        self._paused_acc = 0.0
        self._pause_ts = self._started_at if self.paused else None

    def _set_paused_state(self, new_state: bool):
        if new_state == self.paused:
            return
        now = time.monotonic()
        if new_state:
            self._pause_ts = now
        else:
            if self._pause_ts is not None:
                self._paused_acc += max(0.0, now - self._pause_ts)
            self._pause_ts = None
        self.paused = new_state

    def _clock_elapsed(self) -> float:
        now = time.monotonic()
        pause_window = (now - self._pause_ts) if self._pause_ts is not None else 0.0
        elapsed = self._base_pos_s + (now - self._started_at - self._paused_acc - pause_window)
        if self.duration_s is not None:
            elapsed = min(elapsed, float(self.duration_s))
        return max(0.0, float(elapsed))

    def start(self) -> bool:
        if self.controller.show_youtube_video and self.youtube_video_source:
            proc, player_name = self.controller._start_youtube_video_process(
                self.youtube_video_source
            )
            if proc is not None:
                self.backend = "proc"
                self.proc = proc
                self.player_name = player_name
                self.supports_pause = os.name != "nt"
                self.supports_seek = False
                self.paused = False
                self._reset_timing(0.0)
                return True

        if self.controller._is_url(self.source):
            proc = self.controller._start_ffplay_process(
                self.source,
                "ffplay não encontrado para reproduzir stream.",
            )
            if proc is None:
                return False
            self.backend = "proc"
            self.proc = proc
            self.player_name = "ffplay"
            self.supports_pause = os.name != "nt"
            self.supports_seek = False
            self.paused = False
            self._reset_timing(0.0)
            return True

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(self.source)
            pygame.mixer.music.play()
            self.backend = "pygame"
            self.player_name = "pygame"
            self.supports_pause = True
            self.supports_seek = True
            self.paused = False
            self._reset_timing(0.0)
            return True
        except Exception as exc:
            log_warn(f"pygame falhou ({exc}). Usando ffplay...")
            if pygame.mixer.get_init():
                pygame.mixer.quit()

        proc = self.controller._start_ffplay_process(
            self.source,
            "ffplay não encontrado.",
        )
        if proc is None:
            return False
        self.backend = "proc"
        self.proc = proc
        self.player_name = "ffplay"
        self.supports_pause = os.name != "nt"
        self.supports_seek = False
        self.paused = False
        self._reset_timing(0.0)
        return True

    def is_alive(self) -> bool:
        if self.backend == "pygame":
            return self.paused or pygame.mixer.music.get_busy()
        if self.proc is not None:
            return self.proc.poll() is None
        return False

    def position_s(self) -> float:
        return self._clock_elapsed()

    def pause(self, should_pause: bool) -> bool:
        if not self.supports_pause:
            return False
        if should_pause == self.paused:
            return True

        if self.backend == "pygame":
            try:
                if should_pause:
                    pygame.mixer.music.pause()
                else:
                    pygame.mixer.music.unpause()
            except Exception:
                return False
            self._set_paused_state(should_pause)
            return True

        if self.proc is None or self.proc.poll() is not None or os.name == "nt":
            return False
        try:
            self.proc.send_signal(signal.SIGSTOP if should_pause else signal.SIGCONT)
        except Exception:
            return False
        self._set_paused_state(should_pause)
        return True

    def seek(self, delta_s: int) -> bool:
        if self.backend != "pygame":
            return False

        target_s = self.position_s() + float(delta_s)
        if self.duration_s is not None:
            target_s = min(float(self.duration_s), target_s)
        target_s = max(0.0, target_s)
        was_paused = self.paused

        try:
            pygame.mixer.music.play(start=target_s)
        except Exception:
            return False

        self.paused = False
        self._reset_timing(target_s)
        if was_paused:
            try:
                pygame.mixer.music.pause()
            except Exception:
                return False
            self._set_paused_state(True)
        return True

    def stop(self):
        if self.backend == "pygame":
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            return

        if self.proc and self.proc.poll() is None:
            if os.name != "nt" and self.paused:
                try:
                    self.proc.send_signal(signal.SIGCONT)
                except Exception:
                    pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except Exception:
                self.proc.kill()

    def shutdown(self):
        self.stop()
        if self.backend == "pygame" and pygame.mixer.get_init():
            pygame.mixer.quit()


# ---------------------------------------------------------------------------
# Controller principal
# ---------------------------------------------------------------------------

class MusicLEDController:
    def __init__(
        self,
        port="/dev/ttyACM0",
        baudrate=115200,
        fps=None,
        config=None,
        middle_bpm_multiplier=1.0,
        show_youtube_video=False,
    ):
        self.port      = port
        self.baudrate  = baudrate
        self.config    = config or EffectConfig()
        self.fps       = self.config.FPS if fps is None else int(fps)
        try:
            speed_mult = float(middle_bpm_multiplier)
        except Exception:
            speed_mult = 1.0
        if not np.isfinite(speed_mult) or speed_mult <= 0.0:
            speed_mult = 1.0
        self.middle_bpm_multiplier = speed_mult
        self.show_youtube_video = bool(show_youtube_video)
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
            log_ok(f"Conectado ao Arduino em {_c(self.port, C.BYELLOW)}")
            if self.ser.in_waiting:
                log_sub(self.ser.readline().decode(errors="ignore").strip())
            return True
        except Exception as exc:
            log_err(f"Erro ao conectar ao Arduino: {exc}")
            log_sub("Verifique a porta serial e se o Arduino está conectado")
            return False

    def send_led_command(self, led_values, mode=1):
        if not (self.ser and self.ser.is_open):
            return
        # FIX #8: serial.write sem try/except causava crash ao desconectar o Arduino.
        try:
            num_leds = self.config.NUM_LEDS
            values   = np.zeros(num_leds, dtype=np.int32)
            src      = np.asarray(led_values, dtype=np.int32)
            n        = min(num_leds, src.shape[0])
            values[:n] = src[:n]
            payload  = ",".join(str(int(v)) for v in values)
            self.ser.write(f"P,{int(mode)},{payload}\n".encode())
        except serial.SerialException:
            pass  # Não interrompe o loop de playback

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.send_led_command([0] * self.config.NUM_LEDS, mode=1)
            self.ser.close()
            log_ok("Desconectado do Arduino")

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

    @staticmethod
    def _window_mean(arr, lo, hi):
        lo = int(np.clip(lo, 0, len(arr) - 1))
        hi = int(np.clip(hi, lo, len(arr) - 1))
        return float(np.mean(arr[lo:hi + 1]))

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
        if len(beat_frames) < 2:
            return transitions

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
        for t in transitions:
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
        cfg = self.config
        log_sub("Separando harmônico/percussivo (HPSS)...")
        y_harm, y_perc = librosa.effects.hpss(y, margin=3.0)

        hop_samples = int(cfg.SEG_HOP_SECONDS * sr)

        log_sub("Calculando features espectrais por janela...")

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

        log_sub("Segmentação aglomerativa...")

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

        log_sub(
            f"{_c(len(valid_boundaries), C.BYELLOW, C.BOLD)} fronteiras de seção detectadas"
        )

        p90_perc = float(np.percentile(rms_perc, 90)) + 1e-9
        p90_rms  = float(np.percentile(rms_total, 90)) + 1e-9
        p95_low  = float(np.percentile(low_energy_per_win, 95)) + 1e-9

        transitions = []
        wins_ctx = max(1, int(2.0 / cfg.SEG_HOP_SECONDS))
        wm = self._window_mean

        for b_time in valid_boundaries:
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
                icon_skip = _c("✗", C.DIM)
                log_sub(
                    f"t={_c(f'{b_time:6.2f}s', C.DIM)}  {icon_skip}  "
                    f"{_c('ignorada (silêncio)', C.DIM)}"
                    f"  beat={_c(f'{beat_norm_after:.2f}', C.DIM)}"
                    f"  kick={_c(f'{kick_mean_after:.2f}', C.DIM)}"
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

            icon, color = _EFFECT_ICONS[eff_type]
            log_sub(
                f"t={_c(f'{b_time:6.2f}s', C.BYELLOW)}  "
                f"{_c(icon, color)}  {_c(f'{eff_type.name:<14}', color, C.BOLD)}"
                f"  beat={_c(f'{beat_norm_after:.2f}', C.BCYAN)}"
                f"  kick={_c(f'{kick_mean_after:.2f}', C.BMAGENTA)}"
                f"  flat={_c(f'{flat_aft:.2f}', C.DIM)}"
                f"  dur={_c(f'{dur_s:.2f}s', C.DIM)}"
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
        # No original eram recalculadas em cada um dos 5000+ frames do loop.
        freqs     = np.fft.rfftfreq(samples_per_frame, 1.0 / sr)
        low_mask  = (freqs >= cfg.LOW_BAND_MIN_HZ) & (freqs <= cfg.LOW_BAND_MAX_HZ)
        high_mask = (freqs >= 2000.0)              & (freqs <= 8000.0)
        _has_low  = bool(np.any(low_mask))
        _has_high = bool(np.any(high_mask))

        middle_tempo_bpm = tempo_bpm * self.middle_bpm_multiplier
        if not np.isfinite(middle_tempo_bpm) or middle_tempo_bpm <= 1.0:
            middle_tempo_bpm = tempo_bpm

        base_step_frames = int(np.clip(
            round((60.0 / middle_tempo_bpm) * self.fps * cfg.BPM_STEP_MULT),
            cfg.STEP_FRAMES_MIN, cfg.STEP_FRAMES_MAX,
        ))
        base_speed = 1.0 / float(base_step_frames)

        trans_index  = 0
        n_trans      = len(transitions)
        active_trans = None   # TransitionEvent | None

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

        # FIX #5: variáveis de cache para percentil (atualizado a cada 10 frames)
        p90_energy = 1e-9
        p95_low    = 1e-9

        # ---- BPM resync: grid de batidas reais (librosa) --------------- #
        # Converte beat_times → beat_frame_indices no espaço da partitura
        beat_frame_set = set(int(np.clip(bf, 0, n_frames - 1)) for bf in beat_frames)
        # Sorted list para busca do próximo beat
        beat_frame_sorted = sorted(beat_frame_set)
        _bf_len = len(beat_frame_sorted)

        def _next_beat_phase(current_frame: int) -> float:
            """Retorna a fase [0,1) que o phase_accumulator deveria ter
               para disparar o próximo step exatamente no próximo beat."""
            # Busca binária do próximo beat >= current_frame
            lo, hi = 0, _bf_len
            while lo < hi:
                mid = (lo + hi) // 2
                if beat_frame_sorted[mid] < current_frame:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= _bf_len:
                return phase_accumulator  # fim da música, não altera
            frames_to_beat = beat_frame_sorted[lo] - current_frame
            # Queremos que o acumulador chegue a 1.0 em exatamente frames_to_beat frames
            ideal = float(np.clip(1.0 - frames_to_beat * base_speed, 0.0, 1.0))
            return ideal

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
            # FIX #4: usa masks pré-computadas (freqs, low_mask, high_mask já calculadas)
            low_band_energy  = float(np.mean(spectrum[low_mask]))  if _has_low  else 0.0
            high_band_energy = float(np.mean(spectrum[high_mask])) if _has_high else 0.0

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
            low_band_window.append(low_band_energy)

            # FIX #5: percentile em cache — atualiza a cada 10 frames (era todo frame, O(n log n)).
            if i % 10 == 0:
                p90_energy = float(np.percentile(np.array(energy_window), 90)) + 1e-9
                p95_low    = float(np.percentile(np.array(low_band_window), 95)) + 1e-9
            norm_energy = float(np.clip(rms / p90_energy, 0.0, 1.0))
            low_norm    = float(np.clip(low_band_energy / p95_low, 0.0, 1.0))

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

            # ---- transição agendada ----------------------------------- #
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
                    # ---- ressincroniza fase ao grid de batidas -------- #
                    if cfg.BPM_RESYNC_STRENGTH > 0.0 and _bf_len > 0:
                        ideal = _next_beat_phase(i)
                        phase_accumulator = (
                            phase_accumulator * (1.0 - cfg.BPM_RESYNC_STRENGTH)
                            + ideal * cfg.BPM_RESYNC_STRENGTH
                        )

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

            # ---- pontas ---------------------------------------------- #
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
                progress_bar(i + 1, n_frames, label=f"{i+1}/{n_frames} frames")

        print()
        return led_patterns, dominant_leds

    # ------------------------------------------------------------------ #
    # Ponto de entrada público                                             #
    # ------------------------------------------------------------------ #

    def process_audio_file(self, audio_file):
        source_label = audio_file if not self._is_url(audio_file) else "YouTube stream"
        log_step(f"Carregando áudio: {_c(source_label, C.BYELLOW)}")

        y, sr, duration = self._load_audio_data(audio_file, target_sr=22050)
        log_sub(f"Duração: {_c(f'{duration:.2f}s', C.BYELLOW)}  |  SR: {_c(f'{sr} Hz', C.BCYAN)}")

        n_frames = max(1, int(duration * self.fps))

        tempo, beat_librosa = librosa.beat.beat_track(y=y, sr=sr)
        try:
            tempo_bpm = float(np.asarray(tempo).item())
        except Exception:
            tempo_bpm = 120.0
        if not np.isfinite(tempo_bpm) or tempo_bpm <= 1.0:
            tempo_bpm = 120.0

        # Converte beat frames (espaço librosa) → frames da partitura (FPS)
        beat_times_s  = librosa.frames_to_time(beat_librosa, sr=sr)
        beat_frames   = (beat_times_s * self.fps).astype(np.int32)
        middle_tempo_bpm = tempo_bpm * self.middle_bpm_multiplier
        if not np.isfinite(middle_tempo_bpm) or middle_tempo_bpm <= 1.0:
            middle_tempo_bpm = tempo_bpm

        log_sub(
            f"BPM detectado: {_c(f'{tempo_bpm:.1f}', C.BMAGENTA, C.BOLD)}"
            f"  |  {_c(len(beat_frames), C.BYELLOW)} batidas mapeadas"
        )
        log_sub(
            f"BPM LEDs do meio: {_c(f'{middle_tempo_bpm:.1f}', C.BCYAN, C.BOLD)}"
            f"  ({_c(f'x{self.middle_bpm_multiplier:.1f}', C.BYELLOW)})"
        )

        section_line()
        log_step("Fase 1 — Análise estrutural")
        transitions = self._analyze_structure(y, sr)

        # ---- Snapa cada transição ao beat real mais próximo ----------- #
        transitions = self._snap_to_beats(
            transitions, beat_frames, tempo_bpm, self.fps, self.config, n_frames
        )
        if transitions:
            log_sub(_c("Transições alinhadas ao grid de batidas:", C.BBLUE, C.BOLD))
            for t in transitions:
                t_s   = t.frame_start / self.fps
                dur_s = (t.frame_end - t.frame_start) / self.fps
                icon, color = _EFFECT_ICONS[t.effect_type]
                log_sub(
                    f"  t={_c(f'{t_s:6.2f}s', C.BYELLOW)}  "
                    f"{_c(icon, color)}  {_c(f'{t.effect_type.name:<14}', color, C.BOLD)}"
                    f"  dur={_c(f'{dur_s:.2f}s', C.BCYAN)}"
                )

        section_line()
        log_step(
            f"Fase 2 — Gerando partitura  "
            f"({_c(n_frames, C.BYELLOW)} frames @ {_c(self.fps, C.BCYAN)} FPS)"
        )
        led_patterns, dominant_leds = self._generate_led_score(
            y, sr, n_frames, transitions, tempo_bpm, beat_frames
        )
        log_ok("Partitura gerada com sucesso!")

        # Telemetria
        section_line()
        cfg  = self.config
        pwms = np.array([v for _, v in led_patterns], dtype=np.int32)
        sat  = np.mean(pwms >= 245, axis=0) * 100.0
        dark = np.mean(pwms <= 5,   axis=0) * 100.0
        avg  = np.mean(pwms, axis=0)
        num_leds = cfg.NUM_LEDS
        log_step("Telemetria")

        # Mini bar chart por LED
        _TBLOCKS = " ▁▂▃▄▅▆▇█"
        bar_row = ""
        for i in range(num_leds):
            blk   = _TBLOCKS[min(int(avg[i] / 255 * 8), 8)]
            is_edge = (i == 0 or i == num_leds - 1)
            col   = C.BCYAN if not is_edge else C.BBLUE
            bar_row += f" {_c(blk * 2, col, C.BOLD)}"
        log_sub(f"PWM médio (visual):{bar_row}")

        avg_str  = "  ".join(_c(f"L{i+1}:{avg[i]:.0f}", C.BCYAN)  for i in range(num_leds))
        sat_str  = "  ".join(
            _c(f"L{i+1}:{sat[i]:.0f}%",  C.BRED    if sat[i]  > 30 else C.BGREEN)
            for i in range(num_leds)
        )
        dark_str = "  ".join(
            _c(f"L{i+1}:{dark[i]:.0f}%", C.BYELLOW if dark[i] > 60 else C.BRED if dark[i] > 80 else C.BGREEN)
            for i in range(num_leds)
        )
        log_sub(f"PWM médio:          {avg_str}")
        log_sub(f"Saturado   ≥245:    {sat_str}")
        log_sub(f"Apagado    ≤5:      {dark_str}")

        # Alertas diagnósticos
        mid_dark_avg = float(np.mean(dark[1:-1]))
        edge_sat_avg = float(np.mean(sat[[0, -1]]))
        if mid_dark_avg > 40:
            log_warn(
                f"LEDs do meio apagados {mid_dark_avg:.0f}% do tempo "
                f"— considere reduzir MIDDLE_FADE_DECAY (atual: {cfg.MIDDLE_FADE_DECAY})"
            )
        if edge_sat_avg > 20:
            log_warn(
                f"Pontas saturando {edge_sat_avg:.0f}% do tempo "
                f"— considere reduzir EDGE_PULSE_GAIN ou EDGE_HEAVY_BONUS_GAIN"
            )

        playback_source = audio_file
        if self._is_url(audio_file):
            playback_source = self._create_temp_wav_from_audio(y, sr)

        return led_patterns, duration, playback_source

    # ------------------------------------------------------------------ #
    # Playback                                                             #
    # ------------------------------------------------------------------ #

    def _start_ffplay_process(self, audio_source, missing_msg):
        ffplay_bin = shutil.which("ffplay")
        if not ffplay_bin:
            log_err(missing_msg)
            return None
        cmd = [ffplay_bin, "-nodisp", "-autoexit", "-loglevel", "quiet", audio_source]
        try:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            log_err(f"Falha ao iniciar ffplay: {exc}")
            return None

    def _start_youtube_video_process(self, video_source):
        mpv_bin = shutil.which("mpv")
        if not mpv_bin:
            log_warn(
                "mpv não encontrado para vídeo no terminal. "
                "Mantendo reprodução somente de áudio."
            )
            return None, ""

        vo_candidates = ("tct", "caca")
        for vo in vo_candidates:
            cmd = [
                mpv_bin,
                "--no-config",
                "--really-quiet",
                "--terminal=yes",
                "--force-window=no",
                f"--vo={vo}",
                video_source,
            ]
            try:
                proc = subprocess.Popen(cmd)
            except Exception:
                continue
            time.sleep(0.7)
            if proc.poll() is None:
                return proc, "mpv"

        log_warn(
            "mpv não conseguiu renderizar vídeo no terminal (vo=tct/caca). "
            "Mantendo reprodução somente de áudio."
        )
        return None, ""

    def sync_and_play(self, audio_source, youtube_video_source=None, track_title=None):
        led_patterns, duration, playback_source = self.process_audio_file(audio_source)

        if not self.ser or not self.ser.is_open:
            log_err("Arduino não conectado!")
            self._cleanup_temp_playback_files()
            return

        section_line()
        log_step("Iniciando reprodução sincronizada")

        title = track_title
        if not title:
            if self._is_url(audio_source):
                title = "YouTube"
            else:
                title = os.path.basename(str(audio_source))
        title = str(title) if title else "Faixa"

        session = PlaybackSession(
            self,
            playback_source,
            duration_s=duration,
            youtube_video_source=youtube_video_source,
        )
        if not session.start():
            self._cleanup_temp_playback_files()
            return

        ctrl_parts = ["q sair"]
        if session.supports_pause:
            ctrl_parts.insert(0, "p/⎵ pausa")
        if session.supports_seek:
            ctrl_parts.append("[ ]/←→ seek ±10s")
        print(_c(f"    Controles: {' | '.join(ctrl_parts)}\n", C.DIM))

        self.playing = True
        n_frames = len(led_patterns)
        frame_idx_prev = -1
        frames_skipped = 0
        ended_by_user = False
        show_box = not (self.show_youtube_video and youtube_video_source and session.player_name == "mpv")
        box = PlaybackBox(
            title=title,
            player_name=session.player_name,
            duration_s=duration,
            queue_idx=0,
            queue_total=1,
        )

        try:
            with KeyReader() as keys:
                while self.playing and session.is_alive():
                    elapsed = session.position_s()
                    if n_frames <= 0:
                        break
                    frame_idx = min(max(int(elapsed * self.fps), 0), n_frames - 1)

                    if frame_idx < frame_idx_prev:
                        frame_idx_prev = frame_idx - 1

                    if not session.paused and frame_idx != frame_idx_prev:
                        if frame_idx > frame_idx_prev + 1:
                            frames_skipped += frame_idx - frame_idx_prev - 1
                        mode, led_values = led_patterns[frame_idx]
                        self.send_led_command(led_values, mode=mode)
                        frame_idx_prev = frame_idx

                    if show_box:
                        box.draw(
                            state="PAUSADO" if session.paused else "TOCANDO",
                            frame_idx=frame_idx,
                            elapsed_s=elapsed,
                            paused=session.paused,
                            supports_seek=session.supports_seek,
                        )

                    key = keys.read_key() if keys.enabled else None
                    if key:
                        k = key.lower() if len(key) == 1 else key

                        if k in ("p", " "):
                            session.pause(not session.paused)

                        elif k in ("[", "LEFT") and session.supports_seek:
                            if session.seek(-SEEK_STEP_S):
                                seek_idx = min(
                                    max(int(session.position_s() * self.fps), 0),
                                    n_frames - 1,
                                )
                                frame_idx_prev = seek_idx - 1

                        elif k in ("]", "RIGHT") and session.supports_seek:
                            if session.seek(SEEK_STEP_S):
                                seek_idx = min(
                                    max(int(session.position_s() * self.fps), 0),
                                    n_frames - 1,
                                )
                                frame_idx_prev = seek_idx - 1

                        elif k == "q":
                            ended_by_user = True
                            break

                    if duration > 0 and elapsed >= duration:
                        break
                    time.sleep(0.04)

        except KeyboardInterrupt:
            ended_by_user = True
        finally:
            self.playing = False
            session.shutdown()
            self.send_led_command([0] * self.config.NUM_LEDS, mode=1)
            self._cleanup_temp_playback_files()

        print()
        section_line()
        if ended_by_user:
            log_warn("Reprodução interrompida pelo usuário")
        else:
            if frames_skipped:
                log_warn(f"{frames_skipped} frames pulados por drift de timing")
            log_ok("Reprodução finalizada!")


# ---------------------------------------------------------------------------
# UI / entrada
# ---------------------------------------------------------------------------

def select_audio_file():
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
    except Exception:
        log_err("PySide6 não encontrado. Instale com: pip install PySide6")
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
        log_err("PySide6 não encontrado. Instale com: pip install PySide6")
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


def select_middle_led_speed_multiplier():
    try:
        from PySide6.QtWidgets import QApplication, QInputDialog
    except Exception:
        log_err("PySide6 não encontrado. Instale com: pip install PySide6")
        return None
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)
    options = ["Mais lento (1x BPM)", "Mais rápido (2x BPM)"]
    choice, ok = QInputDialog.getItem(
        None,
        "Velocidade LEDs do meio",
        "Escolha a velocidade dos LEDs do meio:",
        options,
        0,
        False,
    )
    if owns_app:
        app.quit()
    if not ok:
        return None
    return 1.0 if choice == options[0] else 2.0


def select_show_youtube_video_in_terminal():
    try:
        from PySide6.QtWidgets import QApplication, QInputDialog
    except Exception:
        log_err("PySide6 não encontrado. Instale com: pip install PySide6")
        return None
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)
    options = ["Não", "Sim"]
    choice, ok = QInputDialog.getItem(
        None,
        "Vídeo do YouTube",
        "Deseja exibir vídeo do YouTube no terminal?",
        options,
        0,
        False,
    )
    if owns_app:
        app.quit()
    if not ok:
        return None
    return choice == options[1]


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
    formats = info.get("formats") or []
    stream_url = info.get("url")
    if not stream_url:
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

    av_formats = [
        f for f in formats
        if f.get("url")
        and f.get("acodec") not in (None, "none")
        and f.get("vcodec") not in (None, "none")
    ]
    video_stream_url = None
    if av_formats:
        av_formats.sort(
            key=lambda f: (f.get("height") or 0.0, f.get("tbr") or 0.0)
        )
        video_stream_url = av_formats[-1]["url"]
    if not video_stream_url:
        video_stream_url = stream_url

    return stream_url, info.get("title") or "YouTube", video_stream_url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    banner()

    arduino_port = "/dev/ttyACM0"
    selection = select_audio_source()
    if not selection:
        log_warn("Nenhuma fonte selecionada. Encerrando.")
        return

    middle_speed_mult = select_middle_led_speed_multiplier()
    if middle_speed_mult is None:
        log_warn("Nenhuma velocidade para LEDs do meio selecionada. Encerrando.")
        return
    speed_label = "mais rápido (2x BPM)" if middle_speed_mult > 1.0 else "mais lento (padrão)"
    log_ok(f"Velocidade dos LEDs do meio: {_c(speed_label, C.BYELLOW)}")

    source_type, source_value = selection
    show_youtube_video = False
    youtube_video_source = None
    track_title = None
    if source_type == "youtube":
        show_youtube_video = select_show_youtube_video_in_terminal()
        if show_youtube_video is None:
            log_warn("Nenhuma opção de vídeo selecionada. Encerrando.")
            return
        video_label = "ativado (terminal)" if show_youtube_video else "desativado"
        log_ok(f"Vídeo do YouTube: {_c(video_label, C.BYELLOW)}")

        log_step("Resolvendo stream do YouTube via yt-dlp...")
        try:
            audio_file, title, youtube_video_source = resolve_youtube_stream(source_value)
            track_title = title
            log_ok(f"Stream pronto: {_c(title, C.BYELLOW)}")
        except Exception as exc:
            log_err(f"Falha ao resolver stream: {exc}")
            return
    else:
        audio_file = source_value
        track_title = os.path.basename(str(audio_file))

    controller = MusicLEDController(
        port=arduino_port,
        fps=30,
        middle_bpm_multiplier=middle_speed_mult,
        show_youtube_video=show_youtube_video,
    )

    section_line()
    if not controller.connect_arduino():
        log_warn("Configure a porta serial correta e tente novamente")
        return

    try:
        controller.sync_and_play(
            audio_file,
            youtube_video_source=youtube_video_source,
            track_title=track_title,
        )
    except FileNotFoundError:
        log_err(f"Arquivo não encontrado: {audio_file}")
    except Exception as exc:
        log_err(f"Erro inesperado: {exc}")
    finally:
        controller.disconnect()

    section_line()
    print(_c("  Obrigado por usar o LED Music Sync! 🎶\n", C.BMAGENTA, C.BOLD))


if __name__ == "__main__":
    main()

    
