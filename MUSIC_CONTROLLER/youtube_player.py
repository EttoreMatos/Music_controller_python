#!/usr/bin/env python3
"""
YouTube Music Player — v2
Controles: [p/⎵] pause  |  [← →] faixa ant/próx  |  [[ ]] seek ±10s  |  [q] sair
"""

import argparse
import array
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from typing import Optional

try:
    from rich import box as RICH_BOX
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    RICH_ENABLED = True
    RICH_CONSOLE = Console()
except Exception:
    RICH_ENABLED = False
    RICH_BOX = None
    RICH_CONSOLE = None
    Panel = None
    Text = None


# ─── Constantes ───────────────────────────────────────────────────────────────

BLOCKS         = " ▁▂▃▄▅▆▇█"
SPINNER_FRAMES = "|/-\\"
SEEK_STEP_S    = 10


# ─── ANSI ─────────────────────────────────────────────────────────────────────

class C:
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    BRED     = "\033[91m"
    BGREEN   = "\033[92m"
    BYELLOW  = "\033[93m"
    BCYAN    = "\033[96m"
    BMAGENTA = "\033[95m"
    BWHITE   = "\033[97m"


def _c(text, *codes) -> str:
    return "".join(codes) + str(text) + C.RESET


def _bar_ansi(fill: float) -> str:
    """Gradiente de cor: verde → amarelo-verde → amarelo → laranja → vermelho."""
    if fill < 0.40:
        return "\033[38;5;46m"
    elif fill < 0.60:
        return "\033[38;5;82m"
    elif fill < 0.75:
        return "\033[38;5;226m"
    elif fill < 0.88:
        return "\033[38;5;208m"
    else:
        return "\033[38;5;196m"


# ─── Logging ──────────────────────────────────────────────────────────────────

def banner():
    if RICH_ENABLED and RICH_CONSOLE and sys.stdout.isatty():
        RICH_CONSOLE.print(Panel(
            Text.from_markup(
                "[bold magenta]YouTube Music Player[/] [dim]v2[/]\n"
                "[dim]← → faixa  ·  p/⎵ pause  ·  [ ] seek  ·  q sair[/]"
            ),
            border_style="magenta",
            box=RICH_BOX.ROUNDED if RICH_BOX else None,
        ))
        return
    print()
    print(_c("  YouTube Music Player v2", C.BMAGENTA, C.BOLD))
    print(_c("  ← → faixa  ·  p/⎵ pause  ·  [ ] seek  ·  q sair", C.DIM))
    print()


def log_ok(msg):
    if RICH_ENABLED and RICH_CONSOLE and sys.stdout.isatty():
        RICH_CONSOLE.print(f"[bold green]✓[/] {msg}")
        return
    print(_c("  ✓ ", C.BGREEN, C.BOLD) + _c(msg, C.BWHITE))


def log_err(msg):
    if RICH_ENABLED and RICH_CONSOLE and sys.stdout.isatty():
        RICH_CONSOLE.print(f"[bold red]✗[/] {msg}")
        return
    print(_c("  ✗ ", C.BRED, C.BOLD) + _c(msg, C.BWHITE))


def log_warn(msg):
    if RICH_ENABLED and RICH_CONSOLE and sys.stdout.isatty():
        RICH_CONSOLE.print(f"[bold yellow]⚠[/] {msg}")
        return
    print(_c("  ⚠ ", C.BYELLOW, C.BOLD) + _c(msg, C.BWHITE))


def log_step(msg):
    if RICH_ENABLED and RICH_CONSOLE and sys.stdout.isatty():
        RICH_CONSOLE.print(f"\n[bold cyan]▶[/] [bold]{msg}[/]")
        return
    print(_c("\n  ▶ ", C.BCYAN, C.BOLD) + _c(msg, C.BWHITE, C.BOLD))


# ─── Data ─────────────────────────────────────────────────────────────────────

class YouTubeMedia:
    __slots__ = ("title", "webpage_url", "audio_stream_url", "video_stream_url", "duration_s")

    def __init__(self, title, webpage_url, audio_stream_url, video_stream_url, duration_s):
        self.title             = title
        self.webpage_url       = webpage_url
        self.audio_stream_url  = audio_stream_url
        self.video_stream_url  = video_stream_url
        self.duration_s        = duration_s


class PlaybackQueue:
    """Fila de músicas com navegação para frente e para trás."""

    def __init__(self, queries: list):
        self._queries = list(queries)
        self._index   = 0

    @property
    def index(self) -> int:
        return self._index

    @property
    def total(self) -> int:
        return len(self._queries)

    @property
    def current(self) -> Optional[str]:
        if 0 <= self._index < len(self._queries):
            return self._queries[self._index]
        return None

    def advance(self) -> bool:
        if self._index < len(self._queries) - 1:
            self._index += 1
            return True
        return False

    def go_back(self) -> bool:
        if self._index > 0:
            self._index -= 1
            return True
        return False

    def add(self, query: str):
        self._queries.append(query)


# ─── SpinnerLine ──────────────────────────────────────────────────────────────

class SpinnerLine:
    def __init__(self, text: str, interval_s: float = 0.09):
        self.text      = text
        self.interval_s = interval_s
        self.enabled   = sys.stdout.isatty()
        self._stop     = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not self.enabled:
            print(_c(f"  ... {self.text}", C.DIM))
            return
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.4)
        if self.enabled:
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()

    def _animate(self):
        idx = 0
        while not self._stop.is_set():
            frame = SPINNER_FRAMES[idx % len(SPINNER_FRAMES)]
            sys.stdout.write("\r" + _c(f"  {frame} {self.text}", C.BCYAN, C.BOLD))
            sys.stdout.flush()
            idx += 1
            if self._stop.wait(self.interval_s):
                break


def _run_with_spinner(text, func, *args, **kwargs):
    s = SpinnerLine(text)
    s.start()
    try:
        return func(*args, **kwargs)
    finally:
        s.stop()


# ─── KeyReader ────────────────────────────────────────────────────────────────

class KeyReader:
    """Leitura de teclas sem bloqueio; reconhece setas (sequências de escape)."""

    def __init__(self):
        self.enabled   = False
        self._is_win   = os.name == "nt"
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
            self._fd        = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            attrs           = termios.tcgetattr(self._fd)
            attrs[3]       &= ~(termios.ICANON | termios.ECHO)
            attrs[6][termios.VMIN]  = 0
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
                        return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(rest[1:2], "ESC")
                except Exception:
                    pass
            return "ESC"
        return ch


# ─── AudioLevelTracker ────────────────────────────────────────────────────────

class AudioLevelTracker:
    """Captura e analisa o espectro de frequências em tempo real via ffmpeg."""

    def __init__(
        self, source_url: str, *,
        sample_rate: int = 12000,
        chunk_ms: int    = 50,
        base_bars: int   = 40,
        freq_min: float  = 55.0,
        freq_max: float  = 6000.0,
    ):
        self.source_url  = source_url
        self.sample_rate = sample_rate
        self.chunk_ms    = chunk_ms
        self.base_bars   = max(8, base_bars)
        self.freq_min    = freq_min
        self.freq_max    = min(freq_max, sample_rate / 2.0 * 0.92)
        self.enabled     = False
        self._stop       = threading.Event()
        self._lock       = threading.Lock()
        self._levels     = [0.0] * self.base_bars
        self._gain       = 0.12
        self._last_ts    = 0.0
        self._decay_ts   = 0.0
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen]   = None
        self._freqs = self._logspace(self.freq_min, self.freq_max, self.base_bars)

    def start(self) -> bool:
        if not shutil.which("ffmpeg"):
            return False
        self.enabled = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=0.6)
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass

    def get_levels(self, num_bars: int) -> Optional[list]:
        if not self.enabled or num_bars <= 0:
            return None
        with self._lock:
            levels = list(self._levels)
            now    = time.monotonic()
            if self._proc and self._proc.poll() is not None:
                if now - self._last_ts > 0.25:
                    return None
            if now - self._last_ts > 0.20:
                elapsed = (now - self._decay_ts) if self._decay_ts else 0.0
                fall    = min(0.22, max(0.0, elapsed * 0.32))
                if fall > 0.0:
                    self._levels  = [max(0.0, v - fall) for v in self._levels]
                    levels        = list(self._levels)
                    self._decay_ts = now
        return self._resample(levels, num_bars) if levels else None

    def _run(self):
        ffmpeg = shutil.which("ffmpeg")
        chunk_bytes = max(256, int(self.sample_rate * self.chunk_ms / 1000.0)) * 2
        cmd = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", self.source_url,
            "-vn", "-ac", "1", "-ar", str(self.sample_rate), "-f", "s16le", "-",
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception:
            self.enabled = False
            return
        if not self._proc.stdout:
            self.enabled = False
            return
        while not self._stop.is_set() and self._proc.poll() is None:
            chunk = self._proc.stdout.read(chunk_bytes)
            if not chunk:
                time.sleep(0.01)
                continue
            frame = self._analyze(chunk)
            if frame:
                with self._lock:
                    self._levels  = self._merge(self._levels, frame)
                    now           = time.monotonic()
                    self._last_ts = now
                    self._decay_ts = now

    def _analyze(self, raw: bytes) -> list:
        even = len(raw) - (len(raw) % 2)
        if even <= 0:
            return []
        samples = array.array("h")
        samples.frombytes(raw[:even])
        if sys.byteorder != "little":
            samples.byteswap()
        if len(samples) < 32:
            return []
        eff_sr = float(self.sample_rate)
        if len(samples) > 320:
            stride  = max(1, len(samples) // 320)
            samples = samples[::stride]
            eff_sr /= stride
        floats = [s / 32768.0 for s in samples]
        n = len(floats)
        if n <= 1:
            return []
        energies = []
        for freq in self._freqs:
            f     = min(freq, eff_sr / 2.0 * 0.92)
            coeff = 2.0 * math.cos(2.0 * math.pi * f / eff_sr)
            s0 = s1 = s2 = 0.0
            for x in floats:
                s0 = x + coeff * s1 - s2
                s2 = s1
                s1 = s0
            power = s1 * s1 + s2 * s2 - coeff * s1 * s2
            energies.append(math.sqrt(max(0.0, power) / n))
        peak = max(energies) if energies else 0.0
        self._gain = max(peak, self._gain * 0.985)
        if self._gain < 1e-6:
            self._gain = 1e-6
        return [min(1.0, max(0.0, (e / self._gain) * 1.45) ** 0.72) for e in energies]

    def _merge(self, prev: list, cur: list) -> list:
        out = []
        for p, c in zip(prev, cur):
            v = (p * 0.30 + c * 0.70) if c >= p else max(c, p - 0.04)
            out.append(min(1.0, max(0.0, v)))
        n = len(out)
        if len(cur) > n:
            out.extend(cur[n:])
        elif len(prev) > n:
            out.extend(prev[n:])
        return out

    def _resample(self, vals: list, size: int) -> list:
        if not vals:
            return []
        if size <= 1:
            return [vals[0]]
        if len(vals) == size:
            return list(vals)
        mx  = len(vals) - 1
        out = []
        for i in range(size):
            pos   = i * mx / (size - 1)
            left  = int(pos)
            right = min(mx, left + 1)
            frac  = pos - left
            out.append(vals[left] * (1.0 - frac) + vals[right] * frac)
        return out

    def _logspace(self, lo: float, hi: float, n: int) -> list:
        if n <= 1:
            return [max(lo, 1.0)]
        a = math.log(max(lo, 1.0))
        b = math.log(max(hi, lo + 1.0))
        return [math.exp(a + (b - a) * i / (n - 1)) for i in range(n)]


def _create_visualizer(url: str) -> Optional[AudioLevelTracker]:
    if not sys.stdout.isatty():
        return None
    v = AudioLevelTracker(url)
    return v if v.start() else None


# ─── Utilitários ──────────────────────────────────────────────────────────────

def _fit_text(text: str, width: int) -> str:
    s = str(text).replace("\n", " ")
    if width <= 0:
        return ""
    if len(s) > width:
        return s[: width - 3] if width > 3 else s[:width]
    return s.ljust(width)


def _fmt_dur(s: Optional[int]) -> str:
    if s is None:
        return "--:--"
    m, sec = divmod(int(s), 60)
    h, m   = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ─── PlaybackBox ──────────────────────────────────────────────────────────────

class PlaybackBox:
    """HUD de reprodução com visualizador de espectro, peaks e gradiente de cores."""

    BAR_H = 9  # altura do visualizador em linhas

    def __init__(
        self, title: str, player_name: str, duration_s: Optional[int],
        queue_idx: int = 0, queue_total: int = 1,
    ):
        self.title       = title
        self.player_name = player_name
        self.duration_s  = duration_s
        self.queue_idx   = queue_idx
        self.queue_total = queue_total
        self.enabled     = sys.stdout.isatty()
        self._line_count = 0
        self._peaks: list      = []
        self._peak_ts: list    = []

    # ── Peaks ────────────────────────────────────────────────────────────────

    def _update_peaks(self, levels: list) -> list:
        HOLD = 1.5   # segundos segurando no pico
        FALL = 0.45  # queda por segundo após hold
        now  = time.monotonic()
        n    = len(levels)
        if len(self._peaks) != n:
            self._peaks   = list(levels)
            self._peak_ts = [now] * n
            return list(self._peaks)
        for i, lvl in enumerate(levels):
            if lvl >= self._peaks[i]:
                self._peaks[i]   = lvl
                self._peak_ts[i] = now
            else:
                age = now - self._peak_ts[i]
                if age > HOLD:
                    self._peaks[i] = max(0.0, self._peaks[i] - (age - HOLD) * FALL)
        return list(self._peaks)

    # ── Renderização de uma barra ─────────────────────────────────────────────

    def _render_bar(self, level: float, peak: float, height: int) -> list:
        """Retorna lista de `height` strings (topo→base), 1 char cada."""
        lvl_px  = level * height           # altura em "pixels" (float)
        pk_row  = round(peak * height)     # linha do marcador de pico (1-indexed)
        rows    = []
        for row in range(height, 0, -1):   # row = 1 (base) … height (topo)
            if lvl_px >= row:
                ch = "█"
            elif lvl_px > row - 1:
                frac = lvl_px - (row - 1)
                idx  = max(1, min(8, int(frac * 8 + 0.5)))
                ch   = BLOCKS[idx]
            else:
                ch = " "

            # Colorizar
            if ch != " ":
                ch = _bar_ansi(level) + ch + C.RESET
            elif row == pk_row and peak > 0.04:
                # marcador de pico branco
                ch = "\033[38;5;231m▔\033[0m"

            rows.append(ch)
        return rows

    # ── Grid do visualizador ─────────────────────────────────────────────────

    def _build_viz(
        self,
        num_bars: int,
        audio_levels: Optional[list],
        frame_idx: int,
        paused: bool,
    ) -> list:
        if audio_levels:
            levels = [
                min(1.0, max(0.0, audio_levels[i] if i < len(audio_levels) else audio_levels[-1]))
                for i in range(num_bars)
            ]
        else:
            levels = []
            for i in range(num_bars):
                wa = (math.sin(frame_idx * 0.35 + i * 0.75) + 1) / 2
                wb = (math.sin(frame_idx * 0.18 + i * 0.31 + 1.2) + 1) / 2
                levels.append(0.6 * wa + 0.4 * wb)

        if paused:
            levels = [v * 0.18 for v in levels]

        peaks = self._update_peaks(levels)
        cols  = [self._render_bar(l, p, self.BAR_H) for l, p in zip(levels, peaks)]

        # Transpor: linhas de colunas → lista de strings por linha
        return ["".join(col[r] for col in cols) for r in range(self.BAR_H)]

    # ── Barra de progresso ───────────────────────────────────────────────────

    def _progress_bar(self, elapsed: float, inner: int, frame_idx: int) -> str:
        el_str  = _fmt_dur(int(max(0, elapsed)))
        tot_str = _fmt_dur(self.duration_s)
        prefix  = f" {el_str} "
        suffix  = f" {tot_str}"
        space   = max(6, inner - len(prefix) - len(suffix) - 4)
        if self.duration_s and self.duration_s > 0:
            ratio  = min(1.0, max(0.0, elapsed / self.duration_s))
            filled = int(space * ratio)
            bar    = "▓" * filled + "░" * (space - filled)
        else:
            cur = frame_idx % space
            bar = "".join("▓" if i == cur else "░" for i in range(space))
        return f"{prefix}[{bar}]{suffix}"

    # ── Draw ─────────────────────────────────────────────────────────────────

    def draw(
        self, *,
        state: str,
        frame_idx: int,
        elapsed_s: float,
        paused: bool,
        supports_seek: bool = True,
    ):
        if not self.enabled:
            return

        cols  = shutil.get_terminal_size((92, 20)).columns
        width = min(max(cols, 72), 132)
        inner = width - 2  # espaço entre as bordas |…|

        # Dicas de controles
        ctrl_parts = ["p/⎵ pause", "← → faixa"]
        if supports_seek:
            ctrl_parts.append("[ ] seek±10s")
        ctrl_parts.append("q sair")
        ctrl_hint = "  |  ".join(ctrl_parts) + f"  —  {self.player_name}"

        # Cores por estado
        state_color = C.BGREEN if state == "TOCANDO" else C.BYELLOW
        queue_str   = f"[{self.queue_idx + 1}/{self.queue_total}]"

        sep = _c("+" + "─" * inner + "+", C.BCYAN, C.BOLD)

        out = [
            sep,
            _c("|" + _fit_text(f"  {state}  {queue_str}", inner) + "|", state_color, C.BOLD),
            _c("|" + _fit_text(f"  Faixa: {self.title}", inner) + "|", C.BWHITE),
            _c("|" + _fit_text(f"  Progresso: {self._progress_bar(elapsed_s, inner, frame_idx)}", inner) + "]  " + "|", C.BCYAN),
            _c("|" + _fit_text(f"  {ctrl_hint}", inner) + "|", C.DIM),
            sep,
        ]

        if self._line_count:
            sys.stdout.write(f"\033[{self._line_count}F")
        for line in out:
            sys.stdout.write("\r\033[2K" + line + "\n")
        sys.stdout.flush()
        self._line_count = len(out)


# ─── MPV IPC ──────────────────────────────────────────────────────────────────

def _make_mpv_sock() -> Optional[str]:
    if os.name == "nt":
        return None
    name = f"mpv-{os.getpid()}-{int(time.time() * 1000)}.sock"
    return os.path.join(tempfile.gettempdir(), name)


def _wait_mpv_sock(path: str, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.05)
    return os.path.exists(path)


def _mpv_cmd(path: str, command: list) -> bool:
    resp = _mpv_request(path, command)
    return bool(resp and resp.get("error") == "success")


def _mpv_get(path: str, prop: str):
    resp = _mpv_request(path, ["get_property", prop])
    if not resp or resp.get("error") != "success":
        return None
    return resp.get("data")


def _mpv_request(path: str, command: list) -> Optional[dict]:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            s.connect(path)
            s.sendall((json.dumps({"command": command}) + "\n").encode())
            raw = b""
            while b"\n" not in raw:
                chunk = s.recv(4096)
                if not chunk:
                    break
                raw += chunk
        line = raw.splitlines()[0].decode("utf-8", errors="ignore").strip() if raw else ""
        return json.loads(line) if line else {}
    except Exception:
        return None


def _cleanup_sock(path: Optional[str]):
    if path:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


# ─── Resolução de stream ──────────────────────────────────────────────────────

def _require_yt_dlp():
    try:
        import yt_dlp
        return yt_dlp
    except Exception as e:
        raise RuntimeError("yt-dlp não encontrado. Instale: pip install yt-dlp") from e


def _pick_audio(info: dict) -> Optional[str]:
    # URL direta (formato unico ja resolvido)
    if info.get("url"):
        return info["url"]
    fmts = info.get("formats") or []
    # Preferencia: streams so-audio com maior bitrate
    audio_only = [
        f for f in fmts
        if f.get("url")
        and f.get("acodec") not in (None, "none")
        and f.get("vcodec") in (None, "none")
    ]
    if audio_only:
        audio_only.sort(key=lambda f: (f.get("abr") or 0, f.get("tbr") or 0))
        return audio_only[-1]["url"]
    # Fallback: qualquer formato com audio
    any_audio = [f for f in fmts if f.get("url") and f.get("acodec") not in (None, "none")]
    if any_audio:
        any_audio.sort(key=lambda f: (f.get("abr") or 0, f.get("tbr") or 0))
        return any_audio[-1]["url"]
    # Ultimo recurso: primeiro formato disponivel
    for f in fmts:
        if f.get("url"):
            return f["url"]
    return None


def _pick_video(info: dict, fallback: str) -> str:
    fmts = [
        f for f in (info.get("formats") or [])
        if f.get("url")
        and f.get("acodec") not in (None, "none")
        and f.get("vcodec") not in (None, "none")
    ]
    if fmts:
        fmts.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0))
        return fmts[-1]["url"]
    return info.get("webpage_url") or fallback


def resolve_youtube_media(query: str) -> YouTubeMedia:
    yt_dlp = _require_yt_dlp()
    # Tenta formatos do mais rico ao mais simples para máxima compatibilidade
    format_candidates = [
        "bestvideo+bestaudio/bestvideo+bestaudio",
        "bestvideo+bestaudio",
        "best[ext=mp4]/best",
        "best",
    ]
    info = None
    last_err = None
    for fmt in format_candidates:
        opts = {
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "format": fmt, "default_search": "ytsearch1",
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(query, download=False)
            if info:
                break
        except Exception as e:
            last_err = e
            info = None
    if not info:
        raise RuntimeError(last_err or "yt-dlp não retornou informações")
    if not info:
        raise RuntimeError("yt-dlp não retornou informações")
    if "entries" in info and info["entries"]:
        info = next((e for e in info["entries"] if e), None)
        if not info:
            raise RuntimeError("Nenhum resultado válido")
    audio = _pick_audio(info)
    if not audio:
        raise RuntimeError("Stream de áudio não encontrado")
    video = _pick_video(info, audio)
    dur   = info.get("duration")
    try:
        dur = int(dur) if dur is not None else None
    except Exception:
        dur = None
    return YouTubeMedia(
        title            = info.get("title") or "YouTube",
        webpage_url      = info.get("webpage_url") or "",
        audio_stream_url = audio,
        video_stream_url = video,
        duration_s       = dur,
    )


# ─── Loop de controles ────────────────────────────────────────────────────────

def _wait_player(
    proc: subprocess.Popen,
    *,
    title: str,
    player_name: str,
    duration_s: Optional[int]        = None,
    pause_fn    = None,   # fn(should_pause: bool) -> bool
    stop_fn     = None,   # fn() -> None
    seek_fn     = None,   # fn(delta_s: int) -> bool
    position_fn = None,   # fn() -> Optional[float]
    paused_state_fn = None,  # fn() -> Optional[bool]
    queue_idx: int   = 0,
    queue_total: int = 1,
    show_box: bool   = True,
) -> str:
    """
    Gerencia a reprodução e lida com o input do teclado.

    Retorna: "done" | "next" | "prev" | "quit"
    """
    paused     = False
    frame_idx  = 0
    started_at = time.monotonic()
    paused_acc = 0.0
    pause_ts: Optional[float] = None

    box = PlaybackBox(
        title=title, player_name=player_name, duration_s=duration_s,
        queue_idx=queue_idx, queue_total=queue_total,
    )

    def _do_stop() -> None:
        if stop_fn:
            stop_fn()
        elif proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass

    def _set_paused_state(new_state: bool, now_ts: Optional[float] = None) -> None:
        nonlocal paused, pause_ts, paused_acc
        if new_state == paused:
            return
        now_ts = now_ts or time.monotonic()
        if new_state:
            pause_ts = now_ts
        else:
            if pause_ts is not None:
                paused_acc += max(0.0, now_ts - pause_ts)
            pause_ts = None
        paused = new_state

    try:
        with KeyReader() as keys:
            while proc.poll() is None:
                now        = time.monotonic()
                if paused_state_fn:
                    remote_state = paused_state_fn()
                    if isinstance(remote_state, bool):
                        _set_paused_state(remote_state, now)

                remote_elapsed = position_fn() if position_fn else None
                if isinstance(remote_elapsed, (int, float)):
                    elapsed = max(0.0, float(remote_elapsed))
                else:
                    pause_win  = (now - pause_ts) if pause_ts is not None else 0.0
                    elapsed    = max(0.0, now - started_at - paused_acc - pause_win)

                if show_box:
                    box.draw(
                        state         = "PAUSADO" if paused else "TOCANDO",
                        frame_idx     = frame_idx,
                        elapsed_s     = elapsed,
                        paused        = paused,
                        supports_seek = seek_fn is not None,
                    )

                key = keys.read_key() if keys.enabled else None
                if key:
                    k = key.lower() if len(key) == 1 else key

                    # ── Pause / Play
                    if k in ("p", " ") and pause_fn:
                        if pause_fn(not paused):
                            target_state = not paused
                            if paused_state_fn:
                                remote_state = paused_state_fn()
                                if isinstance(remote_state, bool):
                                    target_state = remote_state
                            _set_paused_state(target_state)

                    # ── Seek recuar
                    elif k == "[" and seek_fn:
                        if seek_fn(-SEEK_STEP_S) and not position_fn:
                            started_at += SEEK_STEP_S  # corrige elapsed exibido

                    # ── Seek avançar
                    elif k == "]" and seek_fn:
                        if seek_fn(SEEK_STEP_S) and not position_fn:
                            started_at -= SEEK_STEP_S

                    # ── Próxima faixa
                    elif k == "RIGHT":
                        _do_stop()
                        return "next"

                    # ── Faixa anterior
                    elif k == "LEFT":
                        _do_stop()
                        return "prev"

                    # ── Sair
                    elif k == "q":
                        _do_stop()
                        return "quit"

                frame_idx += 1
                time.sleep(0.09)

    except KeyboardInterrupt:
        _do_stop()
        return "quit"

    # Processo encerrado naturalmente
    if proc.poll() is None:
        try:
            proc.wait(timeout=1.0)
        except Exception:
            proc.terminate()
    return "done"


# ─── Launchers ────────────────────────────────────────────────────────────────

def _start_proc(cmd: list, loading_text: str, delay: float = 0.6) -> subprocess.Popen:
    s = SpinnerLine(loading_text)
    s.start()
    try:
        proc = subprocess.Popen(cmd)
        if delay > 0:
            time.sleep(delay)
        return proc
    finally:
        s.stop()


def _play_audio_mode(
    source_url: str, title: str, duration_s: Optional[int],
    queue_idx: int = 0, queue_total: int = 1,
) -> str:
    # ── mpv (preferido: IPC permite pause e seek, com progresso real) ────────
    mpv = shutil.which("mpv")
    if mpv:
        ipc = _make_mpv_sock()
        cmd = [
            mpv, "--no-config", "--really-quiet", "--no-video",
            "--force-window=no", "--ytdl-format=bestaudio/best",
        ]
        if ipc:
            cmd.append(f"--input-ipc-server={ipc}")
        cmd.append(source_url)

        try:
            proc = _start_proc(cmd, "Iniciando mpv (áudio)...")
        except Exception as e:
            log_err(f"Falha ao iniciar mpv: {e}")
            _cleanup_sock(ipc)
            return "done"

        if proc.poll() is not None:
            _cleanup_sock(ipc)
            return "done"

        has_ipc = bool(ipc and _wait_mpv_sock(ipc))

        def mpv_pause(should: bool) -> bool:
            return _mpv_cmd(ipc, ["set_property", "pause", should]) if has_ipc else False

        def mpv_stop():
            if proc.poll() is not None:
                return
            if has_ipc:
                _mpv_cmd(ipc, ["quit"])
                time.sleep(0.05)
            if proc.poll() is None:
                proc.terminate()

        def mpv_seek(delta: int) -> bool:
            return _mpv_cmd(ipc, ["seek", delta, "relative"]) if has_ipc else False

        def mpv_pos() -> Optional[float]:
            if not has_ipc:
                return None
            val = _mpv_get(ipc, "time-pos")
            try:
                return float(val) if val is not None else None
            except Exception:
                return None

        def mpv_is_paused() -> Optional[bool]:
            if not has_ipc:
                return None
            val = _mpv_get(ipc, "pause")
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            return None

        try:
            return _wait_player(
                proc, title=title, player_name="mpv", duration_s=duration_s,
                pause_fn  = mpv_pause  if has_ipc else None,
                stop_fn   = mpv_stop,
                seek_fn   = mpv_seek   if has_ipc else None,
                position_fn = mpv_pos if has_ipc else None,
                paused_state_fn = mpv_is_paused if has_ipc else None,
                queue_idx=queue_idx, queue_total=queue_total,
            )
        finally:
            _cleanup_sock(ipc)

    # ── ffplay (fallback: sem seek e sem posição real por IPC) ───────────────
    ffplay = shutil.which("ffplay")
    if ffplay:
        cmd = [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", source_url]
        try:
            proc = _start_proc(cmd, "Iniciando ffplay (áudio)...")
        except Exception as e:
            log_err(f"Falha ao iniciar ffplay: {e}")
            return "done"

        if proc.poll() is not None:
            return "done"

        on_unix = os.name != "nt"

        def ff_pause(should: bool) -> bool:
            if not on_unix:
                return False
            try:
                proc.send_signal(signal.SIGSTOP if should else signal.SIGCONT)
                return True
            except Exception:
                return False

        def ff_stop():
            if proc.poll() is not None:
                return
            if on_unix:
                try:
                    proc.send_signal(signal.SIGCONT)
                except Exception:
                    pass
            proc.terminate()

        return _wait_player(
            proc, title=title, player_name="ffplay", duration_s=duration_s,
            pause_fn = ff_pause if on_unix else None,
            stop_fn  = ff_stop,
            queue_idx=queue_idx, queue_total=queue_total,
        )

    log_err("Nenhum player encontrado. Instale mpv ou ffplay.")
    return "done"


def _play_watch_mode(
    source_url: str, title: str, duration_s: Optional[int],
    fallback_url: str = "",
    queue_idx: int = 0, queue_total: int = 1,
) -> str:
    mpv = shutil.which("mpv")
    if mpv:
        vo_candidates = (
            ("tct", 24),
            ("caca", 20),
        )
        for vo, max_fps in vo_candidates:
            cmd = [
                mpv, "--no-config", "--really-quiet", "--terminal=yes",
                "--force-window=no", "--profile=sw-fast", "--framedrop=vo",
                f"--vo={vo}",
            ]
            if max_fps:
                cmd.append(f"--vf=fps={max_fps}")
            cmd.append(source_url)
            try:
                proc = _start_proc(cmd, f"Iniciando mpv (vo={vo})...", delay=0.7)
            except Exception:
                continue
            if proc.poll() is not None:
                if proc.returncode == 0:
                    return "done"
                continue
            try:
                while proc.poll() is None:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                if proc.poll() is None:
                    proc.terminate()
                return "quit"
            if proc.returncode in (0, None):
                return "done"
        log_err("mpv não conseguiu renderizar vídeo no terminal (vo=kitty/sixel/tct/caca).")
        return "done"

    ffplay = shutil.which("ffplay")
    if ffplay:
        url = fallback_url or source_url
        cmd = [ffplay, "-autoexit", "-loglevel", "quiet", url]
        try:
            proc = _start_proc(cmd, "Iniciando ffplay (vídeo)...", delay=0.7)
        except Exception as e:
            log_err(f"Falha ffplay: {e}")
            return "done"
        if proc.poll() is not None:
            return "done"
        on_unix = os.name != "nt"

        def ff_pause(p: bool) -> bool:
            if not on_unix:
                return False
            try:
                proc.send_signal(signal.SIGSTOP if p else signal.SIGCONT)
                return True
            except Exception:
                return False

        def ff_stop():
            if proc.poll() is not None:
                return
            if on_unix:
                try:
                    proc.send_signal(signal.SIGCONT)
                except Exception:
                    pass
            proc.terminate()

        return _wait_player(
            proc, title=title, player_name="ffplay", duration_s=duration_s,
            pause_fn=ff_pause if on_unix else None, stop_fn=ff_stop,
            queue_idx=queue_idx, queue_total=queue_total,
        )

    log_err("Nenhum player para vídeo. Instale mpv ou ffplay.")
    return "done"


# ─── GUI (PySide6) ────────────────────────────────────────────────────────────

def _gui_get_queries() -> list:
    try:
        from PySide6.QtWidgets import (  # type: ignore
            QApplication, QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton,
        )
        from PySide6.QtCore import Qt  # type: ignore
    except Exception:
        log_err("PySide6 não encontrado. Instale: pip install PySide6")
        return []

    app = QApplication.instance() or QApplication(sys.argv)

    dlg = QDialog()
    dlg.setWindowTitle("Playlist — YouTube Music Player")
    dlg.setMinimumWidth(520)
    layout = QVBoxLayout(dlg)

    lbl = QLabel("Digite músicas ou URLs (uma por linha):")
    layout.addWidget(lbl)

    edit = QTextEdit()
    edit.setPlaceholderText(
        "Never Gonna Give You Up\n"
        "https://youtu.be/dQw4w9WgXcQ\n"
        "Bohemian Rhapsody Queen"
    )
    edit.setMinimumHeight(140)
    layout.addWidget(edit)

    btn = QPushButton("▶  Adicionar à fila e tocar")
    btn.clicked.connect(dlg.accept)
    layout.addWidget(btn)

    if dlg.exec() != QDialog.Accepted:
        return []
    return [ln.strip() for ln in edit.toPlainText().splitlines() if ln.strip()]


def _gui_select_mode() -> Optional[str]:
    try:
        from PySide6.QtWidgets import QApplication, QInputDialog  # type: ignore
    except Exception:
        log_err("PySide6 não encontrado.")
        return None
    app     = QApplication.instance() or QApplication(sys.argv)
    options = ["Somente ouvir", "Ver + ouvir"]
    choice, ok = QInputDialog.getItem(None, "Modo de Reprodução", "Como deseja reproduzir:", options, 0, False)
    if not ok:
        return None
    return "watch" if choice == options[1] else "audio"


# ─── Args ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="YouTube Music Player v2 — playlist com pause e seek."
    )
    p.add_argument(
        "query", nargs="*",
        help="Músicas ou URLs (múltiplas aceitas; espaço entre elas ou aspas)",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--watch",      action="store_true", help="Ver + ouvir")
    g.add_argument("--audio-only", action="store_true", help="Somente ouvir (padrão)")
    return p.parse_args()


def _silence():
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false;qt.*.warning=false")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    _silence()
    args = _parse_args()
    banner()

    # Coletar queries
    queries = [q.strip() for q in args.query if q.strip()]
    if not queries:
        queries = _gui_get_queries()
    if not queries:
        log_warn("Nenhuma música informada. Encerrando.")
        return 1

    # Selecionar modo
    if args.watch:
        mode = "watch"
    elif args.audio_only:
        mode = "audio"
    else:
        mode = _gui_select_mode()
    if mode is None:
        log_warn("Nenhum modo selecionado. Encerrando.")
        return 1

    queue = PlaybackQueue(queries)
    log_ok(f"Playlist: {queue.total} música(s)  —  modo: {'ver + ouvir' if mode == 'watch' else 'somente ouvir'}")

    while True:
        query = queue.current
        if not query:
            break

        log_step(f"[{queue.index + 1}/{queue.total}] Resolvendo: {query}")
        try:
            media = _run_with_spinner("Buscando stream no YouTube...", resolve_youtube_media, query)
        except Exception as e:
            log_err(f"Falha: {e}")
            if not queue.advance():
                break
            continue

        log_ok(f"Música: {media.title}")

        if mode == "watch":
            result = _play_watch_mode(
                media.video_stream_url, media.title, media.duration_s,
                fallback_url=media.webpage_url,
                queue_idx=queue.index, queue_total=queue.total,
            )
        else:
            result = _play_audio_mode(
                media.audio_stream_url, media.title, media.duration_s,
                queue_idx=queue.index, queue_total=queue.total,
            )

        if result == "quit":
            log_warn("Interrompido pelo usuário.")
            break
        elif result == "next":
            if not queue.advance():
                log_ok("Fim da playlist.")
                break
        elif result == "prev":
            queue.go_back()
        else:  # "done"
            if not queue.advance():
                log_ok("Fim da playlist.")
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
