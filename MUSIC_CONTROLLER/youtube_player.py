#!/usr/bin/env python3
"""
YouTube Music Player — v3.0 (Enhanced)
Controles: [p/⎵] pause  |  [← →] faixa ant/próx  |  [[ ]] seek ±10s  |  [+/=] [-/_] volume  |  [s] shuffle  |  [r] repeat  |  [l] playlist  |  [f] favoritar  |  [q] sair

Melhorias: Volume control, Shuffle, Repeat, Playlist view, Favorites, Cache, Notificações
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
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

# ─── Config ─────────────────────────────────────────────────────────────────

class RepeatMode(Enum):
    NONE = "none"
    ONE = "one"
    ALL = "all"


class ConfigManager:
    def __init__(self):
        self.config = self._load()
    
    def _load(self):
        config_path = Path(__file__).parent / "config.json"
        if not config_path.exists():
            config_path = Path("config.json")
        defaults = {
            "seek_step": 10, "volume": 75, "volume_step": 5,
            "shuffle": False, "repeat": "none",
            "show_volume": True, "visualizer_bars": 40, "visualizer_height": 9,
            "cache_enabled": True, "history_enabled": True, "favorites_enabled": True,
            "desktop_notifications": True, "use_rich": True
        }
        try:
            with open(config_path, 'r') as f:
                return {**defaults, **json.load(f)}
        except:
            return defaults
    
    def save(self):
        config_path = Path(__file__).parent / "config.json"
        if not config_path.parent.exists():
            config_path = Path("config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except:
            pass

CONFIG = ConfigManager().config
SEEK_STEP_S = CONFIG.get("seek_step", 10)

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
    print(_c("  YouTube Music Player v3", C.BMAGENTA, C.BOLD))
    print(_c("  ← → faixa  ·  p/⎵ pause  ·  [ ] seek  ·  +/- volume  ·  s/r/l/f  ·  q sair", C.DIM))
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


# ─── Cache / History / Favorites ────────────────────────────────────────────

class StreamCache:
    def __init__(self, max_entries=100):
        self.max_entries = max_entries
        self._cache = {}
        self._lock = threading.Lock()
        self._load()
    
    def _load(self):
        try:
            cache_path = Path(__file__).parent / "stream_cache.json"
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    self._cache = json.load(f)
        except:
            self._cache = {}
    
    def save(self):
        try:
            cache_path = Path(__file__).parent / "stream_cache.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(self._cache, f)
        except:
            pass
    
    def get(self, query):
        with self._lock:
            if query in self._cache:
                data = self._cache[query]
                if "timestamp" in data and time.time() - data["timestamp"] < 86400:
                    info = data["info"]
                    return YouTubeMedia(info["title"], info["webpage_url"],
                                       info["audio_stream_url"], info["video_stream_url"],
                                       info.get("duration_s"))
        return None
    
    def set(self, query, media):
        with self._lock:
            self._cache[query] = {
                "info": {
                    "title": media.title, "webpage_url": media.webpage_url,
                    "audio_stream_url": media.audio_stream_url,
                    "video_stream_url": media.video_stream_url,
                    "duration_s": media.duration_s
                },
                "timestamp": time.time()
            }
            if len(self._cache) > self.max_entries:
                self._cache = dict(list(self._cache.items())[-self.max_entries:])
            self.save()


class HistoryManager:
    def __init__(self, max_entries=100):
        self.max_entries = max_entries
        self._entries = []
        self._lock = threading.Lock()
        self._load()
    
    def _load(self):
        try:
            history_path = Path(__file__).parent / "history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    self._entries = json.load(f)
        except:
            self._entries = []
    
    def save(self):
        try:
            history_path = Path(__file__).parent / "history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump(self._entries, f)
        except:
            pass
    
    def add(self, media):
        with self._lock:
            entry = {
                "title": media.title, "webpage_url": media.webpage_url,
                "timestamp": time.time(), "duration_s": media.duration_s
            }
            for e in self._entries[-10:]:
                if e["webpage_url"] == media.webpage_url and time.time() - e["timestamp"] < 3600:
                    return
            self._entries.append(entry)
            self._entries = self._entries[-self.max_entries:]
            self.save()


class FavoritesManager:
    def __init__(self):
        self._favorites = []
        self._lock = threading.Lock()
        self._load()
    
    def _load(self):
        try:
            fav_path = Path(__file__).parent / "favorites.json"
            if fav_path.exists():
                with open(fav_path, 'r') as f:
                    self._favorites = json.load(f)
        except:
            self._favorites = []
    
    def save(self):
        try:
            fav_path = Path(__file__).parent / "favorites.json"
            fav_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fav_path, 'w') as f:
                json.dump(self._favorites, f)
        except:
            pass
    
    def add(self, media) -> bool:
        with self._lock:
            if any(f["webpage_url"] == media.webpage_url for f in self._favorites):
                return False
            self._favorites.append({
                "title": media.title, "webpage_url": media.webpage_url,
                "duration_s": media.duration_s, "added_at": time.time()
            })
            self.save()
            return True
    
    def remove(self, webpage_url) -> bool:
        with self._lock:
            for i, f in enumerate(self._favorites):
                if f["webpage_url"] == webpage_url:
                    self._favorites.pop(i)
                    self.save()
                    return True
            return False
    
    def is_favorite(self, webpage_url) -> bool:
        with self._lock:
            return any(f["webpage_url"] == webpage_url for f in self._favorites)


# ─── Fila aprimorada ─────────────────────────────────────────────────────────

class PlaybackQueue:
    """Fila de músicas com navegação, shuffle e repeat."""

    def __init__(self, queries: list, shuffle: bool = False, repeat: RepeatMode = RepeatMode.NONE):
        self._original_queries = list(queries)
        self._queries = list(queries)
        self._index = 0
        self._shuffle = shuffle
        self._repeat = repeat
        self._shuffled_indices = []
        if self._shuffle:
            self._apply_shuffle()
    
    def _apply_shuffle(self):
        import random
        self._shuffled_indices = list(range(len(self._queries)))
        random.shuffle(self._shuffled_indices)
        self._index = 0
    
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
    
    @property
    def shuffle(self) -> bool:
        return self._shuffle
    
    @shuffle.setter
    def shuffle(self, value: bool):
        self._shuffle = value
        if value:
            self._apply_shuffle()
        else:
            self._queries = list(self._original_queries)
            self._index = min(self._index, len(self._queries) - 1)
    
    @property
    def repeat(self) -> RepeatMode:
        return self._repeat
    
    @repeat.setter
    def repeat(self, value: RepeatMode):
        self._repeat = value
    
    def advance(self) -> bool:
        if self._repeat == RepeatMode.ONE:
            return True
        if self._shuffle:
            if self._index < len(self._shuffled_indices) - 1:
                self._index += 1
                return True
            else:
                if self._repeat == RepeatMode.ALL:
                    self._index = 0
                    return True
                return False
        else:
            if self._index < len(self._queries) - 1:
                self._index += 1
                return True
            else:
                if self._repeat == RepeatMode.ALL:
                    self._index = 0
                    return True
                return False
    
    def go_back(self) -> bool:
        if self._shuffle:
            if self._index > 0:
                self._index -= 1
                return True
        else:
            if self._index > 0:
                self._index -= 1
                return True
        return False
    
    def add(self, query: str):
        self._original_queries.append(query)
        self._queries.append(query)
        if self._shuffle:
            self._apply_shuffle()
    
    def get_playlist_display(self, current_index: int = -1) -> str:
        lines = []
        for i, query in enumerate(self._queries):
            marker = ">" if i == (current_index if current_index >= 0 else self._index) else " "
            display_query = query[:50] + "..." if len(query) > 50 else query
            lines.append(f"    {marker} [{i+1}] {display_query}")
        return "\n".join(lines)


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


# ─── Utilitários ──────────────────────────────────────────────────────────────

def _str_display_width(text: str) -> int:
    """Largura visual real: caracteres CJK contam como 2 colunas."""
    width = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x1100 <= cp <= 0x115F or   # Hangul Jamo
            0x2E80 <= cp <= 0x303E or   # Radicais CJK
            0x3040 <= cp <= 0x33FF or   # Japonês (Hiragana, Katakana, etc.)
            0x3400 <= cp <= 0x4DBF or   # CJK Extensão A
            0x4E00 <= cp <= 0x9FFF or   # CJK Unificado
            0xA000 <= cp <= 0xA4CF or   # Yi
            0xAC00 <= cp <= 0xD7AF or   # Sílabas Hangul
            0xF900 <= cp <= 0xFAFF or   # Compatibilidade CJK
            0xFE10 <= cp <= 0xFE6F or   # Formas verticais / compatibilidade
            0xFF01 <= cp <= 0xFF60 or   # Fullwidth ASCII
            0xFFE0 <= cp <= 0xFFE6 or   # Símbolos fullwidth
            0x20000 <= cp <= 0x2FFFD or # CJK Extensão B+
            0x30000 <= cp <= 0x3FFFD or   # CJK Extensão G+
            0x1F300 <= cp <= 0x1FFFF    # Emojis / símbolos miscelâneos
        ):
            width += 2
        else:
            width += 1
    return width


def _fit_text(text: str, width: int) -> str:
    s = str(text).replace("\n", " ")
    if width <= 0:
        return ""
    display_w = _str_display_width(s)
    if display_w > width:
        result = []
        current_w = 0
        for ch in s:
            ch_w = _str_display_width(ch)
            if current_w + ch_w > width - 3:
                break
            result.append(ch)
            current_w += ch_w
        suffix = "..." if width > 3 else ""
        padding = " " * max(0, width - current_w - len(suffix))
        return "".join(result) + suffix + padding
    return s + " " * (width - display_w)


def _fmt_dur(s: Optional[int]) -> str:
    if s is None:
        return "--:--"
    m, sec = divmod(int(s), 60)
    h, m   = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ─── PlaybackBox ──────────────────────────────────────────────────────────────

class PlaybackBox:
    """HUD de reprodução."""

    def __init__(
        self, title: str, player_name: str, duration_s: Optional[int],
        queue_idx: int = 0, queue_total: int = 1, volume: int = 75,
        shuffle: bool = False, repeat: RepeatMode = RepeatMode.NONE,
    ):
        self.title       = title
        self.player_name = player_name
        self.duration_s  = duration_s
        self.queue_idx   = queue_idx
        self.queue_total = queue_total
        self.volume      = volume
        self.shuffle     = shuffle
        self.repeat      = repeat
        self.enabled     = sys.stdout.isatty()
        self._line_count = 0
        self._action_msg = ""
        self._action_ts = 0.0

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

    def _volume_bar(self, volume: int, width: int = 10) -> str:
        filled = int(width * volume / 100)
        empty = width - filled
        return f"🔊 [{'█' * filled}{'░' * empty}] {volume}%"
    
    def _repeat_status(self) -> str:
        if self.repeat == RepeatMode.NONE:
            return "Repeat: OFF"
        elif self.repeat == RepeatMode.ONE:
            return "Repeat: 1 (música)"
        else:
            return "Repeat: ALL (playlist)"
    
    def _shuffle_status(self) -> str:
        return f"Shuffle: {'ON' if self.shuffle else 'OFF'}"
    
    def show_action(self, msg: str):
        """Mostra uma mensagem de ação temporária."""
        self._action_msg = msg
        self._action_ts = time.monotonic()

    # ── Draw ─────────────────────────────────────────────────────────────────

    def draw(
        self, *,
        state: str,
        frame_idx: int,
        elapsed_s: float,
        paused: bool,
        supports_seek: bool = True,
        show_volume: bool = True,
    ):
        if not self.enabled:
            return

        cols  = shutil.get_terminal_size((92, 20)).columns
        width = min(max(cols, 72), 132)
        inner = width - 2

        ctrl_parts = ["p/⎵ pause", "[ ] seek±10s" if supports_seek else None, "+/- vol", "s shuffle", "r repeat", "l playlist", "f fav", "q sair"]
        ctrl_hint = "  |  ".join(p for p in ctrl_parts if p)

        state_color = C.BGREEN if state == "TOCANDO" else C.BYELLOW
        queue_str   = f"[{self.queue_idx + 1}/{self.queue_total}]"

        sep     = _c("╔" + "═" * inner + "╗", C.BCYAN, C.BOLD)
        sep_bot = _c("╚" + "═" * inner + "╝", C.BCYAN, C.BOLD)

        status_parts = [self._shuffle_status(), self._repeat_status()]
        if show_volume:
            status_parts.append(self._volume_bar(self.volume))
        status_content = f"  {'  │  '.join(status_parts)}"

        action_line = ""
        if self._action_msg and (time.monotonic() - self._action_ts) < 2.0:
            action_line = _c("║" + _fit_text(f"  ⚡ {self._action_msg}", inner) + "║", C.BYELLOW, C.BOLD)

        out = [
            sep,
            _c("║" + _fit_text(f"  {state}  {queue_str}  ♪  {self.title}", inner) + "║", state_color, C.BOLD),
            _c("║" + _fit_text(f"  {self._progress_bar(elapsed_s, inner, frame_idx)}", inner) + "║", C.BCYAN),
            _c("║" + _fit_text(status_content, inner) + "║", C.BWHITE),
        ]
        if action_line:
            out.append(action_line)
        out.append(_c("║" + _fit_text(f"  {ctrl_hint}", inner) + "║", C.DIM))
        out.append(sep_bot)

        if self._line_count:
            sys.stdout.write(f"\033[{self._line_count}F")
        new_count = len(out)
        for line in out:
            sys.stdout.write("\r\033[2K" + line + "\n")
        # Apagar linhas excedentes do frame anterior
        if self._line_count > new_count:
            for _ in range(self._line_count - new_count):
                sys.stdout.write("\r\033[2K\n")
            sys.stdout.write(f"\033[{self._line_count - new_count}F")
        sys.stdout.flush()
        self._line_count = new_count

    def clear(self):
        """Apaga o box do terminal ao encerrar."""
        if not self.enabled or not self._line_count:
            return
        sys.stdout.write(f"\033[{self._line_count}F")
        for _ in range(self._line_count):
            sys.stdout.write("\r\033[2K\n")
        sys.stdout.write(f"\033[{self._line_count}F")
        sys.stdout.flush()
        self._line_count = 0


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


def resolve_youtube_media(query: str, cache: Optional[StreamCache] = None) -> YouTubeMedia:
    # Tentar cache primeiro
    if cache:
        cached = cache.get(query)
        if cached:
            return cached
    
    yt_dlp = _require_yt_dlp()
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
        for attempt in range(CONFIG.get("retry_attempts", 3)):
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(query, download=False)
                if info:
                    break
            except Exception as e:
                last_err = e
                info = None
                if attempt < CONFIG.get("retry_attempts", 3) - 1:
                    time.sleep(1)
        if info:
            break
    
    if not info:
        raise RuntimeError(last_err or "yt-dlp não retornou informações")
    
    if "entries" in info and info["entries"]:
        info = next((e for e in info["entries"] if e), None)
        if not info:
            raise RuntimeError("Nenhum resultado válido")
    
    audio = _pick_audio(info)
    if not audio:
        raise RuntimeError("Stream de áudio não encontrado")
    video = _pick_video(info, audio)
    dur = info.get("duration")
    try:
        dur = int(dur) if dur is not None else None
    except Exception:
        dur = None
    
    media = YouTubeMedia(
        title=info.get("title") or "YouTube",
        webpage_url=info.get("webpage_url") or "",
        audio_stream_url=audio,
        video_stream_url=video,
        duration_s=dur,
    )
    
    # Salvar no cache
    if cache:
        cache.set(query, media)
    
    return media


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
    volume_fn   = None,   # fn(volume: int) -> bool
    get_volume_fn = None, # fn() -> Optional[int]
    audio_url   = None,   # removido (visualizador eliminado)
    queue: PlaybackQueue = None,
    show_playlist_fn = None,
    toggle_favorite_fn = None,
    queue_idx: int   = 0,
    queue_total: int = 1,
    show_box: bool   = True,
    show_volume: bool = True,
) -> str:
    """
    Gerencia a reprodução e lida com o input do teclado.

    Retorna: "done" | "next" | "prev" | "quit"
    """
    paused = False
    frame_idx = 0
    started_at = time.monotonic()
    paused_acc = 0.0
    pause_ts: Optional[float] = None
    current_volume = CONFIG.get("volume", 75)
    
    if get_volume_fn:
        vol = get_volume_fn()
        if vol is not None:
            current_volume = vol

    # Obter status inicial de shuffle/repeat do queue
    current_shuffle = queue.shuffle if queue else False
    current_repeat = queue.repeat if queue else RepeatMode.NONE

    box = PlaybackBox(
        title=title, player_name=player_name, duration_s=duration_s,
        queue_idx=queue_idx, queue_total=queue_total, volume=current_volume,
        shuffle=current_shuffle, repeat=current_repeat,
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
        now_ts = now_ts or time.monotonic()
        if new_state == paused:
            return
        if new_state:
            pause_ts = now_ts
        else:
            if pause_ts is not None:
                paused_acc += max(0.0, now_ts - pause_ts)
            pause_ts = None
        paused = new_state
    
    def _update_volume(delta: int):
        nonlocal current_volume
        step = CONFIG.get("volume_step", 5)
        current_volume = max(0, min(100, current_volume + delta))
        box.volume = current_volume
        if volume_fn:
            success = volume_fn(current_volume)
            status = f"Volume: {current_volume}%" if success else f"Volume: {current_volume}% (MPV sem IPC - modo local)"
        else:
            status = f"Volume: {current_volume}% (modo simulado)"
        box.show_action(status)
    
    def _toggle_shuffle():
        nonlocal current_shuffle
        if queue:
            queue.shuffle = not queue.shuffle
            current_shuffle = queue.shuffle
            box.shuffle = current_shuffle
            box.show_action(f"Shuffle: {'ATIVADO' if current_shuffle else 'DESATIVADO'}")
    
    def _cycle_repeat():
        nonlocal current_repeat
        if queue:
            modes = [RepeatMode.NONE, RepeatMode.ONE, RepeatMode.ALL]
            current_index = modes.index(queue.repeat)
            queue.repeat = modes[(current_index + 1) % len(modes)]
            current_repeat = queue.repeat
            box.repeat = current_repeat
            mode_names = {"none": "NENHUM", "one": "1 MÚSICA", "all": "TODAS"}
            box.show_action(f"Repeat: {mode_names.get(current_repeat.value, current_repeat.value)}")

    try:
        with KeyReader() as keys:
            while proc.poll() is None:
                now = time.monotonic()
                if paused_state_fn:
                    remote_state = paused_state_fn()
                    if isinstance(remote_state, bool):
                        _set_paused_state(remote_state, now)
                
                # Não atualizar volume automaticamente para não sobescrever ajustes manuais
                # if get_volume_fn:
                #     vol = get_volume_fn()
                #     if vol is not None:
                #         current_volume = vol
                #         box.volume = current_volume

                remote_elapsed = position_fn() if position_fn else None
                if isinstance(remote_elapsed, (int, float)):
                    elapsed = max(0.0, float(remote_elapsed))
                else:
                    pause_win = (now - pause_ts) if pause_ts is not None else 0.0
                    elapsed = max(0.0, now - started_at - paused_acc - pause_win)

                if show_box:
                    box.draw(
                        state="PAUSADO" if paused else "TOCANDO",
                        frame_idx=frame_idx,
                        elapsed_s=elapsed,
                        paused=paused,
                        supports_seek=seek_fn is not None,
                        show_volume=show_volume,
                    )

                key = keys.read_key() if keys.enabled else None
                if key:
                    k = key.lower() if len(key) == 1 else key

                    # Pause / Play
                    if k in ("p", " ") and pause_fn:
                        if pause_fn(not paused):
                            target_state = not paused
                            if paused_state_fn:
                                remote_state = paused_state_fn()
                                if isinstance(remote_state, bool):
                                    target_state = remote_state
                            _set_paused_state(target_state)

                    # Seek
                    elif k == "[" and seek_fn:
                        if seek_fn(-SEEK_STEP_S) and not position_fn:
                            started_at += SEEK_STEP_S
                    elif k == "]" and seek_fn:
                        if seek_fn(SEEK_STEP_S) and not position_fn:
                            started_at -= SEEK_STEP_S

                    # Volume
                    elif k in ("+", "="):
                        _update_volume(CONFIG.get("volume_step", 5))
                    elif k in ("-", "_"):
                        _update_volume(-CONFIG.get("volume_step", 5))

                    # Shuffle
                    elif k == "s":
                        _toggle_shuffle()

                    # Repeat
                    elif k == "r":
                        _cycle_repeat()

                    # Playlist
                    elif k == "l" and show_playlist_fn:
                        show_playlist_fn()
                        box.show_action("Playlist aberta")

                    # Favorite
                    elif k == "f" and toggle_favorite_fn:
                        if toggle_favorite_fn():
                            box.show_action("Adicionado aos favoritos! ✓")
                        else:
                            box.show_action("Removido dos favoritos")

                    # Quit
                    elif k == "q":
                        box.clear()
                        _do_stop()
                        return "quit"

                frame_idx += 1
                time.sleep(0.09)

    except KeyboardInterrupt:
        box.clear()
        _do_stop()
        return "quit"

    # Processo encerrado naturalmente
    box.clear()
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
    queue: Optional[PlaybackQueue] = None,
    show_playlist_fn: Optional[Callable] = None,
    toggle_favorite_fn: Optional[Callable] = None,
) -> str:
    # ── mpv (preferido: IPC permite pause e seek, com progresso real) ────────
    mpv = shutil.which("mpv")
    if mpv:
        ipc = _make_mpv_sock()
        cmd = [
            mpv, "--no-config", "--really-quiet", "--no-video",
            "--force-window=no", "--no-input-terminal",
            "--ytdl-format=bestaudio/best",
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

        def mpv_volume(vol: int) -> bool:
            # Tentar com set_property
            if has_ipc:
                result = _mpv_cmd(ipc, ["set_property", "volume", max(0, min(100, vol))])
                if result:
                    return result
                # Tentar com set (alternativo)
                return _mpv_cmd(ipc, ["set", "volume", max(0, min(100, vol))])
            return False

        def mpv_get_volume() -> Optional[int]:
            if not has_ipc:
                return None
            val = _mpv_get(ipc, "volume")
            try:
                return int(float(val)) if val is not None else None
            except Exception:
                return None

        try:
            return _wait_player(
                proc, title=title, player_name="mpv", duration_s=duration_s,
                pause_fn=mpv_pause if has_ipc else None,
                stop_fn=mpv_stop,
                seek_fn=mpv_seek if has_ipc else None,
                position_fn=mpv_pos if has_ipc else None,
                paused_state_fn=mpv_is_paused if has_ipc else None,
                volume_fn=mpv_volume if has_ipc else None,
                get_volume_fn=mpv_get_volume if has_ipc else None,
                audio_url=source_url,
                queue=queue,
                show_playlist_fn=show_playlist_fn,
                toggle_favorite_fn=toggle_favorite_fn,
                queue_idx=queue_idx, queue_total=queue_total,
                show_volume=CONFIG.get("show_volume", True),
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
            queue=queue,
            show_playlist_fn=show_playlist_fn,
            toggle_favorite_fn=toggle_favorite_fn,
            queue_idx=queue_idx, queue_total=queue_total,
        )

    log_err("Nenhum player encontrado. Instale mpv ou ffplay.")
    return "done"


def _play_watch_mode(
    source_url: str, title: str, duration_s: Optional[int],
    fallback_url: str = "",
    queue_idx: int = 0, queue_total: int = 1,
    queue: Optional[PlaybackQueue] = None,
    show_playlist_fn: Optional[Callable] = None,
    toggle_favorite_fn: Optional[Callable] = None,
) -> str:
    mpv = shutil.which("mpv")
    if mpv:
        ipc = _make_mpv_sock()
        vo_candidates = (("tct", 24), ("caca", 20), ("kitty", 30), ("sixel", 30))
        for vo, max_fps in vo_candidates:
            cmd = [
                mpv, "--no-config", "--really-quiet", "--terminal=yes",
                "--force-window=no", "--profile=sw-fast", "--framedrop=vo",
                "--no-input-terminal",
                f"--vo={vo}",
            ]
            if ipc:
                cmd.append(f"--input-ipc-server={ipc}")
            if max_fps:
                cmd.append(f"--vf=fps={max_fps}")
            cmd.append(source_url)
            try:
                proc = _start_proc(cmd, f"Iniciando mpv (vo={vo})...", delay=0.7)
            except Exception:
                _cleanup_sock(ipc)
                continue
            if proc.poll() is not None:
                _cleanup_sock(ipc)
                if proc.returncode == 0:
                    return "done"
                continue
            
            # Configurar IPC para controle
            has_ipc = bool(ipc and _wait_mpv_sock(ipc))
            
            def mpv_watch_pause(should: bool) -> bool:
                return _mpv_cmd(ipc, ["set_property", "pause", should]) if has_ipc else False
            
            def mpv_watch_stop():
                if proc.poll() is not None:
                    return
                if has_ipc:
                    _mpv_cmd(ipc, ["quit"])
                    time.sleep(0.05)
                if proc.poll() is None:
                    proc.terminate()
            
            def mpv_watch_seek(delta: int) -> bool:
                return _mpv_cmd(ipc, ["seek", delta, "relative"]) if has_ipc else False
            
            def mpv_watch_volume(vol: int) -> bool:
                if has_ipc:
                    return _mpv_cmd(ipc, ["set_property", "volume", max(0, min(100, vol))])
                return False
            
            try:
                return _wait_player(
                    proc, title=title, player_name=f"mpv (vo={vo})", duration_s=duration_s,
                    pause_fn=mpv_watch_pause if has_ipc else None,
                    stop_fn=mpv_watch_stop,
                    seek_fn=mpv_watch_seek if has_ipc else None,
                    position_fn=None,
                    paused_state_fn=None,
                    volume_fn=mpv_watch_volume if has_ipc else None,
                    get_volume_fn=None,
                    queue=queue,
                    show_playlist_fn=show_playlist_fn,
                    toggle_favorite_fn=toggle_favorite_fn,
                    queue_idx=queue_idx, queue_total=queue_total,
                    show_volume=CONFIG.get("show_volume", True),
                )
            finally:
                _cleanup_sock(ipc)

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
            queue=queue,
            show_playlist_fn=show_playlist_fn,
            toggle_favorite_fn=toggle_favorite_fn,
            show_volume=CONFIG.get("show_volume", True),
        )

    log_err("Nenhum player para vídeo. Instale mpv ou ffplay.")
    return "done"


# ─── GUI (PyQt5) ────────────────────────────────────────────────────────────

def _gui_get_queries() -> list:
    try:
        from PyQt5.QtWidgets import (  # type: ignore
            QApplication, QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton,
        )
        from PyQt5.QtCore import Qt  # type: ignore
    except Exception:
        log_err("PyQt5 não encontrado. Instale: pip install PyQt5")
        return []

    app = QApplication.instance() or QApplication(sys.argv)

    dlg = QDialog()
    dlg.setWindowTitle("YouTube Music Player v3 — Selecione as músicas")
    dlg.setMinimumSize(550, 350)
    layout = QVBoxLayout(dlg)

    lbl = QLabel("<b>Digite músicas ou URLs do YouTube (uma por linha):</b>")
    lbl.setAlignment(Qt.AlignCenter)
    layout.addWidget(lbl)

    edit = QTextEdit()
    edit.setPlaceholderText(
        "Exemplos:\n"
        "• Never Gonna Give You Up\n"
        "• https://youtu.be/dQw4w9WgXcQ\n"
        "• Bohemian Rhapsody - Queen"
    )
    edit.setMinimumHeight(180)
    edit.setStyleSheet("font-family: Arial; font-size: 12px;")
    layout.addWidget(edit)

    btn = QPushButton("▶  TOCAR")
    btn.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px;")
    btn.setMinimumSize(200, 40)
    btn.clicked.connect(dlg.accept)
    layout.addWidget(btn)

    if dlg.exec() != QDialog.Accepted:
        return []
    return [ln.strip() for ln in edit.toPlainText().splitlines() if ln.strip()]


def _gui_select_mode() -> Optional[str]:
    try:
        from PyQt5.QtWidgets import QApplication, QInputDialog  # type: ignore
    except Exception:
        log_err("PyQt5 não encontrado.")
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
        description="YouTube Music Player v3 — Toca músicas do YouTube. Basta executar e escolher ou passar o nome/URL."
    )
    p.add_argument(
        "query", nargs="*",
        help="Músicas ou URLs (ex: 'Bohemian Rhapsody' ou 'https://youtu.be/...')",
    )
    p.add_argument(
        "--video", action="store_true",
        help="Modo vídeo (requer terminal com suporte a vídeo)",
    )
    return p.parse_args()


def _silence():
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false;qt.*.warning=false")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    _silence()
    args = _parse_args()
    
    # Configurar banner
    banner()

    # Inicializar cache, histórico e favoritos
    cache = StreamCache() if CONFIG.get("cache_enabled", True) else None
    history = HistoryManager() if CONFIG.get("history_enabled", True) else None
    favorites = FavoritesManager() if CONFIG.get("favorites_enabled", True) else None

    # Coletar queries
    queries = [q.strip() for q in args.query if q.strip()]
    if not queries:
        queries = _gui_get_queries()
    if not queries:
        log_warn("Nenhuma música informada. Encerrando.")
        return 1

    # Selecionar modo (padrão: áudio, --video para modo vídeo)
    mode = "watch" if args.video else "audio"

    # Criar fila com shuffle e repeat do config
    queue = PlaybackQueue(
        queries,
        shuffle=CONFIG.get("shuffle", False),
        repeat=RepeatMode(CONFIG.get("repeat", "none"))
    )
    log_ok(f"Playlist: {queue.total} música(s) — modo: {'ver + ouvir' if mode == 'watch' else 'somente ouvir'}")
    
    current_media = None
    
    def show_playlist():
        # Mostrar playlist em um box temporário
        playlist_text = queue.get_playlist_display(queue.index)
        print(f"\n{'+' * 60}")
        print(f"  PLAYLIST ATUAL ({queue.total} músicas)")
        print(f"{'=' * 60}")
        print(playlist_text)
        print(f"{'=' * 60}")
        print("  Pressione qualquer tecla para continuar...", end="", flush=True)
        # Esperar tecla
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.getch()
            else:
                import sys, tty, termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except:
            input()
    
    def toggle_favorite():
        if favorites and current_media:
            if favorites.is_favorite(current_media.webpage_url):
                favorites.remove(current_media.webpage_url)
                return False
            else:
                favorites.add(current_media)
                return True
        return False
    
    while True:
        query = queue.current
        if not query:
            break

        log_step(f"[{queue.index + 1}/{queue.total}] Resolvendo: {query}")
        try:
            current_media = _run_with_spinner(
                "Buscando stream no YouTube...", 
                resolve_youtube_media, query, cache
            )
            if history:
                history.add(current_media)
        except Exception as e:
            log_err(f"Falha: {e}")
            if not queue.advance():
                break
            continue

        log_ok(f"Música: {current_media.title}")

        if mode == "watch":
            result = _play_watch_mode(
                current_media.video_stream_url, current_media.title, current_media.duration_s,
                fallback_url=current_media.webpage_url,
                queue_idx=queue.index, queue_total=queue.total,
                queue=queue,
                show_playlist_fn=show_playlist,
                toggle_favorite_fn=toggle_favorite,
            )
        else:
            result = _play_audio_mode(
                current_media.audio_stream_url, current_media.title, current_media.duration_s,
                queue_idx=queue.index, queue_total=queue.total,
                queue=queue,
                show_playlist_fn=show_playlist,
                toggle_favorite_fn=toggle_favorite,
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
