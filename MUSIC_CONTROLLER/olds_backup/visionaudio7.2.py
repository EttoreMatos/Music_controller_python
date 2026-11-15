from __future__ import annotations

import contextlib
import glob
import importlib
from collections import OrderedDict
import importlib.util
import io
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
import webbrowser
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QPointF, QRectF, QSize, Qt, QProcess, QProcessEnvironment, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QGraphicsDropShadowEffect,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


APP_TITLE = "VisionAudio 7.2"
SEEK_STEP_S = 10
PRIMARY_TIMELINE_COLUMNS = 140


def format_seconds(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--:--"
    total = max(0, int(value))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "--"
    size = float(max(0, value))
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return "--"  # never reached; satisfies type checker


def is_url(value: str) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def dependency_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def normalize_error_message(message: Optional[str], error_type: Optional[str] = None) -> str:
    text = str(message or "").strip()
    if text:
        return text
    if error_type:
        return f"{error_type}: ocorreu um erro sem detalhes adicionais."
    return "Ocorreu um erro sem detalhes adicionais."


def describe_exception(exc: BaseException) -> Tuple[str, str]:
    error_type = type(exc).__name__
    message = normalize_error_message(str(exc), error_type)
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    if not details:
        details = f"{error_type}: {message}"
    return message, details


def qt_message(
    parent: QWidget,
    title: str,
    message: str,
    level: str = "info",
    details: Optional[str] = None,
) -> None:
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(normalize_error_message(message) if level == "error" else str(message or title).strip() or title)
    if level == "error":
        box.setIcon(QMessageBox.Critical)
    elif level == "warn":
        box.setIcon(QMessageBox.Warning)
    else:
        box.setIcon(QMessageBox.Information)
    if details:
        box.setDetailedText(details)
    box.setStandardButtons(QMessageBox.Ok)
    box.setStyleSheet(
        """
        QMessageBox { background-color: #141d2a; }
        QMessageBox QLabel {
            color: #e8f0fa;
            background-color: transparent;
            font-size: 12px;
        }
        QMessageBox QPushButton {
            min-width: 88px;
            min-height: 28px;
            padding: 6px 14px;
            border-radius: 9px;
            border: 1px solid #4a6788;
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2e4660, stop:1 #1a2c40);
            color: #edf5fd;
            font-weight: 600;
        }
        QMessageBox QPushButton:hover {
            border-color: #6b92b8;
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #3a5c78, stop:1 #22384d);
        }
        """
    )
    box.exec()


def apply_drop_shadow(
    widget: QWidget,
    *,
    blur_radius: float,
    x_offset: float = 0.0,
    y_offset: float = 8.0,
    color: str = "#000000",
    alpha: int = 120,
) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur_radius)
    effect.setOffset(x_offset, y_offset)
    shade = QColor(color)
    shade.setAlpha(alpha)
    effect.setColor(shade)
    widget.setGraphicsEffect(effect)


@dataclass(frozen=True)
class EffectConfigCompat:
    FPS: int = 30
    NUM_LEDS: int = 6
    STEP_FRAMES_MIN: int = 2
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
    EDGE_IDLE_DECAY: float = 0.78
    RMS_FAST_ALPHA: float = 0.30
    RMS_SLOW_ALPHA: float = 0.06
    FLUX_FAST_ALPHA: float = 0.35
    FLUX_SLOW_ALPHA: float = 0.07
    STARTUP_RAMP_SECONDS: float = 2.0
    SEG_HOP_SECONDS: float = 0.85
    SEG_K_SEGMENTS: int = 56
    SEG_MIN_GAP_SECONDS: float = 1.4
    SEG_IGNORE_START_SECONDS: float = 2.5
    SEG_NOVELTY_THRESHOLD: float = 0.16
    SEG_STRUCT_CHANGE_THRESHOLD: float = 0.10
    SEG_PERSISTENCE_AHEAD_SECONDS: float = 3.0
    SEG_MIN_PERSIST_DELTA: float = 0.06
    SEG_PERSISTENCE_MIN_RATIO: float = 0.50
    SEG_MIN_ENERGY_THRESHOLD: float = 0.10
    NOISE_FLATNESS_THRESHOLD: float = 0.55
    BRIGHT_CENTROID_THRESHOLD: float = 0.22
    TRANS_MIN_DURATION_S: float = 0.45
    TRANS_MAX_DURATION_S: float = 1.80
    TRANS_BLEND_FRAMES: int = 5
    BPM_RESYNC_STRENGTH: float = 0.85
    BPM_CONTINUOUS_SYNC_STRENGTH: float = 0.06
    PLAYBACK_MAX_DRIFT_FRAMES: int = 2


@dataclass(frozen=True)
class ControlSpec:
    key: str
    label: str
    group: str
    minimum: float
    maximum: float
    default: float
    scale: float = 1.0
    value_type: type = float
    decimals: int = 2
    primary: bool = False
    tooltip: str = ""

    def clamp(self, value: float) -> float:
        clamped = min(self.maximum, max(self.minimum, value))
        if self.value_type is int:
            return int(round(clamped))
        if self.decimals <= 0:
            return round(clamped)
        return round(clamped, self.decimals)

    def slider_range(self) -> Tuple[int, int]:
        return int(round(self.minimum * self.scale)), int(round(self.maximum * self.scale))

    def to_slider(self, value: float) -> int:
        return int(round(self.clamp(value) * self.scale))

    def from_slider(self, raw: int) -> float:
        return self.clamp(raw / self.scale)


_CFG = EffectConfigCompat()

CONTROL_SPECS: Dict[str, ControlSpec] = {
    "ACTIVE_MAX_PWM": ControlSpec("ACTIVE_MAX_PWM", "Brilho", "Main", 120.0, 255.0, _CFG.ACTIVE_MAX_PWM, 1.0, float, 0, True, "Brilho maximo dos LEDs centrais."),
    "EDGE_PULSE_GAIN": ControlSpec("EDGE_PULSE_GAIN", "Pontas", "Main", 80.0, 255.0, _CFG.EDGE_PULSE_GAIN, 1.0, float, 0, True, "Forca das pontas."),
    "BEAT_BOOST_PWM": ControlSpec("BEAT_BOOST_PWM", "Beat boost", "Main", 0.0, 60.0, _CFG.BEAT_BOOST_PWM, 1.0, float, 0, True, "Boost de brilho em batidas."),
    "MIDDLE_FADE_DECAY": ControlSpec("MIDDLE_FADE_DECAY", "Fade meio", "Main", 0.30, 0.95, _CFG.MIDDLE_FADE_DECAY, 100.0, float, 2, True, "Cauda dos LEDs do meio."),
    "MIDDLE_EDGE_SYNC_GAIN": ControlSpec("MIDDLE_EDGE_SYNC_GAIN", "Sync meio", "Main", 0.0, 150.0, _CFG.MIDDLE_EDGE_SYNC_GAIN, 1.0, float, 0, True, "Quanto as pontas empurram o centro."),
    "TAIL_CUTOFF": ControlSpec("TAIL_CUTOFF", "Cutoff", "Main", 0.0, 12.0, _CFG.TAIL_CUTOFF, 10.0, float, 1, True, "Corta ruidos baixos no final."),
    "SEG_NOVELTY_THRESHOLD": ControlSpec("SEG_NOVELTY_THRESHOLD", "TH novidade", "Main", 0.05, 0.35, _CFG.SEG_NOVELTY_THRESHOLD, 100.0, float, 2, True, "Threshold de disparo de transicoes."),
    "SEG_STRUCT_CHANGE_THRESHOLD": ControlSpec("SEG_STRUCT_CHANGE_THRESHOLD", "TH estrutura", "Main", 0.04, 0.25, _CFG.SEG_STRUCT_CHANGE_THRESHOLD, 100.0, float, 2, True, "Minimo de troca estrutural."),
    "TRANS_MAX_DURATION_S": ControlSpec("TRANS_MAX_DURATION_S", "Trans max", "Main", 0.50, 2.80, _CFG.TRANS_MAX_DURATION_S, 100.0, float, 2, True, "Duracao maxima das transicoes."),
    "TRANS_BLEND_FRAMES": ControlSpec("TRANS_BLEND_FRAMES", "Blend", "Main", 1.0, 12.0, _CFG.TRANS_BLEND_FRAMES, 1.0, int, 0, True, "Frames de blend."),
    "ACTIVE_MIN_PWM": ControlSpec("ACTIVE_MIN_PWM", "Brilho base", "Dynamics", 20.0, 140.0, _CFG.ACTIVE_MIN_PWM, 1.0, float, 0),
    "MIDDLE_KICK_BRIGHT_GAIN": ControlSpec("MIDDLE_KICK_BRIGHT_GAIN", "Kick meio", "Dynamics", 20.0, 220.0, _CFG.MIDDLE_KICK_BRIGHT_GAIN, 1.0, float, 0),
    "MIDDLE_STRONG_KICK_BRIGHT_GAIN": ControlSpec("MIDDLE_STRONG_KICK_BRIGHT_GAIN", "Kick forte", "Dynamics", 10.0, 140.0, _CFG.MIDDLE_STRONG_KICK_BRIGHT_GAIN, 1.0, float, 0),
    "HEAVY_BEAT_THRESHOLD": ControlSpec("HEAVY_BEAT_THRESHOLD", "Beat pesado", "Dynamics", 0.10, 0.95, _CFG.HEAVY_BEAT_THRESHOLD, 100.0, float, 2),
    "EDGE_HEAVY_BONUS_GAIN": ControlSpec("EDGE_HEAVY_BONUS_GAIN", "Bonus ponta", "Edges", 30.0, 255.0, _CFG.EDGE_HEAVY_BONUS_GAIN, 1.0, float, 0),
    "EDGE_KICK_WINDOW_BOOST": ControlSpec("EDGE_KICK_WINDOW_BOOST", "Janela kick", "Edges", 50.0, 280.0, _CFG.EDGE_KICK_WINDOW_BOOST, 1.0, float, 0),
    "EDGE_GATE_OPEN_THRESHOLD": ControlSpec("EDGE_GATE_OPEN_THRESHOLD", "Gate open", "Edges", 0.05, 0.60, _CFG.EDGE_GATE_OPEN_THRESHOLD, 100.0, float, 2),
    "EDGE_GATE_CLOSE_THRESHOLD": ControlSpec("EDGE_GATE_CLOSE_THRESHOLD", "Gate close", "Edges", 0.02, 0.40, _CFG.EDGE_GATE_CLOSE_THRESHOLD, 100.0, float, 2),
    "EDGE_GATE_MIN_LOW_DOMINANCE": ControlSpec("EDGE_GATE_MIN_LOW_DOMINANCE", "Low dom", "Edges", 0.02, 0.60, _CFG.EDGE_GATE_MIN_LOW_DOMINANCE, 100.0, float, 2),
    "EDGE_NOISE_FLOOR_PWM": ControlSpec("EDGE_NOISE_FLOOR_PWM", "Noise floor", "Edges", 0.0, 60.0, _CFG.EDGE_NOISE_FLOOR_PWM, 1.0, float, 0),
    "EDGE_IDLE_DECAY": ControlSpec("EDGE_IDLE_DECAY", "Idle decay", "Edges", 0.40, 0.95, _CFG.EDGE_IDLE_DECAY, 100.0, float, 2),
    "NOISE_FLATNESS_THRESHOLD": ControlSpec("NOISE_FLATNESS_THRESHOLD", "Flatness", "Segmentation", 0.20, 0.90, _CFG.NOISE_FLATNESS_THRESHOLD, 100.0, float, 2),
    "BRIGHT_CENTROID_THRESHOLD": ControlSpec("BRIGHT_CENTROID_THRESHOLD", "Centroid", "Segmentation", 0.05, 0.60, _CFG.BRIGHT_CENTROID_THRESHOLD, 100.0, float, 2),
    "SEG_MIN_GAP_SECONDS": ControlSpec("SEG_MIN_GAP_SECONDS", "Gap min", "Segmentation", 0.40, 3.00, _CFG.SEG_MIN_GAP_SECONDS, 100.0, float, 2),
    "SEG_MIN_ENERGY_THRESHOLD": ControlSpec("SEG_MIN_ENERGY_THRESHOLD", "Energia min", "Segmentation", 0.02, 0.40, _CFG.SEG_MIN_ENERGY_THRESHOLD, 100.0, float, 2),
    "TRANS_MIN_DURATION_S": ControlSpec("TRANS_MIN_DURATION_S", "Trans min", "Timing", 0.15, 1.20, _CFG.TRANS_MIN_DURATION_S, 100.0, float, 2),
    "BPM_RESYNC_STRENGTH": ControlSpec("BPM_RESYNC_STRENGTH", "Resync", "Timing", 0.20, 1.00, _CFG.BPM_RESYNC_STRENGTH, 100.0, float, 2),
    "BPM_CONTINUOUS_SYNC_STRENGTH": ControlSpec("BPM_CONTINUOUS_SYNC_STRENGTH", "Sync continuo", "Timing", 0.01, 0.30, _CFG.BPM_CONTINUOUS_SYNC_STRENGTH, 100.0, float, 2),
}

PRIMARY_CONTROL_KEYS = [key for key, spec in CONTROL_SPECS.items() if spec.primary]
ADVANCED_GROUPS = ["Dynamics", "Edges", "Segmentation", "Timing"]
TRANSITION_PROFILE_LABELS = {"low": "Baixa", "medium": "Equilibrada", "high": "Alta"}
EFFECT_PROFILE_LABELS = {"low": "Suave", "medium": "Equilibrada", "high": "Agressiva"}
MIDDLE_SPEED_OPTIONS = {1.0: "1x BPM", 2.0: "2x BPM"}


def default_control_values() -> Dict[str, float]:
    return {key: spec.default for key, spec in CONTROL_SPECS.items()}


def build_profile_overrides(transition_profile: str, effect_profile: str) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if transition_profile == "low":
        overrides.update(
            {
                "SEG_K_SEGMENTS": 42,
                "SEG_MIN_GAP_SECONDS": 2.1,
                "SEG_NOVELTY_THRESHOLD": 0.20,
                "SEG_STRUCT_CHANGE_THRESHOLD": 0.12,
            }
        )
    elif transition_profile == "high":
        overrides.update(
            {
                "SEG_K_SEGMENTS": 74,
                "SEG_MIN_GAP_SECONDS": 0.95,
                "SEG_NOVELTY_THRESHOLD": 0.13,
                "SEG_STRUCT_CHANGE_THRESHOLD": 0.08,
            }
        )

    if effect_profile == "low":
        overrides.update(
            {
                "ACTIVE_MAX_PWM": 172.0,
                "BEAT_BOOST_PWM": 18.0,
                "MIDDLE_KICK_BRIGHT_GAIN": 95.0,
                "MIDDLE_STRONG_KICK_BRIGHT_GAIN": 52.0,
                "EDGE_PULSE_GAIN": 135.0,
                "NOISE_FLATNESS_THRESHOLD": 0.62,
                "BRIGHT_CENTROID_THRESHOLD": 0.26,
            }
        )
    elif effect_profile == "high":
        overrides.update(
            {
                "ACTIVE_MAX_PWM": 220.0,
                "BEAT_BOOST_PWM": 38.0,
                "MIDDLE_KICK_BRIGHT_GAIN": 150.0,
                "MIDDLE_STRONG_KICK_BRIGHT_GAIN": 85.0,
                "EDGE_PULSE_GAIN": 190.0,
                "NOISE_FLATNESS_THRESHOLD": 0.48,
                "BRIGHT_CENTROID_THRESHOLD": 0.18,
                "TRANS_BLEND_FRAMES": 4,
            }
        )
    return overrides


@dataclass
class UIConfigState:
    transition_profile: str = "medium"
    effect_profile: str = "medium"
    middle_speed_multiplier: float = 1.0
    values: Dict[str, float] = field(default_factory=default_control_values)

    def normalized_values(self) -> Dict[str, float]:
        merged = default_control_values()
        merged.update(self.values or {})
        normalized: Dict[str, float] = {}
        for key, spec in CONTROL_SPECS.items():
            value = merged.get(key, spec.default)
            normalized[key] = spec.clamp(float(value))
        return normalized

    def to_effect_config(self, engine_module: Optional[Any] = None) -> Any:
        config_cls = getattr(engine_module, "EffectConfig", EffectConfigCompat)
        cfg = config_cls()
        overrides = build_profile_overrides(self.transition_profile, self.effect_profile)
        overrides.update(self.normalized_values())
        return replace(cfg, **overrides)

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "transition_profile": self.transition_profile,
            "effect_profile": self.effect_profile,
            "middle_speed_multiplier": float(self.middle_speed_multiplier),
            "values": self.normalized_values(),
        }

    @classmethod
    def from_serializable(cls, payload: Dict[str, Any]) -> "UIConfigState":
        if not isinstance(payload, dict):
            raise ValueError("Formato de configuração inválido.")
        transition_profile = str(payload.get("transition_profile", "medium"))
        if transition_profile not in TRANSITION_PROFILE_LABELS:
            transition_profile = "medium"
        effect_profile = str(payload.get("effect_profile", "medium"))
        if effect_profile not in EFFECT_PROFILE_LABELS:
            effect_profile = "medium"
        middle_speed_raw = payload.get("middle_speed_multiplier", 1.0)
        try:
            middle_speed = float(middle_speed_raw)
        except (TypeError, ValueError):
            middle_speed = 1.0
        if middle_speed not in MIDDLE_SPEED_OPTIONS:
            middle_speed = 1.0
        values = default_control_values()
        raw_values = payload.get("values", {})
        if isinstance(raw_values, dict):
            for key, spec in CONTROL_SPECS.items():
                if key not in raw_values:
                    continue
                try:
                    values[key] = spec.clamp(float(raw_values[key]))
                except (TypeError, ValueError):
                    values[key] = spec.default
        return cls(
            transition_profile=transition_profile,
            effect_profile=effect_profile,
            middle_speed_multiplier=middle_speed,
            values=values,
        )


@dataclass
class LoadedTrack:
    source_mode: str
    requested_source: str
    resolved_source: str
    title: str
    display_source: str
    duration_s: Optional[float] = None
    file_size_bytes: Optional[int] = None
    youtube_video_url: Optional[str] = None
    youtube_page_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationResult:
    numeric_values: Dict[str, float]
    categorical_values: Dict[str, Any]
    feature_snapshot: Dict[str, float]
    using_sklearn: bool
    model_label: str = "fallback interno"
    summary_lines: List[str] = field(default_factory=list)


@dataclass
class GeneratedSequence:
    track: LoadedTrack
    config_state: UIConfigState
    effect_config: Any
    led_patterns: List[Tuple[int, Any]]
    dominant_leds: List[int]
    duration_s: float
    transitions: List[Any]
    playback_source: str
    tempo_bpm: float
    beat_frames: List[int]
    feature_snapshot: Dict[str, float]
    preview_columns: List[List[int]]
    temp_files: List[str] = field(default_factory=list)


@dataclass
class CachedTrackData:
    y: Any
    sr: int
    duration_s: float
    feature_snapshot: Optional[Dict[str, float]] = None


def build_preview_columns(led_patterns: Sequence[Tuple[int, Any]], columns: int = PRIMARY_TIMELINE_COLUMNS) -> List[List[int]]:
    if not led_patterns:
        return []
    total = len(led_patterns)
    chunk = max(1, total // columns)
    output: List[List[int]] = []
    for start in range(0, total, chunk):
        block = led_patterns[start : start + chunk]
        accum = [0.0] * 6
        count = 0
        for _, values in block:
            vals = [int(v) for v in list(values)]
            for idx in range(min(6, len(vals))):
                accum[idx] += vals[idx]
            count += 1
        if count:
            output.append([int(round(v / count)) for v in accum])
    return output[:columns]


class DependencyError(RuntimeError):
    pass


class AudioSourceResolver:
    def load(self, source_mode: str, source_value: str) -> LoadedTrack:
        value = (source_value or "").strip()
        if not value:
            raise ValueError("Informe um arquivo ou busca do YouTube.")

        if source_mode == "local":
            path = os.path.abspath(os.path.expanduser(value))
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            return LoadedTrack(
                source_mode="local",
                requested_source=value,
                resolved_source=path,
                title=os.path.basename(path),
                display_source=path,
                file_size_bytes=os.path.getsize(path),
                metadata={"suffix": Path(path).suffix.lower()},
            )

        stream_url, title, video_url, page_url = self.resolve_youtube_stream(value)
        return LoadedTrack(
            source_mode="youtube",
            requested_source=value,
            resolved_source=stream_url,
            title=title,
            display_source=value,
            youtube_video_url=video_url,
            youtube_page_url=page_url,
            metadata={"platform": "youtube"},
        )

    @staticmethod
    def resolve_youtube_stream(query_or_url: str) -> Tuple[str, str, Optional[str], Optional[str]]:
        if not dependency_available("yt_dlp"):
            raise DependencyError("yt-dlp nao encontrado. Instale com: pip install yt-dlp")
        yt_dlp = importlib.import_module("yt_dlp")
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
            raise RuntimeError("yt-dlp nao retornou informacoes do video.")
        if "entries" in info and info["entries"]:
            info = next((entry for entry in info["entries"] if entry), None)
        if not info:
            raise RuntimeError("Nenhum resultado de audio encontrado.")
        formats = info.get("formats") or []
        stream_url = info.get("url")
        if not stream_url:
            audio_only = [
                item
                for item in formats
                if item.get("url")
                and item.get("acodec") not in (None, "none")
                and item.get("vcodec") in (None, "none")
            ]
            if audio_only:
                audio_only.sort(key=lambda item: (item.get("abr") or 0.0, item.get("tbr") or 0.0))
                stream_url = audio_only[-1]["url"]
        if not stream_url:
            raise RuntimeError("Nao foi possivel resolver a URL de audio.")
        av_formats = [
            item
            for item in formats
            if item.get("url")
            and item.get("acodec") not in (None, "none")
            and item.get("vcodec") not in (None, "none")
        ]
        video_formats = [
            item
            for item in formats
            if item.get("url")
            and item.get("vcodec") not in (None, "none")
        ]
        video_url = None
        preferred_video_formats = av_formats or video_formats
        if preferred_video_formats:
            preferred_video_formats.sort(
                key=lambda item: (
                    item.get("height") or 0.0,
                    item.get("fps") or 0.0,
                    1 if item.get("acodec") not in (None, "none") else 0,
                    item.get("tbr") or 0.0,
                )
            )
            video_url = preferred_video_formats[-1]["url"]
        page_url = info.get("webpage_url") or info.get("original_url") or query_or_url
        return stream_url, info.get("title") or "YouTube", video_url, page_url


class AudioReader6EngineLoader:
    def __init__(self) -> None:
        self._module: Optional[Any] = None
        self._error: Optional[BaseException] = None

    def load(self) -> Any:
        if self._module is not None:
            return self._module
        if self._error is not None:
            raise DependencyError(str(self._error))
        try:
            self._module = importlib.import_module("audio_reader6")
            return self._module
        except Exception as exc:
            self._error = exc
            raise DependencyError(
                "Nao foi possivel importar audio_reader6. "
                "Verifique dependencias como librosa, pygame e pyserial.\n"
                f"Detalhe: {exc}"
            ) from exc


def create_temp_wav_from_audio(y: Any, sr: int) -> str:
    import numpy as np

    fd, path = tempfile.mkstemp(prefix="visionaudio72_", suffix=".wav")
    os.close(fd)
    try:
        pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sr))
            wav_file.writeframes(pcm.tobytes())
        return path
    except Exception:
        with contextlib.suppress(OSError):
            os.remove(path)
        raise


class ParameterRecommendationService:
    FEATURE_KEYS = [
        "tempo_bpm",
        "beat_density",
        "rms_mean",
        "rms_p90",
        "bass_ratio",
        "centroid_mean",
        "centroid_std",
        "flatness_mean",
        "flux_mean",
        "boundaries_per_minute",
    ]
    NUMERIC_TARGET_KEYS = [
        "ACTIVE_MAX_PWM",
        "EDGE_PULSE_GAIN",
        "BEAT_BOOST_PWM",
        "MIDDLE_FADE_DECAY",
        "MIDDLE_EDGE_SYNC_GAIN",
        "SEG_NOVELTY_THRESHOLD",
        "SEG_STRUCT_CHANGE_THRESHOLD",
        "TRANS_MAX_DURATION_S",
        "TRANS_BLEND_FRAMES",
        "EDGE_HEAVY_BONUS_GAIN",
    ]
    CATEGORICAL_TARGET_KEYS = [
        "transition_profile",
        "effect_profile",
        "middle_speed_multiplier",
    ]

    def __init__(self) -> None:
        self._archetypes = self._build_archetypes()
        self._using_sklearn = False
        self._model_label = "fallback interno"
        self._numeric_model = None
        self._numeric_scaler = None
        self._numeric_target_scaler = None
        self._categorical_models: Dict[str, Any] = {}
        self._categorical_scalers: Dict[str, Any] = {}
        self._feature_matrix: List[List[float]] = []
        self._train()

    @property
    def using_sklearn(self) -> bool:
        return self._using_sklearn

    def _build_archetypes(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "ambient",
                "features": [82, 0.35, 0.18, 0.28, 0.32, 0.18, 0.06, 0.18, 0.10, 1.2],
                "numeric": [165, 125, 10, 0.78, 48, 0.19, 0.13, 1.85, 6, 90],
                "categorical": {"transition_profile": "low", "effect_profile": "low", "middle_speed_multiplier": 1.0},
            },
            {
                "name": "cinematic",
                "features": [98, 0.42, 0.24, 0.38, 0.48, 0.21, 0.08, 0.24, 0.16, 1.7],
                "numeric": [182, 148, 16, 0.72, 62, 0.17, 0.11, 1.95, 6, 118],
                "categorical": {"transition_profile": "low", "effect_profile": "medium", "middle_speed_multiplier": 1.0},
            },
            {
                "name": "pop",
                "features": [116, 0.58, 0.34, 0.52, 0.51, 0.29, 0.11, 0.30, 0.24, 2.3],
                "numeric": [198, 162, 26, 0.62, 84, 0.16, 0.10, 1.55, 5, 142],
                "categorical": {"transition_profile": "medium", "effect_profile": "medium", "middle_speed_multiplier": 1.0},
            },
            {
                "name": "house",
                "features": [124, 0.74, 0.42, 0.64, 0.60, 0.32, 0.12, 0.34, 0.30, 2.8],
                "numeric": [210, 182, 34, 0.56, 102, 0.14, 0.09, 1.30, 4, 165],
                "categorical": {"transition_profile": "high", "effect_profile": "high", "middle_speed_multiplier": 2.0},
            },
            {
                "name": "rock",
                "features": [132, 0.68, 0.48, 0.72, 0.66, 0.28, 0.15, 0.38, 0.36, 3.0],
                "numeric": [214, 188, 32, 0.54, 96, 0.13, 0.08, 1.22, 4, 178],
                "categorical": {"transition_profile": "high", "effect_profile": "high", "middle_speed_multiplier": 1.0},
            },
            {
                "name": "trap",
                "features": [146, 0.62, 0.40, 0.68, 0.74, 0.24, 0.09, 0.33, 0.34, 2.6],
                "numeric": [220, 198, 38, 0.52, 112, 0.14, 0.08, 1.12, 4, 192],
                "categorical": {"transition_profile": "high", "effect_profile": "high", "middle_speed_multiplier": 2.0},
            },
            {
                "name": "latin",
                "features": [102, 0.61, 0.32, 0.50, 0.58, 0.34, 0.13, 0.28, 0.26, 2.4],
                "numeric": [194, 168, 24, 0.60, 88, 0.16, 0.10, 1.40, 5, 150],
                "categorical": {"transition_profile": "medium", "effect_profile": "medium", "middle_speed_multiplier": 2.0},
            },
            {
                "name": "noise",
                "features": [110, 0.44, 0.30, 0.54, 0.40, 0.38, 0.22, 0.62, 0.42, 3.6],
                "numeric": [188, 158, 18, 0.66, 74, 0.20, 0.12, 1.45, 6, 122],
                "categorical": {"transition_profile": "medium", "effect_profile": "low", "middle_speed_multiplier": 1.0},
            },
        ]

    def _feature_ranges(self) -> List[float]:
        columns = list(zip(*self._feature_matrix))
        return [max(max(col) - min(col), 1e-6) for col in columns]

    def _build_augmented_samples(self) -> List[Tuple[List[float], List[float], Dict[str, Any]]]:
        if not self._feature_matrix:
            return []
        rng = random.Random(7)
        feature_ranges = self._feature_ranges()
        samples: List[Tuple[List[float], List[float], Dict[str, Any]]] = []

        for archetype in self._archetypes:
            base_features = [float(value) for value in archetype["features"]]
            base_numeric = [float(value) for value in archetype["numeric"]]
            base_categorical = dict(archetype["categorical"])
            samples.append((base_features, base_numeric, base_categorical))

            for _ in range(18):
                noisy_features: List[float] = []
                for idx, (value, spread) in enumerate(zip(base_features, feature_ranges)):
                    jitter_ratio = 0.045 if idx in (0, 9) else 0.032
                    noisy_features.append(float(value + rng.gauss(0.0, spread * jitter_ratio)))
                noisy_numeric: List[float] = []
                for idx, key in enumerate(self.NUMERIC_TARGET_KEYS):
                    spec = CONTROL_SPECS[key]
                    span = max(spec.maximum - spec.minimum, 1e-6)
                    nudged = base_numeric[idx] + rng.gauss(0.0, span * 0.022)
                    noisy_numeric.append(float(spec.clamp(nudged)))
                samples.append((noisy_features, noisy_numeric, dict(base_categorical)))

        total = len(self._archetypes)
        for left_idx in range(total):
            left = self._archetypes[left_idx]
            for right_idx in range(left_idx + 1, total):
                right = self._archetypes[right_idx]
                if abs(float(left["features"][0]) - float(right["features"][0])) > 54.0:
                    continue
                for blend in (0.25, 0.5, 0.75):
                    blended_features: List[float] = []
                    blended_numeric: List[float] = []
                    for idx, (left_value, right_value) in enumerate(zip(left["features"], right["features"])):
                        mixed = float(left_value) * (1.0 - blend) + float(right_value) * blend
                        mixed += rng.gauss(0.0, feature_ranges[idx] * 0.012)
                        blended_features.append(mixed)
                    for idx, key in enumerate(self.NUMERIC_TARGET_KEYS):
                        spec = CONTROL_SPECS[key]
                        mixed = float(left["numeric"][idx]) * (1.0 - blend) + float(right["numeric"][idx]) * blend
                        blended_numeric.append(float(spec.clamp(mixed)))
                    blended_categorical: Dict[str, Any] = {}
                    for key in self.CATEGORICAL_TARGET_KEYS:
                        left_value = left["categorical"][key]
                        right_value = right["categorical"][key]
                        if left_value == right_value:
                            blended_categorical[key] = left_value
                        elif key == "middle_speed_multiplier":
                            blended_categorical[key] = left_value if blend < 0.5 else right_value
                        else:
                            blended_categorical[key] = left_value if blend <= 0.5 else right_value
                    samples.append((blended_features, blended_numeric, blended_categorical))
        return samples

    def _train(self) -> None:
        self._feature_matrix = [row["features"] for row in self._archetypes]
        if not dependency_available("sklearn"):
            return
        try:
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            from sklearn.preprocessing import StandardScaler
        except Exception:
            return

        samples = self._build_augmented_samples()
        if not samples:
            return
        train_x = [row[0] for row in samples]
        train_numeric_y = [row[1] for row in samples]
        self._numeric_scaler = StandardScaler()
        scaled_x = self._numeric_scaler.fit_transform(train_x)
        numeric_target_scaler = StandardScaler()
        scaled_numeric_y = numeric_target_scaler.fit_transform(train_numeric_y)

        numeric_neural_ok = False
        with warnings.catch_warnings(record=True) as numeric_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            numeric_model = MLPRegressor(
                hidden_layer_sizes=(16, 8),
                activation="tanh",
                solver="lbfgs",
                alpha=2e-3,
                max_iter=12000,
                random_state=7,
            )
            numeric_model.fit(scaled_x, scaled_numeric_y)
        numeric_has_warning = any(issubclass(item.category, ConvergenceWarning) for item in numeric_warnings)
        try:
            numeric_predictions = numeric_target_scaler.inverse_transform(numeric_model.predict(scaled_x))
        except Exception:
            numeric_predictions = []
        numeric_error = self._normalized_numeric_mae(numeric_predictions, train_numeric_y)
        if (not numeric_has_warning) and math.isfinite(numeric_error) and numeric_error <= 0.10:
            self._numeric_model = numeric_model
            self._numeric_target_scaler = numeric_target_scaler
            numeric_neural_ok = True
        else:
            self._numeric_model = KNeighborsRegressor(n_neighbors=5, weights="distance")
            self._numeric_model.fit(scaled_x, train_numeric_y)
            self._numeric_target_scaler = None

        categorical_neural_count = 0
        for key in self.CATEGORICAL_TARGET_KEYS:
            scaler = StandardScaler()
            sx = scaler.fit_transform(train_x)
            y = [row[2][key] for row in samples]
            with warnings.catch_warnings(record=True) as class_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                model = MLPClassifier(
                    hidden_layer_sizes=(12,),
                    activation="tanh",
                    solver="lbfgs",
                    alpha=2e-3,
                    max_iter=12000,
                    random_state=11 + len(self._categorical_models),
                )
                model.fit(sx, y)
            class_has_warning = any(issubclass(item.category, ConvergenceWarning) for item in class_warnings)
            try:
                predicted_labels = list(model.predict(sx))
            except Exception:
                predicted_labels = []
            accuracy = self._categorical_accuracy(predicted_labels, y)
            if class_has_warning or accuracy < 0.84:
                model = KNeighborsClassifier(n_neighbors=5, weights="distance")
                model.fit(sx, y)
            else:
                categorical_neural_count += 1
            self._categorical_scalers[key] = scaler
            self._categorical_models[key] = model

        self._using_sklearn = True
        if numeric_neural_ok and categorical_neural_count == len(self.CATEGORICAL_TARGET_KEYS):
            self._model_label = "rede neural scikit-learn"
        elif numeric_neural_ok or categorical_neural_count:
            self._model_label = "hibrido scikit-learn"
        else:
            self._model_label = "fallback scikit-learn"

    def _distance_weights(self, feature_values: List[float]) -> List[Tuple[float, Dict[str, Any]]]:
        if not self._archetypes:
            return []
        ranges = self._feature_ranges()
        pairs = []
        for row, archetype in zip(self._feature_matrix, self._archetypes):
            squared = 0.0
            for value, target, r in zip(feature_values, row, ranges):
                squared += ((value - target) / r) ** 2
            distance = math.sqrt(squared)
            pairs.append((distance, archetype))
        pairs.sort(key=lambda item: item[0])
        return pairs[:3]

    def _fallback_prediction(self, feature_values: List[float]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        neighbors = self._distance_weights(feature_values)
        total_weight = 0.0
        accumulator = [0.0] * len(self.NUMERIC_TARGET_KEYS)
        categorical_votes: Dict[str, Dict[Any, float]] = {key: {} for key in self.CATEGORICAL_TARGET_KEYS}
        for distance, archetype in neighbors:
            weight = 1.0 / max(distance, 1e-6)
            total_weight += weight
            for idx, value in enumerate(archetype["numeric"]):
                accumulator[idx] += weight * float(value)
            for key, value in archetype["categorical"].items():
                categorical_votes[key][value] = categorical_votes[key].get(value, 0.0) + weight
        numeric_values = {}
        for idx, key in enumerate(self.NUMERIC_TARGET_KEYS):
            value = accumulator[idx] / max(total_weight, 1e-9)
            numeric_values[key] = CONTROL_SPECS[key].clamp(value)
        categorical_values = {}
        for key, votes in categorical_votes.items():
            categorical_values[key] = max(votes.items(), key=lambda item: item[1])[0]
        return numeric_values, categorical_values

    def _normalized_numeric_mae(self, predicted_rows: Sequence[Sequence[float]], expected_rows: Sequence[Sequence[float]]) -> float:
        if len(predicted_rows) == 0 or len(expected_rows) == 0:
            return float("inf")
        total_error = 0.0
        total_count = 0
        for predicted, expected in zip(predicted_rows, expected_rows):
            for key, predicted_value, expected_value in zip(self.NUMERIC_TARGET_KEYS, predicted, expected):
                spec = CONTROL_SPECS[key]
                span = max(spec.maximum - spec.minimum, 1e-6)
                total_error += abs(float(predicted_value) - float(expected_value)) / span
                total_count += 1
        return total_error / max(total_count, 1)

    def _categorical_accuracy(self, predicted: Sequence[Any], expected: Sequence[Any]) -> float:
        if len(predicted) == 0 or len(expected) == 0:
            return 0.0
        correct = sum(1 for predicted_value, expected_value in zip(predicted, expected) if predicted_value == expected_value)
        return correct / max(len(expected), 1)

    def recommend(self, feature_snapshot: Dict[str, float]) -> RecommendationResult:
        feature_values = [float(feature_snapshot.get(key, 0.0)) for key in self.FEATURE_KEYS]
        fallback_numeric, fallback_categorical = self._fallback_prediction(feature_values)
        if self._using_sklearn and self._numeric_scaler and self._numeric_model:
            scaled_x = self._numeric_scaler.transform([feature_values])
            raw_numeric = self._numeric_model.predict(scaled_x)
            if self._numeric_target_scaler is not None:
                predicted_numeric = self._numeric_target_scaler.inverse_transform(raw_numeric)[0]
            else:
                predicted_numeric = raw_numeric[0]
            numeric_values = {
                key: CONTROL_SPECS[key].clamp(float(value) * 0.72 + float(fallback_numeric[key]) * 0.28)
                for key, value in zip(self.NUMERIC_TARGET_KEYS, predicted_numeric)
            }
            categorical_values: Dict[str, Any] = {}
            for key in self.CATEGORICAL_TARGET_KEYS:
                scaler = self._categorical_scalers[key]
                model = self._categorical_models[key]
                scaled_features = scaler.transform([feature_values])
                predicted = model.predict(scaled_features)[0]
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(scaled_features)[0]
                    confidence = max(float(value) for value in probabilities)
                    categorical_values[key] = predicted if confidence >= 0.56 else fallback_categorical[key]
                else:
                    categorical_values[key] = predicted
        else:
            numeric_values = fallback_numeric
            categorical_values = fallback_categorical

        summary_lines = [
            f"BPM {feature_snapshot.get('tempo_bpm', 0.0):.1f} | densidade {feature_snapshot.get('beat_density', 0.0):.2f}",
            f"Graves {feature_snapshot.get('bass_ratio', 0.0):.2f} | flatness {feature_snapshot.get('flatness_mean', 0.0):.2f}",
            f"Transicoes por minuto {feature_snapshot.get('boundaries_per_minute', 0.0):.2f}",
        ]
        return RecommendationResult(
            numeric_values=numeric_values,
            categorical_values=categorical_values,
            feature_snapshot=feature_snapshot,
            using_sklearn=self._using_sklearn,
            model_label=self._model_label,
            summary_lines=summary_lines,
        )


class GenerationService:
    def __init__(self, engine_loader: Optional[AudioReader6EngineLoader] = None) -> None:
        self.engine_loader = engine_loader or AudioReader6EngineLoader()
        self._track_cache: OrderedDict[str, CachedTrackData] = OrderedDict()
        self._track_cache_max = 5

    def _track_cache_key(self, track: LoadedTrack) -> str:
        parts = [track.source_mode, track.resolved_source]
        if track.source_mode == "local" and os.path.isfile(track.resolved_source):
            stat = os.stat(track.resolved_source)
            parts.extend([str(stat.st_size), str(stat.st_mtime_ns)])
        else:
            parts.append(track.requested_source)
        return "|".join(parts)

    def _cache_track_data(self, key: str, data: CachedTrackData) -> CachedTrackData:
        if key in self._track_cache:
            self._track_cache.move_to_end(key)
        self._track_cache[key] = data
        while len(self._track_cache) > self._track_cache_max:
            self._track_cache.popitem(last=False)
        return data

    def _get_or_load_track_data(
        self,
        track: LoadedTrack,
        librosa_module: Any,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> CachedTrackData:
        cache_key = self._track_cache_key(track)
        cached = self._track_cache.get(cache_key)
        if cached is not None:
            if progress_cb:
                progress_cb(10, "Usando áudio em cache…")
            self._track_cache.move_to_end(cache_key)
            return cached
        if progress_cb:
            progress_cb(10, "Carregando audio...")
        y, sr = self._load_audio(track.resolved_source, target_sr=22050, librosa_module=librosa_module)
        duration = max(librosa_module.get_duration(y=y, sr=sr), 1e-6)
        cached = CachedTrackData(y=y, sr=sr, duration_s=float(duration))
        return self._cache_track_data(cache_key, cached)

    def _peak_pick_onsets(self, librosa_module: Any, onset_env: Any) -> Any:
        peak_pick = librosa_module.util.peak_pick
        kwargs = {
            "pre_max": 3,
            "post_max": 3,
            "pre_avg": 3,
            "post_avg": 5,
            "delta": 0.5,
            "wait": 5,
        }
        try:
            return peak_pick(onset_env, **kwargs)
        except TypeError:
            return peak_pick(
                onset_env,
                kwargs["pre_max"],
                kwargs["post_max"],
                kwargs["pre_avg"],
                kwargs["post_avg"],
                kwargs["delta"],
                kwargs["wait"],
            )

    def _require_modules(self) -> Tuple[Any, Any]:
        if not dependency_available("librosa"):
            raise DependencyError("librosa nao encontrado. Instale com: pip install librosa")
        if not dependency_available("numpy"):
            raise DependencyError("numpy nao encontrado.")
        librosa = importlib.import_module("librosa")
        np = importlib.import_module("numpy")
        return librosa, np

    def extract_features(
        self,
        track: LoadedTrack,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, float]:
        librosa, np = self._require_modules()
        track_data = self._get_or_load_track_data(track, librosa, progress_cb)
        if track_data.feature_snapshot is not None:
            if progress_cb:
                progress_cb(100, "Features em cache prontas.")
            return dict(track_data.feature_snapshot)
        y, sr, duration = track_data.y, track_data.sr, track_data.duration_s
        if progress_cb:
            progress_cb(45, "Extraindo features principais...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo_bpm = float(np.asarray(tempo).item()) if np.size(tempo) else 120.0
        if not math.isfinite(tempo_bpm) or tempo_bpm <= 1.0:
            tempo_bpm = 120.0
        rms = librosa.feature.rms(y=y)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0] / max(sr / 2.0, 1.0)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = self._peak_pick_onsets(librosa, onset_env)
        spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        low_mask = (freqs >= 30.0) & (freqs <= 140.0)
        bass_energy = float(np.mean(spectrum[low_mask, :])) if np.any(low_mask) else 0.0
        total_energy = float(np.mean(spectrum)) or 1e-9
        flux = np.mean(np.maximum(np.diff(spectrum, axis=1), 0.0), axis=0) if spectrum.shape[1] > 1 else np.zeros(1)
        beat_density = float(len(beat_frames)) / duration
        boundaries_per_minute = float(len(peaks)) / (duration / 60.0)
        features = {
            "tempo_bpm": float(tempo_bpm),
            "beat_density": beat_density,
            "rms_mean": float(np.mean(rms)) if len(rms) else 0.0,
            "rms_p90": float(np.percentile(rms, 90)) if len(rms) else 0.0,
            "bass_ratio": float(bass_energy / total_energy),
            "centroid_mean": float(np.mean(centroid)) if len(centroid) else 0.0,
            "centroid_std": float(np.std(centroid)) if len(centroid) else 0.0,
            "flatness_mean": float(np.mean(flatness)) if len(flatness) else 0.0,
            "flux_mean": float(np.mean(flux)) if len(flux) else 0.0,
            "boundaries_per_minute": boundaries_per_minute,
        }
        track_data.feature_snapshot = dict(features)
        if progress_cb:
            progress_cb(100, "Features prontas.")
        return features

    def generate(
        self,
        track: LoadedTrack,
        config_state: UIConfigState,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> GeneratedSequence:
        if progress_cb:
            progress_cb(5, "Importando motor do audio_reader6...")
        module = self.engine_loader.load()
        librosa, np = self._require_modules()
        effect_config = config_state.to_effect_config(module)
        controller = module.MusicLEDController(
            port="/dev/null",
            fps=effect_config.FPS,
            config=effect_config,
            middle_bpm_multiplier=config_state.middle_speed_multiplier,
            show_youtube_video=False,
        )
        track_data = self._get_or_load_track_data(track, librosa, progress_cb)
        y, sr, duration = track_data.y, track_data.sr, track_data.duration_s
        n_frames = max(1, int(duration * controller.fps))
        if progress_cb:
            progress_cb(24, "Calculando BPM...")
        tempo, beat_librosa = librosa.beat.beat_track(y=y, sr=sr)
        try:
            tempo_bpm = float(np.asarray(tempo).item())
        except Exception:
            tempo_bpm = 120.0
        if not math.isfinite(tempo_bpm) or tempo_bpm <= 1.0:
            tempo_bpm = 120.0
        beat_times_s = librosa.frames_to_time(beat_librosa, sr=sr)
        beat_frames = (beat_times_s * controller.fps).astype(np.int32)
        if progress_cb:
            progress_cb(42, "Detectando estrutura e transicoes...")
        with contextlib.redirect_stdout(io.StringIO()):
            transitions = controller._analyze_structure(y, sr)
            transitions = controller._snap_to_beats(
                transitions, beat_frames, tempo_bpm, controller.fps, controller.config, n_frames
            )
        if progress_cb:
            progress_cb(72, "Gerando score dos LEDs...")
        with contextlib.redirect_stdout(io.StringIO()):
            led_patterns, dominant_leds = controller._generate_led_score(
                y, sr, n_frames, transitions, tempo_bpm, beat_frames
            )
        playback_source = track.resolved_source
        temp_files: List[str] = []
        if is_url(playback_source):
            playback_source = create_temp_wav_from_audio(y, sr)
            temp_files.append(playback_source)
        if progress_cb:
            progress_cb(86, "Montando preview e resumo...")
        features = dict(track_data.feature_snapshot) if track_data.feature_snapshot is not None else self.extract_features(track)
        preview_columns = build_preview_columns(led_patterns)
        if progress_cb:
            progress_cb(100, "Sequencia pronta.")
        return GeneratedSequence(
            track=track,
            config_state=config_state,
            effect_config=effect_config,
            led_patterns=led_patterns,
            dominant_leds=dominant_leds,
            duration_s=float(duration),
            transitions=list(transitions),
            playback_source=playback_source,
            tempo_bpm=float(tempo_bpm),
            beat_frames=[int(frame) for frame in list(beat_frames)],
            feature_snapshot=features,
            preview_columns=preview_columns,
            temp_files=temp_files,
        )

    def _load_audio(self, source: str, target_sr: int, librosa_module: Any) -> Tuple[Any, int]:
        if is_url(source):
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                raise DependencyError("ffmpeg nao encontrado para decodificar stream.")
            cmd = [
                ffmpeg_bin,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                source,
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
                error = result.stderr.decode(errors="ignore").strip()
                raise RuntimeError(f"Falha ao decodificar stream: {error}")
            np = importlib.import_module("numpy")
            pcm = np.frombuffer(result.stdout, dtype=np.int16)
            y = pcm.astype(np.float32) / 32768.0
            return y, target_sr
        y, sr = librosa_module.load(source, sr=target_sr, mono=True)
        return y, sr


class SerialHardwareController:
    def __init__(self) -> None:
        self._serial = None
        self._port: Optional[str] = None
        self.num_leds = 6

    def available_ports(self) -> List[str]:
        def port_key(port: str) -> Tuple[int, str]:
            if "/dev/ttyACM" in port:
                return (0, port)
            if "/dev/ttyUSB" in port:
                return (1, port)
            return (2, port)

        if dependency_available("serial.tools.list_ports"):
            try:
                list_ports = importlib.import_module("serial.tools.list_ports")
                ports = sorted({item.device for item in list_ports.comports()})
                preferred = [port for port in ports if "/dev/ttyACM" in port or "/dev/ttyUSB" in port]
                if preferred:
                    return sorted(preferred, key=port_key)
                if ports:
                    return sorted(ports, key=port_key)
            except Exception:
                pass
        candidates = sorted(set(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")), key=port_key)
        return candidates

    @property
    def is_connected(self) -> bool:
        return bool(self._serial and getattr(self._serial, "is_open", False))

    @property
    def port(self) -> Optional[str]:
        return self._port

    def connect_port(self, port: str, baudrate: int = 115200) -> None:
        if not dependency_available("serial"):
            raise DependencyError("pyserial nao encontrado. Instale com: pip install pyserial")
        serial = importlib.import_module("serial")
        self.disconnect()
        self._serial = serial.Serial(port, baudrate, timeout=1)
        self._port = port
        time.sleep(1.5)

    def disconnect(self) -> None:
        if self.is_connected:
            try:
                self.send_led_command([0] * self.num_leds)
            except Exception:
                pass
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self._port = None

    def send_led_command(self, led_values: Sequence[int], mode: int = 1) -> None:
        if not self.is_connected:
            return
        values = [0] * self.num_leds
        for idx, value in enumerate(list(led_values)[: self.num_leds]):
            values[idx] = int(max(0, min(255, int(value))))
        payload = ",".join(str(value) for value in values)
        self._serial.write(f"P,{int(mode)},{payload}\n".encode())


class PlaybackBackendSession:
    def __init__(self, source: str, duration_s: Optional[float]) -> None:
        self.source = source
        self.duration_s = duration_s
        self.backend = "none"
        self.proc: Optional[subprocess.Popen] = None
        self.supports_pause = False
        self.supports_seek = False
        self.player_name = "none"
        self.paused = False
        self._base_pos_s = 0.0
        self._started_at = 0.0
        self._paused_acc = 0.0
        self._pause_ts: Optional[float] = None
        self._pygame = None

    def _reset_timing(self, base_pos_s: float = 0.0) -> None:
        self._base_pos_s = max(0.0, float(base_pos_s))
        self._started_at = time.monotonic()
        self._paused_acc = 0.0
        self._pause_ts = self._started_at if self.paused else None

    def _set_paused_state(self, new_state: bool) -> None:
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
        return max(0.0, elapsed)

    def _start_ffplay_process(self) -> subprocess.Popen:
        ffplay_bin = shutil.which("ffplay")
        if not ffplay_bin:
            raise DependencyError("ffplay nao encontrado.")
        return subprocess.Popen(
            [ffplay_bin, "-nodisp", "-autoexit", "-loglevel", "quiet", self.source],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def start(self) -> bool:
        if dependency_available("pygame") and not is_url(self.source):
            try:
                self._pygame = importlib.import_module("pygame")
                if not self._pygame.mixer.get_init():
                    self._pygame.mixer.init()
                self._pygame.mixer.music.load(self.source)
                self._pygame.mixer.music.play()
                self.backend = "pygame"
                self.player_name = "pygame"
                self.supports_pause = True
                self.supports_seek = True
                self.paused = False
                self._reset_timing(0.0)
                return True
            except Exception:
                if self._pygame and self._pygame.mixer.get_init():
                    self._pygame.mixer.quit()
                self._pygame = None
        self.proc = self._start_ffplay_process()
        self.backend = "proc"
        self.player_name = "ffplay"
        self.supports_pause = os.name != "nt"
        self.supports_seek = False
        self.paused = False
        self._reset_timing(0.0)
        return True

    def is_alive(self) -> bool:
        if self.backend == "pygame" and self._pygame is not None:
            return self.paused or self._pygame.mixer.music.get_busy()
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
        if self.backend == "pygame" and self._pygame is not None:
            try:
                if should_pause:
                    self._pygame.mixer.music.pause()
                else:
                    self._pygame.mixer.music.unpause()
            except Exception:
                return False
            self._set_paused_state(should_pause)
            return True
        if self.proc is None or self.proc.poll() is not None or os.name == "nt":
            return False
        self.proc.send_signal(signal.SIGSTOP if should_pause else signal.SIGCONT)
        self._set_paused_state(should_pause)
        return True

    def seek(self, delta_s: int) -> bool:
        if self.backend != "pygame" or self._pygame is None:
            return False
        target = self.position_s() + float(delta_s)
        if self.duration_s is not None:
            target = min(float(self.duration_s), target)
        target = max(0.0, target)
        was_paused = self.paused
        try:
            self._pygame.mixer.music.play(start=target)
        except Exception:
            return False
        self.paused = False
        self._reset_timing(target)
        if was_paused:
            self._pygame.mixer.music.pause()
            self._set_paused_state(True)
        return True

    def stop(self) -> None:
        if self.backend == "pygame" and self._pygame is not None:
            try:
                self._pygame.mixer.music.stop()
            except Exception:
                pass
            return
        if self.proc and self.proc.poll() is None:
            if os.name != "nt" and self.paused:
                with contextlib.suppress(Exception):
                    self.proc.send_signal(signal.SIGCONT)
            self.proc.terminate()
            with contextlib.suppress(Exception):
                self.proc.wait(timeout=1.0)

    def shutdown(self) -> None:
        self.stop()
        if self.backend == "pygame" and self._pygame is not None and self._pygame.mixer.get_init():
            self._pygame.mixer.quit()


class PlaybackController(QWidget):
    state_changed = pyqtSignal(str)
    position_changed = pyqtSignal(float, float)
    frame_changed = pyqtSignal(object)
    playback_finished = pyqtSignal()
    playback_error = pyqtSignal(str)

    def __init__(self, hardware_controller: Optional[SerialHardwareController] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.hardware_controller = hardware_controller or SerialHardwareController()
        self.sequence: Optional[GeneratedSequence] = None
        self.session: Optional[PlaybackBackendSession] = None
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self._frame_idx_prev = -1
        self._state = "idle"

    @property
    def state(self) -> str:
        return self._state

    def set_hardware_controller(self, hardware_controller: SerialHardwareController) -> None:
        self.hardware_controller = hardware_controller

    def current_position_s(self) -> float:
        if not self.session:
            return 0.0
        return self.session.position_s()

    def play(self, sequence: GeneratedSequence) -> None:
        if self.session and self.sequence is sequence and self._state == "paused":
            self.pause_toggle()
            return
        self.stop()
        self.sequence = sequence
        self.session = PlaybackBackendSession(sequence.playback_source, sequence.duration_s)
        try:
            if not self.session.start():
                raise RuntimeError("Nao foi possivel iniciar a reproducao.")
        except Exception as exc:
            self.session = None
            self._state = "error"
            message, _details = describe_exception(exc)
            self.playback_error.emit(message)
            return
        self._frame_idx_prev = -1
        self._state = "playing"
        self.state_changed.emit(self._state)
        self.timer.start()

    def pause_toggle(self) -> None:
        if not self.session:
            return
        target = not self.session.paused
        if self.session.pause(target):
            self._state = "paused" if target else "playing"
            self.state_changed.emit(self._state)

    def stop(self) -> None:
        self.timer.stop()
        if self.session:
            self.session.shutdown()
        self.session = None
        self._frame_idx_prev = -1
        if self.sequence:
            with contextlib.suppress(Exception):
                self.hardware_controller.send_led_command([0] * 6)
        self._state = "idle"
        self.state_changed.emit(self._state)
        self.position_changed.emit(0.0, self.sequence.duration_s if self.sequence else 0.0)
        self.frame_changed.emit([0] * 6)

    def seek_relative(self, delta_s: int) -> None:
        if self.session and self.session.seek(delta_s):
            self._frame_idx_prev = -1

    def seek_ratio(self, ratio: float) -> None:
        if not self.session or not self.session.supports_seek or not self.sequence:
            return
        target = max(0.0, min(1.0, ratio)) * self.sequence.duration_s
        delta = int(round(target - self.session.position_s()))
        # Ensure at least 1 second delta so seek is never silently skipped
        # when the user clicks very close to the current position.
        if delta == 0:
            delta = 1 if target > self.session.position_s() else -1
        self.seek_relative(delta)

    def _finish(self) -> None:
        self.timer.stop()
        if self.session:
            self.session.shutdown()
            self.session = None
        with contextlib.suppress(Exception):
            self.hardware_controller.send_led_command([0] * 6)
        self._state = "idle"
        self.state_changed.emit(self._state)
        self.playback_finished.emit()

    def _tick(self) -> None:
        if not self.session or not self.sequence:
            self._finish()
            return
        if not self.session.is_alive():
            self._finish()
            return
        elapsed = self.session.position_s()
        duration = max(self.sequence.duration_s, 1e-6)
        frame_idx = min(max(int(elapsed * self.sequence.effect_config.FPS), 0), len(self.sequence.led_patterns) - 1)
        if not self.session.paused and frame_idx != self._frame_idx_prev:
            mode, led_values = self.sequence.led_patterns[frame_idx]
            values = [int(value) for value in list(led_values)]
            with contextlib.suppress(Exception):
                self.hardware_controller.send_led_command(values, mode=mode)
            self.frame_changed.emit(values)
            self._frame_idx_prev = frame_idx
        self.position_changed.emit(elapsed, duration)


class ExternalVideoWindowController(QWidget):
    state_changed = pyqtSignal(bool)
    error = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._capture_process_output)
        self._process.finished.connect(self._on_process_finished)
        self._process.errorOccurred.connect(self._on_process_error)
        self._process_output = ""
        self._backend_name = ""

    @property
    def is_open(self) -> bool:
        return self._process.state() != QProcess.NotRunning

    @property
    def backend_name(self) -> str:
        return self._backend_name or "desconhecido"

    def open(self, source: str, title: str, start_s: float = 0.0) -> None:
        self.close()
        self._process_output = ""
        program, args = self._build_player_command(source, title, start_s)
        env = QProcessEnvironment.systemEnvironment()
        if env.contains("DISPLAY"):
            self._process.setProcessEnvironment(env)
        self._process.start(program, args)
        if not self._process.waitForStarted(2500):
            raise RuntimeError("Nao foi possivel abrir a janela externa de video.")
        self._process.waitForFinished(350)
        self._capture_process_output()
        if self._process.state() == QProcess.NotRunning:
            details = self._process_output.strip()
            raise RuntimeError(details or "O reprodutor de video fechou logo apos iniciar.")
        self.state_changed.emit(True)

    def _build_player_command(self, source: str, title: str, start_s: float) -> Tuple[str, List[str]]:
        mpv_bin = shutil.which("mpv")
        if mpv_bin:
            self._backend_name = "mpv"
            args = [
                "--force-window=yes",
                "--idle=no",
                "--no-terminal",
                "--no-audio",
                "--title=Video - " + title,
                "--autofit=960x540",
            ]
            if start_s > 0.05:
                args.append(f"--start={max(0.0, start_s):.3f}")
            args.append(source)
            return mpv_bin, args

        ffplay_bin = shutil.which("ffplay")
        if not ffplay_bin:
            raise DependencyError("Nem mpv nem ffplay foram encontrados para abrir a janela de video.")
        self._backend_name = "ffplay"
        args = ["-autoexit", "-loglevel", "error", "-window_title", f"Video - {title}", "-an", "-x", "960", "-y", "540"]
        if start_s > 0.05:
            args.extend(["-ss", f"{max(0.0, start_s):.3f}"])
        args.append(source)
        return ffplay_bin, args

    def close(self) -> None:
        if not self.is_open:
            self.state_changed.emit(False)
            return
        self._process.blockSignals(True)
        self._process.terminate()
        if not self._process.waitForFinished(1200):
            self._process.kill()
            self._process.waitForFinished(1200)
        self._process.blockSignals(False)
        self.state_changed.emit(False)

    def _capture_process_output(self) -> None:
        chunk = bytes(self._process.readAllStandardOutput()).decode(errors="ignore")
        if chunk:
            self._process_output += chunk

    def _on_process_finished(self, _exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._capture_process_output()
        if self._process_output.strip():
            self.error.emit(self._process_output.strip())
        self.state_changed.emit(False)

    def _on_process_error(self, _error: QProcess.ProcessError) -> None:
        self._capture_process_output()
        message = self._process_output.strip() or self._process.errorString().strip() or "Falha ao abrir a janela de video."
        self.error.emit(message)
        self.state_changed.emit(False)


class LoadWorker(QThread):
    progress = pyqtSignal(int, str)
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, resolver: AudioSourceResolver, source_mode: str, source_value: str) -> None:
        super().__init__()
        self.resolver = resolver
        self.source_mode = source_mode
        self.source_value = source_value

    def run(self) -> None:
        try:
            self.progress.emit(20, "Resolvendo fonte...")
            track = self.resolver.load(self.source_mode, self.source_value)
            self.progress.emit(100, "Faixa carregada.")
            self.loaded.emit(track)
        except Exception as exc:
            self.failed.emit(*describe_exception(exc))


class RecommendationWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_result = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, generation_service: GenerationService, recommendation_service: ParameterRecommendationService, track: LoadedTrack) -> None:
        super().__init__()
        self.generation_service = generation_service
        self.recommendation_service = recommendation_service
        self.track = track

    def run(self) -> None:
        try:
            features = self.generation_service.extract_features(self.track, self.progress.emit)
            result = self.recommendation_service.recommend(features)
            self.finished_result.emit(result)
        except Exception as exc:
            self.failed.emit(*describe_exception(exc))


class GenerationWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_result = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, generation_service: GenerationService, track: LoadedTrack, config_state: UIConfigState) -> None:
        super().__init__()
        self.generation_service = generation_service
        self.track = track
        self.config_state = config_state

    def run(self) -> None:
        try:
            result = self.generation_service.generate(self.track, self.config_state, self.progress.emit)
            self.finished_result.emit(result)
        except Exception as exc:
            self.failed.emit(*describe_exception(exc))


class LedPreviewWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(92)
        self._levels = [0] * 6
        self._anim_phase = 0.0
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(40)  # 25 fps idle animation
        self._anim_timer.timeout.connect(self._anim_tick)
        self._anim_timer.start()

    def _anim_tick(self) -> None:
        self._anim_phase = (self._anim_phase + 0.07) % (2 * math.pi)
        self.update()

    def set_levels(self, levels: Sequence[int]) -> None:
        values = [0] * 6
        for idx, value in enumerate(list(levels)[:6]):
            values[idx] = int(max(0, min(255, int(value))))
        self._levels = values
        self.update()

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(8, 8, -8, -8)

        bg = QLinearGradient(0, rect.top(), 0, rect.bottom())
        bg.setColorAt(0.0, QColor("#1a2b40"))
        bg.setColorAt(0.5, QColor("#0f1d2e"))
        bg.setColorAt(1.0, QColor("#07111c"))
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 20, 20)
        painter.fillPath(path, bg)

        hl = QLinearGradient(rect.left(), 0, rect.right(), 0)
        hl.setColorAt(0.0, QColor(80, 140, 210, 0))
        hl.setColorAt(0.35, QColor(100, 170, 240, 70))
        hl.setColorAt(0.65, QColor(100, 170, 240, 70))
        hl.setColorAt(1.0, QColor(80, 140, 210, 0))
        hl_pen = QPen()
        hl_pen.setBrush(hl)
        hl_pen.setWidthF(1.5)
        painter.setPen(hl_pen)
        painter.drawLine(int(rect.left() + 22), int(rect.top()), int(rect.right() - 22), int(rect.top()))

        painter.setPen(QPen(QColor("#253d58"), 1.2))
        painter.drawPath(path)

        any_active = any(v > 8 for v in self._levels)
        w = rect.width() / 6.0
        for idx, level in enumerate(self._levels):
            cx = rect.left() + w * (idx + 0.5)
            cy = rect.center().y() - 3
            center = QPointF(cx, cy)
            radius = min(w * 0.31, rect.height() * 0.28)
            is_edge = idx in (0, 5)
            base_color = QColor("#f79435") if is_edge else QColor("#28c8f8")

            if any_active:
                raw_level = level / 255.0
                intensity = min(1.0, (raw_level ** 0.70) * 1.42)
                glow_strength = min(1.0, raw_level * 1.72 + 0.16) if level > 0 else 0.0
            else:
                offset = idx * (math.pi / 3.0)
                intensity = (math.sin(self._anim_phase + offset) + 1.0) * 0.5 * 0.18
                glow_strength = intensity

            painter.setPen(Qt.NoPen)
            for ring in (3, 2, 1):
                scale = 1.0 + ring * 0.26
                alpha = int(max(0, min(190, (0.08 + 0.20 * glow_strength) * (4 - ring) * 86)))
                gc = QColor(base_color)
                gc.setAlpha(alpha)
                painter.setBrush(gc)
                painter.drawEllipse(center, radius * scale, radius * scale)

            flat_fill = QColor(base_color).lighter(int(135 + intensity * 80))
            painter.setBrush(flat_fill)
            painter.drawEllipse(center, radius, radius)

            if intensity > 0.02:
                inner_fill = QColor(base_color).lighter(int(180 + intensity * 70))
                inner_fill.setAlpha(int(205 + intensity * 50))
                painter.setBrush(inner_fill)
                painter.drawEllipse(center, radius * 0.62, radius * 0.62)

            rim = QColor(base_color).lighter(165)
            rim.setAlpha(int(130 + intensity * 95))
            painter.setPen(QPen(rim, 1.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, radius, radius)

            painter.setPen(QColor(90, 140, 190, 100))
            painter.setFont(QFont("DejaVu Sans", 6))
            painter.drawText(QRectF(cx - 10, rect.bottom() - 14, 20, 12), Qt.AlignCenter, str(idx + 1))


class TimelineWidget(QWidget):
    transition_moved = pyqtSignal(int, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setMouseTracking(True)
        self.setToolTip("Arraste os marcadores âmbar para reposicionar transições.")
        self.preview_columns: List[List[int]] = []
        self.transitions: List[Any] = []
        self.duration_s = 0.0
        self.position_s = 0.0
        self._hover_transition_index: Optional[int] = None
        self._drag_transition_index: Optional[int] = None
        self._ph_phase = 0.0
        self._ph_timer = QTimer(self)
        self._ph_timer.setInterval(40)
        self._ph_timer.timeout.connect(self._ph_tick)
        self._ph_timer.start()

    def _ph_tick(self) -> None:
        self._ph_phase = (self._ph_phase + 0.10) % (2 * math.pi)
        if self.duration_s > 0:
            self.update()

    def set_sequence(self, sequence: Optional[GeneratedSequence]) -> None:
        if sequence is None:
            self.preview_columns = []
            self.transitions = []
            self.duration_s = 0.0
            self.position_s = 0.0
        else:
            self.preview_columns = sequence.preview_columns
            self.transitions = list(sequence.transitions)
            self.duration_s = float(sequence.duration_s)
        self.update()

    def set_position(self, position_s: float) -> None:
        self.position_s = max(0.0, float(position_s))
        self.update()

    def _timeline_rect(self) -> QRectF:
        rect = QRectF(self.rect()).adjusted(8, 8, -8, -14)
        return rect.adjusted(18, 34, -18, -32)

    def _transition_x(self, index: int) -> Optional[float]:
        if self.duration_s <= 0 or index < 0 or index >= len(self.transitions):
            return None
        inner = self._timeline_rect()
        transition = self.transitions[index]
        start_s = float(getattr(transition, "frame_start", 0)) / _CFG.FPS
        ratio = max(0.0, min(1.0, start_s / self.duration_s))
        return inner.left() + inner.width() * ratio

    def _transition_at_pos(self, pos: QPointF) -> Optional[int]:
        if not self.transitions or self.duration_s <= 0:
            return None
        inner = self._timeline_rect()
        if pos.y() < inner.top() - 18 or pos.y() > inner.bottom() + 18:
            return None
        best_idx: Optional[int] = None
        best_dist = 9999.0
        for idx in range(len(self.transitions)):
            tx = self._transition_x(idx)
            if tx is None:
                continue
            dist = abs(float(pos.x()) - tx)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx if best_dist <= 12.0 else None

    def _move_transition_to_x(self, index: int, x_value: float) -> None:
        if self.duration_s <= 0 or index < 0 or index >= len(self.transitions):
            return
        inner = self._timeline_rect()
        ratio = (float(x_value) - inner.left()) / max(1.0, inner.width())
        start_s = max(0.0, min(self.duration_s, ratio * self.duration_s))
        transition = self.transitions[index]
        fps = _CFG.FPS
        duration_frames = max(1, int(getattr(transition, "frame_end", 0)) - int(getattr(transition, "frame_start", 0)))
        max_start = max(0, int(round(self.duration_s * fps)) - duration_frames)
        new_start = max(0, min(max_start, int(round(start_s * fps))))
        transition.frame_start = new_start
        transition.frame_end = new_start + duration_frames
        self.transition_moved.emit(index, new_start / fps)
        self.update()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton:
            idx = self._transition_at_pos(event.pos())
            if idx is not None:
                self._drag_transition_index = idx
                self.setCursor(Qt.SizeHorCursor)
                self._move_transition_to_x(idx, event.pos().x())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drag_transition_index is not None:
            self._move_transition_to_x(self._drag_transition_index, event.pos().x())
            event.accept()
            return
        self._hover_transition_index = self._transition_at_pos(event.pos())
        self.setCursor(Qt.SizeHorCursor if self._hover_transition_index is not None else Qt.ArrowCursor)
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self._drag_transition_index is not None:
            self._move_transition_to_x(self._drag_transition_index, event.pos().x())
            self._drag_transition_index = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._hover_transition_index = None
        if self._drag_transition_index is None:
            self.setCursor(Qt.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(8, 8, -8, -14)

        bg = QLinearGradient(0, rect.top(), 0, rect.bottom())
        bg.setColorAt(0.0, QColor("#141f2e"))
        bg.setColorAt(0.5, QColor("#0e1824"))
        bg.setColorAt(1.0, QColor("#08111a"))
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 18, 18)
        painter.fillPath(path, bg)

        hl = QLinearGradient(rect.left(), 0, rect.right(), 0)
        hl.setColorAt(0.0, QColor(60, 120, 180, 0))
        hl.setColorAt(0.4, QColor(80, 150, 210, 55))
        hl.setColorAt(0.6, QColor(80, 150, 210, 55))
        hl.setColorAt(1.0, QColor(60, 120, 180, 0))
        hl_pen = QPen()
        hl_pen.setBrush(hl)
        hl_pen.setWidthF(1.4)
        painter.setPen(hl_pen)
        painter.drawLine(int(rect.left() + 20), int(rect.top()), int(rect.right() - 20), int(rect.top()))

        painter.setPen(QPen(QColor("#1e3248"), 1.2))
        painter.drawPath(path)

        if not self.preview_columns:
            painter.setPen(QColor("#6b7e96"))
            painter.setFont(QFont("DejaVu Sans", 10))
            painter.drawText(QRectF(rect.adjusted(0, 0, 0, -8)), Qt.AlignCenter, "O preview da sequência aparecerá aqui")
            return

        inner = QRectF(self._timeline_rect())

        if self.duration_s > 0:
            ruler_y = rect.top() + 5
            ruler_h = 13
            n_marks = min(8, max(2, int(self.duration_s // 30)))
            if n_marks < 2:
                n_marks = 4
            painter.setFont(QFont("DejaVu Sans", 6))
            painter.setPen(QColor("#3a5a7a"))
            for i in range(n_marks + 1):
                rx = int(inner.left() + inner.width() * i / n_marks)
                painter.drawLine(rx, int(ruler_y), rx, int(ruler_y + ruler_h))
                if 0 < i < n_marks:
                    ts = self.duration_s * i / n_marks
                    painter.drawText(QRectF(rx - 14, ruler_y, 28, ruler_h), Qt.AlignCenter, format_seconds(ts))

        num_cols = len(self.preview_columns)
        if num_cols > 0:
            cw = max(1.5, inner.width() / num_cols)
            for idx, levels in enumerate(self.preview_columns):
                avg = sum(levels) / max(1, len(levels))
                ratio = avg / 255.0
                bh = inner.height() * ratio
                bx = inner.left() + idx * cw
                hue = 0.58 - ratio * 0.12
                bar_c = QColor.fromHsvF(hue, 0.72, 0.45 + ratio * 0.55)
                bar_c.setAlphaF(0.16 + 0.66 * ratio)
                br = QRectF(bx, inner.bottom() - bh, max(1.2, cw - 0.8), bh)
                painter.fillRect(br, bar_c)
                if bh > 2:
                    cap_c = QColor.fromHsvF(hue, 0.5, 1.0)
                    cap_c.setAlphaF(0.50 * ratio)
                    painter.fillRect(QRectF(bx, inner.bottom() - bh, max(1.2, cw - 0.8), 1.5), cap_c)

        fps = _CFG.FPS
        for index, transition in enumerate(self.transitions):
            start_s = float(getattr(transition, "frame_start", 0)) / fps
            if self.duration_s <= 0:
                continue
            tr = max(0.0, min(1.0, start_s / self.duration_s))
            tx = inner.left() + inner.width() * tr
            active = index == self._hover_transition_index or index == self._drag_transition_index
            painter.setPen(QPen(QColor(245, 158, 11, 190 if active else 115), 1.4 if active else 1.0, Qt.DashLine))
            painter.drawLine(int(tx), int(inner.top()), int(tx), int(inner.bottom()))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#fbbf24" if active else "#f59e0b"))
            dm = QPainterPath()
            marker = 5 if active else 4
            dm.moveTo(tx, inner.top() - 3)
            dm.lineTo(tx + marker, inner.top() + 5)
            dm.lineTo(tx, inner.top() + 13)
            dm.lineTo(tx - marker, inner.top() + 5)
            dm.closeSubpath()
            painter.drawPath(dm)

        if self.duration_s > 0:
            pr = max(0.0, min(1.0, self.position_s / self.duration_s))
            px = inner.left() + inner.width() * pr
            g_alpha = int(30 + 22 * math.sin(self._ph_phase))
            for rw in range(10, 0, -2):
                gc = QColor(240, 248, 255, int(g_alpha * rw / 10))
                painter.setPen(QPen(gc, float(rw)))
                painter.drawLine(int(px), int(rect.top() + 10), int(px), int(rect.bottom() - 10))
            painter.setPen(QPen(QColor("#eef4fc"), 2.0))
            painter.drawLine(int(px), int(rect.top() + 10), int(px), int(rect.bottom() - 10))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#eef4fc"))
            tri = QPainterPath()
            tri.moveTo(px - 5, rect.top() + 8)
            tri.lineTo(px + 5, rect.top() + 8)
            tri.lineTo(px, rect.top() + 16)
            tri.closeSubpath()
            painter.drawPath(tri)


class VerticalControl(QFrame):
    value_changed = pyqtSignal(str, float)

    def __init__(self, spec: ControlSpec, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("VerticalControl")
        self.setMinimumWidth(92)
        self.setMaximumWidth(104)
        self.setMinimumHeight(236)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(6)
        self.title_label = QLabel(spec.label)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-size:10px;font-weight:700;color:#eef3fa;")
        self.value_label = QLabel(self._format_value(spec.default))
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("font-size:12px;font-weight:700;color:#ffd26d;")
        self.slider = QSlider(Qt.Vertical)
        lo, hi = spec.slider_range()
        self.slider.setRange(lo, hi)
        self.slider.setValue(spec.to_slider(spec.default))
        self.slider.valueChanged.connect(self._on_value_changed)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(max(1, int((hi - lo) / 8)))
        self.slider.setMinimumHeight(148)
        layout.addWidget(self.title_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.setToolTip(spec.tooltip or spec.label)
        self.set_recommended(False)
        apply_drop_shadow(self, blur_radius=28, y_offset=8, alpha=110)

    def _format_value(self, value: float) -> str:
        if self.spec.value_type is int or self.spec.decimals == 0:
            return str(int(round(value)))
        return f"{value:.{self.spec.decimals}f}"

    def _on_value_changed(self, raw: int) -> None:
        value = self.spec.from_slider(raw)
        self.value_label.setText(self._format_value(value))
        self.value_changed.emit(self.spec.key, value)

    def value(self) -> float:
        return self.spec.from_slider(self.slider.value())

    def set_value(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self.spec.to_slider(value))
        self.slider.blockSignals(False)
        self.value_label.setText(self._format_value(self.value()))

    def set_recommended(self, enabled: bool) -> None:
        if enabled:
            self.setStyleSheet(
                """
                QFrame#VerticalControl {
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 #1e3246, stop:0.38 #162535, stop:1 #0c1520);
                    border-top: 2px solid #40d4ff;
                    border-left: 1px solid #28a8d8;
                    border-right: 1px solid #1a7898;
                    border-bottom: 1px solid #104858;
                    border-radius: 20px;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame#VerticalControl {
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 #222e40, stop:0.36 #141e2e, stop:1 #090f1a);
                    border-top: 1px solid #384f68;
                    border-left: 1px solid #28394e;
                    border-right: 1px solid #1a2838;
                    border-bottom: 1px solid #10182a;
                    border-radius: 20px;
                }
                """
            )


class CollapsibleSection(QFrame):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("CollapsibleSection")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QHBoxLayout()
        self.button = QToolButton()
        self.button.setObjectName("CollapsibleHeaderButton")
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(False)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.RightArrow)
        self.button.clicked.connect(self._toggle)
        header.addWidget(self.button)
        header.addStretch(1)
        layout.addLayout(header)
        self.content = QWidget()
        self.content.setVisible(False)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)
        layout.addWidget(self.content)

    def _toggle(self) -> None:
        checked = self.button.isChecked()
        self.button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)


class VisionAudioWindow(QMainWindow):
    def __init__(
        self,
        source_resolver: Optional[AudioSourceResolver] = None,
        generation_service: Optional[GenerationService] = None,
        recommendation_service: Optional[ParameterRecommendationService] = None,
        playback_controller: Optional[PlaybackController] = None,
        hardware_controller: Optional[SerialHardwareController] = None,
    ) -> None:
        super().__init__()
        self.source_resolver = source_resolver or AudioSourceResolver()
        self.generation_service = generation_service or GenerationService()
        self.recommendation_service = recommendation_service or ParameterRecommendationService()
        self.hardware_controller = hardware_controller or SerialHardwareController()
        self.playback_controller = playback_controller or PlaybackController(self.hardware_controller)
        self.playback_controller.set_hardware_controller(self.hardware_controller)
        self.video_window_controller = ExternalVideoWindowController(self)

        self.current_track: Optional[LoadedTrack] = None
        self.current_recommendation: Optional[RecommendationResult] = None
        self.current_sequence: Optional[GeneratedSequence] = None
        self._load_worker: Optional[LoadWorker] = None
        self._recommend_worker: Optional[RecommendationWorker] = None
        self._generation_worker: Optional[GenerationWorker] = None
        self._scrubbing = False
        self._spin_controls: Dict[str, QWidget] = {}
        self._primary_controls: Dict[str, VerticalControl] = {}
        self._group_fields: Dict[str, List[str]] = {group: [] for group in ADVANCED_GROUPS}
        self._ui_busy = False
        self._last_background_error = False
        self.content_splitter: Optional[QSplitter] = None
        self.monitor_splitter: Optional[QSplitter] = None
        self.preview_panel: Optional[QFrame] = None
        self.monitor_column: Optional[QFrame] = None
        self.controls_column: Optional[QFrame] = None
        self.player_bar: Optional[QFrame] = None
        self.player_indent: Optional[QWidget] = None

        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 980)
        self._apply_theme()
        self._build_ui()
        self._build_menus()
        self._apply_depth_effects()
        self._bind_playback()
        self.refresh_ports()
        self._set_status_badge("ready")
        self._update_controls_enabled()
        self.log(f"{APP_TITLE} pronto.")
        # Startup fade-in animation
        self.setWindowOpacity(0.0)
        QTimer.singleShot(60, self._start_fade_in)

    def _apply_theme(self) -> None:
        self.setFont(QFont("DejaVu Sans", 9))
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#07080f"))
        palette.setColor(QPalette.Base, QColor("#0b0d18"))
        palette.setColor(QPalette.AlternateBase, QColor("#0f1220"))
        palette.setColor(QPalette.Text, QColor("#dce8f8"))
        palette.setColor(QPalette.WindowText, QColor("#dce8f8"))
        palette.setColor(QPalette.Button, QColor("#131624"))
        palette.setColor(QPalette.ButtonText, QColor("#dce8f8"))
        self.setPalette(palette)
        self.setStyleSheet(
            """
            /* ═══════════════════════════════════════════════
               VISIONAUDIO 7.2  —  DEEP SPACE DESIGN SYSTEM
               Accent palette:
                 Cyan-ice  : #00d4ff / #38b8d8
                 Violet    : #8b5cf6 / #a78bfa
                 Amber     : #f59e0b / #fbbf24   (primary CTA only)
                 Slate-bg  : #07080f → #0d1022
               ═══════════════════════════════════════════════ */

            QMainWindow {
                background: qradialgradient(cx:0.15, cy:0.06, radius:1.20,
                    stop:0   #151a2e,
                    stop:0.12 #0e1122,
                    stop:0.45 #090c18,
                    stop:1   #040509);
            }

            /* ── Labels ── */
            QLabel { color: #ccd8ee; }

            QLabel#SectionTitle {
                font-size: 11px;
                font-weight: 800;
                color: #00d4ff;
                letter-spacing: 1.2px;
                text-transform: uppercase;
            }
            QLabel#SectionMeta {
                color: #5a6f8a;
                font-size: 10px;
                font-style: italic;
            }
            QLabel#PlayerTitle {
                font-size: 12px;
                font-weight: 700;
                color: #e8f2ff;
                letter-spacing: 0.4px;
            }
            QLabel#PlayerState {
                font-size: 11px;
                color: #00d4ff;
                font-weight: 700;
                letter-spacing: 0.8px;
            }

            /* ── Video button ── */
            QToolButton#VideoWindowButton {
                background: rgba(0, 212, 255, 0.06);
                color: #5a8fa8;
                border: 1px solid rgba(0, 180, 220, 0.28);
                border-radius: 10px;
                padding: 4px 10px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            QToolButton#VideoWindowButton:hover {
                color: #a0dcf0;
                border-color: rgba(0, 212, 255, 0.55);
                background: rgba(0, 212, 255, 0.12);
            }
            QToolButton#VideoWindowButton:checked {
                color: #fbbf24;
                border-color: #f59e0b;
                background: rgba(245, 158, 11, 0.15);
            }
            QToolButton#VideoWindowButton:disabled {
                color: #2e3d4e;
                border-color: rgba(40, 55, 70, 0.40);
                background: transparent;
            }

            QLabel#PositionLabel {
                font-family: "DejaVu Sans Mono";
                font-size: 13px;
                font-weight: 700;
                color: #e8f4ff;
                padding: 4px 8px 6px 0px;
                min-height: 22px;
                max-height: 28px;
                background: transparent;
                letter-spacing: 0.5px;
            }

            /* ── Cards — deep glass with cyan rim ── */
            QFrame#Card, QFrame#PlayerBar {
                background: qlineargradient(x1:0, y1:0, x2:0.1, y2:1,
                    stop:0.000 #161b2e,
                    stop:0.015 #111624,
                    stop:0.500 #0b0f1c,
                    stop:0.985 #080c16,
                    stop:1.000 #050810);
                border-top: 1px solid rgba(0, 200, 240, 0.30);
                border-left: 1px solid rgba(0, 180, 220, 0.16);
                border-right: 1px solid rgba(0, 160, 200, 0.10);
                border-bottom: 1px solid rgba(0, 120, 160, 0.10);
                border-radius: 18px;
            }

            QFrame#InsetCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0c1020, stop:1 #070a14);
                border-top: 1px solid rgba(0, 180, 220, 0.18);
                border-left: 1px solid rgba(0, 160, 200, 0.10);
                border-right: 1px solid rgba(0, 120, 160, 0.07);
                border-bottom: 1px solid rgba(0, 100, 140, 0.07);
                border-radius: 12px;
            }

            /* ── Default buttons ── */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2038, stop:1 #10162a);
                border-top: 1px solid rgba(0, 200, 240, 0.35);
                border-left: 1px solid rgba(0, 180, 220, 0.20);
                border-right: 1px solid rgba(0, 140, 180, 0.12);
                border-bottom: 1px solid rgba(0, 100, 140, 0.12);
                border-radius: 10px;
                color: #9ab8d8;
                padding: 6px 14px;
                min-height: 18px;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.3px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e2a46, stop:1 #121c32);
                border-top-color: rgba(0, 212, 255, 0.60);
                color: #c8e0f4;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #090e1c, stop:1 #141c30);
                border-top-color: rgba(0, 160, 200, 0.25);
                color: #7099b8;
            }
            QPushButton:disabled {
                background: #090c18;
                color: #2a3a4e;
                border-color: rgba(30, 45, 65, 0.50);
            }

            /* Primary CTA — amber fire */
            QPushButton[variant="primary"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fbbf24, stop:0.45 #f59e0b, stop:1 #d97706);
                border-top: 2px solid #fcd34d;
                border-left: 1px solid #f59e0b;
                border-right: 1px solid #b45309;
                border-bottom: 1px solid #92400e;
                color: #0d0800;
                font-weight: 800;
                letter-spacing: 0.5px;
            }
            QPushButton[variant="primary"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fde68a, stop:0.45 #fbbf24, stop:1 #f59e0b);
                border-top-color: #fef3c7;
            }
            QPushButton[variant="primary"]:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #b45309, stop:1 #78350f);
                border-top-color: #d97706;
            }
            QPushButton[variant="primary"]:disabled {
                background: #2a1f0a;
                color: #5a4020;
                border-color: #3a2a10;
            }

            /* Secondary — ghost violet tint */
            QPushButton[variant="secondary"] {
                background: rgba(139, 92, 246, 0.06);
                color: #7c6db0;
                border-top-color: rgba(139, 92, 246, 0.30);
                border-left-color: rgba(139, 92, 246, 0.18);
                border-right-color: rgba(100, 70, 200, 0.10);
                border-bottom-color: rgba(80, 55, 170, 0.10);
            }
            QPushButton[variant="secondary"]:hover {
                background: rgba(139, 92, 246, 0.14);
                color: #a78bfa;
                border-top-color: rgba(167, 139, 250, 0.55);
            }

            /* ── Inputs ── */
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0c1020, stop:1 #070a14);
                border-top: 1px solid rgba(0, 180, 220, 0.25);
                border-left: 1px solid rgba(0, 160, 200, 0.14);
                border-right: 1px solid rgba(0, 120, 160, 0.09);
                border-bottom: 1px solid rgba(0, 100, 140, 0.09);
                border-radius: 10px;
                padding: 7px 14px 7px 14px;
                color: #c8dcf0;
                selection-background-color: #f59e0b;
                selection-color: #080600;
                min-height: 18px;
            }
            QComboBox QLineEdit {
                background: transparent;
                border: none;
                color: #c8dcf0;
                padding: 2px 10px 2px 10px;
                selection-background-color: #f59e0b;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-top-color: rgba(0, 212, 255, 0.60);
                border-left-color: rgba(0, 200, 240, 0.35);
            }
            QComboBox { color: #c8dcf0; }

            /* ComboBox chrome */
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 26px;
                margin: 1px 1px 1px 0;
                border: none;
                border-left: 1px solid rgba(0, 150, 190, 0.20);
                background: transparent;
                border-top-right-radius: 9px;
                border-bottom-right-radius: 9px;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0; height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #38b8d8;
                margin-right: 8px;
            }
            QComboBox::down-arrow:on { border-top-color: #fbbf24; }
            QComboBox QAbstractItemView {
                background: #0c1020;
                border: 1px solid rgba(0, 180, 220, 0.30);
                border-radius: 10px;
                selection-background-color: rgba(0, 180, 220, 0.18);
                color: #c8dcf0;
                padding: 4px 0px;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 7px 16px;
                min-height: 22px;
            }
            QComboBox QAbstractItemView::item:selected {
                color: #00d4ff;
            }

            /* ── List ── */
            QListWidget::item {
                border-bottom: 1px solid rgba(0, 140, 180, 0.12);
                padding: 8px 8px;
                color: #8aa8c8;
            }
            QListWidget::item:selected {
                background: rgba(0, 180, 220, 0.14);
                border-radius: 7px;
                color: #00d4ff;
                border-bottom-color: transparent;
            }
            QListWidget::item:hover {
                background: rgba(0, 160, 200, 0.08);
                color: #9cbad4;
            }

            /* ── Log console ── */
            QPlainTextEdit {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #080c18, stop:1 #050810);
                border: 1px solid rgba(0, 160, 200, 0.18);
                border-radius: 10px;
                padding: 8px 10px;
                color: #6888a8;
                selection-background-color: #f59e0b;
                selection-color: #080600;
                font-family: "DejaVu Sans Mono";
                font-size: 10px;
                line-height: 1.5;
            }

            /* ── Scrollbars ── */
            QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
            QScrollBar:vertical {
                width: 8px;
                background: rgba(0, 0, 0, 0.0);
                border: none;
                margin: 4px 2px 4px 0;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                min-height: 28px;
                border-radius: 4px;
                background: rgba(0, 180, 220, 0.22);
                border: none;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(0, 212, 255, 0.40);
            }
            QScrollBar::handle:vertical:pressed {
                background: rgba(0, 160, 200, 0.55);
            }
            QScrollBar:horizontal {
                height: 8px;
                background: rgba(0, 0, 0, 0.0);
                border: none;
                margin: 0 4px 2px 4px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                min-width: 28px;
                border-radius: 4px;
                background: rgba(0, 180, 220, 0.22);
                border: none;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(0, 212, 255, 0.40);
            }
            QScrollBar::handle:horizontal:pressed {
                background: rgba(0, 160, 200, 0.55);
            }
            QScrollBar::add-line, QScrollBar::sub-line,
            QScrollBar::add-page, QScrollBar::sub-page {
                background: transparent; border: none;
            }

            QScrollArea#ControlColumnScroll QScrollBar:vertical {
                width: 5px;
                margin: 3px 1px 3px 0;
            }
            QScrollArea#ControlColumnScroll QScrollBar::handle:vertical {
                min-height: 22px;
                border-radius: 3px;
                background: rgba(0, 160, 200, 0.20);
            }
            QScrollArea#ControlColumnScroll QScrollBar::handle:vertical:hover {
                background: rgba(0, 212, 255, 0.38);
            }

            /* ── Progress bar ── */
            QProgressBar {
                background: rgba(0, 0, 0, 0.35);
                border: 1px solid rgba(0, 160, 200, 0.22);
                border-radius: 5px;
                color: #6888a8;
                text-align: center;
                font-size: 9px;
                font-weight: 700;
                letter-spacing: 0.4px;
                min-height: 9px;
                max-height: 9px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0070a0, stop:0.55 #00b8e0, stop:1 #fbbf24);
                border-radius: 4px;
            }

            /* ── Menu bar ── */
            QMenuBar {
                background: rgba(8, 10, 20, 0.92);
                color: #5a7898;
                padding: 1px 6px;
                font-size: 11px;
                font-weight: 600;
                border-bottom: 1px solid rgba(0, 160, 200, 0.18);
                spacing: 2px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background: transparent;
                border-radius: 5px;
                letter-spacing: 0.3px;
            }
            QMenuBar::item:selected {
                background: rgba(0, 180, 220, 0.12);
                color: #00d4ff;
            }
            QMenu {
                background: #0b0f1e;
                border: 1px solid rgba(0, 180, 220, 0.28);
                border-radius: 10px;
                padding: 6px 4px;
            }
            QMenu::item {
                padding: 8px 28px 8px 16px;
                color: #8aa8c8;
                border-radius: 6px;
                margin: 1px 4px;
            }
            QMenu::item:selected {
                background: rgba(0, 180, 220, 0.15);
                color: #00d4ff;
            }
            QMenu::separator {
                height: 1px;
                background: rgba(0, 160, 200, 0.15);
                margin: 4px 12px;
            }

            /* ── MessageBox ── */
            QMessageBox {
                background-color: #0b0f1e;
            }
            QMessageBox QLabel {
                color: #9ab8d4;
                background-color: transparent;
                font-size: 12px;
            }
            QMessageBox QPushButton {
                min-width: 84px;
                min-height: 28px;
                padding: 6px 14px;
                border-radius: 8px;
                border-top: 1px solid rgba(0, 200, 240, 0.35);
                border-left: 1px solid rgba(0, 180, 220, 0.20);
                border-right: 1px solid rgba(0, 140, 180, 0.12);
                border-bottom: 1px solid rgba(0, 100, 140, 0.12);
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1a2038, stop:1 #10162a);
                color: #9ab8d8;
                font-weight: 600;
            }
            QMessageBox QPushButton:hover {
                border-top-color: rgba(0, 212, 255, 0.60);
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1e2a46, stop:1 #121c32);
                color: #c8e0f4;
            }

            /* ── Status badge ── */
            QLabel#StatusBadge {
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 1.0px;
                text-transform: uppercase;
                border-radius: 8px;
                padding: 6px 12px;
                color: #002010;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #34d399, stop:1 #059669);
                border: 1px solid #10b981;
            }
            QLabel#StatusBadge[state="busy"] {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #fbbf24, stop:1 #d97706);
                border: 1px solid #f59e0b;
                color: #1a0e00;
            }
            QLabel#StatusBadge[state="error"] {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #f87171, stop:1 #dc2626);
                border: 1px solid #ef4444;
                color: #1a0000;
            }

            /* ── Header round buttons ── */
            QToolButton#HeaderRoundBtn {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #182240, stop:1 #0e1628);
                border: 1px solid rgba(0, 180, 220, 0.38);
                border-radius: 16px;
                color: #6898b8;
                font-size: 15px;
                font-weight: 700;
                min-width: 34px; max-width: 34px;
                min-height: 34px; max-height: 34px;
                padding: 0px;
            }
            QToolButton#HeaderRoundBtn:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #1e2e52, stop:1 #121e36);
                border-color: rgba(0, 212, 255, 0.65);
                color: #00d4ff;
            }
            QToolButton#HeaderRoundBtn:pressed {
                background: #080c18;
                border-color: rgba(0, 160, 200, 0.35);
                color: #0090b8;
            }
            QToolButton#HeaderRoundBtn:disabled {
                color: #1e2c3c;
                border-color: rgba(30, 48, 66, 0.40);
                background: #060a14;
            }

            QToolButton#HeaderRoundBtnPrimary {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #fbbf24, stop:0.45 #f59e0b, stop:1 #d97706);
                border: 1px solid #fcd34d;
                color: #0d0800;
                border-radius: 16px;
                font-size: 15px;
                font-weight: 800;
                min-width: 34px; max-width: 34px;
                min-height: 34px; max-height: 34px;
                padding: 0px;
            }
            QToolButton#HeaderRoundBtnPrimary:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #fde68a, stop:0.45 #fbbf24, stop:1 #f59e0b);
                border-color: #fef3c7;
            }
            QToolButton#HeaderRoundBtnPrimary:pressed {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #b45309, stop:1 #78350f);
                border-color: #d97706;
            }
            QToolButton#HeaderRoundBtnPrimary:disabled {
                color: #4a3510;
                border-color: #2e1f08;
                background: #1a1005;
            }

            /* ── Player chrome ── */
            QFrame#PlayerChrome {
                background: rgba(0, 8, 20, 0.60);
                border: 1px solid rgba(0, 180, 220, 0.18);
                border-radius: 14px;
            }

            /* ── Transport buttons ── */
            QPushButton#TransportBtn {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #182240, stop:1 #0e1628);
                border: 1px solid rgba(0, 180, 220, 0.32);
                border-radius: 18px;
                color: #4a7898;
                font-size: 14px;
                font-weight: 700;
                min-width: 36px; max-width: 36px;
                min-height: 36px; max-height: 36px;
                padding: 0px;
            }
            QPushButton#TransportBtn:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #1e2e52, stop:1 #121e36);
                border-color: rgba(0, 212, 255, 0.60);
                color: #00d4ff;
            }
            QPushButton#TransportBtn:pressed {
                background: #060a16;
                border-color: rgba(0, 150, 190, 0.30);
                color: #0080a8;
            }
            QPushButton#TransportBtn:disabled {
                background: #060a14;
                border-color: rgba(20, 35, 55, 0.40);
                color: #1a2838;
            }
            QPushButton#TransportBtn[accent="true"] {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #fbbf24, stop:0.45 #f59e0b, stop:1 #d97706);
                border: 1px solid #fcd34d;
                color: #0d0800;
                min-width: 34px; max-width: 34px;
                min-height: 34px; max-height: 34px;
                border-radius: 17px;
            }
            QPushButton#TransportBtn[accent="true"]:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #fde68a, stop:0.45 #fbbf24, stop:1 #f59e0b);
                border-color: #fef3c7;
            }
            QPushButton#TransportBtn[accent="true"]:pressed {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #b45309, stop:1 #78350f);
            }
            QPushButton#TransportBtn[accent="true"]:disabled {
                background: #1a1005;
                border-color: #2e1f08;
                color: #3a2808;
            }

            /* ── Scrubber ── */
            QSlider#Scrubber::groove:horizontal {
                height: 6px;
                border-radius: 3px;
                background: rgba(0, 0, 0, 0.50);
                border: 1px solid rgba(0, 140, 180, 0.22);
            }
            QSlider#Scrubber::sub-page:horizontal {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #005878, stop:0.60 #00b8e0, stop:1 #fbbf24);
                border-radius: 3px;
            }
            QSlider#Scrubber::handle:horizontal {
                width: 12px; height: 12px;
                margin: -4px 0;
                border-radius: 6px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #69e4ff, stop:0.48 #00b8e0, stop:1 #007fb8);
                border: 1px solid #08111a;
            }
            QSlider#Scrubber::handle:horizontal:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #b8f4ff, stop:1 #00d4ff);
                border-color: #102034;
            }

            /* ── Profile slot ── */
            QFrame#ProfileSlot {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #101626, stop:1 #080c18);
                border: 1px solid rgba(0, 160, 200, 0.18);
                border-radius: 10px;
            }

            /* ── GroupBox (advanced panel) ── */
            QGroupBox {
                color: #38b8d8;
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 1.2px;
                text-transform: uppercase;
                border: 1px solid rgba(0, 160, 200, 0.18);
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 8px;
                left: 12px;
                background: transparent;
            }

            /* ── CollapsibleSection header button ── */
            QToolButton#CollapsibleHeaderButton {
                background: transparent;
                border: none;
                color: #38b8d8;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.8px;
                padding: 4px 6px;
            }
            QToolButton#CollapsibleHeaderButton:hover {
                color: #00d4ff;
            }
            QToolButton#CollapsibleHeaderButton:checked {
                color: #a78bfa;
            }

            /* ── SpinBox chrome ── */
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                background: rgba(0, 160, 200, 0.10);
                border-left: 1px solid rgba(0, 140, 180, 0.20);
                border-radius: 0 5px 0 0;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                background: rgba(0, 160, 200, 0.10);
                border-left: 1px solid rgba(0, 140, 180, 0.20);
                border-radius: 0 0 5px 0;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none;
                width: 0; height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #38b8d8;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none;
                width: 0; height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #38b8d8;
            }

            /* ── Splitter handle ── */
            QSplitter::handle {
                background: transparent;
            }
            QSplitter::handle:horizontal {
                width: 10px;
                background: transparent;
                border-left: 1px solid rgba(0, 140, 180, 0.08);
                border-right: 1px solid rgba(0, 140, 180, 0.08);
            }
            QSplitter::handle:vertical {
                height: 10px;
                background: transparent;
                border-top: 1px solid rgba(0, 140, 180, 0.08);
                border-bottom: 1px solid rgba(0, 140, 180, 0.08);
            }
            """
        )
    def _build_ui(self) -> None:

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(14)

        main_layout.addWidget(self._build_header())
        workspace = QHBoxLayout()
        workspace.setContentsMargins(0, 0, 0, 0)
        workspace.setSpacing(12)
        self.monitor_column = self._build_monitor_column()
        workspace.addWidget(self.monitor_column)

        right_area = QWidget()
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.content_splitter = QSplitter(Qt.Horizontal)
        self.content_splitter.setObjectName("MainContentSplitter")
        self.content_splitter.setHandleWidth(10)
        self.content_splitter.setChildrenCollapsible(False)
        self.controls_column = self._build_controls_column()
        self.preview_panel = self._build_preview_column()
        self.content_splitter.addWidget(self.controls_column)
        self.content_splitter.addWidget(self.preview_panel)
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 0)
        self.content_splitter.setCollapsible(0, False)
        self.content_splitter.setCollapsible(1, True)
        right_layout.addWidget(self.content_splitter, 1)
        self.player_bar = self._build_player_bar()
        right_layout.addWidget(self.player_bar)
        workspace.addWidget(right_area, 1)
        main_layout.addLayout(workspace, 1)
        QTimer.singleShot(0, self._configure_initial_splitters)

    def _make_header_round_button(
        self, standard_pixmap: QStyle.StandardPixmap, tooltip: str, *, primary: bool = False
    ) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName("HeaderRoundBtnPrimary" if primary else "HeaderRoundBtn")
        btn.setIcon(self.style().standardIcon(standard_pixmap))
        btn.setIconSize(QSize(18, 18))
        btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFocusPolicy(Qt.StrongFocus)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return btn

    def _make_transport_button(
        self, standard_pixmap: QStyle.StandardPixmap, tooltip: str, *, accent: bool = False
    ) -> QPushButton:
        btn = QPushButton()
        btn.setObjectName("TransportBtn")
        btn.setText("")
        btn.setIcon(self.style().standardIcon(standard_pixmap))
        btn.setIconSize(QSize(16 if not accent else 14, 16 if not accent else 14))
        btn.setToolTip(tooltip)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn.setFocusPolicy(Qt.StrongFocus)
        if accent:
            btn.setProperty("accent", True)
        return btn

    def _build_header(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.status_badge = QLabel("Pronto")
        self.status_badge.setObjectName("StatusBadge")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setFixedWidth(112)
        card.setMinimumHeight(118)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Arquivo local", "local")
        self.source_combo.addItem("YouTube", "youtube")
        self.source_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        self.source_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Ficheiro ou pesquisa / URL do YouTube")
        self.browse_button = self._make_header_round_button(
            QStyle.SP_DirOpenIcon, "Escolher ficheiro de áudio…"
        )
        self.browse_button.clicked.connect(self.browse_audio_file)
        self.load_button = self._make_header_round_button(QStyle.SP_ArrowDown, "Carregar áudio")
        self.load_button.clicked.connect(self.load_track)
        self.suggest_button = self._make_header_round_button(
            QStyle.SP_FileDialogInfoView, "Sugerir parâmetros"
        )
        self.suggest_button.clicked.connect(self.request_recommendation)
        self.generate_button = self._make_header_round_button(
            QStyle.SP_BrowserReload, "Gerar sequência", primary=True
        )
        self.generate_button.clicked.connect(self.generate_sequence)

        mini = QHBoxLayout()
        mini.setSpacing(6)
        mini.addWidget(self.browse_button)
        mini.addWidget(self.load_button)
        mini.addWidget(self.suggest_button)
        mini.addWidget(self.generate_button)

        row.addWidget(self.source_combo)
        row.addWidget(self.source_input, 1)
        row.addLayout(mini)
        row.addWidget(self.status_badge)
        layout.addLayout(row)
        # Sync initial placeholder/browse-button visibility after all widgets exist
        QTimer.singleShot(0, self._on_source_mode_changed)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self.port_combo.lineEdit():
            self.port_combo.lineEdit().setPlaceholderText("/dev/ttyACM0")
        self.refresh_ports_button = QPushButton("Atualizar portas")
        self.refresh_ports_button.setProperty("variant", "secondary")
        self.refresh_ports_button.clicked.connect(self.refresh_ports)
        self.connect_button = QPushButton("Conectar Arduino")
        self.connect_button.setProperty("variant", "primary")
        self.connect_button.clicked.connect(self.toggle_connection)
        self.connection_label = QLabel("")
        self.connection_label.setObjectName("SectionMeta")
        self.connection_label.setVisible(False)
        self.progress_label = QLabel("Aguardando ação")
        self.progress_label.setObjectName("SectionMeta")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        bottom.addWidget(self.port_combo)
        bottom.addWidget(self.refresh_ports_button)
        bottom.addWidget(self.connect_button)
        bottom.addWidget(self.progress_label, 1)
        bottom.addWidget(self.progress_bar, 1)
        layout.addLayout(bottom)
        return card

    def _build_monitor_column(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        card.setMinimumWidth(236)

        title = QLabel("Monitores")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.led_preview = LedPreviewWidget()
        self.led_preview.setMinimumHeight(78)
        self.led_preview.setMaximumHeight(98)
        layout.addWidget(self.led_preview)

        self.monitor_splitter = QSplitter(Qt.Vertical)
        self.monitor_splitter.setObjectName("MonitorSplitter")
        self.monitor_splitter.setHandleWidth(10)
        self.monitor_splitter.setChildrenCollapsible(False)
        transitions_card = self._build_transition_panel()
        logs_card = self._build_log_panel()
        self.monitor_splitter.addWidget(transitions_card)
        self.monitor_splitter.addWidget(logs_card)
        self.monitor_splitter.setStretchFactor(0, 0)
        self.monitor_splitter.setStretchFactor(1, 1)
        self.monitor_splitter.setCollapsible(0, False)
        self.monitor_splitter.setCollapsible(1, False)
        layout.addWidget(self.monitor_splitter, 1)
        return card

    def _build_preview_column(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        card.setMinimumWidth(0)

        self.track_title_label = QLabel("Nenhuma faixa carregada")
        self.track_title_label.setStyleSheet("font-size:15px;font-weight:700;color:#f3f7fd;")
        self.track_meta_label = QLabel("")
        self.track_meta_label.setObjectName("SectionMeta")
        layout.addWidget(self.track_title_label)
        layout.addWidget(self.track_meta_label)

        self.feature_summary_label = QLabel("Sem análise ainda.")
        self.feature_summary_label.setWordWrap(True)
        self.feature_summary_label.setStyleSheet(
            "color:#bfd1e4;background:#0d1621;border:1px solid #22364a;border-radius:12px;padding:10px;font-size:10px;"
        )
        layout.addWidget(self.feature_summary_label)

        timeline_header = QHBoxLayout()
        timeline_header.setSpacing(8)
        timeline_title = QLabel("Preview geral")
        timeline_title.setObjectName("SectionTitle")
        timeline_header.addWidget(timeline_title)
        timeline_header.addStretch(1)
        layout.addLayout(timeline_header)

        self.timeline_widget = TimelineWidget()
        self.timeline_widget.setMinimumHeight(240)
        self.timeline_widget.transition_moved.connect(self._on_timeline_transition_moved)
        layout.addWidget(self.timeline_widget, 1)
        return card

    def _build_transition_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        title = QLabel("Transições")
        title.setObjectName("SectionTitle")
        self.transition_list = QListWidget()
        self.transition_list.setMinimumHeight(126)
        layout.addWidget(title)
        layout.addWidget(self.transition_list, 1)
        return card

    def _build_log_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        title = QLabel("Logs e status")
        title.setObjectName("SectionTitle")
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        card.setMinimumHeight(180)
        card.setMaximumHeight(280)
        self.log_output.setMinimumHeight(120)
        self.log_output.setMaximumHeight(220)
        layout.addWidget(title)
        layout.addWidget(self.log_output, 1)
        return card

    def _build_controls_column(self) -> QFrame:
        wrapper = QFrame()
        wrapper.setObjectName("Card")
        wrapper.setMinimumWidth(540)
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setObjectName("ControlColumnScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        inner = QWidget()
        inner.setObjectName("ControlViewport")
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_profile_card())
        layout.addWidget(self._build_primary_controls_card())
        layout.addWidget(self._build_advanced_controls_card())
        layout.addStretch(1)
        return wrapper

    def _build_profile_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        card.setMaximumHeight(132)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        title = QLabel("Rack de comportamento")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.transition_profile_combo = QComboBox()
        self.effect_profile_combo = QComboBox()
        self.middle_speed_combo = QComboBox()
        for key, label in TRANSITION_PROFILE_LABELS.items():
            self.transition_profile_combo.addItem(label, key)
        for key, label in EFFECT_PROFILE_LABELS.items():
            self.effect_profile_combo.addItem(label, key)
        for key, label in MIDDLE_SPEED_OPTIONS.items():
            self.middle_speed_combo.addItem(label, key)

        row = QHBoxLayout()
        row.setSpacing(8)
        for label_text, combo in (
            ("Transições", self.transition_profile_combo),
            ("Agressividade", self.effect_profile_combo),
            ("Velocidade meio", self.middle_speed_combo),
        ):
            slot = QFrame()
            slot.setObjectName("ProfileSlot")
            slot_layout = QVBoxLayout(slot)
            slot_layout.setContentsMargins(7, 6, 7, 6)
            slot_layout.setSpacing(3)
            slot_label = QLabel(label_text)
            slot_label.setObjectName("SectionMeta")
            slot_layout.addWidget(slot_label)
            slot_layout.addWidget(combo)
            row.addWidget(slot, 1)
        layout.addLayout(row)
        return card

    def _build_primary_controls_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        title = QLabel("Controles principais")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        columns = 5
        for index, key in enumerate(PRIMARY_CONTROL_KEYS):
            control = VerticalControl(CONTROL_SPECS[key])
            control.value_changed.connect(self._on_control_value_changed)
            self._primary_controls[key] = control
            grid.addWidget(control, index // columns, index % columns)
        layout.addLayout(grid)
        return card

    def _build_advanced_controls_card(self) -> QFrame:
        section = CollapsibleSection("Painel avançado")
        for group in ADVANCED_GROUPS:
            group_box = QGroupBox(group)
            group_layout = QVBoxLayout(group_box)
            form = QFormLayout()
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(10)
            for key, spec in CONTROL_SPECS.items():
                if spec.primary or spec.group != group:
                    continue
                editor: QWidget
                if spec.value_type is int:
                    widget = QSpinBox()
                    widget.setRange(int(spec.minimum), int(spec.maximum))
                    widget.setValue(int(spec.default))
                    widget.valueChanged.connect(lambda _value, control_key=key: self._on_spin_control_changed(control_key))
                    editor = widget
                else:
                    widget = QDoubleSpinBox()
                    widget.setDecimals(spec.decimals)
                    widget.setSingleStep(1.0 / spec.scale if spec.scale > 1.0 else 0.05)
                    widget.setRange(spec.minimum, spec.maximum)
                    widget.setValue(float(spec.default))
                    widget.valueChanged.connect(lambda _value, control_key=key: self._on_spin_control_changed(control_key))
                    editor = widget
                self._spin_controls[key] = editor
                self._group_fields[group].append(key)
                form.addRow(spec.label, editor)
            group_layout.addLayout(form)
            restore_button = QPushButton(f"Restaurar padrões de {group}")
            restore_button.setProperty("variant", "secondary")
            restore_button.clicked.connect(lambda _checked=False, group_name=group: self.restore_group_defaults(group_name))
            group_layout.addWidget(restore_button)
            section.content_layout.addWidget(group_box)
        return section

    def _build_player_bar(self) -> QFrame:
        card = QFrame()
        card.setObjectName("PlayerBar")
        card.setMinimumHeight(112)
        card.setMaximumHeight(118)
        outer = QVBoxLayout(card)
        outer.setContentsMargins(12, 6, 12, 6)
        outer.setSpacing(3)

        top = QHBoxLayout()
        top.setSpacing(10)
        self.player_title_label = QLabel("Player pronto para sequência gerada")
        self.player_title_label.setObjectName("PlayerTitle")
        self.player_state_label = QLabel("Idle")
        self.player_state_label.setObjectName("PlayerState")
        top.addWidget(self.player_title_label, 1)
        self.video_window_button = QToolButton()
        self.video_window_button.setObjectName("VideoWindowButton")
        self.video_window_button.setText("Vídeo")
        self.video_window_button.setCheckable(True)
        self.video_window_button.setToolTip("Abrir vídeo em janela separada (mpv ou ffplay)")
        self.video_window_button.clicked.connect(self.toggle_video_window)
        top.addWidget(self.video_window_button)
        top.addWidget(self.player_state_label)
        outer.addLayout(top)

        chrome = QFrame()
        chrome.setObjectName("PlayerChrome")
        chrome.setMinimumHeight(66)
        chrome.setMaximumHeight(72)
        inner = QVBoxLayout(chrome)
        inner.setContentsMargins(14, 3, 14, 6)
        inner.setSpacing(2)

        transport_row = QWidget()
        transport_row.setMinimumHeight(42)
        transport_row.setMaximumHeight(42)
        transport = QHBoxLayout(transport_row)
        transport.setContentsMargins(0, 0, 0, 0)
        transport.setSpacing(10)
        transport.addStretch(1)

        self.seek_back_button = self._make_transport_button(
            QStyle.SP_MediaSeekBackward, "Retroceder 10 s"
        )
        self.seek_back_button.clicked.connect(lambda: self.playback_controller.seek_relative(-SEEK_STEP_S))

        self.play_button = self._make_transport_button(QStyle.SP_MediaPlay, "Reproduzir / pausar", accent=True)
        self.play_button.clicked.connect(self.toggle_playback)

        self.stop_button = self._make_transport_button(QStyle.SP_MediaStop, "Parar")
        self.stop_button.clicked.connect(self.stop_playback)

        self.seek_forward_button = self._make_transport_button(
            QStyle.SP_MediaSeekForward, "Avançar 10 s"
        )
        self.seek_forward_button.clicked.connect(lambda: self.playback_controller.seek_relative(SEEK_STEP_S))

        transport.addWidget(self.seek_back_button, 0, Qt.AlignCenter)
        transport.addWidget(self.play_button, 0, Qt.AlignCenter)
        transport.addWidget(self.stop_button, 0, Qt.AlignCenter)
        transport.addWidget(self.seek_forward_button, 0, Qt.AlignCenter)
        transport.addStretch(1)
        inner.addWidget(transport_row)

        time_row_w = QWidget()
        time_row_w.setMinimumHeight(18)
        time_row = QHBoxLayout(time_row_w)
        time_row.setContentsMargins(0, 0, 0, 0)
        self.position_label = QLabel("0:00 / --:--")
        self.position_label.setObjectName("PositionLabel")
        self.position_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        time_row.addWidget(self.position_label, 0, Qt.AlignLeft)
        time_row.addStretch(1)
        inner.addWidget(time_row_w)

        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setObjectName("Scrubber")
        self.scrubber.setRange(0, 1000)
        self.scrubber.sliderPressed.connect(self._begin_scrub)
        self.scrubber.sliderReleased.connect(self._end_scrub)
        inner.addWidget(self.scrubber)

        outer.addWidget(chrome)
        self._refresh_widget_style(self.play_button)
        return card

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        file_menu = menu_bar.addMenu("File")
        act_load_cfg = QAction("Carregar configuração…", self)
        act_load_cfg.triggered.connect(self.load_config_json)
        file_menu.addAction(act_load_cfg)
        act_save_cfg = QAction("Guardar configuração…", self)
        act_save_cfg.triggered.connect(self.save_config_json)
        file_menu.addAction(act_save_cfg)

        edit_menu = menu_bar.addMenu("Edit")
        act_clear_log = QAction("Limpar consola de log", self)
        act_clear_log.triggered.connect(self.clear_log_console)
        edit_menu.addAction(act_clear_log)

        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("Sobre o VisionAudio…", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def clear_log_console(self) -> None:
        self.log_output.clear()

    def _show_about(self) -> None:
        qt_message(
            self,
            "Sobre",
            f"{APP_TITLE}\n\n"
            "Gera sequências de PWM para LEDs a partir de áudio (arquivo local ou YouTube), "
            "com motor em audio_reader6, recomendação de parâmetros e pré-visualização em tempo real.\n\n"
            "Interface construída em PyQt5.",
            "info",
        )

    def _apply_depth_effects(self) -> None:
        for widget in self.findChildren(QFrame):
            if widget.objectName() in {"Card", "PlayerBar"}:
                apply_drop_shadow(widget, blur_radius=36, y_offset=12, alpha=110)
        for button in self.findChildren(QPushButton):
            if button.objectName() == "TransportBtn":
                continue
            apply_drop_shadow(button, blur_radius=14, y_offset=3, alpha=70)
        apply_drop_shadow(self.status_badge, blur_radius=22, y_offset=4, alpha=80)

    # ── Startup fade-in ────────────────────────────────────
    def _start_fade_in(self) -> None:
        self._fade_value = 0.0
        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(14)
        self._fade_timer.timeout.connect(self._fade_step)
        self._fade_timer.start()

    def _fade_step(self) -> None:
        self._fade_value = min(1.0, self._fade_value + 0.04)
        self.setWindowOpacity(self._fade_value)
        if self._fade_value >= 1.0:
            self._fade_timer.stop()

    def _bind_playback(self) -> None:
        self.playback_controller.state_changed.connect(self._on_playback_state_changed)
        self.playback_controller.position_changed.connect(self._on_playback_position_changed)
        self.playback_controller.frame_changed.connect(self.led_preview.set_levels)
        self.playback_controller.playback_finished.connect(self._on_playback_finished)
        self.playback_controller.playback_error.connect(self._on_playback_error)
        self.video_window_controller.state_changed.connect(self._on_video_window_state_changed)
        self.video_window_controller.error.connect(self._on_video_window_error)

    def _refresh_widget_style(self, widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _set_status_badge(self, state: str) -> None:
        normalized = state if state in {"ready", "busy", "error"} else "ready"
        label = {"ready": "Pronto", "busy": "Ocupado", "error": "Erro"}[normalized]
        self.status_badge.setProperty("state", normalized)
        self.status_badge.setText(label)
        self._refresh_widget_style(self.status_badge)

    def _configure_initial_splitters(self) -> None:
        if self.content_splitter is not None:
            self.content_splitter.setSizes([820, 360])
        if self.monitor_splitter is not None:
            self.monitor_splitter.setSizes([150, 250])
        self._sync_header_combo_widths()

    def _sync_header_combo_widths(self) -> None:
        combo_width = max(self.source_combo.sizeHint().width(), self.port_combo.sizeHint().width())
        self.source_combo.setFixedWidth(combo_width)
        self.port_combo.setFixedWidth(combo_width)
 
    def _on_source_mode_changed(self, _index: int = 0) -> None:
        mode = self.source_combo.currentData()
        is_local = mode == "local"
        self.browse_button.setVisible(is_local)
        if is_local:
            self.source_input.setPlaceholderText("Selecione um arquivo de áudio local")
        else:
            self.source_input.setPlaceholderText("Digite uma busca ou cole uma URL do YouTube")
        self._update_controls_enabled()

    def browse_audio_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecione o audio",
            "",
            "Audio (*.mp3 *.wav *.flac *.ogg *.m4a *.aac);;Todos os arquivos (*)",
        )
        if path:
            self.source_input.setText(path)

    def refresh_ports(self) -> None:
        ports = self.hardware_controller.available_ports()
        current = self.port_combo.currentText()
        self.port_combo.clear()
        if ports:
            self.port_combo.addItems(ports)
            index = self.port_combo.findText(current)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)
            elif current:
                self.port_combo.setEditText(current)
            elif self.port_combo.findText("/dev/ttyACM0") >= 0:
                self.port_combo.setCurrentText("/dev/ttyACM0")
        else:
            self.port_combo.addItem("/dev/ttyACM0")
            self.port_combo.setCurrentText(current or "/dev/ttyACM0")

    def toggle_connection(self) -> None:
        try:
            if self.hardware_controller.is_connected:
                self.hardware_controller.disconnect()
                self.connection_label.setText("")
                self.connect_button.setText("Conectar Arduino")
                self.connect_button.setProperty("variant", "primary")
                self._refresh_widget_style(self.connect_button)
                self.progress_label.setText("Arduino desconectado.")
                self.progress_bar.setValue(0)
                self._set_status_badge("ready")
                self.log("Arduino desconectado.")
                return
            port = self.port_combo.currentText().strip()
            if not port:
                raise ValueError("Selecione uma porta serial.")
            self.hardware_controller.connect_port(port)
            self.connection_label.setText("")
            self.connect_button.setText("Desconectar Arduino")
            self.connect_button.setProperty("variant", "secondary")
            self._refresh_widget_style(self.connect_button)
            self.progress_label.setText("Arduino conectado.")
            self.progress_bar.setValue(0)
            self._set_status_badge("ready")
            self.log(f"Arduino conectado em {port}.")
        except Exception as exc:
            message, details = describe_exception(exc)
            self.progress_label.setText(message)
            self.progress_bar.setValue(0)
            self._set_status_badge("error")
            self.log(f"Falha ao conectar Arduino: {message}", "error")
            qt_message(self, "Arduino", message, "error", details)

    def _set_busy(
        self,
        busy: bool,
        label: Optional[str] = None,
        value: Optional[int] = None,
        status: Optional[str] = None,
    ) -> None:
        self._ui_busy = busy
        if label is not None:
            self.progress_label.setText(label)
        if value is not None:
            self.progress_bar.setValue(int(value))
        self._set_status_badge(status or ("busy" if busy else "ready"))
        self._update_controls_enabled()

    def _set_progress(self, value: int, label: str) -> None:
        self._set_busy(True, label, int(value), "busy")
        self.log(label)

    def _update_controls_enabled(self) -> None:
        busy = self._ui_busy
        self.source_combo.setEnabled(not busy)
        self.source_input.setEnabled(not busy)
        self.browse_button.setEnabled(not busy and self.source_combo.currentData() == "local")
        self.load_button.setEnabled(not busy)
        self.port_combo.setEnabled(not busy)
        self.refresh_ports_button.setEnabled(not busy)
        self.connect_button.setEnabled(not busy)
        self.suggest_button.setEnabled(not busy and self.current_track is not None)
        self.generate_button.setEnabled(not busy and self.current_track is not None)
        self.seek_back_button.setEnabled(not busy and self.current_sequence is not None)
        self.seek_forward_button.setEnabled(not busy and self.current_sequence is not None)
        self.play_button.setEnabled(not busy and self.current_sequence is not None)
        player_state = getattr(self.playback_controller, "state", "idle")
        self.stop_button.setEnabled((not busy and self.current_sequence is not None) or player_state in ("playing", "paused"))
        video_track = self._current_video_track()
        self.video_window_button.setEnabled(not busy and video_track is not None and bool(video_track.youtube_video_url or video_track.youtube_page_url))

    def _finish_background_task(self) -> None:
        self._set_busy(False, status="error" if self._last_background_error else "ready")

    def load_track(self) -> None:
        self.stop_playback()
        self.video_window_controller.close()
        mode = self.source_combo.currentData()
        value = self.source_input.text().strip()
        self._last_background_error = False
        self._load_worker = LoadWorker(self.source_resolver, mode, value)
        self._load_worker.progress.connect(self._set_progress)
        self._load_worker.loaded.connect(self.handle_loaded_track)
        self._load_worker.failed.connect(self._handle_worker_error)
        self._load_worker.finished.connect(self._finish_background_task)
        self._set_busy(True, "Carregando faixa...", 0)
        self._load_worker.start()

    def request_recommendation(self) -> None:
        if not self.current_track:
            qt_message(self, APP_TITLE, "Carregue uma faixa primeiro.", "warn")
            return
        self._recommend_worker = RecommendationWorker(
            self.generation_service,
            self.recommendation_service,
            self.current_track,
        )
        self._last_background_error = False
        self._recommend_worker.progress.connect(self._set_progress)
        self._recommend_worker.finished_result.connect(self.handle_recommendation)
        self._recommend_worker.failed.connect(self._handle_worker_error)
        self._recommend_worker.finished.connect(self._finish_background_task)
        self._set_busy(True, "Analisando faixa para recomendação...", 0)
        self._recommend_worker.start()

    def generate_sequence(self) -> None:
        if not self.current_track:
            qt_message(self, APP_TITLE, "Carregue uma faixa primeiro.", "warn")
            return
        self.stop_playback()
        config_state = self.current_ui_state()
        self._last_background_error = False
        self._generation_worker = GenerationWorker(self.generation_service, self.current_track, config_state)
        self._generation_worker.progress.connect(self._set_progress)
        self._generation_worker.finished_result.connect(self.handle_generated_sequence)
        self._generation_worker.failed.connect(self._handle_worker_error)
        self._generation_worker.finished.connect(self._finish_background_task)
        self._set_busy(True, "Gerando sequência...", 0)
        self._generation_worker.start()

    def current_ui_state(self) -> UIConfigState:
        values = default_control_values()
        for key, control in self._primary_controls.items():
            values[key] = control.value()
        for key, control in self._spin_controls.items():
            values[key] = float(control.value())
        return UIConfigState(
            transition_profile=self.transition_profile_combo.currentData(),
            effect_profile=self.effect_profile_combo.currentData(),
            middle_speed_multiplier=float(self.middle_speed_combo.currentData()),
            values=values,
        )

    def apply_ui_state(self, state: UIConfigState) -> None:
        normalized = state.normalized_values()
        self.transition_profile_combo.setCurrentIndex(max(0, self.transition_profile_combo.findData(state.transition_profile)))
        self.effect_profile_combo.setCurrentIndex(max(0, self.effect_profile_combo.findData(state.effect_profile)))
        self.middle_speed_combo.setCurrentIndex(max(0, self.middle_speed_combo.findData(float(state.middle_speed_multiplier))))
        for key, value in normalized.items():
            if key in self._primary_controls:
                self._primary_controls[key].set_value(value)
                self._primary_controls[key].set_recommended(False)
            elif key in self._spin_controls:
                control = self._spin_controls[key]
                control.blockSignals(True)
                control.setValue(value)
                control.blockSignals(False)
                control.setStyleSheet("")

    def save_config_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar configuracao",
            os.path.join(os.getcwd(), "visionaudio_config.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        payload = {
            "app": APP_TITLE,
            "version": 2,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.current_ui_state().to_serializable(),
        }
        try:
            with open(path, "w", encoding="utf-8") as file_obj:
                json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        except Exception as exc:
            message, details = describe_exception(exc)
            qt_message(self, "Salvar configuracao", message, "error", details)
            self.log(f"Falha ao salvar configuracao: {message}", "error")
            return
        self.log(f"Configuracao salva em {path}.")

    def load_config_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Carregar configuracao",
            os.getcwd(),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            config_payload = payload.get("config", payload)
            state = UIConfigState.from_serializable(config_payload)
            self.apply_ui_state(state)
        except Exception as exc:
            message, details = describe_exception(exc)
            qt_message(self, "Carregar configuracao", message, "error", details)
            self.log(f"Falha ao carregar configuracao: {message}", "error")
            return
        self.log(f"Configuracao carregada de {path}.")

    def handle_loaded_track(self, track: LoadedTrack) -> None:
        self._cleanup_sequence_temp_files(self.current_sequence)
        self.current_track = track
        self.current_recommendation = None
        self.current_sequence = None
        self.video_window_controller.close()
        self.track_title_label.setText(track.title)
        meta = [track.source_mode.upper(), track.display_source]
        if track.file_size_bytes is not None:
            meta.append(format_bytes(track.file_size_bytes))
        self.track_meta_label.setText(" | ".join(meta))
        self.player_title_label.setText(f"{track.title} carregada")
        self.feature_summary_label.setText("Faixa carregada. Clique em 'Sugerir parâmetros' ou 'Gerar sequência'.")
        self.transition_list.clear()
        self.timeline_widget.set_sequence(None)
        self.led_preview.set_levels([0] * 6)
        self.log(f"Faixa carregada: {track.title}")
        self._update_controls_enabled()

    def handle_recommendation(self, result: RecommendationResult) -> None:
        self.current_recommendation = result
        for key, value in result.numeric_values.items():
            if key in self._primary_controls:
                self._primary_controls[key].set_value(value)
                self._primary_controls[key].set_recommended(True)
            elif key in self._spin_controls:
                control = self._spin_controls[key]
                control.blockSignals(True)
                control.setValue(value)
                control.blockSignals(False)
                control.setStyleSheet("border:1px solid #40c6ff; border-radius: 10px;")
        transition_profile = result.categorical_values.get("transition_profile", "medium")
        effect_profile = result.categorical_values.get("effect_profile", "medium")
        middle_speed = float(result.categorical_values.get("middle_speed_multiplier", 1.0))
        self.transition_profile_combo.setCurrentIndex(max(0, self.transition_profile_combo.findData(transition_profile)))
        self.effect_profile_combo.setCurrentIndex(max(0, self.effect_profile_combo.findData(effect_profile)))
        self.middle_speed_combo.setCurrentIndex(max(0, self.middle_speed_combo.findData(middle_speed)))
        engine_label = result.model_label or ("scikit-learn" if result.using_sklearn else "fallback interno")
        self.feature_summary_label.setText("\n".join(result.summary_lines + [f"Modelo: {engine_label}"]))
        self.log(f"Parâmetros sugeridos com {engine_label}.")

    def handle_generated_sequence(self, sequence: GeneratedSequence) -> None:
        self._cleanup_sequence_temp_files(self.current_sequence)
        self.current_sequence = sequence
        self.timeline_widget.set_sequence(sequence)
        self._refresh_transition_list()
        self.player_title_label.setText(f"{sequence.track.title} | {sequence.tempo_bpm:.1f} BPM")
        self.feature_summary_label.setText(
            "\n".join(
                [
                    f"BPM detectado: {sequence.tempo_bpm:.1f}",
                    f"Frames: {len(sequence.led_patterns)}",
                    f"Transições: {len(sequence.transitions)}",
                    f"Densidade: {sequence.feature_snapshot.get('beat_density', 0.0):.2f}",
                ]
            )
        )
        self.log(f"Sequencia gerada para {sequence.track.title}.")
        self._update_controls_enabled()

    def _refresh_transition_list(self, selected_index: Optional[int] = None) -> None:
        self.transition_list.clear()
        if not self.current_sequence:
            return
        sequence = self.current_sequence
        for transition in sequence.transitions:
            start_s = float(getattr(transition, "frame_start", 0)) / sequence.effect_config.FPS
            end_s = float(getattr(transition, "frame_end", 0)) / sequence.effect_config.FPS
            effect = getattr(getattr(transition, "effect_type", None), "name", "TRANS")
            item = QListWidgetItem(f"{format_seconds(start_s)}  {effect}  dur {max(0.0, end_s - start_s):.2f}s")
            self.transition_list.addItem(item)
        if selected_index is not None and 0 <= selected_index < self.transition_list.count():
            self.transition_list.setCurrentRow(selected_index)

    def _on_timeline_transition_moved(self, index: int, start_s: float) -> None:
        _ = start_s
        self._refresh_transition_list(index)

    def toggle_playback(self) -> None:
        if not self.current_sequence:
            qt_message(self, APP_TITLE, "Gere uma sequencia antes de tocar.", "warn")
            return
        if self.playback_controller.state in ("playing", "paused"):
            self.playback_controller.pause_toggle()
            return
        self.playback_controller.play(self.current_sequence)

    def stop_playback(self) -> None:
        self.playback_controller.stop()
        self.video_window_controller.close()

    def _current_video_track(self) -> Optional[LoadedTrack]:
        if self.current_sequence and (
            self.current_sequence.track.youtube_video_url or self.current_sequence.track.youtube_page_url
        ):
            return self.current_sequence.track
        if self.current_track and (self.current_track.youtube_video_url or self.current_track.youtube_page_url):
            return self.current_track
        return None

    def toggle_video_window(self, checked: bool) -> None:
        if not checked:
            self.video_window_controller.close()
            return
        track = self._current_video_track()
        if not track:
            self.video_window_button.setChecked(False)
            qt_message(self, "Video", "Nao ha video disponivel para a faixa atual.", "warn")
            return
        start_s = self.playback_controller.current_position_s()
        try:
            if track.youtube_video_url:
                self.video_window_controller.open(track.youtube_video_url, track.title, start_s=start_s)
                self.log(f"Janela de video aberta para {track.title} com {self.video_window_controller.backend_name}.")
                return
            if track.youtube_page_url:
                if not webbrowser.open(track.youtube_page_url):
                    raise RuntimeError("Nao foi possivel abrir o navegador para o video.")
                self.log(f"Video aberto no navegador: {track.title}.")
                self.video_window_button.setChecked(False)
                return
            raise RuntimeError("A faixa atual nao possui URL de video.")
        except Exception as exc:
            if track.youtube_page_url:
                self.log("Falha ao abrir no ffplay. Tentando navegador...", "warn")
                if webbrowser.open(track.youtube_page_url):
                    self.video_window_button.setChecked(False)
                    self.log(f"Video aberto no navegador: {track.title}.")
                    return
            self.video_window_button.setChecked(False)
            message, details = describe_exception(exc)
            qt_message(self, "Video", message, "error", details)
            self.log(f"Falha ao abrir video: {message}", "error")

    def _on_playback_state_changed(self, state: str) -> None:
        labels = {
            "idle": "Idle",
            "playing": "Tocando",
            "paused": "Pausado",
            "error": "Erro",
        }
        self.player_state_label.setText(labels.get(state, state))
        st = self.style()
        if state == "playing":
            self.play_button.setIcon(st.standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(st.standardIcon(QStyle.SP_MediaPlay))
        self._refresh_widget_style(self.play_button)
        self._update_controls_enabled()

    def _on_video_window_state_changed(self, opened: bool) -> None:
        self.video_window_button.blockSignals(True)
        self.video_window_button.setChecked(opened)
        self.video_window_button.blockSignals(False)

    def _on_video_window_error(self, message: str) -> None:
        normalized = normalize_error_message(message, "VideoWindowError")
        self.log(f"Falha na janela de video: {normalized}", "error")
        qt_message(self, "Video", normalized, "error")

    def _on_playback_position_changed(self, elapsed: float, duration: float) -> None:
        self.position_label.setText(f"{format_seconds(elapsed)} / {format_seconds(duration)}")
        if duration > 0 and not self._scrubbing:
            ratio = elapsed / duration
            self.scrubber.blockSignals(True)
            self.scrubber.setValue(int(ratio * 1000))
            self.scrubber.blockSignals(False)
        self.timeline_widget.set_position(elapsed)

    def _on_playback_finished(self) -> None:
        self.log("Reproducao finalizada.")

    def _on_playback_error(self, message: str) -> None:
        normalized = normalize_error_message(message, "PlaybackError")
        self.progress_label.setText(normalized)
        self.progress_bar.setValue(0)
        self._set_status_badge("error")
        self.log(f"Falha no player: {normalized}", "error")
        qt_message(self, "Player", normalized, "error")

    def _begin_scrub(self) -> None:
        self._scrubbing = True

    def _end_scrub(self) -> None:
        self._scrubbing = False
        self.playback_controller.seek_ratio(self.scrubber.value() / 1000.0)

    def _handle_worker_error(self, message: str, details: str = "") -> None:
        normalized = normalize_error_message(message, "WorkerError")
        self._last_background_error = True
        self.progress_label.setText(normalized)
        self.progress_bar.setValue(0)
        self._set_status_badge("error")
        self.log(normalized, "error")
        qt_message(self, APP_TITLE, normalized, "error", details or None)

    def _on_control_value_changed(self, key: str, _value: float) -> None:
        control = self._primary_controls[key]
        control.set_recommended(False)

    def _on_spin_control_changed(self, key: str) -> None:
        control = self._spin_controls.get(key)
        if control is not None:
            control.setStyleSheet("")

    def _cleanup_sequence_temp_files(self, sequence: Optional[GeneratedSequence]) -> None:
        if not sequence:
            return
        for path in sequence.temp_files:
            with contextlib.suppress(OSError):
                os.remove(path)

    def restore_group_defaults(self, group_name: str) -> None:
        for key in self._group_fields.get(group_name, []):
            spec = CONTROL_SPECS[key]
            control = self._spin_controls[key]
            control.blockSignals(True)
            control.setValue(spec.default)
            control.blockSignals(False)
            control.setStyleSheet("")
        self.log(f"Padrões restaurados em {group_name}.")

    def log(self, message: str, level: str = "info") -> None:
        prefix = {"info": "[INFO]", "warn": "[WARN]", "error": "[ERRO]"}.get(level, "[INFO]")
        stamp = time.strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"{stamp} {prefix} {message}")

    def closeEvent(self, event: Any) -> None:
        self.stop_playback()
        self.video_window_controller.close()
        self.hardware_controller.disconnect()
        self._cleanup_sequence_temp_files(self.current_sequence)
        super().closeEvent(event)


def collect_dependency_summary() -> str:
    checks = [
        ("PyQt5", dependency_available("PyQt5")),
        ("numpy", dependency_available("numpy")),
        ("librosa", dependency_available("librosa")),
        ("pygame", dependency_available("pygame")),
        ("pyserial", dependency_available("serial")),
        ("scikit-learn", dependency_available("sklearn")),
        ("yt-dlp", dependency_available("yt_dlp")),
        ("ffplay", shutil.which("ffplay") is not None),
        ("ffmpeg", shutil.which("ffmpeg") is not None),
    ]
    return " | ".join(f"{name}:{'ok' if ok else 'missing'}" for name, ok in checks)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = VisionAudioWindow()
    window.log("Dependências: " + collect_dependency_summary())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
