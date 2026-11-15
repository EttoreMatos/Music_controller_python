"""VisionAudio — estação de trabalho para transformar música em sequências de 6 LEDs.

Fluxo:  Carregar áudio → (Sugerir parâmetros) → Gerar sequência → Reproduzir.
Requer o motor `audio_reader6.py` na mesma pasta.
"""
from __future__ import annotations

import contextlib
import glob
import html
import importlib
import importlib.util
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
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QPointF, QProcess, QProcessEnvironment, QRectF, QSettings, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QFont, QIcon, QImage, QKeySequence, QLinearGradient, QPainter, QPalette,
                         QPen, QPixmap, QPolygonF)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

APP_TITLE = "VisionAudio"
APP_VERSION = "7.4"
SEEK_STEP_S = 10
PREVIEW_COLUMNS = 8000


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
    box.setIcon({"error": QMessageBox.Critical, "warn": QMessageBox.Warning}.get(level, QMessageBox.Information))
    if details:
        box.setDetailedText(details)
    box.setStandardButtons(QMessageBox.Ok)
    box.exec()


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

    def format_value(self, value: float) -> str:
        if self.value_type is int or self.decimals <= 0:
            return str(int(round(value)))
        return f"{value:.{self.decimals}f}"


CONTROL_SPECS: Dict[str, ControlSpec] = {
    "ACTIVE_MAX_PWM": ControlSpec("ACTIVE_MAX_PWM", "Brilho", "Main", 120.0, 255.0, 190.0, 1.0, float, 0, True, "Brilho maximo dos LEDs centrais."),
    "EDGE_PULSE_GAIN": ControlSpec("EDGE_PULSE_GAIN", "Pontas", "Main", 80.0, 255.0, 160.0, 1.0, float, 0, True, "Forca das pontas."),
    "BEAT_BOOST_PWM": ControlSpec("BEAT_BOOST_PWM", "Beat boost", "Main", 0.0, 60.0, 25.0, 1.0, float, 0, True, "Boost de brilho em batidas."),
    "MIDDLE_FADE_DECAY": ControlSpec("MIDDLE_FADE_DECAY", "Fade meio", "Main", 0.30, 0.95, 0.6, 100.0, float, 2, True, "Cauda dos LEDs do meio."),
    "MIDDLE_EDGE_SYNC_GAIN": ControlSpec("MIDDLE_EDGE_SYNC_GAIN", "Sync meio", "Main", 0.0, 150.0, 85.0, 1.0, float, 0, True, "Quanto as pontas empurram o centro."),
    "TAIL_CUTOFF": ControlSpec("TAIL_CUTOFF", "Cutoff", "Main", 0.0, 12.0, 2.0, 10.0, float, 1, True, "Corta ruidos baixos no final."),
    "SEG_NOVELTY_THRESHOLD": ControlSpec("SEG_NOVELTY_THRESHOLD", "TH novidade", "Main", 0.05, 0.35, 0.16, 100.0, float, 2, True, "Threshold de disparo de transicoes."),
    "SEG_STRUCT_CHANGE_THRESHOLD": ControlSpec("SEG_STRUCT_CHANGE_THRESHOLD", "TH estrutura", "Main", 0.04, 0.25, 0.1, 100.0, float, 2, True, "Minimo de troca estrutural."),
    "TRANS_MAX_DURATION_S": ControlSpec("TRANS_MAX_DURATION_S", "Trans max", "Main", 0.50, 2.80, 1.8, 100.0, float, 2, True, "Duracao maxima das transicoes."),
    "TRANS_BLEND_FRAMES": ControlSpec("TRANS_BLEND_FRAMES", "Blend", "Main", 1.0, 12.0, 5, 1.0, int, 0, True, "Frames de blend."),
    "ACTIVE_MIN_PWM": ControlSpec("ACTIVE_MIN_PWM", "Brilho base", "Dynamics", 20.0, 140.0, 68.0, 1.0, float, 0),
    "MIDDLE_KICK_BRIGHT_GAIN": ControlSpec("MIDDLE_KICK_BRIGHT_GAIN", "Kick meio", "Dynamics", 20.0, 220.0, 120.0, 1.0, float, 0),
    "MIDDLE_STRONG_KICK_BRIGHT_GAIN": ControlSpec("MIDDLE_STRONG_KICK_BRIGHT_GAIN", "Kick forte", "Dynamics", 10.0, 140.0, 65.0, 1.0, float, 0),
    "HEAVY_BEAT_THRESHOLD": ControlSpec("HEAVY_BEAT_THRESHOLD", "Beat pesado", "Dynamics", 0.10, 0.95, 0.58, 100.0, float, 2),
    "EDGE_HEAVY_BONUS_GAIN": ControlSpec("EDGE_HEAVY_BONUS_GAIN", "Bonus ponta", "Edges", 30.0, 255.0, 150.0, 1.0, float, 0),
    "EDGE_KICK_WINDOW_BOOST": ControlSpec("EDGE_KICK_WINDOW_BOOST", "Janela kick", "Edges", 50.0, 280.0, 200.0, 1.0, float, 0),
    "EDGE_GATE_OPEN_THRESHOLD": ControlSpec("EDGE_GATE_OPEN_THRESHOLD", "Gate open", "Edges", 0.05, 0.60, 0.18, 100.0, float, 2),
    "EDGE_GATE_CLOSE_THRESHOLD": ControlSpec("EDGE_GATE_CLOSE_THRESHOLD", "Gate close", "Edges", 0.02, 0.40, 0.1, 100.0, float, 2),
    "EDGE_GATE_MIN_LOW_DOMINANCE": ControlSpec("EDGE_GATE_MIN_LOW_DOMINANCE", "Low dom", "Edges", 0.02, 0.60, 0.16, 100.0, float, 2),
    "EDGE_NOISE_FLOOR_PWM": ControlSpec("EDGE_NOISE_FLOOR_PWM", "Noise floor", "Edges", 0.0, 60.0, 12.0, 1.0, float, 0),
    "EDGE_IDLE_DECAY": ControlSpec("EDGE_IDLE_DECAY", "Idle decay", "Edges", 0.40, 0.95, 0.78, 100.0, float, 2),
    "NOISE_FLATNESS_THRESHOLD": ControlSpec("NOISE_FLATNESS_THRESHOLD", "Flatness", "Segmentation", 0.20, 0.90, 0.55, 100.0, float, 2),
    "BRIGHT_CENTROID_THRESHOLD": ControlSpec("BRIGHT_CENTROID_THRESHOLD", "Centroid", "Segmentation", 0.05, 0.60, 0.22, 100.0, float, 2),
    "SEG_MIN_GAP_SECONDS": ControlSpec("SEG_MIN_GAP_SECONDS", "Gap min", "Segmentation", 0.40, 3.00, 1.4, 100.0, float, 2),
    "SEG_MIN_ENERGY_THRESHOLD": ControlSpec("SEG_MIN_ENERGY_THRESHOLD", "Energia min", "Segmentation", 0.02, 0.40, 0.1, 100.0, float, 2),
    "TRANS_MIN_DURATION_S": ControlSpec("TRANS_MIN_DURATION_S", "Trans min", "Timing", 0.15, 1.20, 0.45, 100.0, float, 2),
    "BPM_RESYNC_STRENGTH": ControlSpec("BPM_RESYNC_STRENGTH", "Resync", "Timing", 0.20, 1.00, 0.85, 100.0, float, 2),
    "BPM_CONTINUOUS_SYNC_STRENGTH": ControlSpec("BPM_CONTINUOUS_SYNC_STRENGTH", "Sync continuo", "Timing", 0.01, 0.30, 0.06, 100.0, float, 2),
    "MID_SUPPORT_DEPTH": ControlSpec("MID_SUPPORT_DEPTH", "Dinâmica do passo", "Middle", 0.0, 1.0, 0.38, 100.0, float, 2, False, "Quanto o brilho de cada passo segue o ataque real da música (0 = todos os passos iguais)."),
    "MID_STEP_FLOOR": ControlSpec("MID_STEP_FLOOR", "Piso do passo", "Middle", 0.3, 0.95, 0.78, 100.0, float, 2, False, "Brilho que o passo mantém até o próximo (menor = mais pulsante)."),
    "MID_HIT_THRESHOLD": ControlSpec("MID_HIT_THRESHOLD", "Sensib. de ataque", "Middle", 0.15, 0.75, 0.38, 100.0, float, 2, False, "Menor = detecta mais ataques; maior = só os mais fortes."),
    "MID_MAX_LIT_AVG": ControlSpec("MID_MAX_LIT_AVG", "Máx. de LEDs acesos", "Middle", 1.2, 3.0, 1.8, 100.0, float, 2, False, "Trava anti-saturação: média máxima de LEDs do meio acesos ao mesmo tempo."),
    "DROP_ENERGY_JUMP": ControlSpec("DROP_ENERGY_JUMP", "Limiar de drop", "Middle", 0.1, 0.6, 0.22, 100.0, float, 2, False, "Quão forte precisa ser a virada de energia para disparar o flash de drop."),
    "TRANS_LIVE_GAIN": ControlSpec("TRANS_LIVE_GAIN", "Efeito × energia", "Middle", 0.0, 1.0, 0.35, 100.0, float, 2, False, "Quanto o brilho das transições acompanha a energia da música ao vivo."),
}

ADVANCED_GROUPS = ["Dynamics", "Edges", "Segmentation", "Timing", "Middle"]
GROUP_LABELS = {
    "Dynamics": "Dinâmica",
    "Edges": "Pontas",
    "Segmentation": "Segmentação",
    "Timing": "Tempo",
    "Middle": "Meio",
}
MIXER_GROUPS = [
    ("LEDs", ["ACTIVE_MAX_PWM", "EDGE_PULSE_GAIN", "BEAT_BOOST_PWM", "MIDDLE_FADE_DECAY", "MIDDLE_EDGE_SYNC_GAIN", "TAIL_CUTOFF"]),
    ("Transições", ["SEG_NOVELTY_THRESHOLD", "SEG_STRUCT_CHANGE_THRESHOLD", "TRANS_MAX_DURATION_S", "TRANS_BLEND_FRAMES"]),
]
PARAM_HELP = {
    "ACTIVE_MIN_PWM": "Brilho base dos LEDs do meio em trechos de baixa energia.",
    "MIDDLE_KICK_BRIGHT_GAIN": "Acréscimo de brilho do meio a cada kick.",
    "MIDDLE_STRONG_KICK_BRIGHT_GAIN": "Acréscimo extra de brilho para kicks fortes.",
    "HEAVY_BEAT_THRESHOLD": "Acima deste nível a batida é tratada como pesada (boost extra).",
    "EDGE_HEAVY_BONUS_GAIN": "Bônus de intensidade das pontas em batidas pesadas.",
    "EDGE_KICK_WINDOW_BOOST": "Reforço das pontas na janela curta logo após um kick.",
    "EDGE_GATE_OPEN_THRESHOLD": "Nível de grave que abre o gate das pontas (elas começam a pulsar).",
    "EDGE_GATE_CLOSE_THRESHOLD": "Nível de grave abaixo do qual o gate das pontas fecha.",
    "EDGE_GATE_MIN_LOW_DOMINANCE": "Predomínio mínimo dos graves sobre os agudos para as pontas reagirem.",
    "EDGE_NOISE_FLOOR_PWM": "Pulsos das pontas abaixo deste valor são descartados.",
    "EDGE_IDLE_DECAY": "Velocidade com que as pontas apagam quando o gate está fechado.",
    "NOISE_FLATNESS_THRESHOLD": "Planura espectral a partir da qual o trecho conta como ruído (escolha do efeito).",
    "BRIGHT_CENTROID_THRESHOLD": "Centroide espectral a partir do qual o trecho conta como brilhante (escolha do efeito).",
    "SEG_MIN_GAP_SECONDS": "Intervalo mínimo entre duas transições.",
    "SEG_MIN_ENERGY_THRESHOLD": "Ignora transições em trechos quase silenciosos.",
    "TRANS_MIN_DURATION_S": "Duração mínima de um efeito de transição.",
    "BPM_RESYNC_STRENGTH": "Força do reencaixe do relógio dos passos no grid de batidas após uma transição.",
    "BPM_CONTINUOUS_SYNC_STRENGTH": "Correção contínua do relógio para manter os passos exatamente na batida.",
}
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

    def to_effect_config(self, engine_module: Any) -> Any:
        cfg = engine_module.EffectConfig()
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
    duration_s: float
    transitions: List[Any]
    playback_source: str
    tempo_bpm: float
    beat_count: int
    preview_lanes: List[List[int]]
    report: Dict[str, Any] = field(default_factory=dict)
    report_lines: List[str] = field(default_factory=list)
    temp_files: List[str] = field(default_factory=list)

    @property
    def fps(self) -> int:
        return int(self.effect_config.FPS)


@dataclass
class CachedTrackData:
    y: Any
    sr: int
    duration_s: float
    feature_snapshot: Optional[Dict[str, float]] = None
    work_dir: Optional[str] = None          # pasta temporaria do download remoto
    playback_wav: Optional[str] = None      # WAV 44,1 kHz estereo (so faixas remotas)


def build_preview_lanes(led_patterns: Sequence[Tuple[int, Any]], columns: int = PREVIEW_COLUMNS) -> List[List[int]]:
    """Resume a partitura em `columns` colunas de 6 LEDs (pico por coluna, para não perder flashes)."""
    total = len(led_patterns)
    if total == 0:
        return []
    chunk = max(1, -(-total // columns))
    lanes: List[List[int]] = []
    for start in range(0, total, chunk):
        peak = [0] * 6
        for _, values in led_patterns[start:start + chunk]:
            for idx in range(6):
                level = int(values[idx])
                if level > peak[idx]:
                    peak[idx] = level
        lanes.append(peak)
    return lanes


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

        if not dependency_available("yt_dlp"):
            raise DependencyError("yt-dlp não encontrado. Instale com: pip install -U \"yt-dlp[default]\"")
        # O motor registra headers/página do stream: é o que evita o HTTP 403 no download.
        stream = AudioReader6EngineLoader().load().resolve_youtube_stream_info(value)
        return LoadedTrack(
            source_mode="youtube",
            requested_source=value,
            resolved_source=stream.audio_url,
            title=stream.title,
            display_source=value,
            duration_s=stream.duration_s,
            youtube_video_url=stream.video_url,
            youtube_page_url=stream.page_url,
            metadata={
                "platform": "youtube",
                "audio_headers": dict(stream.audio_headers or {}),
                "video_headers": dict(stream.video_headers or {}),
            },
        )


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
                "Verifique dependências do motor (librosa, numpy, scipy).\n"
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
            _, evicted = self._track_cache.popitem(last=False)
            self._dispose_track_data(evicted)
        return data

    @staticmethod
    def _dispose_track_data(data: CachedTrackData) -> None:
        if data.work_dir:
            shutil.rmtree(data.work_dir, ignore_errors=True)

    def cleanup(self) -> None:
        for data in list(self._track_cache.values()):
            self._dispose_track_data(data)
        self._track_cache.clear()

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
        work_dir: Optional[str] = None
        playback_wav: Optional[str] = None
        if is_url(track.resolved_source):
            # FIX 403: yt-dlp baixa o audio; ffmpeg decodifica o arquivo LOCAL.
            engine = self.engine_loader.load()
            work_dir = tempfile.mkdtemp(prefix="visionaudio_yt_")
            try:
                remote = engine.load_remote_audio(track.resolved_source, 22050, work_dir=work_dir)
            except Exception:
                shutil.rmtree(work_dir, ignore_errors=True)
                raise
            y, sr = remote.y, remote.sr
            if remote.local_path:
                try:
                    playback_wav = engine.make_playback_wav(remote.local_path, os.path.join(work_dir, "playback.wav"))
                except Exception:
                    playback_wav = None
                for leftover in glob.glob(os.path.join(work_dir, "audio*")):
                    try:
                        os.remove(leftover)
                    except OSError:
                        pass
        else:
            y, sr = self._load_audio(track.resolved_source, target_sr=22050, librosa_module=librosa_module)
        duration = max(librosa_module.get_duration(y=y, sr=sr), 1e-6)
        cached = CachedTrackData(y=y, sr=sr, duration_s=float(duration), work_dir=work_dir, playback_wav=playback_wav)
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
        def report(percent: float, label: str) -> None:
            if progress_cb:
                progress_cb(int(percent), label)

        report(3, "Carregando motor de análise...")
        engine = self.engine_loader.load()
        librosa, _np = self._require_modules()
        effect_config = config_state.to_effect_config(engine)
        score_engine = engine.LedScoreEngine(
            fps=effect_config.FPS,
            config=effect_config,
            middle_bpm_multiplier=config_state.middle_speed_multiplier,
        )
        track_data = self._get_or_load_track_data(track, librosa, progress_cb)
        y, sr, duration = track_data.y, track_data.sr, track_data.duration_s
        result = score_engine.compute_score(
            y, sr, progress_cb=lambda fraction, label: report(20 + 72 * fraction, label)
        )
        playback_source = track.resolved_source
        temp_files: List[str] = []
        if is_url(playback_source):
            hq_wav = track_data.playback_wav
            if hq_wav and os.path.isfile(hq_wav):
                # copia por sequencia: o cache pode descartar o original a qualquer momento
                fd, copy_path = tempfile.mkstemp(prefix="visionaudio72_", suffix=".wav")
                os.close(fd)
                shutil.copyfile(hq_wav, copy_path)
                playback_source = copy_path
            else:
                playback_source = create_temp_wav_from_audio(y, sr)
            temp_files.append(playback_source)
        report(94, "Montando visão geral...")
        report(100, "Sequência pronta.")
        return GeneratedSequence(
            track=track,
            config_state=config_state,
            effect_config=effect_config,
            led_patterns=result.led_patterns,
            duration_s=float(duration),
            transitions=list(result.transitions),
            playback_source=playback_source,
            tempo_bpm=float(result.tempo_bpm),
            beat_count=len(result.beat_frames),
            preview_lanes=build_preview_lanes(result.led_patterns),
            report=dict(result.report),
            report_lines=list(result.report_lines),
            temp_files=temp_files,
        )

    def _load_audio(self, source: str, target_sr: int, librosa_module: Any) -> Tuple[Any, int]:
        if is_url(source):
            remote = self.engine_loader.load().load_remote_audio(source, target_sr)
            if remote.work_dir:
                shutil.rmtree(remote.work_dir, ignore_errors=True)
            return remote.y, remote.sr
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

    def current_position_s(self) -> float:
        if not self.session:
            return 0.0
        return self.session.position_s()

    def play(self, sequence: GeneratedSequence, start_s: float = 0.0) -> None:
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
        if start_s > 0.05 and self.session.supports_seek:
            self.session.seek(float(start_s))
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

    def seek_to(self, seconds: float) -> None:
        """Vai exatamente para `seconds` (precisão de décimos, não de segundos inteiros)."""
        if not self.session or not self.session.supports_seek or not self.sequence:
            return
        delta = float(seconds) - self.session.position_s()
        if self.session.seek(delta):
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

    def open(self, source: str, title: str, start_s: float = 0.0, page_url: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None:
        self.close()
        self._process_output = ""
        program, args = self._build_player_command(source, title, start_s, page_url, headers)
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

    def _build_player_command(self, source: str, title: str, start_s: float, page_url: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Tuple[str, List[str]]:
        headers = dict(headers or {})
        user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        for key in list(headers):
            if key.lower() == "user-agent":
                user_agent = headers.pop(key)
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
            if page_url:
                # o hook yt-dlp do mpv cuida de headers/chunks (evita 403 na URL crua)
                args.append("--ytdl=yes")
                args.append("--ytdl-format=bestvideo[height<=720]/best[height<=720]/best")
                ytdlp_exe = shutil.which("yt-dlp") or os.path.join(os.path.dirname(sys.executable), "yt-dlp")
                if os.path.isfile(ytdlp_exe):
                    args.append(f"--script-opts=ytdl_hook-ytdl_path={ytdlp_exe}")
                args.append(page_url)
            else:
                args.append(f"--user-agent={user_agent}")
                if headers:
                    args.append("--http-header-fields=" + ",".join(f"{k}: {v}" for k, v in headers.items()))
                args.append(source)
            return mpv_bin, args

        ffplay_bin = shutil.which("ffplay")
        if not ffplay_bin:
            raise DependencyError("Nem mpv nem ffplay foram encontrados para abrir a janela de video.")
        self._backend_name = "ffplay"
        args = ["-autoexit", "-loglevel", "error", "-window_title", f"Video - {title}", "-an", "-x", "960", "-y", "540"]
        if start_s > 0.05:
            args.extend(["-ss", f"{max(0.0, start_s):.3f}"])
        args.extend(["-user_agent", user_agent])
        if headers:
            args.extend(["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())])
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


# ============================================================================
# Tema, ícones e widgets
# ============================================================================

PALETTE = {
    "bg0": "#07080f", "bg1": "#0b0f1c", "bg2": "#111624", "bg3": "#1a2038",
    "border": "#16293d", "border2": "#25425c",
    "text": "#ccd8ee", "dim": "#6b84a0", "faint": "#3a4d63",
    "accent": "#f59e0b", "accent2": "#00d4ff",
    "ok": "#34d399", "err": "#f87171", "warn": "#fbbf24",
    "edge": "#f79435", "mid": "#28c8f8", "drop": "#a78bfa",
}

# Tema "deep space" do 7.2: fundo azul-noite com gradiente radial, cartões de vidro com
# aro ciano, botões glossy e CTA âmbar.
STYLESHEET = Template("""
QWidget { color: $text; font-size: 9pt; }
QMainWindow {
    background: qradialgradient(cx:0.15, cy:0.06, radius:1.20,
        stop:0 #151a2e, stop:0.12 #0e1122, stop:0.45 #090c18, stop:1 #040509);
}
QMainWindow::separator { background: transparent; width: 6px; height: 6px; }
QDialog { background: #0b0f1e; }
QToolTip { background: #0b0f1e; color: $text; border: 1px solid rgba(0, 180, 220, 0.40); border-radius: 6px; padding: 5px 8px; }

QMenuBar { background: rgba(8, 10, 20, 0.92); color: #7f9bb8; padding: 1px 6px; font-weight: 600; border-bottom: 1px solid rgba(0, 160, 200, 0.18); spacing: 2px; }
QMenuBar::item { padding: 6px 12px; background: transparent; border-radius: 5px; }
QMenuBar::item:selected { background: rgba(0, 180, 220, 0.12); color: $accent2; }
QMenu { background: #0b0f1e; border: 1px solid rgba(0, 180, 220, 0.28); border-radius: 10px; padding: 6px 4px; }
QMenu::item { padding: 7px 28px 7px 16px; color: #8aa8c8; border-radius: 6px; margin: 1px 4px; }
QMenu::item:selected { background: rgba(0, 180, 220, 0.15); color: $accent2; }
QMenu::item:disabled { color: $faint; }
QMenu::separator { height: 1px; background: rgba(0, 160, 200, 0.15); margin: 4px 12px; }

QFrame#Panel, QFrame#TopBar, QFrame#TransportBar, QFrame#Card {
    background: qlineargradient(x1:0, y1:0, x2:0.1, y2:1,
        stop:0.000 #161b2e, stop:0.015 #111624, stop:0.500 #0b0f1c, stop:0.985 #080c16, stop:1.000 #050810);
    border-top: 1px solid rgba(0, 200, 240, 0.30);
    border-left: 1px solid rgba(0, 180, 220, 0.16);
    border-right: 1px solid rgba(0, 160, 200, 0.10);
    border-bottom: 1px solid rgba(0, 120, 160, 0.10);
    border-radius: 14px;
}
QFrame#DockTitle {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #161b2e, stop:1 #0b0f1c);
    border: none;
    border-bottom: 1px solid rgba(0, 200, 240, 0.22);
}
QFrame#Divider { background: rgba(0, 160, 200, 0.22); border: none; max-width: 1px; min-width: 1px; }

QLabel { background: transparent; }
QLabel[role="muted"] { color: $dim; }
QLabel[role="faint"] { color: $faint; }
QLabel[role="title"] { font-size: 12pt; font-weight: 700; color: #e8f2ff; }
QLabel[role="section"] { color: $accent2; font-size: 8pt; font-weight: 800; }
QLabel[role="value"] { color: $text; }
QLabel[role="metric"] { color: #ffd26d; font-weight: 700; }
QLabel[role="clock"] { font-size: 15pt; font-weight: 700; font-family: "DejaVu Sans Mono", monospace; color: #e8f4ff; }
QLabel[role="chip"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0c1020, stop:1 #070a14);
    border-top: 1px solid rgba(0, 180, 220, 0.28);
    border-left: 1px solid rgba(0, 160, 200, 0.14);
    border-right: 1px solid rgba(0, 120, 160, 0.09);
    border-bottom: 1px solid rgba(0, 100, 140, 0.09);
    border-radius: 11px; padding: 3px 11px; color: #c8dcf0;
}
QLabel#DockTitleLabel { font-weight: 800; color: $accent2; }
QLabel#StatusBadge {
    padding: 4px 12px; border-radius: 9px; font-weight: 800; color: #002010;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #34d399, stop:1 #059669);
    border: 1px solid #10b981;
}
QLabel#StatusBadge[state="busy"] {
    color: #1a1000; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fbbf24, stop:1 #d97706); border: 1px solid #f59e0b;
}
QLabel#StatusBadge[state="error"] {
    color: #fff0f0; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f87171, stop:1 #b91c1c); border: 1px solid #ef4444;
}
QLabel#HwDot { background: $faint; border-radius: 5px; min-width: 10px; max-width: 10px; min-height: 10px; max-height: 10px; }
QLabel#HwDot[connected="true"] { background: $ok; border: 1px solid #a7f3d0; }

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1a2038, stop:1 #10162a);
    border-top: 1px solid rgba(0, 200, 240, 0.35);
    border-left: 1px solid rgba(0, 180, 220, 0.20);
    border-right: 1px solid rgba(0, 140, 180, 0.12);
    border-bottom: 1px solid rgba(0, 100, 140, 0.12);
    border-radius: 10px; color: #9ab8d8; padding: 6px 14px; min-height: 18px; font-weight: 600;
}
QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e2a46, stop:1 #121c32); border-top-color: rgba(0, 212, 255, 0.60); color: #c8e0f4; }
QPushButton:pressed { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #090e1c, stop:1 #141c30); border-top-color: rgba(0, 160, 200, 0.25); color: #7099b8; }
QPushButton:disabled { background: #090c18; color: #2f4156; border: 1px solid rgba(30, 45, 65, 0.50); }
QPushButton[variant="primary"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fbbf24, stop:0.45 #f59e0b, stop:1 #d97706);
    border-top: 2px solid #fcd34d; border-left: 1px solid #f59e0b; border-right: 1px solid #b45309; border-bottom: 1px solid #92400e;
    color: #0d0800; font-weight: 800;
}
QPushButton[variant="primary"]:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fde68a, stop:0.45 #fbbf24, stop:1 #f59e0b); border-top-color: #fef3c7; }
QPushButton[variant="primary"]:pressed { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #b45309, stop:1 #78350f); border-top-color: #d97706; }
QPushButton[variant="primary"]:disabled { background: #2a1f0a; color: #5a4020; border: 1px solid #3a2a10; }
QPushButton[variant="ghost"] { background: transparent; border: 1px solid transparent; color: $dim; padding: 3px 8px; }
QPushButton[variant="ghost"]:hover { background: rgba(0, 180, 220, 0.10); color: $accent2; }

QToolButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1a2038, stop:1 #10162a);
    border-top: 1px solid rgba(0, 200, 240, 0.30);
    border-left: 1px solid rgba(0, 180, 220, 0.16);
    border-right: 1px solid rgba(0, 140, 180, 0.10);
    border-bottom: 1px solid rgba(0, 100, 140, 0.10);
    border-radius: 10px; padding: 5px; color: #9ab8d8;
}
QToolButton:hover { border-top-color: rgba(0, 212, 255, 0.60); background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e2a46, stop:1 #121c32); }
QToolButton:pressed { background: #090e1c; }
QToolButton:disabled { color: $faint; background: #090c18; border: 1px solid rgba(30, 45, 65, 0.50); }
QToolButton:checked { color: #fbbf24; border: 1px solid #f59e0b; background: rgba(245, 158, 11, 0.15); }
QToolButton[role="play"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fbbf24, stop:0.45 #f59e0b, stop:1 #d97706);
    border: 1px solid #b45309; border-top: 2px solid #fcd34d; border-radius: 23px;
}
QToolButton[role="play"]:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fde68a, stop:0.45 #fbbf24, stop:1 #f59e0b); }
QToolButton[role="play"]:disabled { background: #2a1f0a; border: 1px solid #3a2a10; }
QToolButton#DockClose { background: transparent; border: none; color: $dim; padding: 2px 7px; border-radius: 6px; }
QToolButton#DockClose:hover { background: rgba(0, 180, 220, 0.14); color: $accent2; }

QToolButton[role="transport"] { border-radius: 17px; padding: 0px; }
QLabel#PlayerState { color: $dim; font-weight: 700; padding: 3px 11px; border-radius: 10px; background: rgba(255, 255, 255, 0.05); }
QLabel#PlayerState[state="playing"] { color: $ok; background: rgba(52, 211, 153, 0.14); }
QLabel#PlayerState[state="paused"] { color: $warn; background: rgba(251, 191, 36, 0.14); }
QLabel#PlayerState[state="error"] { color: $err; background: rgba(248, 113, 113, 0.14); }

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0c1020, stop:1 #070a14);
    border-top: 1px solid rgba(0, 180, 220, 0.25);
    border-left: 1px solid rgba(0, 160, 200, 0.14);
    border-right: 1px solid rgba(0, 120, 160, 0.09);
    border-bottom: 1px solid rgba(0, 100, 140, 0.09);
    border-radius: 10px; padding: 6px 12px; color: #c8dcf0; min-height: 18px;
    selection-background-color: #f59e0b; selection-color: #080600;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-top-color: rgba(0, 212, 255, 0.60); border-left-color: rgba(0, 200, 240, 0.35); }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: $faint; background: #090c18; }
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: top right; width: 26px; margin: 1px 1px 1px 0; border: none;
    border-left: 1px solid rgba(0, 150, 190, 0.20); background: transparent;
    border-top-right-radius: 9px; border-bottom-right-radius: 9px;
}
QComboBox::down-arrow { image: none; width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid #38b8d8; margin-right: 8px; }
QComboBox::down-arrow:on { border-top-color: #fbbf24; }
QComboBox QAbstractItemView {
    background: #0c1020; border: 1px solid rgba(0, 180, 220, 0.30); border-radius: 10px; padding: 4px 0px; outline: none;
    selection-background-color: rgba(0, 180, 220, 0.18); selection-color: $accent2;
}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0px; border: none; }

QTabWidget::pane {
    border-top: 1px solid rgba(0, 200, 240, 0.30); border-left: 1px solid rgba(0, 180, 220, 0.16);
    border-right: 1px solid rgba(0, 160, 200, 0.10); border-bottom: 1px solid rgba(0, 120, 160, 0.10);
    border-radius: 14px; top: -1px;
    background: qlineargradient(x1:0, y1:0, x2:0.1, y2:1, stop:0 #161b2e, stop:0.015 #111624, stop:0.5 #0b0f1c, stop:1 #050810);
}
QTabBar::tab { background: transparent; color: $dim; padding: 8px 18px; border: none; border-bottom: 2px solid transparent; font-weight: 700; }
QTabBar::tab:selected { color: $accent2; border-bottom: 2px solid $accent2; }
QTabBar::tab:hover:!selected { color: #a0dcf0; }

QTableWidget, QPlainTextEdit {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #080c18, stop:1 #050810);
    border: 1px solid rgba(0, 160, 200, 0.18); border-radius: 10px; color: #8aa8c8;
    gridline-color: rgba(0, 140, 180, 0.12);
    selection-background-color: rgba(0, 180, 220, 0.18); selection-color: $accent2;
    alternate-background-color: rgba(0, 160, 200, 0.04);
}
QPlainTextEdit { padding: 8px 10px; color: #6888a8; }
QTableWidget::item { padding: 3px 8px; }
QHeaderView::section { background: #0b0f1c; color: $accent2; border: none; border-bottom: 1px solid rgba(0, 160, 200, 0.25); padding: 5px 8px; font-weight: 700; }

QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { width: 8px; background: transparent; border: none; margin: 4px 2px 4px 0; border-radius: 4px; }
QScrollBar::handle:vertical { min-height: 28px; border-radius: 4px; background: rgba(0, 180, 220, 0.22); border: none; }
QScrollBar::handle:vertical:hover { background: rgba(0, 212, 255, 0.40); }
QScrollBar:horizontal { height: 8px; background: transparent; border: none; margin: 0 4px 2px 4px; border-radius: 4px; }
QScrollBar::handle:horizontal { min-width: 28px; border-radius: 4px; background: rgba(0, 180, 220, 0.22); border: none; }
QScrollBar::handle:horizontal:hover { background: rgba(0, 212, 255, 0.40); }
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: transparent; border: none; }

QProgressBar { background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(0, 160, 200, 0.22); border-radius: 5px; max-height: 9px; min-height: 9px; text-align: center; color: transparent; }
QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0070a0, stop:0.55 #00b8e0, stop:1 #fbbf24); border-radius: 4px; }

QStatusBar { background: rgba(8, 10, 20, 0.92); border: none; border-top: 1px solid rgba(0, 160, 200, 0.22); color: $dim; }
QStatusBar::item { border: none; margin: 0; padding: 0; }
QStatusBar QLabel { background: transparent; }
QSplitter::handle { background: transparent; }
QSplitter::handle:vertical { height: 8px; }

QFrame#Strip {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #222e40, stop:0.36 #141e2e, stop:1 #090f1a);
    border-top: 1px solid #384f68; border-left: 1px solid #28394e; border-right: 1px solid #1a2838; border-bottom: 1px solid #10182a;
    border-radius: 18px;
}
QFrame#Strip[recommended="true"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e3246, stop:0.38 #162535, stop:1 #0c1520);
    border-top: 2px solid #40d4ff; border-left: 1px solid #28a8d8; border-right: 1px solid #1a7898; border-bottom: 1px solid #104858;
}
QLabel#StripName { color: #eef3fa; font-size: 8pt; font-weight: 700; }
QLabel#StripValue { color: #dce8f8; font-weight: 700; }
QLabel#StripValue[modified="true"] { color: #ffd26d; }
QFrame#ParamRow { background: transparent; border: none; border-radius: 8px; }
QFrame#ParamRow:hover { background: rgba(0, 180, 220, 0.06); }
QFrame#ParamRow[recommended="true"] { border-left: 2px solid $accent2; }
QLabel#ParamName[modified="true"] { color: #ffd26d; }
""")


def build_stylesheet() -> str:
    return STYLESHEET.substitute(PALETTE)


def refresh_style(widget: QWidget) -> None:
    """Reaplica o QSS depois de mudar uma propriedade dinâmica."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    p = PALETTE
    roles = {
        QPalette.Window: p["bg0"], QPalette.WindowText: p["text"],
        QPalette.Base: p["bg0"], QPalette.AlternateBase: p["bg1"],
        QPalette.Text: p["text"], QPalette.Button: p["bg2"], QPalette.ButtonText: p["text"],
        QPalette.ToolTipBase: p["bg3"], QPalette.ToolTipText: p["text"],
        QPalette.Highlight: p["accent2"], QPalette.HighlightedText: "#04202b",
        QPalette.BrightText: "#ffffff", QPalette.Link: p["accent2"],
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(p["faint"]))
    app.setPalette(palette)
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)
    app.setStyleSheet(build_stylesheet())


# ----------------------------------------------------------------------------
# Ícones vetoriais (desenhados em tempo de execução: iguais em qualquer sistema)
# ----------------------------------------------------------------------------

def _poly(painter: QPainter, points: Sequence[Tuple[float, float]]) -> None:
    painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))


def _rrect(painter: QPainter, x: float, y: float, w: float, h: float, r: float = 1.5) -> None:
    painter.drawRoundedRect(QRectF(x, y, w, h), r, r)


def _icon_play(p: QPainter, color: QColor) -> None:
    _poly(p, [(7, 4), (20, 12), (7, 20)])


def _icon_pause(p: QPainter, color: QColor) -> None:
    _rrect(p, 6, 4, 4.5, 16)
    _rrect(p, 13.5, 4, 4.5, 16)


def _icon_stop(p: QPainter, color: QColor) -> None:
    _rrect(p, 6, 6, 12, 12, 2)


def _icon_back(p: QPainter, color: QColor) -> None:
    _poly(p, [(12, 5), (3, 12), (12, 19)])
    _poly(p, [(21, 5), (12, 12), (21, 19)])


def _icon_forward(p: QPainter, color: QColor) -> None:
    _poly(p, [(12, 5), (21, 12), (12, 19)])
    _poly(p, [(3, 5), (12, 12), (3, 19)])


def _icon_folder(p: QPainter, color: QColor) -> None:
    _rrect(p, 3, 8, 18, 12, 2)
    _rrect(p, 3, 5, 8, 5, 1.5)


def _icon_download(p: QPainter, color: QColor) -> None:
    _rrect(p, 10.5, 3, 3, 9, 1)
    _poly(p, [(5.5, 11), (18.5, 11), (12, 18)])
    _rrect(p, 4, 19.5, 16, 2.5, 1)


def _icon_spark(p: QPainter, color: QColor) -> None:
    _poly(p, [(12, 2), (14.5, 9.5), (22, 12), (14.5, 14.5), (12, 22), (9.5, 14.5), (2, 12), (9.5, 9.5)])


def _icon_bolt(p: QPainter, color: QColor) -> None:
    _poly(p, [(13, 2), (5, 13.5), (11, 13.5), (10, 22), (19, 10), (13, 10)])


def _icon_refresh(p: QPainter, color: QColor) -> None:
    pen = QPen(color, 2.4)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawArc(QRectF(5, 5, 14, 14), 40 * 16, 280 * 16)
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    _poly(p, [(15.5, 2.5), (20.5, 8), (13, 9)])


def _icon_video(p: QPainter, color: QColor) -> None:
    _rrect(p, 3, 6, 13, 12, 2)
    _poly(p, [(17, 10), (22, 7), (22, 17), (17, 14)])


def _icon_trash(p: QPainter, color: QColor) -> None:
    _rrect(p, 6, 8, 12, 13, 1.5)
    _rrect(p, 4, 5, 16, 2, 1)
    _rrect(p, 9, 3, 6, 2.5, 1)


def _icon_copy(p: QPainter, color: QColor) -> None:
    _rrect(p, 8, 8, 12, 13, 2)
    pen = QPen(color, 1.8)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(QRectF(4, 3, 11, 13), 2, 2)


def _icon_zoom_in(p: QPainter, color: QColor) -> None:
    pen = QPen(color, 2.2)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QRectF(3.5, 3.5, 12.0, 12.0))
    p.drawLine(QPointF(14.0, 14.0), QPointF(20.5, 20.5))
    p.drawLine(QPointF(9.5, 6.8), QPointF(9.5, 12.2))
    p.drawLine(QPointF(6.8, 9.5), QPointF(12.2, 9.5))


def _icon_zoom_out(p: QPainter, color: QColor) -> None:
    pen = QPen(color, 2.2)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QRectF(3.5, 3.5, 12.0, 12.0))
    p.drawLine(QPointF(14.0, 14.0), QPointF(20.5, 20.5))
    p.drawLine(QPointF(6.8, 9.5), QPointF(12.2, 9.5))


def _icon_fit(p: QPainter, color: QColor) -> None:
    pen = QPen(color, 2.2)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPolyline(QPolygonF([QPointF(8, 5), QPointF(4, 5), QPointF(4, 19), QPointF(8, 19)]))
    p.drawPolyline(QPolygonF([QPointF(16, 5), QPointF(20, 5), QPointF(20, 19), QPointF(16, 19)]))
    p.drawLine(QPointF(8.5, 12), QPointF(15.5, 12))
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    _poly(p, [(8.0, 12), (11.0, 9.2), (11.0, 14.8)])
    _poly(p, [(16.0, 12), (13.0, 9.2), (13.0, 14.8)])


def _icon_follow(p: QPainter, color: QColor) -> None:
    pen = QPen(color, 2.2)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(7.0, 4.0), QPointF(7.0, 20.0))
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    _poly(p, [(11.5, 7.0), (21.0, 12.0), (11.5, 17.0)])


def _icon_prev_marker(p: QPainter, color: QColor) -> None:
    _rrect(p, 4, 5, 2.8, 14, 1)
    _poly(p, [(19.5, 5), (8.5, 12), (19.5, 19)])


def _icon_next_marker(p: QPainter, color: QColor) -> None:
    _rrect(p, 17.2, 5, 2.8, 14, 1)
    _poly(p, [(4.5, 5), (15.5, 12), (4.5, 19)])


_ICON_SHAPES = {
    "zoom_in": _icon_zoom_in, "zoom_out": _icon_zoom_out, "fit": _icon_fit, "follow": _icon_follow,
    "prev_marker": _icon_prev_marker, "next_marker": _icon_next_marker,
    "play": _icon_play, "pause": _icon_pause, "stop": _icon_stop, "back": _icon_back,
    "forward": _icon_forward, "folder": _icon_folder, "download": _icon_download,
    "spark": _icon_spark, "bolt": _icon_bolt, "refresh": _icon_refresh,
    "video": _icon_video, "trash": _icon_trash, "copy": _icon_copy,
}


class IconFactory:
    _cache: Dict[Tuple[str, str], QIcon] = {}

    @classmethod
    def icon(cls, name: str, color: Optional[str] = None) -> QIcon:
        color = color or PALETTE["text"]
        key = (name, color)
        if key not in cls._cache:
            icon = QIcon()
            try:
                icon.addPixmap(cls._pixmap(name, color), QIcon.Normal)
                icon.addPixmap(cls._pixmap(name, PALETTE["faint"]), QIcon.Disabled)
            except Exception:
                icon = QIcon()
            cls._cache[key] = icon
        return cls._cache[key]

    @staticmethod
    def _pixmap(name: str, color: str, size: int = 48) -> QPixmap:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.scale(size / 24.0, size / 24.0)
        qcolor = QColor(color)
        painter.setBrush(qcolor)
        painter.setPen(Qt.NoPen)
        _ICON_SHAPES[name](painter, qcolor)
        painter.end()
        return pixmap


def format_clock(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--:--.-"
    value = max(0.0, float(value))
    minutes, seconds = divmod(value, 60.0)
    return f"{int(minutes)}:{seconds:04.1f}"


def mix_color(base: str, other: str, t: float) -> QColor:
    a, b = QColor(base), QColor(other)
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )


# ----------------------------------------------------------------------------
# Monitor dos 6 LEDs
# ----------------------------------------------------------------------------

class LedMonitorWidget(QWidget):
    """Os 6 LEDs como orbes numa pílula de vidro. Parados: cinza com um leve tom da cor, sem efeito."""
    LABELS = ("E", "1", "2", "3", "4", "E")
    EDGE_COLOR = "#2fd27a"          # pontas: verdes
    IDLE_BODY = "#3a4553"           # cinza-azulado base dos orbes apagados
    IDLE_RIM = "#2a3441"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._levels = [0] * 6
        self.setMinimumHeight(98)

    def sizeHint(self) -> QSize:
        return QSize(250, 102)

    def set_levels(self, levels: Sequence[int]) -> None:
        values = [int(v) for v in list(levels)[:6]]
        values += [0] * (6 - len(values))
        self._levels = [max(0, min(255, v)) for v in values]
        self.update()

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)

        bg = QLinearGradient(0, rect.top(), 0, rect.bottom())
        bg.setColorAt(0.0, QColor("#1a2b40"))
        bg.setColorAt(0.5, QColor("#0f1d2e"))
        bg.setColorAt(1.0, QColor("#07111c"))
        painter.setPen(QPen(QColor("#253d58"), 1.2))
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(rect, 18, 18)

        hl = QLinearGradient(rect.left(), 0, rect.right(), 0)
        hl.setColorAt(0.0, QColor(80, 140, 210, 0))
        hl.setColorAt(0.35, QColor(100, 170, 240, 70))
        hl.setColorAt(0.65, QColor(100, 170, 240, 70))
        hl.setColorAt(1.0, QColor(80, 140, 210, 0))
        hl_pen = QPen()
        hl_pen.setBrush(QBrush(hl))
        hl_pen.setWidthF(1.5)
        painter.setPen(hl_pen)
        painter.drawLine(QPointF(rect.left() + 22, rect.top()), QPointF(rect.right() - 22, rect.top()))

        slot = rect.width() / 6.0
        radius = max(5.0, min(slot * 0.30, (rect.height() - 40.0) / 2.0))
        cy = rect.top() + 18.0 + radius          # orbes mais abaixo do título da seção
        font = painter.font()
        font.setPointSizeF(7.0)
        for idx, level in enumerate(self._levels):
            cx = rect.left() + slot * (idx + 0.5)
            center = QPointF(cx, cy)
            base = QColor(self.EDGE_COLOR if idx in (0, 5) else PALETTE["mid"])
            if level > 8:
                raw = level / 255.0
                intensity = min(1.0, (raw ** 0.70) * 1.42)
                glow = min(1.0, raw * 1.72 + 0.16)
                painter.setPen(Qt.NoPen)
                for ring in (3, 2, 1):
                    scale = 1.0 + ring * 0.26
                    halo = QColor(base)
                    halo.setAlpha(int(max(0, min(190, (0.08 + 0.20 * glow) * (4 - ring) * 86))))
                    painter.setBrush(halo)
                    painter.drawEllipse(center, radius * scale, radius * scale)
                painter.setBrush(QColor(base).lighter(int(135 + intensity * 80)))
                painter.drawEllipse(center, radius, radius)
                inner = QColor(base).lighter(int(180 + intensity * 70))
                inner.setAlpha(int(205 + intensity * 50))
                painter.setBrush(inner)
                painter.drawEllipse(center, radius * 0.62, radius * 0.62)
                rim = QColor(base).lighter(165)
                rim.setAlpha(int(130 + intensity * 95))
                painter.setPen(QPen(rim, 1.2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(center, radius, radius)
            else:
                # apagado: cinza liso com um toque da cor do LED — sem brilho, sem anel, sem animação
                painter.setPen(QPen(mix_color(self.IDLE_RIM, base.name(), 0.14), 1.0))
                painter.setBrush(mix_color(self.IDLE_BODY, base.name(), 0.16))
                painter.drawEllipse(center, radius, radius)
            painter.setFont(font)
            painter.setPen(QColor(90, 140, 190, 140))
            painter.drawText(QRectF(cx - 12, rect.bottom() - 17, 24, 13), Qt.AlignCenter, self.LABELS[idx])
        painter.end()


# ----------------------------------------------------------------------------
# Linha do tempo: 6 faixas (uma por LED), régua, transições, drops e playhead
# ----------------------------------------------------------------------------

class TimelineWidget(QWidget):
    """Partitura dos 6 LEDs com zoom, rolagem e navegação.

    Zoom:      roda do mouse (no ponto do cursor) · botões −/+/Ajustar · duplo clique
    Rolagem:   Shift+roda ou gesto horizontal · botão do meio/direito arrastando · setas ← →
    Minimapa:  barra inferior com o perfil de energia, transições e drops; arraste a janela para
               navegar e as bordas dela para dar zoom
    Pads:      textura de step sequencer (estilo FL Studio) sobre a grade de tempos/compassos
    Pontos:    o indicador de cada faixa pisca com o LED correspondente durante a reprodução
    """
    seek_requested = pyqtSignal(float)              # segundos
    view_changed = pyqtSignal(float, float, float)  # início, janela e duração (s)
    follow_changed = pyqtSignal(bool)

    GLOSSY = True

    LANE_NAMES = ("PONTA E", "LED 1", "LED 2", "LED 3", "LED 4", "PONTA D")
    GUTTER = 92.0
    RULER_H = 34.0
    OVERVIEW_H = 24.0
    LANE_GAP = 2.0
    MIN_SPAN_S = 3.0            # zoom máximo: janela de 3 s
    MIN_CELL_PX = 4.0           # largura mínima de cada pad
    PAD_LEVELS = 24             # níveis de brilho quantizados (cache dos gradientes)
    EDGE_GRAB_PX = 6.0
    ENVELOPE_BUCKETS = 480
    TICK_STEPS = (0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self._lanes: List[List[int]] = []
        self._columns = 0
        self._duration = 0.0
        self._fps = 30
        self._bpm = 0.0
        self._position = 0.0
        self._cue = 0.0
        self._live = [0] * 6
        self._transitions: List[Tuple[float, float, str]] = []
        self._drops: List[float] = []
        self._envelope: List[float] = []
        self._hover_x: Optional[float] = None
        self._seek_x: Optional[float] = None
        self._hover_marker: Optional[Tuple[str, int]] = None
        self._view_start = 0.0
        self._view_span = 0.0
        self._follow = True
        self._pan_anchor: Optional[Tuple[float, float]] = None
        self._ov_mode: Optional[str] = None
        self._ov_grab = 0.0
        self._serial = 0
        self._cache: Optional[QPixmap] = None
        self._cache_key: Optional[tuple] = None
        self._ov_cache: Optional[QPixmap] = None
        self._ov_key: Optional[tuple] = None

    # -- dados --------------------------------------------------------------
    def set_sequence(self, sequence: Optional[GeneratedSequence]) -> None:
        self._lanes = []
        self._transitions = []
        self._drops = []
        self._envelope = []
        self._position = 0.0
        self._cue = 0.0
        self._duration = 0.0
        self._columns = 0
        self._bpm = 0.0
        self._live = [0] * 6
        self._serial += 1
        self._cache = None
        self._cache_key = None
        if sequence is not None and sequence.preview_lanes:
            self._fps = max(1, sequence.fps)
            self._duration = max(0.0, float(sequence.duration_s))
            self._lanes = sequence.preview_lanes
            self._columns = len(sequence.preview_lanes)
            self._bpm = float(sequence.tempo_bpm or 0.0)
            for transition in sequence.transitions:
                start = float(getattr(transition, "frame_start", 0)) / self._fps
                end = float(getattr(transition, "frame_end", 0)) / self._fps
                name = getattr(getattr(transition, "effect_type", None), "name", "TRANS")
                self._transitions.append((start, end, name))
            self._drops = [float(d) for d in (sequence.report or {}).get("drops_s", [])]
            buckets = self.ENVELOPE_BUCKETS
            envelope = [0.0] * buckets
            for index, column in enumerate(self._lanes):
                bucket = min(buckets - 1, index * buckets // self._columns)
                energy = sum(column) / (6.0 * 255.0)
                if energy > envelope[bucket]:
                    envelope[bucket] = energy
            self._envelope = envelope
        self._view_start = 0.0
        self._view_span = self._duration
        self.set_follow(True)
        self._clamp_view()

    def set_position(self, seconds: float) -> None:
        self._position = max(0.0, float(seconds))
        self._chase_playhead()
        self.update()

    def set_cue(self, seconds: float) -> None:
        self._cue = max(0.0, float(seconds))
        self.update()

    def set_live_levels(self, levels: Sequence[int]) -> None:
        values = [max(0, min(255, int(v))) for v in list(levels)[:6]]
        values += [0] * (6 - len(values))
        if values == self._live:
            return
        self._live = values
        self.update(0, 0, int(self.GUTTER), self.height())      # só a coluna dos nomes

    # -- zoom / rolagem -----------------------------------------------------
    def is_zoomed(self) -> bool:
        return self._duration > 0 and self._view_span < self._duration - 1e-6

    def set_follow(self, flag: bool) -> None:
        flag = bool(flag)
        changed = flag != self._follow
        self._follow = flag
        if changed:
            self.follow_changed.emit(flag)
        if flag:
            self._chase_playhead()

    def _release_follow(self) -> None:
        if self._follow:
            self._follow = False
            self.follow_changed.emit(False)

    def _chase_playhead(self) -> None:
        if not (self._follow and self.is_zoomed()) or self._pan_anchor is not None or self._ov_mode is not None:
            return
        margin = self._view_span * 0.08
        if self._position > self._view_start + self._view_span - margin or self._position < self._view_start:
            self._view_start = self._position - margin
            self._clamp_view()

    def _clamp_view(self) -> None:
        if self._duration <= 0 or self._columns == 0:
            self._view_start = 0.0
            self._view_span = 0.0
        else:
            low = min(self.MIN_SPAN_S, self._duration)
            self._view_span = max(low, min(self._duration, self._view_span))
            self._view_start = max(0.0, min(self._duration - self._view_span, self._view_start))
        self.view_changed.emit(self._view_start, self._view_span, self._duration)
        self.update()

    def zoom_by(self, factor: float, anchor_x: Optional[float] = None) -> None:
        """factor < 1 aproxima; factor > 1 afasta. O ponto sob `anchor_x` fica parado."""
        if self._duration <= 0 or self._columns == 0:
            return
        rect = self._lanes_rect()
        if anchor_x is None:
            frac = 0.5
        else:
            frac = max(0.0, min(1.0, (anchor_x - rect.left()) / max(1.0, rect.width())))
        anchor_t = self._view_start + frac * self._view_span
        low = min(self.MIN_SPAN_S, self._duration)
        new_span = max(low, min(self._duration, self._view_span * factor))
        self._view_start = anchor_t - frac * new_span
        self._view_span = new_span
        self._clamp_view()

    def zoom_in(self) -> None:
        self.zoom_by(0.6)

    def zoom_out(self) -> None:
        self.zoom_by(1.0 / 0.6)

    def zoom_fit(self) -> None:
        self._view_start = 0.0
        self._view_span = self._duration
        self._clamp_view()
        self.set_follow(True)

    def pan_by(self, seconds: float) -> None:
        self._view_start += seconds
        self._release_follow()
        self._clamp_view()

    def reveal(self, seconds: float) -> None:
        """Garante que `seconds` esteja visível (sem mexer no modo de seguir)."""
        if self._view_span <= 0:
            return
        if seconds < self._view_start or seconds > self._view_start + self._view_span:
            self._view_start = seconds - self._view_span * 0.3
            self._clamp_view()

    # -- geometria ----------------------------------------------------------
    def _lanes_rect(self) -> QRectF:
        height = self.height() - self.RULER_H - self.OVERVIEW_H - 12.0
        return QRectF(self.GUTTER, self.RULER_H, max(10.0, self.width() - self.GUTTER - 10.0), max(10.0, height))

    def _lane_height(self, rect: QRectF) -> float:
        return (rect.height() - self.LANE_GAP * 5) / 6.0

    def _x_of(self, seconds: float) -> float:
        rect = self._lanes_rect()
        if self._view_span <= 0:
            return rect.left()
        return rect.left() + (seconds - self._view_start) / self._view_span * rect.width()

    def _seconds_of(self, x: float) -> float:
        rect = self._lanes_rect()
        if rect.width() <= 0 or self._view_span <= 0:
            return 0.0
        frac = max(0.0, min(1.0, (x - rect.left()) / rect.width()))
        return self._view_start + frac * self._view_span

    def _marker_at(self, x: float, y: float) -> Optional[Tuple[str, int]]:
        if y > self.RULER_H + 2:
            return None
        rect = self._lanes_rect()
        best: Optional[Tuple[str, int]] = None
        best_dist = 7.0
        candidates = [("t", i, start) for i, (start, _e, _n) in enumerate(self._transitions)]
        candidates += [("d", i, sec) for i, sec in enumerate(self._drops)]
        for kind, index, seconds in candidates:
            mx = self._x_of(seconds)
            if rect.left() - 4 <= mx <= rect.right() + 4 and abs(mx - x) < best_dist:
                best, best_dist = (kind, index), abs(mx - x)
        return best

    def _marker_seconds(self, key: Tuple[str, int]) -> float:
        kind, index = key
        return self._transitions[index][0] if kind == "t" else self._drops[index]

    def _overview_rect(self) -> QRectF:
        rect = self._lanes_rect()
        return QRectF(rect.left(), rect.bottom() + 6.0, rect.width(), self.OVERVIEW_H - 4.0)

    def _handle_rect(self) -> QRectF:
        ov = self._overview_rect()
        if self._duration <= 0:
            return QRectF(ov)
        width = min(ov.width(), max(10.0, self._view_span / self._duration * ov.width()))
        left = ov.left() + self._view_start / self._duration * ov.width()
        left = max(ov.left(), min(ov.right() - width, left))
        return QRectF(left, ov.top(), width, ov.height())

    def _in_overview(self, y: float) -> bool:
        return y >= self._overview_rect().top() - 4.0

    def _overview_zone(self, x: float) -> Tuple[str, float]:
        """Qual parte da janela do minimapa está sob x: borda esquerda 'l', direita 'r' ou corpo 'm'."""
        handle = self._handle_rect()
        grab = self.EDGE_GRAB_PX
        if handle.width() >= 3 * grab:
            if abs(x - handle.left()) <= grab:
                return "l", 0.0
            if abs(x - handle.right()) <= grab:
                return "r", 0.0
        if handle.left() - 1 <= x <= handle.right() + 1:
            return "m", x - handle.left()
        return "m", handle.width() / 2.0

    def _drag_overview(self, x: float) -> None:
        ov = self._overview_rect()
        if ov.width() <= 0 or self._duration <= 0:
            return
        low = min(self.MIN_SPAN_S, self._duration)
        t = max(0.0, min(self._duration, (x - ov.left()) / ov.width() * self._duration))
        if self._ov_mode == "m":
            self._view_start = (x - self._ov_grab - ov.left()) / ov.width() * self._duration
        elif self._ov_mode == "l":
            end = self._view_start + self._view_span
            start = min(t, end - low)
            self._view_start, self._view_span = start, end - start
        elif self._ov_mode == "r":
            end = max(t, self._view_start + low)
            self._view_span = end - self._view_start
        self._release_follow()
        self._clamp_view()

    def _update_hover_cursor(self, x: float, y: float) -> None:
        if self._in_overview(y):
            zone, _ = self._overview_zone(x)
            if zone in ("l", "r"):
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.OpenHandCursor if self.is_zoomed() else Qt.ArrowCursor)
        elif self._hover_marker is not None:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.unsetCursor()

    # -- mouse / teclado ----------------------------------------------------
    def wheelEvent(self, event: Any) -> None:
        if not self._columns:
            event.ignore()
            return
        delta = event.angleDelta()
        horizontal = abs(delta.x()) > abs(delta.y())
        if horizontal or event.modifiers() & Qt.ShiftModifier:
            amount = delta.x() if horizontal else delta.y()
            self.pan_by(-amount / 120.0 * self._view_span * 0.10)
        else:
            self.zoom_by(0.85 ** (delta.y() / 120.0), float(event.pos().x()))
        event.accept()

    def keyPressEvent(self, event: Any) -> None:
        if not self._columns or int(event.modifiers()) & ~int(Qt.KeypadModifier):
            super().keyPressEvent(event)
            return
        key = event.key()
        if key == Qt.Key_Left:
            self.pan_by(-self._view_span * 0.15)
        elif key == Qt.Key_Right:
            self.pan_by(self._view_span * 0.15)
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.zoom_out()
        elif key == Qt.Key_0:
            self.zoom_fit()
        elif key == Qt.Key_Home:
            self._view_start = 0.0
            self._release_follow()
            self._clamp_view()
        elif key == Qt.Key_End:
            self._view_start = self._duration
            self._release_follow()
            self._clamp_view()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        if not self._columns:
            return
        pos = event.pos()
        x, y = float(pos.x()), float(pos.y())
        if self._ov_mode is not None:
            self._drag_overview(x)
            return
        if self._pan_anchor is not None:
            anchor_x, anchor_start = self._pan_anchor
            rect = self._lanes_rect()
            self._view_start = anchor_start - (x - anchor_x) / max(1.0, rect.width()) * self._view_span
            self._release_follow()
            self._clamp_view()
            return
        self._hover_x = x
        marker = self._marker_at(x, y)
        if marker != self._hover_marker:
            self._hover_marker = marker
            if marker is not None:
                kind, index = marker
                if kind == "t":
                    start, end, name = self._transitions[index]
                    text = f"{name.title()}  ·  {format_clock(start)}  ·  duração {max(0.0, end - start):.2f}s"
                else:
                    text = f"DROP  ·  {format_clock(self._drops[index])}"
                QToolTip.showText(event.globalPos(), text)
            else:
                QToolTip.hideText()
        self._update_hover_cursor(x, y)
        if self._seek_x is not None:
            self._seek_x = x
        self.update()

    def mousePressEvent(self, event: Any) -> None:
        if not self._columns:
            return
        self.setFocus(Qt.MouseFocusReason)
        x, y = float(event.pos().x()), float(event.pos().y())
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_anchor = (x, self._view_start)
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        if self._in_overview(y):
            self._ov_mode, self._ov_grab = self._overview_zone(x)
            self.setCursor(Qt.SizeHorCursor if self._ov_mode in ("l", "r") else Qt.ClosedHandCursor)
            self._drag_overview(x)
            return
        rect = self._lanes_rect()
        if rect.left() <= x <= rect.right() and y <= rect.bottom():
            self._seek_x = x
            self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        x, y = float(event.pos().x()), float(event.pos().y())
        if event.button() in (Qt.MiddleButton, Qt.RightButton) and self._pan_anchor is not None:
            self._pan_anchor = None
            self._update_hover_cursor(x, y)
            return
        if event.button() != Qt.LeftButton:
            return
        if self._ov_mode is not None:
            self._ov_mode = None
            self._update_hover_cursor(x, y)
            return
        if self._seek_x is None or not self._columns:
            return
        marker = self._marker_at(x, y)
        seconds = self._marker_seconds(marker) if marker is not None else self._seconds_of(x)
        self._seek_x = None
        self.set_follow(True)
        self.update()
        self.seek_requested.emit(seconds)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if not self._columns or event.button() != Qt.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        x, y = float(event.pos().x()), float(event.pos().y())
        rect = self._lanes_rect()
        if self._in_overview(y) or (rect.left() <= x <= rect.right() and y <= rect.bottom()):
            if self.is_zoomed():
                self.zoom_fit()                       # duplo clique: ajustar à janela
            else:
                self.zoom_by(0.2, x if not self._in_overview(y) else None)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, _event: Any) -> None:
        self._hover_x = None
        self._hover_marker = None
        self.unsetCursor()
        self.update()

    # -- pintura ------------------------------------------------------------
    def _tick_step(self, px_per_second: float) -> float:
        for step in self.TICK_STEPS:
            if step * px_per_second >= 64.0:
                return step
        return self.TICK_STEPS[-1]

    @staticmethod
    def _tick_label(seconds: float, step: float) -> str:
        if step >= 1:
            return format_seconds(seconds)
        minutes, secs = divmod(seconds, 60.0)
        return f"{int(minutes)}:{secs:04.1f}"

    def _lanes_pixmap(self, rect: QRectF) -> QPixmap:
        """Faixas já pintadas (pads + grade). Só é refeito quando zoom/rolagem/tamanho mudam."""
        dpr = self.devicePixelRatioF()
        key = (int(rect.width()), int(rect.height()), round(self._view_start, 4), round(self._view_span, 4), dpr, self._serial)
        if self._cache is not None and self._cache_key == key:
            return self._cache
        pixmap = QPixmap(max(1, int(rect.width() * dpr)), max(1, int(rect.height() * dpr)))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        self._paint_lanes(painter, rect.width(), rect.height())
        painter.end()
        self._cache, self._cache_key = pixmap, key
        return pixmap

    def _paint_lanes(self, p: QPainter, w: float, h: float) -> None:
        gap = self.LANE_GAP
        lane_h = (h - gap * 5) / 6.0
        n = self._columns
        span = self._view_span
        if n <= 0 or span <= 0 or self._duration <= 0:
            return
        view_start = self._view_start
        view_end = view_start + span
        col_s = self._duration / n
        pps = w / span
        group = max(1, int(math.ceil(self.MIN_CELL_PX / max(1e-6, col_s * pps))))
        first = max(0, int(view_start / col_s) - group)
        first -= first % group
        last = min(n, int(view_end / col_s) + group + 1)

        beat_s = 60.0 / self._bpm if self._bpm > 1.0 else 0.0
        show_grid = beat_s > 0 and beat_s * pps >= 7.0
        bar_shade = QColor(255, 255, 255, 8)
        beat_line = QColor(255, 255, 255, 16)
        bar_line = QColor(255, 255, 255, 44)
        empty_pad = QColor(255, 255, 255, 10)
        if self.GLOSSY:
            lane_grad = QLinearGradient(0.0, 0.0, 0.0, lane_h)
            lane_grad.setColorAt(0.0, QColor("#0c1020"))
            lane_grad.setColorAt(1.0, QColor("#070a14"))
            lane_bg = QBrush(lane_grad)
        else:
            lane_bg = QBrush(QColor(PALETTE["bg0"]))
        levels = self.PAD_LEVELS
        brushes: Dict[Tuple[bool, int], QBrush] = {}

        def pad_brush(edge: bool, q: int) -> QBrush:
            cached = brushes.get((edge, q))
            if cached is None:
                base = QColor(PALETTE["edge"] if edge else PALETTE["mid"])
                alpha = 0.30 + 0.70 * (q / float(levels)) ** 0.75
                top_c, bottom_c = base.lighter(120), base.darker(150)
                top_c.setAlphaF(alpha)
                bottom_c.setAlphaF(alpha)
                grad = QLinearGradient(0.0, 1.0, 0.0, max(2.0, lane_h - 1.0))
                grad.setColorAt(0.0, top_c)
                grad.setColorAt(1.0, bottom_c)
                cached = QBrush(grad)
                brushes[(edge, q)] = cached
            return cached

        p.setRenderHint(QPainter.Antialiasing, False)
        for lane in range(6):
            edge = lane in (0, 5)
            p.save()
            p.translate(0.0, lane * (lane_h + gap))
            p.setClipRect(QRectF(0.0, 0.0, w, lane_h))
            p.fillRect(QRectF(0.0, 0.0, w, lane_h), lane_bg)
            if self.GLOSSY:
                p.fillRect(QRectF(0.0, 0.0, w, 1.0), QColor(0, 200, 240, 24))

            if show_grid:
                b0 = int(math.floor(view_start / beat_s))
                b1 = int(math.ceil(view_end / beat_s))
                for bar in range(b0 // 4, b1 // 4 + 1):
                    if bar % 2 == 1:
                        x0 = (bar * 4 * beat_s - view_start) * pps
                        p.fillRect(QRectF(x0, 0.0, 4 * beat_s * pps, lane_h), bar_shade)
                for beat in range(b0, b1 + 1):
                    x = (beat * beat_s - view_start) * pps
                    p.fillRect(QRectF(x, 0.0, 1.0, lane_h), bar_line if beat % 4 == 0 else beat_line)

            for c0 in range(first, last, group):
                c1 = min(n, c0 + group)
                level = 0
                for column in self._lanes[c0:c1]:
                    if column[lane] > level:
                        level = column[lane]
                x0 = (c0 * col_s - view_start) * pps
                x1 = (c1 * col_s - view_start) * pps
                cw = x1 - x0 - 1.0
                if cw < 1.0:
                    cw = max(0.8, x1 - x0)
                cell = QRectF(x0 + 0.5, 1.0, cw, lane_h - 2.0)
                t = max(0.0, min(1.0, level / 255.0))
                q = int(round(t * levels))
                if q <= 0:
                    p.fillRect(cell, empty_pad)
                    continue
                p.fillRect(cell, pad_brush(edge, q))
                p.fillRect(QRectF(cell.left(), cell.top(), cell.width(), 1.0), QColor(255, 255, 255, int(70 * t)))
            p.restore()

    def _overview_pixmap(self, ov: QRectF) -> QPixmap:
        """Minimapa: perfil de energia da música + transições e drops."""
        dpr = self.devicePixelRatioF()
        key = (int(ov.width()), int(ov.height()), dpr, self._serial)
        if self._ov_cache is not None and self._ov_key == key:
            return self._ov_cache
        w, h = max(1, int(ov.width())), max(1, int(ov.height()))
        pixmap = QPixmap(int(w * dpr), int(h * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(QRectF(0, 0, w, h), QColor(0, 0, 0, 120))
        if self._envelope and self._duration > 0:
            count = len(self._envelope)
            for px in range(w):
                energy = self._envelope[min(count - 1, int(px / float(w) * count))]
                bar_h = max(1.0, (h - 3.0) * min(1.0, energy * 1.15))
                p.fillRect(QRectF(px, h - 1.0 - bar_h, 1.0, bar_h), QColor(40, 200, 248, 150))
            for start, _end, _name in self._transitions:
                p.fillRect(QRectF(int(start / self._duration * w), 0.0, 1.0, h), QColor(245, 158, 11, 190))
            for seconds in self._drops:
                p.fillRect(QRectF(int(seconds / self._duration * w) - 1, 0.0, 2.0, h), QColor(167, 139, 250, 220))
        p.end()
        self._ov_cache, self._ov_key = pixmap, key
        return pixmap

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self._lanes_rect()
        font = painter.font()
        font.setPointSizeF(8.0)
        painter.setFont(font)

        if not self._columns:
            painter.setPen(QColor(PALETTE["dim"]))
            painter.drawText(QRectF(self.rect()), Qt.AlignCenter,
                             "1  Carregue um áudio      2  Gere a sequência      3  Reproduza (Espaço)")
            painter.end()
            return

        # faixas (pads + grade em cache)
        painter.drawPixmap(rect.topLeft(), self._lanes_pixmap(rect))

        # indicadores das faixas: piscam com o LED correspondente
        lane_h = self._lane_height(rect)
        for lane in range(6):
            top = rect.top() + lane * (lane_h + self.LANE_GAP)
            base = QColor(PALETTE["edge"] if lane in (0, 5) else PALETTE["mid"])
            t = (self._live[lane] / 255.0) ** 0.7
            center = QPointF(17.0, top + lane_h / 2.0)
            radius = 5.0 + 1.3 * t
            painter.setPen(Qt.NoPen)
            if t > 0.03:
                for scale, alpha in ((2.5, 0.14), (1.8, 0.30)):
                    halo = QColor(base)
                    halo.setAlphaF(alpha * t)
                    painter.setBrush(halo)
                    painter.drawEllipse(center, radius * scale, radius * scale)
            unlit = mix_color(PALETTE["bg0"], base.name(), 0.38)
            painter.setBrush(mix_color(unlit.name(), base.lighter(125).name(), t))
            rim = base.lighter(150)
            rim.setAlpha(int(70 + 150 * t))
            painter.setPen(QPen(rim, 1.1))
            painter.drawEllipse(center, radius, radius)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, int(40 + 150 * t)))
            painter.drawEllipse(QPointF(center.x() - radius * 0.30, center.y() - radius * 0.32), radius * 0.28, radius * 0.28)
            painter.setPen(mix_color(PALETTE["dim"], PALETTE["text"], t))
            painter.drawText(QRectF(34.0, top, self.GUTTER - 38.0, lane_h), Qt.AlignVCenter | Qt.AlignLeft, self.LANE_NAMES[lane])

        # régua
        view_end = self._view_start + self._view_span
        pps = rect.width() / self._view_span if self._view_span > 0 else 1.0
        step = self._tick_step(pps)
        painter.setPen(QPen(QColor(PALETTE["border2"]), 1))
        painter.drawLine(QPointF(rect.left(), self.RULER_H - 2), QPointF(rect.right(), self.RULER_H - 2))
        for k in range(int(math.floor(self._view_start / step)), int(math.ceil(view_end / step)) + 1):
            t_s = k * step
            x = self._x_of(t_s)
            if x < rect.left() - 0.5 or x > rect.right() + 0.5:
                continue
            painter.setPen(QPen(QColor(PALETTE["border2"]), 1))
            painter.drawLine(QPointF(x, self.RULER_H - 7), QPointF(x, self.RULER_H - 2))
            label = self._tick_label(t_s, step)
            if x + 3 + painter.fontMetrics().horizontalAdvance(label) > self.width() - 2:
                continue
            painter.setPen(QColor(PALETTE["dim"]))
            painter.drawText(QRectF(x + 3, 3, 56, 13), Qt.AlignLeft | Qt.AlignVCenter, label)

        # transições
        xs = [self._x_of(start) for start, _e, _n in self._transitions]
        for index, (start, end, name) in enumerate(self._transitions):
            x = xs[index]
            x2 = self._x_of(end)
            if x2 < rect.left() or x > rect.right() + 1:
                continue
            band = QColor(PALETTE["accent"])
            band.setAlphaF(0.10)
            bx0, bx1 = max(rect.left(), x), min(rect.right(), x2)
            painter.setPen(Qt.NoPen)
            painter.setBrush(band)
            painter.drawRect(QRectF(bx0, rect.top(), max(1.0, bx1 - bx0), rect.height()))
            if x < rect.left() - 0.5:
                continue
            painter.setPen(QPen(QColor(PALETTE["accent"]), 1, Qt.DashLine))
            painter.drawLine(QPointF(x, self.RULER_H - 2), QPointF(x, rect.bottom()))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(PALETTE["accent"]))
            _poly(painter, [(x - 4, self.RULER_H - 10), (x + 4, self.RULER_H - 10), (x, self.RULER_H - 3)])
            next_x = xs[index + 1] if index + 1 < len(xs) else rect.right() + 100.0
            if next_x - x > 74.0:
                painter.setPen(QColor(PALETTE["accent"]))
                painter.drawText(QRectF(x + 6, 18, 70, 13), Qt.AlignLeft | Qt.AlignVCenter, name.title())

        # drops
        for seconds in self._drops:
            x = self._x_of(seconds)
            if x < rect.left() - 0.5 or x > rect.right() + 0.5:
                continue
            painter.setPen(QPen(QColor(PALETTE["drop"]), 1.6))
            painter.drawLine(QPointF(x, self.RULER_H - 2), QPointF(x, rect.bottom()))
            painter.setPen(QColor(PALETTE["drop"]))
            painter.drawText(QRectF(x - 46, 18, 42, 13), Qt.AlignRight | Qt.AlignVCenter, "DROP")

        # hover / seek
        guide_x = self._seek_x if self._seek_x is not None else self._hover_x
        if guide_x is not None and rect.left() <= guide_x <= rect.right() and self._ov_mode is None:
            painter.setPen(QPen(QColor(PALETTE["text"]), 1, Qt.DotLine))
            painter.drawLine(QPointF(guide_x, self.RULER_H - 2), QPointF(guide_x, rect.bottom()))
            label = format_clock(self._seconds_of(guide_x))
            painter.fillRect(QRectF(guide_x + 4, rect.bottom() - 18, 56, 15), QColor(PALETTE["bg3"]))
            painter.setPen(QColor(PALETTE["text"]))
            painter.drawText(QRectF(guide_x + 4, rect.bottom() - 18, 56, 15), Qt.AlignCenter, label)

        # ponto de início (cue): onde a reprodução começa quando está parada
        if self._cue > 0.01:
            cx = self._x_of(self._cue)
            if rect.left() - 1 <= cx <= rect.right() + 1:
                ok = QColor(PALETTE["ok"])
                painter.setPen(QPen(ok, 1.2, Qt.DashLine))
                painter.drawLine(QPointF(cx, self.RULER_H - 2), QPointF(cx, rect.bottom()))
                painter.setPen(Qt.NoPen)
                painter.setBrush(ok)
                _poly(painter, [(cx, self.RULER_H - 14), (cx + 9, self.RULER_H - 10), (cx, self.RULER_H - 6)])

        # playhead
        px = self._x_of(self._position)
        if rect.left() - 1 <= px <= rect.right() + 1:
            if self.GLOSSY:
                for width, alpha in ((7.0, 22), (4.0, 38)):
                    painter.setPen(QPen(QColor(240, 248, 255, alpha), width))
                    painter.drawLine(QPointF(px, self.RULER_H - 2), QPointF(px, rect.bottom()))
                head = QColor("#eef4fc")
            else:
                head = QColor(PALETTE["accent2"])
            painter.setPen(QPen(head, 1.8))
            painter.drawLine(QPointF(px, self.RULER_H - 2), QPointF(px, rect.bottom()))
            painter.setPen(Qt.NoPen)
            painter.setBrush(head)
            _poly(painter, [(px - 5, self.RULER_H - 12), (px + 5, self.RULER_H - 12), (px, self.RULER_H - 3)])

        # minimapa de navegação
        ov = self._overview_rect()
        painter.drawPixmap(ov.topLeft(), self._overview_pixmap(ov))
        handle = self._handle_rect()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.drawRect(QRectF(ov.left(), ov.top(), max(0.0, handle.left() - ov.left()), ov.height()))
        painter.drawRect(QRectF(handle.right(), ov.top(), max(0.0, ov.right() - handle.right()), ov.height()))
        accent = QColor(PALETTE["accent2"])
        fill = QColor(accent)
        fill.setAlphaF(0.22 if self._ov_mode is not None else (0.12 if self.is_zoomed() else 0.05))
        outline = QColor(accent)
        outline.setAlphaF(0.95 if self.is_zoomed() else 0.35)
        painter.setPen(QPen(outline, 1.2))
        painter.setBrush(fill)
        painter.drawRoundedRect(handle.adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)
        if handle.width() >= 18 and self.is_zoomed():
            painter.setPen(QPen(outline, 1.6, Qt.SolidLine, Qt.RoundCap))
            mid = handle.center().y()
            for gx in (handle.left() + 3.5, handle.right() - 3.5):
                painter.drawLine(QPointF(gx, mid - 3.5), QPointF(gx, mid + 3.5))
        if self._duration > 0:
            ox = ov.left() + max(0.0, min(1.0, self._position / self._duration)) * ov.width()
            painter.setPen(QPen(QColor("#eef4fc"), 1.4))
            painter.drawLine(QPointF(ox, ov.top()), QPointF(ox, ov.bottom()))
        painter.end()


class SeekBar(QWidget):
    """Barra de progresso do player: clique/arraste para navegar, com marcas de transições e drops."""
    seek_requested = pyqtSignal(float)              # segundos

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self._duration = 0.0
        self._position = 0.0
        self._cue = 0.0
        self._transitions: List[float] = []
        self._drops: List[float] = []
        self._hover_x: Optional[float] = None
        self._drag_x: Optional[float] = None

    def set_sequence(self, sequence: Optional[GeneratedSequence]) -> None:
        self._transitions = []
        self._drops = []
        self._position = 0.0
        self._cue = 0.0
        self._duration = 0.0
        if sequence is not None:
            fps = float(max(1, sequence.fps))
            self._duration = max(0.0, float(sequence.duration_s))
            self._transitions = [float(getattr(t, "frame_start", 0)) / fps for t in sequence.transitions]
            self._drops = [float(d) for d in (sequence.report or {}).get("drops_s", [])]
        self.update()

    def set_position(self, seconds: float) -> None:
        self._position = max(0.0, float(seconds))
        self.update()

    def set_cue(self, seconds: float) -> None:
        self._cue = max(0.0, float(seconds))
        self.update()

    def _track(self) -> QRectF:
        return QRectF(12.0, self.height() - 15.0, max(10.0, self.width() - 24.0), 6.0)

    def _x_of(self, seconds: float) -> float:
        track = self._track()
        if self._duration <= 0:
            return track.left()
        return track.left() + max(0.0, min(1.0, seconds / self._duration)) * track.width()

    def _seconds_of(self, x: float) -> float:
        track = self._track()
        return max(0.0, min(1.0, (x - track.left()) / max(1.0, track.width()))) * self._duration

    def mouseMoveEvent(self, event: Any) -> None:
        self._hover_x = float(event.pos().x())
        if self._drag_x is not None:
            self._drag_x = self._hover_x
        self.update()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self._duration > 0:
            self._drag_x = float(event.pos().x())
            self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self._drag_x is not None:
            seconds = self._seconds_of(float(event.pos().x()))
            self._drag_x = None
            self.update()
            self.seek_requested.emit(seconds)

    def leaveEvent(self, _event: Any) -> None:
        self._hover_x = None
        self.update()

    def paintEvent(self, _event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        track = self._track()
        active = self._duration > 0

        # ranhura
        p.setPen(QPen(QColor(255, 255, 255, 16), 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(track.adjusted(-0.6, -0.6, 0.6, 1.2), 3.5, 3.5)
        groove = QLinearGradient(track.topLeft(), track.bottomLeft())
        groove.setColorAt(0.0, QColor("#010204"))
        groove.setColorAt(1.0, QColor("#0e1521"))
        p.setPen(QPen(QColor("#04070c"), 1.0))
        p.setBrush(QBrush(groove))
        p.drawRoundedRect(track, 3, 3)
        if not active:
            p.end()
            return

        # marcas: transições (âmbar) e drops (violeta)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(245, 158, 11, 200))
        for seconds in self._transitions:
            p.drawRect(QRectF(self._x_of(seconds) - 0.75, track.top() - 5.0, 1.5, 4.0))
        p.setBrush(QColor(PALETTE["drop"]))
        for seconds in self._drops:
            p.drawRect(QRectF(self._x_of(seconds) - 1.0, track.top() - 8.0, 2.0, 7.0))

        shown = self._seconds_of(self._drag_x) if self._drag_x is not None else self._position
        px = self._x_of(shown)

        # progresso luminoso
        fill = QRectF(track.left(), track.top() + 1.0, max(0.0, px - track.left()), track.height() - 2.0)
        if fill.width() > 0.5:
            grad = QLinearGradient(fill.topLeft(), fill.topRight())
            grad.setColorAt(0.0, QColor("#0891b2"))
            grad.setColorAt(1.0, QColor("#7defff"))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 212, 255, 46))
            p.drawRoundedRect(fill.adjusted(0.0, -2.0, 0.0, 2.0), 4, 4)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(fill, 2.5, 2.5)

        # ponto de início (cue)
        if self._cue > 0.01:
            cx = self._x_of(self._cue)
            ok = QColor(PALETTE["ok"])
            p.setPen(Qt.NoPen)
            p.setBrush(ok)
            _poly(p, [(cx - 4, track.bottom() + 6.0), (cx + 4, track.bottom() + 6.0), (cx, track.bottom() + 1.5)])

        # alça
        hot = self._drag_x is not None or (self._hover_x is not None and abs(self._hover_x - px) < 12.0)
        radius = 8.0 if hot else 6.5
        center = QPointF(px, track.center().y())
        p.setPen(Qt.NoPen)
        for scale, alpha in ((2.2, 26), (1.6, 44)):
            p.setBrush(QColor(0, 212, 255, alpha + (20 if hot else 0)))
            p.drawEllipse(center, radius * scale, radius * scale)
        knob = QLinearGradient(center.x(), center.y() - radius, center.x(), center.y() + radius)
        knob.setColorAt(0.0, QColor("#ffffff"))
        knob.setColorAt(1.0, QColor("#9fb0c4"))
        p.setPen(QPen(QColor("#00d4ff"), 1.8))
        p.setBrush(QBrush(knob))
        p.drawEllipse(center, radius, radius)

        # balão de tempo (hover ou arraste)
        bubble_x = self._drag_x if self._drag_x is not None else self._hover_x
        if bubble_x is not None:
            label = format_clock(self._seconds_of(bubble_x))
            font = p.font()
            font.setPointSizeF(8.0)
            font.setBold(True)
            p.setFont(font)
            width = p.fontMetrics().horizontalAdvance(label) + 14.0
            left = max(2.0, min(self.width() - width - 2.0, bubble_x - width / 2.0))
            box = QRectF(left, 1.0, width, 16.0)
            p.setPen(QPen(QColor(0, 180, 220, 130), 1.0))
            p.setBrush(QColor("#0b0f1e"))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(QColor("#e8f4ff"))
            p.drawText(box, Qt.AlignCenter, label)
        p.end()


# ----------------------------------------------------------------------------
# Controles de parâmetro
# ----------------------------------------------------------------------------

class FaderSlider(QSlider):
    """Fader de mesa de som desenhado à mão: ranhura rebaixada, escala graduada, preenchimento
    luminoso, marca do valor padrão e knob metálico com estrias e linha indicadora.
    Mantém a API do QSlider (valueChanged, setRange, setValue...)."""

    def __init__(self, orientation: Qt.Orientation, parent: Optional[QWidget] = None) -> None:
        super().__init__(orientation, parent)
        self._default_raw: Optional[int] = None
        self._grab: Optional[float] = None
        self._hover = False
        self.setMouseTracking(True)
        if self._vertical():
            self._cap_long, self._cap_short = 24.0, 38.0     # altura x largura do knob
            self.setFixedWidth(62)
            self.setMinimumHeight(110)
        else:
            self._cap_long, self._cap_short = 16.0, 24.0     # largura x altura do knob
            self.setFixedHeight(28)
            self.setMinimumWidth(90)

    # -- geometria ----------------------------------------------------------
    def _vertical(self) -> bool:
        return self.orientation() == Qt.Vertical

    def set_default_raw(self, raw: int) -> None:
        self._default_raw = int(raw)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(62, 150) if self._vertical() else QSize(160, 28)

    def minimumSizeHint(self) -> QSize:
        return QSize(62, 90) if self._vertical() else QSize(90, 28)

    def _travel(self) -> Tuple[float, float]:
        """Posição (em px, ao longo do eixo) do centro do knob no mínimo e no máximo."""
        pad = self._cap_long / 2.0 + 3.0
        if self._vertical():
            return float(self.height()) - pad, pad
        return pad, float(self.width()) - pad

    def _pos_of(self, raw: float) -> float:
        a, b = self._travel()
        lo, hi = self.minimum(), self.maximum()
        frac = (raw - lo) / float(hi - lo) if hi > lo else 0.0
        return a + (b - a) * max(0.0, min(1.0, frac))

    def _raw_at(self, axis: float) -> int:
        a, b = self._travel()
        span = b - a
        if abs(span) < 1e-6:
            return self.minimum()
        frac = max(0.0, min(1.0, (axis - a) / span))
        return int(round(self.minimum() + frac * (self.maximum() - self.minimum())))

    def _cap_rect(self) -> QRectF:
        c = self._pos_of(self.value())
        if self._vertical():
            cx = self.width() / 2.0
            return QRectF(cx - self._cap_short / 2.0, c - self._cap_long / 2.0, self._cap_short, self._cap_long)
        cy = self.height() / 2.0
        return QRectF(c - self._cap_long / 2.0, cy - self._cap_short / 2.0, self._cap_long, self._cap_short)

    # -- mouse --------------------------------------------------------------
    def _axis(self, pos: Any) -> float:
        return float(pos.y()) if self._vertical() else float(pos.x())

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        axis = self._axis(event.pos())
        cap = self._cap_rect()
        center = cap.center().y() if self._vertical() else cap.center().x()
        if cap.contains(QPointF(event.pos())):
            self._grab = axis - center
        else:
            self._grab = 0.0
            self.setValue(self._raw_at(axis))
        self.setSliderDown(True)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        if self._grab is not None:
            self.setValue(self._raw_at(self._axis(event.pos()) - self._grab))
            event.accept()
            return
        hover = self._cap_rect().contains(QPointF(event.pos()))
        if hover != self._hover:
            self._hover = hover
            self.setCursor(Qt.PointingHandCursor if hover else Qt.ArrowCursor)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self._grab is not None:
            self._grab = None
            self.setSliderDown(False)
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: Any) -> None:
        if self._hover:
            self._hover = False
            self.update()
        super().leaveEvent(event)

    # -- pintura ------------------------------------------------------------
    def paintEvent(self, _event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        vertical = self._vertical()
        enabled = self.isEnabled()
        a, b = self._travel()
        cx, cy = self.width() / 2.0, self.height() / 2.0
        thick = 9.0 if vertical else 7.0
        if vertical:
            slot = QRectF(cx - thick / 2.0, b - 3.0, thick, a - b + 6.0)
        else:
            slot = QRectF(a - 3.0, cy - thick / 2.0, b - a + 6.0, thick)
        cap = self._cap_rect()
        cap_c = self._pos_of(self.value())

        if vertical:
            self._paint_scale(p, cx, a, b, cap_c, enabled)

        # moldura metálica da ranhura (placa do fader)
        bezel = slot.adjusted(-4.5, -4.5, 4.5, 4.5)
        plate = QLinearGradient(bezel.topLeft(), bezel.topRight() if vertical else bezel.bottomLeft())
        plate.setColorAt(0.0, QColor("#3a4556"))
        plate.setColorAt(0.45, QColor("#1c2330"))
        plate.setColorAt(1.0, QColor("#0d121a"))
        p.setPen(QPen(QColor("#04070c"), 1.0))
        p.setBrush(QBrush(plate))
        p.drawRoundedRect(bezel, bezel.width() / 2.0 if vertical else bezel.height() / 2.0,
                          bezel.width() / 2.0 if vertical else bezel.height() / 2.0)
        p.setPen(QPen(QColor(255, 255, 255, 34), 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(bezel.adjusted(1.0, 1.0, -1.0, -1.0), bezel.width() / 2.0 - 1.0 if vertical else bezel.height() / 2.0 - 1.0,
                          bezel.width() / 2.0 - 1.0 if vertical else bezel.height() / 2.0 - 1.0)

        # ranhura rebaixada
        groove = QLinearGradient(slot.topLeft(), slot.topRight() if vertical else slot.bottomLeft())
        groove.setColorAt(0.0, QColor("#010204"))
        groove.setColorAt(0.5, QColor("#05080d"))
        groove.setColorAt(1.0, QColor("#0b111a"))
        p.setPen(QPen(QColor("#020305"), 1.0))
        p.setBrush(QBrush(groove))
        p.drawRoundedRect(slot, thick / 2.0, thick / 2.0)

        # preenchimento luminoso do mínimo até o knob
        if vertical:
            fill = QRectF(slot.left() + 2.0, cap_c, slot.width() - 4.0, max(0.0, a - cap_c + 1.0))
            fill_grad = QLinearGradient(0.0, b, 0.0, a)
            glow = fill.adjusted(-2.0, 0.0, 2.0, 0.0)
        else:
            fill = QRectF(a - 1.0, slot.top() + 2.0, max(0.0, cap_c - a + 1.0), slot.height() - 4.0)
            fill_grad = QLinearGradient(a, 0.0, b, 0.0)
            glow = fill.adjusted(0.0, -2.0, 0.0, 2.0)
        if enabled:
            fill_grad.setColorAt(0.0 if vertical else 1.0, QColor("#7defff"))
            fill_grad.setColorAt(1.0 if vertical else 0.0, QColor("#0891b2"))
            glow_color = QColor(0, 212, 255, 46)
        else:
            fill_grad.setColorAt(0.0, QColor(90, 100, 115))
            fill_grad.setColorAt(1.0, QColor(60, 68, 80))
            glow_color = QColor(0, 0, 0, 0)
        if fill.width() > 0 and fill.height() > 0:
            p.setPen(Qt.NoPen)
            p.setBrush(glow_color)
            p.drawRoundedRect(glow, 3, 3)
            p.setBrush(QBrush(fill_grad))
            p.drawRoundedRect(fill, 2, 2)

        # marca do valor padrão (entalhe âmbar na moldura)
        if self._default_raw is not None:
            dpos = self._pos_of(self._default_raw)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(251, 191, 36, 220))
            if vertical:
                p.drawRect(QRectF(bezel.left() - 3.0, dpos - 0.75, 3.0, 1.5))
                p.drawRect(QRectF(bezel.right(), dpos - 0.75, 3.0, 1.5))
            else:
                p.drawRect(QRectF(dpos - 0.75, bezel.top() - 3.0, 1.5, 3.0))
                p.drawRect(QRectF(dpos - 0.75, bezel.bottom(), 1.5, 3.0))

        self._paint_cap(p, cap, vertical, enabled)
        p.end()

    def _paint_scale(self, p: QPainter, cx: float, a: float, b: float, cap_c: float, enabled: bool) -> None:
        """Escala graduada à esquerda e coluna de LEDs segmentados à direita (fora da largura do knob)."""
        xl = cx - self._cap_short / 2.0 - 3.0
        for i in range(11):
            pos = a + (b - a) * i / 10.0
            major = i % 5 == 0
            length = 8.0 if major else 4.0
            p.setPen(QPen(QColor(150, 190, 230, 165 if major else 80), 1.0))
            p.drawLine(QPointF(xl - length, pos), QPointF(xl, pos))
        segments = 18
        seg_h = (a - b) / segments
        lx = cx + self._cap_short / 2.0 + 4.0
        p.setPen(Qt.NoPen)
        for i in range(segments):
            bottom = a - i * seg_h
            seg = QRectF(lx, bottom - seg_h + 1.0, 4.0, seg_h - 2.0)
            frac = (i + 0.5) / segments
            lit = enabled and cap_c <= bottom - seg_h / 2.0
            if lit:
                color = QColor("#22d3ee" if frac < 0.62 else "#67e8f9" if frac < 0.80 else "#fbbf24" if frac < 0.92 else "#f97316")
                halo = QColor(color)
                halo.setAlpha(46)
                p.setBrush(halo)
                p.drawRoundedRect(seg.adjusted(-1.5, -1.0, 1.5, 1.0), 2.0, 2.0)
                p.setBrush(color)
            else:
                p.setBrush(QColor(255, 255, 255, 16))
            p.drawRoundedRect(seg, 1.0, 1.0)

    def _paint_cap(self, p: QPainter, r: QRectF, vertical: bool, enabled: bool) -> None:
        pressed = self._grab is not None
        # sombra projetada em camadas (o knob "afunda" quando pressionado)
        drop = 1.5 if pressed else 3.5
        p.setPen(Qt.NoPen)
        for grow, alpha in ((6.0, 22), (3.5, 34), (1.5, 56)):
            p.setBrush(QColor(0, 0, 0, alpha))
            p.drawRoundedRect(r.adjusted(-grow, -grow + drop, grow, grow + drop), 5.0 + grow, 5.0 + grow)
        # corpo metálico escovado (luz vindo de cima)
        body = QLinearGradient(r.topLeft(), r.bottomLeft())
        if enabled:
            for stop, color in ((0.00, "#f7f9fc"), (0.08, "#dbe2ec"), (0.30, "#aab5c4"),
                                (0.50, "#7d8998"), (0.56, "#8d99a8"), (1.00, "#3c4552")):
                body.setColorAt(stop, QColor(color))
        else:
            for stop, color in ((0.0, "#5b6470"), (0.5, "#3b434e"), (1.0, "#262c35")):
                body.setColorAt(stop, QColor(color))
        p.setPen(QPen(QColor("#080b10"), 1.0))
        p.setBrush(QBrush(body))
        p.drawRoundedRect(r, 5.0, 5.0)
        # chanfro: luz no alto/esquerda, sombra embaixo
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
        p.drawRoundedRect(r.adjusted(1.0, 1.0, -1.0, -1.0), 4.0, 4.0)
        p.setPen(QPen(QColor(0, 0, 0, 70), 1.0))
        p.drawLine(QPointF(r.left() + 3.0, r.bottom() - 1.5), QPointF(r.right() - 3.0, r.bottom() - 1.5))
        # almofada de pegada côncava (borracha escura) com estrias
        pad = r.adjusted(5.0, 4.5, -5.0, -4.5) if vertical else r.adjusted(4.5, 5.0, -4.5, -5.0)
        grip = QLinearGradient(pad.topLeft(), pad.bottomLeft() if vertical else pad.topRight())
        for stop, color in ((0.0, "#262d38"), (0.5, "#0a0d13"), (1.0, "#262d38")):
            grip.setColorAt(stop, QColor(color))
        p.setPen(QPen(QColor(0, 0, 0, 170), 1.0))
        p.setBrush(QBrush(grip))
        p.drawRoundedRect(pad, 3.0, 3.0)
        center = r.center()
        for k in (-0.30, -0.19, 0.19, 0.30):
            off = k * self._cap_long
            if vertical:
                y = center.y() + off
                line_a = (QPointF(pad.left() + 3.0, y), QPointF(pad.right() - 3.0, y))
                line_b = (QPointF(pad.left() + 3.0, y + 1.0), QPointF(pad.right() - 3.0, y + 1.0))
            else:
                x = center.x() + off
                line_a = (QPointF(x, pad.top() + 3.0), QPointF(x, pad.bottom() - 3.0))
                line_b = (QPointF(x + 1.0, pad.top() + 3.0), QPointF(x + 1.0, pad.bottom() - 3.0))
            p.setPen(QPen(QColor(0, 0, 0, 150), 1.0))
            p.drawLine(*line_a)
            p.setPen(QPen(QColor(255, 255, 255, 34), 1.0))
            p.drawLine(*line_b)
        if (self._hover or pressed) and enabled:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 26 if pressed else 14))
            p.drawRoundedRect(r.adjusted(1.0, 1.0, -1.0, -1.0), 4.0, 4.0)
        # linha indicadora central, atravessando o knob
        if vertical:
            line = (QPointF(r.left() + 2.5, center.y()), QPointF(r.right() - 2.5, center.y()))
        else:
            line = (QPointF(center.x(), r.top() + 2.5), QPointF(center.x(), r.bottom() - 2.5))
        if enabled:
            p.setPen(QPen(QColor(251, 191, 36, 150 if pressed else 80), 4.5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(*line)
            p.setPen(QPen(QColor("#fff3c4"), 1.7, Qt.SolidLine, Qt.RoundCap))
        else:
            p.setPen(QPen(QColor(120, 128, 140), 1.7, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(*line)


class ResetSlider(FaderSlider):
    """Fader que volta ao padrão com duplo clique (como numa mesa de som)."""
    reset_requested = pyqtSignal()

    def mouseDoubleClickEvent(self, event: Any) -> None:
        self.reset_requested.emit()
        event.accept()


class FaderStrip(QFrame):
    value_changed = pyqtSignal(str, float)

    def __init__(self, spec: ControlSpec, help_text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Strip")
        self.spec = spec
        self.setProperty("recommended", False)
        self.setFixedWidth(84)
        self.setToolTip(f"{help_text}\n\nDuplo clique no fader restaura o padrão ({spec.format_value(spec.default)}).")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(6)
        self.name_label = QLabel(spec.label)
        self.name_label.setObjectName("StripName")
        self.name_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.name_label.setWordWrap(True)
        self.name_label.setMinimumHeight(30)
        layout.addWidget(self.name_label)
        self.slider = ResetSlider(Qt.Vertical)
        low, high = spec.slider_range()
        self.slider.setRange(low, high)
        self.slider.set_default_raw(spec.to_slider(spec.default))
        self.slider.setMinimumHeight(110)
        layout.addWidget(self.slider, 1, Qt.AlignHCenter)
        self.value_label = QLabel()
        self.value_label.setObjectName("StripValue")
        self.value_label.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.reset_requested.connect(self.reset)
        self.set_value(spec.default)

    def value(self) -> float:
        return self.spec.from_slider(self.slider.value())

    def set_value(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self.spec.to_slider(value))
        self.slider.blockSignals(False)
        self._refresh_label()

    def reset(self) -> None:
        self.set_value(self.spec.default)
        self.set_recommended(False)
        self.value_changed.emit(self.spec.key, self.value())

    def set_recommended(self, flag: bool) -> None:
        self.setProperty("recommended", bool(flag))
        refresh_style(self)

    def _refresh_label(self) -> None:
        current = self.value()
        self.value_label.setText(self.spec.format_value(current))
        self.value_label.setProperty("modified", abs(current - self.spec.default) > 1e-9)
        refresh_style(self.value_label)

    def _on_slider(self, _raw: int) -> None:
        self._refresh_label()
        self.set_recommended(False)
        self.value_changed.emit(self.spec.key, self.value())


class ParamRow(QFrame):
    value_changed = pyqtSignal(str, float)

    def __init__(self, spec: ControlSpec, help_text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ParamRow")
        self.spec = spec
        self.help_text = help_text
        self.setProperty("recommended", False)
        self.setToolTip(f"{help_text}\n\nPadrão: {spec.format_value(spec.default)}  ·  duplo clique no slider restaura.")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(10)
        self.name_label = QLabel(spec.label)
        self.name_label.setObjectName("ParamName")
        self.name_label.setMinimumWidth(150)
        layout.addWidget(self.name_label)
        self.slider = ResetSlider(Qt.Horizontal)
        low, high = spec.slider_range()
        self.slider.setRange(low, high)
        self.slider.set_default_raw(spec.to_slider(spec.default))
        layout.addWidget(self.slider, 1)
        if spec.value_type is int:
            self.spin: Any = QSpinBox()
            self.spin.setRange(int(spec.minimum), int(spec.maximum))
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(spec.decimals)
            self.spin.setRange(float(spec.minimum), float(spec.maximum))
            self.spin.setSingleStep(10 ** (-spec.decimals) if spec.decimals > 0 else 1.0)
        self.spin.setFixedWidth(78)
        self.spin.setAlignment(Qt.AlignRight)
        layout.addWidget(self.spin)
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.slider.reset_requested.connect(self.reset)
        self.set_value(spec.default)

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, value: float) -> None:
        value = self.spec.clamp(float(value))
        for widget, payload in ((self.slider, self.spec.to_slider(value)), (self.spin, value)):
            widget.blockSignals(True)
            widget.setValue(payload)
            widget.blockSignals(False)
        self._refresh_modified()

    def reset(self) -> None:
        self.set_value(self.spec.default)
        self.set_recommended(False)
        self.value_changed.emit(self.spec.key, self.value())

    def set_recommended(self, flag: bool) -> None:
        self.setProperty("recommended", bool(flag))
        refresh_style(self)

    def matches(self, needle: str) -> bool:
        haystack = f"{self.spec.label} {self.spec.key} {self.help_text}".lower()
        return needle in haystack

    def _refresh_modified(self) -> None:
        self.name_label.setProperty("modified", abs(self.value() - self.spec.default) > 1e-9)
        refresh_style(self.name_label)

    def _on_slider(self, raw: int) -> None:
        value = self.spec.from_slider(raw)
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self._after_user_change()

    def _on_spin(self, _value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self.spec.to_slider(self.value()))
        self.slider.blockSignals(False)
        self._after_user_change()

    def _after_user_change(self) -> None:
        self._refresh_modified()
        self.set_recommended(False)
        self.value_changed.emit(self.spec.key, self.value())


class MixerPanel(QWidget):
    """Os controles principais como canais de uma mesa de som."""
    value_changed = pyqtSignal(str, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.strips: Dict[str, FaderStrip] = {}
        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(18)
        for title, keys in MIXER_GROUPS:
            column = QVBoxLayout()
            column.setSpacing(6)
            header = QLabel(title.upper())
            header.setProperty("role", "section")
            column.addWidget(header)
            row = QHBoxLayout()
            row.setSpacing(6)
            for key in keys:
                spec = CONTROL_SPECS[key]
                strip = FaderStrip(spec, spec.tooltip or PARAM_HELP.get(key, spec.label))
                strip.value_changed.connect(self.value_changed.emit)
                self.strips[key] = strip
                row.addWidget(strip)
            column.addLayout(row, 1)
            outer.addLayout(column)
        outer.addStretch(1)


class ParamPanel(QWidget):
    """Parâmetros avançados agrupados, com busca e restauração por grupo."""
    value_changed = pyqtSignal(str, float)
    group_reset = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.rows: Dict[str, ParamRow] = {}
        self.groups: Dict[str, List[str]] = {}
        self._group_headers: Dict[str, QWidget] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filtrar parâmetros…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        top.addWidget(self.search, 1)
        root.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        body = QVBoxLayout(host)
        body.setContentsMargins(0, 0, 4, 0)
        body.setSpacing(2)
        for group in ADVANCED_GROUPS:
            keys = [k for k, s in CONTROL_SPECS.items() if not s.primary and s.group == group]
            if not keys:
                continue
            self.groups[group] = keys
            header = QWidget()
            hl = QHBoxLayout(header)
            hl.setContentsMargins(8, 10, 4, 2)
            title = QLabel(GROUP_LABELS.get(group, group).upper())
            title.setProperty("role", "section")
            hl.addWidget(title)
            hl.addStretch(1)
            reset = QPushButton("Restaurar padrões")
            reset.setProperty("variant", "ghost")
            reset.clicked.connect(lambda _checked=False, g=group: self.group_reset.emit(g))
            hl.addWidget(reset)
            body.addWidget(header)
            self._group_headers[group] = header
            for key in keys:
                spec = CONTROL_SPECS[key]
                row = ParamRow(spec, spec.tooltip or PARAM_HELP.get(key, spec.label))
                row.value_changed.connect(self.value_changed.emit)
                self.rows[key] = row
                body.addWidget(row)
        body.addStretch(1)
        scroll.setWidget(host)
        root.addWidget(scroll, 1)

    def _apply_filter(self, text: str) -> None:
        needle = (text or "").strip().lower()
        for group, keys in self.groups.items():
            visible = 0
            for key in keys:
                show = not needle or self.rows[key].matches(needle)
                self.rows[key].setVisible(show)
                visible += 1 if show else 0
            self._group_headers[group].setVisible(visible > 0)

    def reset_group(self, group: str) -> None:
        for key in self.groups.get(group, []):
            row = self.rows[key]
            row.set_value(row.spec.default)
            row.set_recommended(False)


# ============================================================================
# Janela principal
# ============================================================================

AUDIO_FILTER = "Áudio (*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.opus *.wma);;Todos os arquivos (*)"
AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma")
LOG_COLORS = {"info": PALETTE["text"], "warn": PALETTE["warn"], "error": PALETTE["err"]}
PLAYER_STATE_LABELS = {"idle": "Parado", "playing": "Tocando", "paused": "Pausado", "error": "Erro"}
SHORTCUTS_TEXT = (
    "Ctrl+O\tAbrir áudio\n"
    "Ctrl+L\tCarregar a faixa informada\n"
    "Ctrl+G\tGerar sequência\n"
    "Espaço\tReproduzir / pausar\n"
    "Ctrl+.\tParar\n"
    "Ctrl+←  /  Ctrl+→\tVoltar / avançar 10 s\n"
    "Clique na partitura ou na barra do player\tIr para o ponto (parado: define o início)\n"
    "Alt+←  /  Alt+→\tMarcador (transição/drop) anterior / próximo\n"
    "Setas ← →  (partitura em foco)\tRolar a partitura\n"
    "Duplo clique na partitura\tAproximar / ajustar à janela\n"
    "Arrastar a janela do minimapa\tNavegar; as bordas dão zoom\n"
    "Roda do mouse na partitura\tZoom no ponto sob o cursor\n"
    "Shift+roda  /  botão do meio\tRolar a partitura\n"
    "Ctrl+=  /  Ctrl+-  /  Ctrl+0\tAproximar / afastar / ajustar à janela\n"
    "Duplo clique em um fader\tRestaura o valor padrão\n"
    "Arrastar um arquivo de áudio para a janela\tCarrega a faixa"
)


class VisionAudioWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setAcceptDrops(True)
        self.resize(1480, 920)
        self.setMinimumSize(1120, 720)

        self.source_resolver = AudioSourceResolver()
        self.generation_service = GenerationService()
        self.recommendation_service = ParameterRecommendationService()
        self.hardware_controller = SerialHardwareController()
        self.playback_controller = PlaybackController(self.hardware_controller, self)
        self.video_window_controller = ExternalVideoWindowController(self)

        self.current_track: Optional[LoadedTrack] = None
        self.current_sequence: Optional[GeneratedSequence] = None
        self.current_recommendation: Optional[RecommendationResult] = None
        self._ui_busy = False
        self._cue_s = 0.0           # onde a reprodução começa quando está parada
        self._last_background_error = False
        self._last_dir = ""
        self._editors: Dict[str, Any] = {}
        self._load_worker: Optional[LoadWorker] = None
        self._recommend_worker: Optional[RecommendationWorker] = None
        self._generation_worker: Optional[GenerationWorker] = None

        self._build_actions()
        self._build_menus()
        self._build_ui()
        self._build_session_dock()
        self._build_status_bar()
        self._connect_signals()
        self._populate_ports()
        self.apply_ui_state(UIConfigState())
        self._restore_settings()
        self._on_source_mode_changed()
        self._set_status_badge("ready")
        self._update_controls_enabled()
        self.log(f"{APP_TITLE} {APP_VERSION} pronto.")
        self.log("Dependências: " + collect_dependency_summary())
        if not any(shutil.which(exe) for exe in ("deno", "node", "bun", "qjs")):
            self.log(
                "Nenhum runtime JavaScript (deno/node) no PATH: o YouTube pode negar os formatos de áudio "
                "(HTTP 403). Instale deno ou node e rode: pip install -U \"yt-dlp[default]\".",
                "warn",
            )

    # ------------------------------------------------------------------ #
    # Construção                                                           #
    # ------------------------------------------------------------------ #

    def _action(self, text: str, slot: Any, shortcut: Optional[Any] = None, tip: str = "") -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setStatusTip(tip)
        action.triggered.connect(lambda _checked=False: slot())
        return action

    def _build_actions(self) -> None:
        self.act_open = self._action("Abrir áudio…", self.browse_audio, "Ctrl+O")
        self.act_load = self._action("Carregar faixa", self.load_track, "Ctrl+L")
        self.act_load_preset = self._action("Carregar preset…", self.load_config_json)
        self.act_save_preset = self._action("Salvar preset…", self.save_config_json)
        self.act_quit = self._action("Sair", self.close, "Ctrl+Q")
        self.act_generate = self._action("Gerar sequência", self.generate_sequence, "Ctrl+G")
        self.act_play = self._action("Reproduzir / pausar", self.toggle_playback, Qt.Key_Space)
        self.act_stop = self._action("Parar", self.stop_playback, "Ctrl+.")
        self.act_back = self._action(f"Voltar {SEEK_STEP_S} s", lambda: self.seek_by(-SEEK_STEP_S), "Ctrl+Left")
        self.act_forward = self._action(f"Avançar {SEEK_STEP_S} s", lambda: self.seek_by(SEEK_STEP_S), "Ctrl+Right")
        self.act_prev_marker = self._action("Marcador anterior", lambda: self._jump_marker(-1), "Alt+Left")
        self.act_next_marker = self._action("Próximo marcador", lambda: self._jump_marker(1), "Alt+Right")
        self.act_zoom_in = self._action("Aproximar a partitura", lambda: self.timeline.zoom_in(), "Ctrl+=")
        self.act_zoom_out = self._action("Afastar a partitura", lambda: self.timeline.zoom_out(), "Ctrl+-")
        self.act_zoom_fit = self._action("Ajustar partitura à janela", lambda: self.timeline.zoom_fit(), "Ctrl+0")
        self.act_suggest = self._action("Sugerir parâmetros", self.request_recommendation)
        self.act_reset_all = self._action("Restaurar todos os parâmetros", self.restore_all_defaults)
        self.act_clear_log = self._action("Limpar log", self.clear_log)
        self.act_shortcuts = self._action("Atalhos de teclado", self.show_shortcuts)
        self.act_about = self._action("Sobre", self.show_about)
        for action in (self.act_open, self.act_load, self.act_generate, self.act_play, self.act_stop,
                       self.act_back, self.act_forward, self.act_zoom_in, self.act_zoom_out, self.act_zoom_fit,
                       self.act_prev_marker, self.act_next_marker):
            self.addAction(action)       # atalhos valem mesmo com os menus escondidos

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&Arquivo")
        for action in (self.act_open, self.act_load):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.act_load_preset)
        file_menu.addAction(self.act_save_preset)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        param_menu = bar.addMenu("&Parâmetros")
        param_menu.addAction(self.act_suggest)
        param_menu.addAction(self.act_reset_all)

        self.view_menu = bar.addMenu("&Ver")
        for action in (self.act_prev_marker, self.act_next_marker):
            self.view_menu.addAction(action)
        self.view_menu.addSeparator()
        for action in (self.act_zoom_in, self.act_zoom_out, self.act_zoom_fit):
            self.view_menu.addAction(action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.act_clear_log)

        help_menu = bar.addMenu("A&juda")
        help_menu.addAction(self.act_shortcuts)
        help_menu.addAction(self.act_about)

    def _tool_button(self, icon: str, tip: str, role: Optional[str] = None, size: int = 18) -> QToolButton:
        button = QToolButton()
        button.setIcon(IconFactory.icon(icon))
        button.setIconSize(QSize(size, size))
        button.setToolTip(tip)
        button.setCursor(Qt.PointingHandCursor)
        if role:
            button.setProperty("role", role)
        return button

    def _divider(self) -> QFrame:
        line = QFrame()
        line.setObjectName("Divider")
        line.setFixedHeight(22)
        return line

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        root.addWidget(self._build_top_bar())

        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self._build_track_area())
        self.main_splitter.addWidget(self._build_tabs())
        self.main_splitter.setStretchFactor(0, 5)
        self.main_splitter.setStretchFactor(1, 4)
        root.addWidget(self.main_splitter, 1)
        root.addWidget(self._build_transport_bar())

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("TopBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        self.source_combo = QComboBox()
        self.source_combo.addItem("Arquivo local", "local")
        self.source_combo.addItem("YouTube", "youtube")
        self.source_combo.setFixedWidth(130)
        row.addWidget(self.source_combo)

        self.source_input = QLineEdit()
        self.source_input.setClearButtonEnabled(True)
        row.addWidget(self.source_input, 1)

        self.browse_button = self._tool_button("folder", "Escolher arquivo de áudio (Ctrl+O)")
        row.addWidget(self.browse_button)

        self.load_button = QPushButton(" Carregar")
        self.load_button.setIcon(IconFactory.icon("download"))
        self.load_button.setToolTip("Carregar a faixa informada (Ctrl+L)")
        row.addWidget(self.load_button)

        row.addWidget(self._divider())

        self.suggest_button = QPushButton(" Sugerir parâmetros")
        self.suggest_button.setIcon(IconFactory.icon("spark"))
        self.suggest_button.setToolTip("Analisa a faixa e ajusta os parâmetros automaticamente")
        row.addWidget(self.suggest_button)

        self.generate_button = QPushButton(" Gerar sequência")
        self.generate_button.setProperty("variant", "primary")
        self.generate_button.setIcon(IconFactory.icon("bolt", "#1a1305"))
        self.generate_button.setToolTip("Analisa o áudio e monta a partitura dos LEDs (Ctrl+G)")
        row.addWidget(self.generate_button)
        return bar

    def _build_track_area(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(10)
        names = QVBoxLayout()
        names.setSpacing(0)
        self.track_title_label = QLabel("Nenhuma faixa carregada")
        self.track_title_label.setProperty("role", "title")
        self.track_meta_label = QLabel("Abra um arquivo (Ctrl+O), arraste-o para cá ou busque no YouTube.")
        self.track_meta_label.setProperty("role", "muted")
        names.addWidget(self.track_title_label)
        names.addWidget(self.track_meta_label)
        head.addLayout(names, 1)
        self.chip_bpm = self._chip("— BPM")
        self.chip_duration = self._chip("—")
        self.chip_transitions = self._chip("— transições")
        for chip in (self.chip_bpm, self.chip_duration, self.chip_transitions):
            head.addWidget(chip, 0, Qt.AlignVCenter)
        layout.addLayout(head)

        tools = QHBoxLayout()
        tools.setSpacing(4)
        self.prev_marker_button = self._tool_button("prev_marker", "Marcador anterior (Alt+←)", size=16)
        self.next_marker_button = self._tool_button("next_marker", "Próximo marcador (Alt+→)", size=16)
        marker_label = QLabel("Marcadores")
        marker_label.setProperty("role", "faint")
        tools.addWidget(self.prev_marker_button)
        tools.addWidget(self.next_marker_button)
        tools.addWidget(marker_label)
        tools.addStretch(1)
        self.zoom_out_button = self._tool_button("zoom_out", "Afastar (Ctrl+-)", size=16)
        self.zoom_label = QLabel("1×")
        self.zoom_label.setProperty("role", "muted")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setMinimumWidth(46)
        self.zoom_in_button = self._tool_button("zoom_in", "Aproximar (Ctrl+=)", size=16)
        self.zoom_fit_button = self._tool_button("fit", "Ajustar à janela (Ctrl+0 · duplo clique)", size=16)
        self.follow_button = self._tool_button("follow", "Seguir o cursor durante a reprodução", size=16)
        self.follow_button.setCheckable(True)
        self.follow_button.setChecked(True)
        for widget in (self.zoom_out_button, self.zoom_label, self.zoom_in_button, self.zoom_fit_button, self.follow_button):
            tools.addWidget(widget)
        for button in (self.prev_marker_button, self.next_marker_button, self.zoom_out_button,
                       self.zoom_in_button, self.zoom_fit_button, self.follow_button):
            button.setFixedSize(30, 26)
        layout.addLayout(tools)

        self.timeline = TimelineWidget()
        layout.addWidget(self.timeline, 1)
        return panel

    def _chip(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("role", "chip")
        return label

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()

        # Mixer
        mixer_page = QWidget()
        mixer_layout = QVBoxLayout(mixer_page)
        mixer_layout.setContentsMargins(0, 6, 0, 0)
        mixer_layout.setSpacing(0)
        self.mixer = MixerPanel()
        mixer_scroll = QScrollArea()
        mixer_scroll.setWidgetResizable(True)
        mixer_scroll.setWidget(self.mixer)
        mixer_layout.addWidget(mixer_scroll, 1)
        self.tabs.addTab(mixer_page, "Mixer")

        # Parâmetros
        self.param_panel = ParamPanel()
        self.tabs.addTab(self.param_panel, "Parâmetros")

        # Transições
        trans_page = QWidget()
        trans_layout = QVBoxLayout(trans_page)
        trans_layout.setContentsMargins(10, 8, 10, 8)
        trans_layout.setSpacing(6)
        trans_hint = QLabel("Duplo clique em uma linha para ir até a transição (durante a reprodução).")
        trans_hint.setProperty("role", "muted")
        trans_layout.addWidget(trans_hint)
        self.transition_table = QTableWidget(0, 4)
        self.transition_table.setHorizontalHeaderLabels(["Início", "Efeito", "Duração", "Brilho"])
        self.transition_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.transition_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.transition_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.transition_table.setAlternatingRowColors(True)
        self.transition_table.verticalHeader().setVisible(False)
        self.transition_table.horizontalHeader().setStretchLastSection(True)
        trans_layout.addWidget(self.transition_table, 1)
        self.tabs.addTab(trans_page, "Transições")

        # Log
        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        log_layout.setContentsMargins(10, 8, 10, 8)
        log_layout.setSpacing(6)
        log_tools = QHBoxLayout()
        log_tools.addStretch(1)
        copy_button = QPushButton(" Copiar")
        copy_button.setProperty("variant", "ghost")
        copy_button.setIcon(IconFactory.icon("copy"))
        copy_button.clicked.connect(self.copy_log)
        clear_button = QPushButton(" Limpar")
        clear_button.setProperty("variant", "ghost")
        clear_button.setIcon(IconFactory.icon("trash"))
        clear_button.clicked.connect(self.clear_log)
        log_tools.addWidget(copy_button)
        log_tools.addWidget(clear_button)
        log_layout.addLayout(log_tools)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(3000)
        log_font = self.log_output.font()
        log_font.setFamily("monospace")
        log_font.setStyleHint(QFont.Monospace)
        self.log_output.setFont(log_font)
        log_layout.addWidget(self.log_output, 1)
        self.tabs.addTab(log_page, "Log")
        return self.tabs

    def _build_transport_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("TransportBar")
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(14, 8, 14, 10)
        outer.setSpacing(4)
        self.seek_bar = SeekBar()
        outer.addWidget(self.seek_bar)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        outer.addLayout(row)

        self.seek_back_button = self._tool_button("back", f"Voltar {SEEK_STEP_S} s (Ctrl+←)")
        self.play_button = self._tool_button("play", "Reproduzir / pausar (Espaço)", role="play", size=22)
        self.play_button.setFixedSize(46, 46)
        self.play_button.setIcon(IconFactory.icon("play", "#1a1305"))   # contraste sobre o fundo âmbar
        self.stop_button = self._tool_button("stop", "Parar (Ctrl+.)")
        self.seek_forward_button = self._tool_button("forward", f"Avançar {SEEK_STEP_S} s (Ctrl+→)")
        for button in (self.seek_back_button, self.stop_button, self.seek_forward_button):
            button.setProperty("role", "transport")
            button.setFixedSize(34, 34)
        glow = QGraphicsDropShadowEffect(self.play_button)
        glow.setBlurRadius(24)
        glow.setOffset(0, 0)
        glow.setColor(QColor(245, 158, 11, 150))
        self.play_button.setGraphicsEffect(glow)
        for button in (self.seek_back_button, self.play_button, self.stop_button, self.seek_forward_button):
            row.addWidget(button)

        self.time_label = QLabel(f"{format_clock(0)} / {format_clock(None)}")
        self.time_label.setProperty("role", "clock")
        self.time_label.setMinimumWidth(170)
        self.time_label.setContentsMargins(12, 0, 8, 0)
        row.addWidget(self.time_label)
        self.state_label = QLabel(PLAYER_STATE_LABELS["idle"])
        self.state_label.setObjectName("PlayerState")
        self.state_label.setProperty("state", "idle")
        row.addWidget(self.state_label)
        row.addStretch(1)

        self.video_button = QToolButton()
        self.video_button.setText(" Vídeo")
        self.video_button.setIcon(IconFactory.icon("video"))
        self.video_button.setIconSize(QSize(18, 18))
        self.video_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.video_button.setCheckable(True)
        self.video_button.setToolTip("Abre o vídeo do YouTube em uma janela separada")
        row.addWidget(self.video_button)
        row.addWidget(self._divider())

        self.hw_dot = QLabel()
        self.hw_dot.setObjectName("HwDot")
        self.hw_dot.setProperty("connected", False)
        row.addWidget(self.hw_dot)
        hw_title = QLabel("Arduino")
        hw_title.setProperty("role", "muted")
        row.addWidget(hw_title)
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(170)
        self.port_combo.setToolTip("Porta serial do Arduino")
        row.addWidget(self.port_combo)
        self.refresh_ports_button = self._tool_button("refresh", "Atualizar lista de portas", size=16)
        row.addWidget(self.refresh_ports_button)
        self.connect_button = QPushButton("Conectar")
        row.addWidget(self.connect_button)
        return bar

    def _form_section(self, title: str, widget: QWidget, spacing: int = 8) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        box = QVBoxLayout(card)
        box.setContentsMargins(12, 10, 12, 12)
        box.setSpacing(spacing)
        label = QLabel(title.upper())
        label.setProperty("role", "section")
        box.addWidget(label)
        box.addWidget(widget)
        return card

    def _build_session_dock(self) -> None:
        self.session_dock = QDockWidget("Sessão", self)
        self.session_dock.setObjectName("SessionDock")
        self.session_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.session_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )

        title_bar = QFrame()
        title_bar.setObjectName("DockTitle")
        title_row = QHBoxLayout(title_bar)
        title_row.setContentsMargins(14, 8, 6, 8)
        title_label = QLabel("Sessão")
        title_label.setObjectName("DockTitleLabel")
        title_row.addWidget(title_label)
        title_row.addStretch(1)
        close_button = QToolButton()
        close_button.setObjectName("DockClose")
        close_button.setText("✕")
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.setToolTip("Ocultar painel (reabra em Ver ▸ Sessão)")
        close_button.clicked.connect(self.session_dock.hide)
        title_row.addWidget(close_button)
        self.session_dock.setTitleBarWidget(title_bar)

        host = QWidget()
        column = QVBoxLayout(host)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(8)

        self.led_monitor = LedMonitorWidget()
        column.addWidget(self._form_section("Monitor de LEDs", self.led_monitor, 14))

        behavior = QWidget()
        form = QFormLayout(behavior)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.transition_profile_combo = QComboBox()
        for key, label in TRANSITION_PROFILE_LABELS.items():
            self.transition_profile_combo.addItem(label, key)
        self.transition_profile_combo.setToolTip("Quantas transições detectar: baixa = só as mais claras.")
        self.effect_profile_combo = QComboBox()
        for key, label in EFFECT_PROFILE_LABELS.items():
            self.effect_profile_combo.addItem(label, key)
        self.effect_profile_combo.setToolTip("Intensidade visual dos efeitos de transição.")
        self.middle_speed_combo = QComboBox()
        for value, label in MIDDLE_SPEED_OPTIONS.items():
            self.middle_speed_combo.addItem(label, float(value))
        self.middle_speed_combo.setToolTip("Velocidade base do chase dos LEDs do meio em relação ao BPM.")
        form.addRow("Transições", self.transition_profile_combo)
        form.addRow("Efeitos", self.effect_profile_combo)
        form.addRow("Meio", self.middle_speed_combo)
        column.addWidget(self._form_section("Comportamento", behavior, 10))

        analysis = QWidget()
        self.analysis_form = QFormLayout(analysis)
        self.analysis_form.setContentsMargins(0, 0, 0, 0)
        self.analysis_form.setHorizontalSpacing(12)
        self.analysis_form.setVerticalSpacing(7)
        self.analysis_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.analysis_form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self.analysis_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.analysis_values: Dict[str, QLabel] = {}
        for key, title in (
            ("bpm", "BPM"), ("beats", "Batidas"), ("pulse", "Confiança do pulso"),
            ("bar", "Confiança do compasso"), ("transitions", "Transições"),
            ("drops", "Drops"), ("patterns", "Trocas de sequência"), ("governor", "Trava anti-saturação"),
        ):
            name = QLabel(title)
            name.setProperty("role", "muted")
            value = QLabel("—")
            value.setProperty("role", "metric" if key == "bpm" else "value")
            value.setWordWrap(True)
            value.setAlignment(Qt.AlignRight | Qt.AlignTop)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.analysis_form.addRow(name, value)
            self.analysis_values[key] = value
        column.addWidget(self._form_section("Análise do motor", analysis, 10))
        column.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(host)
        self.session_dock.setWidget(scroll)
        self.session_dock.setMinimumWidth(280)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.session_dock)
        self.view_menu.insertAction(self.act_clear_log, self.session_dock.toggleViewAction())

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        bar.setSizeGripEnabled(False)
        bar.setContentsMargins(8, 3, 8, 3)
        self.setStatusBar(bar)
        self.status_label = QLabel("Pronto.")
        self.status_label.setProperty("role", "muted")
        self.status_label.setContentsMargins(4, 0, 0, 0)
        bar.addWidget(self.status_label, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        bar.addPermanentWidget(self.progress_bar)
        self.status_badge = QLabel("Pronto")
        self.status_badge.setObjectName("StatusBadge")
        self.status_badge.setProperty("state", "ready")
        bar.addPermanentWidget(self.status_badge)

        for key, strip in self.mixer.strips.items():
            self._editors[key] = strip
        for key, row in self.param_panel.rows.items():
            self._editors[key] = row

    def _connect_signals(self) -> None:
        self.source_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        self.source_input.textChanged.connect(lambda _text: self._update_controls_enabled())
        self.source_input.returnPressed.connect(self.load_track)
        self.browse_button.clicked.connect(self.browse_audio)
        self.load_button.clicked.connect(self.load_track)
        self.suggest_button.clicked.connect(self.request_recommendation)
        self.generate_button.clicked.connect(self.generate_sequence)

        self.play_button.clicked.connect(self.toggle_playback)
        self.stop_button.clicked.connect(self.stop_playback)
        self.seek_back_button.clicked.connect(lambda _checked=False: self.seek_by(-SEEK_STEP_S))
        self.seek_forward_button.clicked.connect(lambda _checked=False: self.seek_by(SEEK_STEP_S))
        self.video_button.toggled.connect(self.toggle_video_window)
        self.refresh_ports_button.clicked.connect(self._populate_ports)
        self.connect_button.clicked.connect(self.toggle_connection)

        self.timeline.seek_requested.connect(self._on_timeline_seek)
        self.seek_bar.seek_requested.connect(self._on_timeline_seek)
        self.timeline.view_changed.connect(self._on_timeline_view)
        self.timeline.follow_changed.connect(self._on_follow_changed)
        self.prev_marker_button.clicked.connect(lambda _checked=False: self._jump_marker(-1))
        self.next_marker_button.clicked.connect(lambda _checked=False: self._jump_marker(1))
        self.zoom_in_button.clicked.connect(lambda _checked=False: self.timeline.zoom_in())
        self.zoom_out_button.clicked.connect(lambda _checked=False: self.timeline.zoom_out())
        self.zoom_fit_button.clicked.connect(lambda _checked=False: self.timeline.zoom_fit())
        self.follow_button.toggled.connect(self.timeline.set_follow)
        self.transition_table.cellDoubleClicked.connect(self._on_transition_activated)
        self.param_panel.group_reset.connect(self.restore_group_defaults)

        self.playback_controller.state_changed.connect(self._on_playback_state_changed)
        self.playback_controller.position_changed.connect(self._on_playback_position_changed)
        self.playback_controller.frame_changed.connect(self._set_live_levels)
        self.playback_controller.playback_finished.connect(self._on_playback_finished)
        self.playback_controller.playback_error.connect(self._on_playback_error)
        self.video_window_controller.state_changed.connect(self._on_video_window_state_changed)
        self.video_window_controller.error.connect(self._on_video_window_error)

    # ------------------------------------------------------------------ #
    # Estado visual                                                        #
    # ------------------------------------------------------------------ #

    def _set_status_badge(self, state: str) -> None:
        normalized = state if state in {"ready", "busy", "error"} else "ready"
        self.status_badge.setText({"ready": "Pronto", "busy": "Ocupado", "error": "Erro"}[normalized])
        self.status_badge.setProperty("state", normalized)
        refresh_style(self.status_badge)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_busy(self, busy: bool, label: Optional[str] = None, value: Optional[int] = None,
                  status: Optional[str] = None) -> None:
        self._ui_busy = busy
        if label is not None:
            self._set_status(label)
        if value is not None:
            self.progress_bar.setValue(int(value))
        self.progress_bar.setVisible(busy)
        self._set_status_badge(status or ("busy" if busy else "ready"))
        self._update_controls_enabled()

    def _set_progress(self, value: int, label: str) -> None:
        self._set_busy(True, label, int(value), "busy")
        self.log(label)

    def _update_controls_enabled(self) -> None:
        busy = self._ui_busy
        has_track = self.current_track is not None
        has_sequence = self.current_sequence is not None
        player_state = self.playback_controller.state
        playing = player_state in ("playing", "paused")
        is_local = self.source_combo.currentData() == "local"
        has_source = bool(self.source_input.text().strip())

        self.source_combo.setEnabled(not busy)
        self.source_input.setEnabled(not busy)
        self.browse_button.setEnabled(not busy and is_local)
        self.load_button.setEnabled(not busy and has_source)
        self.suggest_button.setEnabled(not busy and has_track)
        self.generate_button.setEnabled(not busy and has_track)
        self.play_button.setEnabled(not busy and has_sequence)
        self.stop_button.setEnabled(playing or (not busy and has_sequence))
        self.seek_back_button.setEnabled(playing)
        self.seek_forward_button.setEnabled(playing)
        self.video_button.setEnabled(not busy and self._current_video_track() is not None)
        self.connect_button.setEnabled(not busy)
        self.port_combo.setEnabled(not self.hardware_controller.is_connected)
        self.refresh_ports_button.setEnabled(not self.hardware_controller.is_connected)
        self.transition_profile_combo.setEnabled(not busy)
        self.effect_profile_combo.setEnabled(not busy)
        self.middle_speed_combo.setEnabled(not busy)

        self.act_open.setEnabled(not busy and is_local)
        self.act_load.setEnabled(not busy and has_source)
        self.act_generate.setEnabled(not busy and has_track)
        self.act_suggest.setEnabled(not busy and has_track)
        self.act_play.setEnabled(not busy and has_sequence)
        self.act_stop.setEnabled(self.stop_button.isEnabled())
        self.act_back.setEnabled(playing)
        self.act_forward.setEnabled(playing)
        for widget in (self.prev_marker_button, self.next_marker_button, self.zoom_in_button,
                       self.zoom_out_button, self.zoom_fit_button, self.follow_button,
                       self.act_prev_marker, self.act_next_marker, self.act_zoom_in,
                       self.act_zoom_out, self.act_zoom_fit):
            widget.setEnabled(has_sequence)

    def _on_source_mode_changed(self, _index: int = 0) -> None:
        is_local = self.source_combo.currentData() == "local"
        self.browse_button.setVisible(is_local)
        self.source_input.setPlaceholderText(
            "Caminho do arquivo de áudio…" if is_local else "URL do YouTube ou termo de busca…"
        )
        self._update_controls_enabled()

    # ------------------------------------------------------------------ #
    # Arquivo / faixa                                                      #
    # ------------------------------------------------------------------ #

    def browse_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar áudio", self._last_dir or os.path.expanduser("~"), AUDIO_FILTER)
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        self.source_combo.setCurrentIndex(max(0, self.source_combo.findData("local")))
        self.source_input.setText(path)
        self.load_track()

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and path.lower().endswith(AUDIO_SUFFIXES) and not self._ui_busy:
                self._last_dir = os.path.dirname(path)
                self.source_combo.setCurrentIndex(max(0, self.source_combo.findData("local")))
                self.source_input.setText(path)
                self.load_track()
                event.acceptProposedAction()
                return

    def load_track(self) -> None:
        if self._ui_busy or not self.source_input.text().strip():
            return
        self.stop_playback()
        mode = self.source_combo.currentData()
        value = self.source_input.text().strip()
        self._last_background_error = False
        self._load_worker = LoadWorker(self.source_resolver, mode, value)
        self._load_worker.progress.connect(self._set_progress)
        self._load_worker.loaded.connect(self.handle_loaded_track)
        self._load_worker.failed.connect(self._handle_worker_error)
        self._load_worker.finished.connect(self._finish_background_task)
        self._set_busy(True, "Carregando faixa…", 0)
        self._load_worker.start()

    def request_recommendation(self) -> None:
        if not self.current_track:
            qt_message(self, APP_TITLE, "Carregue uma faixa primeiro.", "warn")
            return
        self._recommend_worker = RecommendationWorker(
            self.generation_service, self.recommendation_service, self.current_track
        )
        self._last_background_error = False
        self._recommend_worker.progress.connect(self._set_progress)
        self._recommend_worker.finished_result.connect(self.handle_recommendation)
        self._recommend_worker.failed.connect(self._handle_worker_error)
        self._recommend_worker.finished.connect(self._finish_background_task)
        self._set_busy(True, "Analisando a faixa para sugerir parâmetros…", 0)
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
        self._set_busy(True, "Gerando sequência…", 0)
        self._generation_worker.start()

    def _finish_background_task(self) -> None:
        if self._last_background_error:
            self._set_busy(False, None, 0, "error")
        else:
            self._set_busy(False, "Pronto.", 0, "ready")

    def _handle_worker_error(self, message: str, details: str = "") -> None:
        normalized = normalize_error_message(message, "WorkerError")
        self._last_background_error = True
        self._set_status(normalized)
        self.log(normalized, "error")
        qt_message(self, APP_TITLE, normalized, "error", details or None)

    # ------------------------------------------------------------------ #
    # Parâmetros / presets                                                 #
    # ------------------------------------------------------------------ #

    def current_ui_state(self) -> UIConfigState:
        values = default_control_values()
        for key, editor in self._editors.items():
            values[key] = float(editor.value())
        return UIConfigState(
            transition_profile=self.transition_profile_combo.currentData(),
            effect_profile=self.effect_profile_combo.currentData(),
            middle_speed_multiplier=float(self.middle_speed_combo.currentData()),
            values=values,
        )

    def apply_ui_state(self, state: UIConfigState) -> None:
        self.transition_profile_combo.setCurrentIndex(max(0, self.transition_profile_combo.findData(state.transition_profile)))
        self.effect_profile_combo.setCurrentIndex(max(0, self.effect_profile_combo.findData(state.effect_profile)))
        self.middle_speed_combo.setCurrentIndex(max(0, self.middle_speed_combo.findData(float(state.middle_speed_multiplier))))
        for key, value in state.normalized_values().items():
            editor = self._editors.get(key)
            if editor is not None:
                editor.set_value(value)
                editor.set_recommended(False)

    def restore_group_defaults(self, group_name: str) -> None:
        self.param_panel.reset_group(group_name)
        self.log(f"Padrões restaurados em {GROUP_LABELS.get(group_name, group_name)}.")

    def restore_all_defaults(self) -> None:
        self.apply_ui_state(UIConfigState())
        self.log("Todos os parâmetros voltaram ao padrão.")

    def save_config_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar preset", os.path.join(self._last_dir or os.getcwd(), "visionaudio_preset.json"), "JSON (*.json)"
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
            qt_message(self, "Salvar preset", message, "error", details)
            self.log(f"Falha ao salvar preset: {message}", "error")
            return
        self.log(f"Preset salvo em {path}.")

    def load_config_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Carregar preset", self._last_dir or os.getcwd(), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            self.apply_ui_state(UIConfigState.from_serializable(payload.get("config", payload)))
        except Exception as exc:
            message, details = describe_exception(exc)
            qt_message(self, "Carregar preset", message, "error", details)
            self.log(f"Falha ao carregar preset: {message}", "error")
            return
        self.log(f"Preset carregado de {path}.")

    # ------------------------------------------------------------------ #
    # Resultados                                                           #
    # ------------------------------------------------------------------ #

    def _reset_analysis(self) -> None:
        for label in self.analysis_values.values():
            label.setText("—")

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
        self.track_meta_label.setText("  ·  ".join(meta))
        self.chip_bpm.setText("— BPM")
        self.chip_duration.setText(format_seconds(track.duration_s) if track.duration_s else "—")
        self.chip_transitions.setText("— transições")
        self.transition_table.setRowCount(0)
        self.timeline.set_sequence(None)
        self.seek_bar.set_sequence(None)
        self._cue_s = 0.0
        self._set_live_levels([0] * 6)
        self.time_label.setText(f"{format_clock(0)} / {format_clock(track.duration_s)}")
        self._reset_analysis()
        self.log(f"Faixa carregada: {track.title}")
        self._update_controls_enabled()

    def handle_recommendation(self, result: RecommendationResult) -> None:
        self.current_recommendation = result
        for key, value in result.numeric_values.items():
            editor = self._editors.get(key)
            if editor is not None:
                editor.set_value(value)
                editor.set_recommended(True)
        categorical = result.categorical_values
        self.transition_profile_combo.setCurrentIndex(
            max(0, self.transition_profile_combo.findData(categorical.get("transition_profile", "medium"))))
        self.effect_profile_combo.setCurrentIndex(
            max(0, self.effect_profile_combo.findData(categorical.get("effect_profile", "medium"))))
        self.middle_speed_combo.setCurrentIndex(
            max(0, self.middle_speed_combo.findData(float(categorical.get("middle_speed_multiplier", 1.0)))))
        model = result.model_label or ("scikit-learn" if result.using_sklearn else "fallback interno")
        for line in result.summary_lines:
            self.log(line)
        self.log(f"Parâmetros sugeridos com {model}.")
        self.tabs.setCurrentIndex(0)

    def handle_generated_sequence(self, sequence: GeneratedSequence) -> None:
        self._cleanup_sequence_temp_files(self.current_sequence)
        self.current_sequence = sequence
        self.timeline.set_sequence(sequence)
        self.seek_bar.set_sequence(sequence)
        self._cue_s = 0.0
        self._fill_transition_table()
        report = sequence.report or {}
        self.chip_bpm.setText(f"{sequence.tempo_bpm:.1f} BPM")
        self.chip_duration.setText(format_seconds(sequence.duration_s))
        self.chip_transitions.setText(f"{len(sequence.transitions)} transições")
        self.time_label.setText(f"{format_clock(0)} / {format_clock(sequence.duration_s)}")
        values = self.analysis_values
        values["bpm"].setText(f"{sequence.tempo_bpm:.1f}")
        values["beats"].setText(str(sequence.beat_count))
        values["pulse"].setText(f"{report.get('beat_confidence', 0.0):.2f}")
        values["bar"].setText(f"{report.get('downbeat_confidence', 0.0):.2f}")
        values["transitions"].setText(
            f"{len(sequence.transitions)}  ({report.get('transitions_on_downbeat', 0)} no tempo 1)")
        drops = report.get("drops_s", [])
        values["drops"].setText(", ".join(f"{d:.1f}s" for d in drops) if drops else "nenhum")
        values["patterns"].setText(
            f"{report.get('pattern_switches', 0)}  ({report.get('section_switches', 0)} em início de seção)")
        values["governor"].setText(f"{report.get('governor_pct', 0.0):.1f}% do tempo")
        for line in sequence.report_lines:
            self.log(line)
        self.log(f"Sequência gerada para {sequence.track.title}.")
        self._update_controls_enabled()

    def _fill_transition_table(self) -> None:
        sequence = self.current_sequence
        self.transition_table.setRowCount(0)
        if not sequence:
            return
        fps = float(sequence.fps)
        for transition in sequence.transitions:
            start = float(getattr(transition, "frame_start", 0)) / fps
            end = float(getattr(transition, "frame_end", 0)) / fps
            effect = getattr(getattr(transition, "effect_type", None), "name", "TRANS").title()
            row = self.transition_table.rowCount()
            self.transition_table.insertRow(row)
            cells = (format_clock(start), effect, f"{max(0.0, end - start):.2f} s",
                     f"{float(getattr(transition, 'pwm_brightness', 0.0)):.0f}")
            for column, text in enumerate(cells):
                self.transition_table.setItem(row, column, QTableWidgetItem(text))

    # ------------------------------------------------------------------ #
    # Reprodução                                                           #
    # ------------------------------------------------------------------ #

    def toggle_playback(self) -> None:
        if not self.current_sequence:
            self._set_status("Gere uma sequência antes de reproduzir.")
            return
        if self.playback_controller.state in ("playing", "paused"):
            self.playback_controller.pause_toggle()
            return
        self.playback_controller.play(self.current_sequence, self._cue_s)

    def stop_playback(self) -> None:
        if self.playback_controller.state == "idle" and self._cue_s > 0:
            self._cue_s = 0.0                      # Parar de novo volta ao começo da faixa
            self._apply_cue_display()
        self.playback_controller.stop()
        self.video_window_controller.close()

    def seek_by(self, delta_s: int) -> None:
        if self.playback_controller.state in ("playing", "paused"):
            self.playback_controller.seek_relative(delta_s)

    def _on_timeline_seek(self, seconds: float, reveal: bool = False) -> None:
        sequence = self.current_sequence
        if not sequence or sequence.duration_s <= 0:
            return
        seconds = max(0.0, min(float(sequence.duration_s), float(seconds)))
        if self.playback_controller.state in ("playing", "paused"):
            self.playback_controller.seek_to(seconds)
        else:
            self._cue_s = seconds                  # parado: define de onde a reprodução começa
            self._apply_cue_display()
            self.statusBar().showMessage(
                f"Início em {format_clock(seconds)} — pressione Espaço para tocar a partir daqui.", 4000)
        if reveal:
            self.timeline.reveal(seconds)

    def _apply_cue_display(self) -> None:
        sequence = self.current_sequence
        duration = sequence.duration_s if sequence else None
        for widget in (self.timeline, self.seek_bar):
            widget.set_cue(self._cue_s)
            widget.set_position(self._cue_s)
        self.time_label.setText(f"{format_clock(self._cue_s)} / {format_clock(duration)}")

    def _current_position(self) -> float:
        if self.playback_controller.state in ("playing", "paused"):
            return self.playback_controller.current_position_s()
        return self._cue_s

    def _marker_times(self) -> List[float]:
        sequence = self.current_sequence
        if not sequence:
            return []
        fps = float(sequence.fps)
        times = [float(getattr(t, "frame_start", 0)) / fps for t in sequence.transitions]
        times += [float(d) for d in (sequence.report or {}).get("drops_s", [])]
        return sorted({round(t, 3) for t in times})

    def _jump_marker(self, direction: int) -> None:
        times = self._marker_times()
        if not times:
            return
        now = self._current_position()
        if direction > 0:
            target = next((t for t in times if t > now + 0.30), None)
        else:
            target = next((t for t in reversed(times) if t < now - 0.60), None)
            if target is None and now > 0.05:
                target = 0.0
        if target is None:
            self.statusBar().showMessage("Não há mais marcadores nessa direção.", 2500)
            return
        self._on_timeline_seek(target, reveal=True)

    def _on_timeline_view(self, _start: float, span: float, duration: float) -> None:
        zoom = duration / span if span > 0 and duration > 0 else 1.0
        self.zoom_label.setText("1×" if zoom < 1.05 else f"{zoom:.1f}×")

    def _on_follow_changed(self, flag: bool) -> None:
        self.follow_button.blockSignals(True)
        self.follow_button.setChecked(flag)
        self.follow_button.blockSignals(False)

    def _set_live_levels(self, levels: Sequence[int]) -> None:
        self.led_monitor.set_levels(levels)
        self.timeline.set_live_levels(levels)

    def _on_transition_activated(self, row: int, _column: int) -> None:
        sequence = self.current_sequence
        if sequence and 0 <= row < len(sequence.transitions):
            start = float(getattr(sequence.transitions[row], "frame_start", 0)) / sequence.fps
            self._on_timeline_seek(start, reveal=True)

    def _on_playback_state_changed(self, state: str) -> None:
        self.state_label.setText(PLAYER_STATE_LABELS.get(state, state))
        self.state_label.setProperty("state", state)
        refresh_style(self.state_label)
        self.play_button.setIcon(IconFactory.icon("pause" if state == "playing" else "play",
                                                  "#1a1305"))
        if state == "idle":
            self._set_live_levels([0] * 6)
            self._apply_cue_display()
        self._update_controls_enabled()

    def _on_playback_position_changed(self, elapsed: float, duration: float) -> None:
        if self.playback_controller.state == "idle":
            elapsed = self._cue_s                  # parado: o cursor fica no ponto de início
        self.time_label.setText(f"{format_clock(elapsed)} / {format_clock(duration)}")
        self.timeline.set_position(elapsed)
        self.seek_bar.set_position(elapsed)

    def _on_playback_finished(self) -> None:
        self.log("Reprodução finalizada.")

    def _on_playback_error(self, message: str) -> None:
        normalized = normalize_error_message(message, "PlaybackError")
        self._set_status(normalized)
        self._set_status_badge("error")
        self.log(f"Falha no player: {normalized}", "error")
        qt_message(self, "Player", normalized, "error")

    # ------------------------------------------------------------------ #
    # Vídeo                                                                #
    # ------------------------------------------------------------------ #

    def _current_video_track(self) -> Optional[LoadedTrack]:
        for track in (self.current_sequence.track if self.current_sequence else None, self.current_track):
            if track and (track.youtube_video_url or track.youtube_page_url):
                return track
        return None

    def toggle_video_window(self, checked: bool) -> None:
        if not checked:
            self.video_window_controller.close()
            return
        track = self._current_video_track()
        if not track:
            self.video_button.setChecked(False)
            qt_message(self, "Vídeo", "Não há vídeo disponível para a faixa atual.", "warn")
            return
        start_s = self.playback_controller.current_position_s()
        try:
            if track.youtube_video_url:
                self.video_window_controller.open(
                    track.youtube_video_url, track.title, start_s=start_s,
                    page_url=track.youtube_page_url,
                    headers=(track.metadata or {}).get("video_headers"),
                )
                self.log(f"Janela de vídeo aberta para {track.title} com {self.video_window_controller.backend_name}.")
                return
            if track.youtube_page_url:
                if not webbrowser.open(track.youtube_page_url):
                    raise RuntimeError("Não foi possível abrir o navegador para o vídeo.")
                self.log(f"Vídeo aberto no navegador: {track.title}.")
                self.video_button.setChecked(False)
                return
            raise RuntimeError("A faixa atual não possui URL de vídeo.")
        except Exception as exc:
            if track.youtube_page_url:
                self.log("Falha ao abrir o player de vídeo. Tentando o navegador…", "warn")
                if webbrowser.open(track.youtube_page_url):
                    self.video_button.setChecked(False)
                    self.log(f"Vídeo aberto no navegador: {track.title}.")
                    return
            self.video_button.setChecked(False)
            message, details = describe_exception(exc)
            qt_message(self, "Vídeo", message, "error", details)
            self.log(f"Falha ao abrir vídeo: {message}", "error")

    def _on_video_window_state_changed(self, opened: bool) -> None:
        self.video_button.blockSignals(True)
        self.video_button.setChecked(opened)
        self.video_button.blockSignals(False)

    def _on_video_window_error(self, message: str) -> None:
        normalized = normalize_error_message(message, "VideoWindowError")
        self.log(f"Falha na janela de vídeo: {normalized}", "error")
        qt_message(self, "Vídeo", normalized, "error")

    # ------------------------------------------------------------------ #
    # Arduino                                                              #
    # ------------------------------------------------------------------ #

    def _populate_ports(self) -> None:
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = self.hardware_controller.available_ports()
        for port in ports:
            self.port_combo.addItem(port)
        if current and current in ports:
            self.port_combo.setCurrentIndex(ports.index(current))
        if not ports:
            self.port_combo.setEditable(True)
            self.port_combo.setEditText("")
            self.port_combo.lineEdit().setPlaceholderText("nenhuma porta encontrada")
        else:
            self.port_combo.setEditable(False)

    def _set_hardware_state(self, connected: bool) -> None:
        self.hw_dot.setProperty("connected", connected)
        refresh_style(self.hw_dot)
        self.connect_button.setText("Desconectar" if connected else "Conectar")
        self.connect_button.setProperty("variant", "ghost" if connected else "primary")
        refresh_style(self.connect_button)
        self._update_controls_enabled()

    def toggle_connection(self) -> None:
        try:
            if self.hardware_controller.is_connected:
                self.hardware_controller.disconnect()
                self._set_hardware_state(False)
                self._set_status("Arduino desconectado.")
                self.log("Arduino desconectado.")
                return
            port = self.port_combo.currentText().strip()
            if not port:
                raise ValueError("Selecione uma porta serial.")
            self.hardware_controller.connect_port(port)
            self._set_hardware_state(True)
            self._set_status(f"Arduino conectado em {port}.")
            self.log(f"Arduino conectado em {port}.")
        except Exception as exc:
            message, details = describe_exception(exc)
            self._set_status(message)
            self._set_status_badge("error")
            self.log(f"Falha ao conectar Arduino: {message}", "error")
            qt_message(self, "Arduino", message, "error", details)

    # ------------------------------------------------------------------ #
    # Log / diálogos / ciclo de vida                                       #
    # ------------------------------------------------------------------ #

    def log(self, message: str, level: str = "info") -> None:
        color = LOG_COLORS.get(level, LOG_COLORS["info"])
        text = html.escape(str(message)).replace("  ", "&nbsp;&nbsp;")
        stamp = time.strftime("%H:%M:%S")
        self.log_output.appendHtml(
            f'<span style="color:{PALETTE["faint"]}">{stamp}</span>&nbsp;&nbsp;<span style="color:{color}">{text}</span>'
        )

    def clear_log(self) -> None:
        self.log_output.clear()

    def copy_log(self) -> None:
        QApplication.clipboard().setText(self.log_output.toPlainText())
        self.statusBar().showMessage("Log copiado para a área de transferência.", 3000)

    def show_shortcuts(self) -> None:
        qt_message(self, "Atalhos de teclado", SHORTCUTS_TEXT, "info")

    def show_about(self) -> None:
        qt_message(
            self, f"Sobre o {APP_TITLE}",
            f"{APP_TITLE} {APP_VERSION}\nMúsica → 6 LEDs (2 pontas + 4 no meio) via Arduino.\n\n"
            f"Dependências: {collect_dependency_summary()}", "info",
        )

    def _cleanup_sequence_temp_files(self, sequence: Optional[GeneratedSequence]) -> None:
        if not sequence:
            return
        for path in sequence.temp_files:
            with contextlib.suppress(OSError):
                os.remove(path)

    def _restore_settings(self) -> None:
        try:
            settings = QSettings("VisionAudio", "VisionAudio")
            geometry = settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
            state = settings.value("windowState")
            if state:
                self.restoreState(state)
            self._last_dir = str(settings.value("last_dir", "") or "")
            mode_index = self.source_combo.findData(str(settings.value("source_mode", "local")))
            if mode_index >= 0:
                self.source_combo.setCurrentIndex(mode_index)
            last_port = str(settings.value("port", "") or "")
            index = self.port_combo.findText(last_port) if last_port else -1
            if index >= 0:
                self.port_combo.setCurrentIndex(index)
        except Exception:
            pass

    def _save_settings(self) -> None:
        try:
            settings = QSettings("VisionAudio", "VisionAudio")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState())
            settings.setValue("last_dir", self._last_dir)
            settings.setValue("source_mode", self.source_combo.currentData())
            settings.setValue("port", self.port_combo.currentText())
        except Exception:
            pass

    def closeEvent(self, event: Any) -> None:
        self._save_settings()
        self.stop_playback()
        self.video_window_controller.close()
        self.hardware_controller.disconnect()
        self._cleanup_sequence_temp_files(self.current_sequence)
        self.generation_service.cleanup()
        super().closeEvent(event)


def collect_dependency_summary() -> str:
    checks = [
        ("PyQt5", dependency_available("PyQt5")),
        ("numpy", dependency_available("numpy")),
        ("scipy", dependency_available("scipy")),
        ("librosa", dependency_available("librosa")),
        ("pygame", dependency_available("pygame")),
        ("pyserial", dependency_available("serial")),
        ("yt-dlp", dependency_available("yt_dlp")),
        ("scikit-learn", dependency_available("sklearn")),
        ("ffmpeg", shutil.which("ffmpeg") is not None),
        ("js-runtime(yt-dlp)", any(shutil.which(exe) for exe in ("deno", "node", "bun", "qjs"))),
    ]
    return "  ".join(f"{name}={'ok' if ok else 'ausente'}" for name, ok in checks)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    apply_dark_theme(app)
    window = VisionAudioWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
