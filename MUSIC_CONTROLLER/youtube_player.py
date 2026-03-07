#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BRED = "\033[91m"
    BGREEN = "\033[92m"
    BYELLOW = "\033[93m"
    BCYAN = "\033[96m"
    BMAGENTA = "\033[95m"
    BWHITE = "\033[97m"


def _c(text, *codes):
    return "".join(codes) + str(text) + C.RESET


def banner():
    print()
    print(_c("  YouTube Music Player", C.BMAGENTA, C.BOLD))
    print(_c("  ver/ouvir ou somente ouvir", C.DIM))
    print()


def log_ok(msg):
    print(_c("  ✓ ", C.BGREEN, C.BOLD) + _c(msg, C.BWHITE))


def log_err(msg):
    print(_c("  ✗ ", C.BRED, C.BOLD) + _c(msg, C.BWHITE))


def log_warn(msg):
    print(_c("  ⚠ ", C.BYELLOW, C.BOLD) + _c(msg, C.BWHITE))


def log_step(msg):
    print(_c("\n  ▶ ", C.BCYAN, C.BOLD) + _c(msg, C.BWHITE, C.BOLD))


@dataclass
class YouTubeMedia:
    title: str
    webpage_url: str
    audio_stream_url: str
    video_stream_url: str
    duration_s: int | None


def _require_yt_dlp():
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Biblioteca yt-dlp não encontrada. Instale com: pip install yt-dlp"
        ) from exc
    return yt_dlp


def _pick_audio_stream(info):
    stream_url = info.get("url")
    formats = info.get("formats") or []
    if stream_url:
        return stream_url

    audio_only = [
        f
        for f in formats
        if f.get("url")
        and f.get("acodec") not in (None, "none")
        and f.get("vcodec") in (None, "none")
    ]
    if not audio_only:
        return None
    audio_only.sort(key=lambda f: (f.get("abr") or 0.0, f.get("tbr") or 0.0))
    return audio_only[-1]["url"]


def _pick_video_stream(info, fallback_audio_url):
    formats = info.get("formats") or []
    av_formats = [
        f
        for f in formats
        if f.get("url")
        and f.get("acodec") not in (None, "none")
        and f.get("vcodec") not in (None, "none")
    ]
    if av_formats:
        av_formats.sort(key=lambda f: (f.get("height") or 0.0, f.get("tbr") or 0.0))
        return av_formats[-1]["url"]
    return info.get("webpage_url") or fallback_audio_url


def resolve_youtube_media(query_or_url: str) -> YouTubeMedia:
    yt_dlp = _require_yt_dlp()

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestvideo+bestaudio/best",
        "default_search": "ytsearch1",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query_or_url, download=False)

    if not info:
        raise RuntimeError("yt-dlp não retornou informações do vídeo")

    if "entries" in info and info["entries"]:
        info = next((entry for entry in info["entries"] if entry), None)
        if not info:
            raise RuntimeError("Nenhum resultado válido encontrado no YouTube")

    audio_stream_url = _pick_audio_stream(info)
    if not audio_stream_url:
        raise RuntimeError("Não foi possível resolver stream de áudio")

    video_stream_url = _pick_video_stream(info, fallback_audio_url=audio_stream_url)
    title = info.get("title") or "YouTube"
    webpage_url = info.get("webpage_url") or ""
    duration_s = info.get("duration")
    if duration_s is not None:
        try:
            duration_s = int(duration_s)
        except Exception:
            duration_s = None

    return YouTubeMedia(
        title=title,
        webpage_url=webpage_url,
        audio_stream_url=audio_stream_url,
        video_stream_url=video_stream_url,
        duration_s=duration_s,
    )


def _run_blocking(cmd):
    proc = subprocess.Popen(cmd)
    return proc.wait()


def _play_watch_mode(source_url: str, fallback_stream_url: str | None = None) -> bool:
    mpv_bin = shutil.which("mpv")
    if mpv_bin:
        vo_candidates = ("tct", "caca")
        for vo in vo_candidates:
            cmd = [
                mpv_bin,
                "--no-config",
                "--really-quiet",
                "--terminal=yes",
                "--force-window=no",
                "--video-sync=audio",
                "--framedrop=vo",
                "--vf=fps=60",
                f"--vo={vo}",
                source_url,
            ]
            proc = subprocess.Popen(cmd)
            time.sleep(0.7)
            if proc.poll() is not None:
                if proc.returncode == 0:
                    return True
                continue
            try:
                while proc.poll() is None:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                if proc.poll() is None:
                    proc.terminate()
                return False
            if proc.returncode in (0, None):
                return True
        log_err(
            "mpv não conseguiu renderizar vídeo no terminal "
            "(vo=tct/caca)."
        )
        return False

    ffplay_bin = shutil.which("ffplay")
    if ffplay_bin:
        ffplay_source = fallback_stream_url or source_url
        cmd = [ffplay_bin, "-autoexit", "-loglevel", "quiet", ffplay_source]
        return _run_blocking(cmd) == 0

    log_err("Nenhum player encontrado para vídeo. Instale mpv ou ffplay.")
    return False


def _play_audio_mode(source_url: str) -> bool:
    mpv_bin = shutil.which("mpv")
    if mpv_bin:
        cmd = [
            mpv_bin,
            "--no-config",
            "--really-quiet",
            "--no-video",
            "--force-window=no",
            "--ytdl-format=bestaudio/best",
            source_url,
        ]
        return _run_blocking(cmd) == 0

    ffplay_bin = shutil.which("ffplay")
    if ffplay_bin:
        cmd = [ffplay_bin, "-nodisp", "-autoexit", "-loglevel", "quiet", source_url]
        return _run_blocking(cmd) == 0

    log_err("Nenhum player encontrado para áudio. Instale mpv ou ffplay.")
    return False


def _format_duration(duration_s: int | None) -> str:
    if duration_s is None:
        return "desconhecida"
    m, s = divmod(int(duration_s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def select_youtube_query() -> str | None:
    try:
        from PyQt5.QtWidgets import QApplication, QInputDialog
    except Exception:
        log_err("PyQt5 não encontrado. Instale com: pip install PyQt5")
        return None

    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    query, ok = QInputDialog.getText(
        None,
        "Buscar no YouTube",
        "Digite o nome da música ou cole a URL:",
    )
    if owns_app:
        app.quit()

    if not ok or not query.strip():
        return None
    return query.strip()


def select_play_mode() -> str | None:
    try:
        from PyQt5.QtWidgets import QApplication, QInputDialog
    except Exception:
        log_err("PyQt5 não encontrado. Instale com: pip install PyQt5")
        return None

    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    options = ["Ver + ouvir", "Somente ouvir"]
    choice, ok = QInputDialog.getItem(
        None,
        "Modo de Reprodução",
        "Escolha como deseja reproduzir:",
        options,
        0,
        False,
    )
    if owns_app:
        app.quit()

    if not ok:
        return None
    return "watch" if choice == options[0] else "audio"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduz músicas do YouTube (ver+ouvir ou somente ouvir)."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Nome da música ou URL do YouTube",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--watch",
        action="store_true",
        help="Ver e ouvir (vídeo + áudio).",
    )
    mode_group.add_argument(
        "--audio-only",
        action="store_true",
        help="Somente ouvir.",
    )
    return parser.parse_args()


def _silence_terminal_warnings():
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault(
        "QT_LOGGING_RULES",
        "*.debug=false;qt.qpa.*=false;qt.*.warning=false",
    )


def main():
    _silence_terminal_warnings()
    args = _parse_args()
    banner()

    query = " ".join(args.query).strip()
    if not query:
        query = select_youtube_query()
    if not query:
        log_warn("Nenhuma música informada. Encerrando.")
        return 1

    if args.watch:
        mode = "watch"
    elif args.audio_only:
        mode = "audio"
    else:
        mode = select_play_mode()
    if mode is None:
        log_warn("Nenhum modo selecionado. Encerrando.")
        return 1

    log_step("Resolvendo stream do YouTube via yt-dlp...")
    try:
        media = resolve_youtube_media(query)
    except Exception as exc:
        log_err(f"Falha ao resolver stream: {exc}")
        return 1

    log_ok(f"Música: {_c(media.title, C.BYELLOW)}")
    log_ok(f"Duração: {_c(_format_duration(media.duration_s), C.BYELLOW)}")
    if media.webpage_url:
        log_ok(f"Link: {_c(media.webpage_url, C.DIM)}")

    log_step(
        "Reprodução: "
        + (_c("ver + ouvir", C.BCYAN) if mode == "watch" else _c("somente ouvir", C.BCYAN))
    )

    ok = (
        _play_watch_mode(
            media.video_stream_url,
            fallback_stream_url=media.video_stream_url or media.webpage_url,
        )
        if mode == "watch"
        else _play_audio_mode(media.audio_stream_url)
    )
    if not ok:
        log_err("Falha na reprodução.")
        return 1

    log_ok("Reprodução finalizada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
