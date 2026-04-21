import os
import unittest
from unittest import mock
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

from visionaudio7 import (
    CONTROL_SPECS,
    EffectConfigCompat,
    GenerationService,
    GeneratedSequence,
    LoadWorker,
    LoadedTrack,
    ParameterRecommendationService,
    RecommendationResult,
    UIConfigState,
    VisionAudioWindow,
)


def get_app() -> QApplication:
    return QApplication.instance() or QApplication([])


class FakeHardwareController:
    def __init__(self) -> None:
        self._connected = False
        self.last_led_values = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def available_ports(self):
        return ["/dev/ttyACM0"]

    def connect_port(self, _port: str, baudrate: int = 115200) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def send_led_command(self, led_values, mode: int = 1) -> None:
        self.last_led_values = (mode, list(led_values))


class FakePlaybackController(QObject):
    state_changed = pyqtSignal(str)
    position_changed = pyqtSignal(float, float)
    frame_changed = pyqtSignal(object)
    playback_finished = pyqtSignal()
    playback_error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.hardware_controller = None
        self.state = "idle"
        self.play_calls = 0
        self.pause_calls = 0
        self.stop_calls = 0

    def set_hardware_controller(self, hardware_controller):
        self.hardware_controller = hardware_controller

    def play(self, sequence):
        self.play_calls += 1
        self.state = "playing"
        self.state_changed.emit("playing")
        self.position_changed.emit(1.0, sequence.duration_s)
        self.frame_changed.emit([10, 50, 90, 90, 50, 10])

    def pause_toggle(self):
        self.pause_calls += 1
        self.state = "paused" if self.state == "playing" else "playing"
        self.state_changed.emit(self.state)

    def stop(self):
        self.stop_calls += 1
        self.state = "idle"
        self.state_changed.emit("idle")
        self.position_changed.emit(0.0, 0.0)
        self.frame_changed.emit([0, 0, 0, 0, 0, 0])

    def seek_relative(self, _delta: int):
        return None

    def seek_ratio(self, _ratio: float):
        return None


class FakeSourceResolver:
    def load(self, source_mode: str, source_value: str):
        return LoadedTrack(
            source_mode=source_mode,
            requested_source=source_value,
            resolved_source=source_value,
            title="Track Fake",
            display_source=source_value,
            file_size_bytes=1234,
        )


class FakeGenerationService:
    def extract_features(self, _track, progress_cb=None):
        if progress_cb:
            progress_cb(100, "ok")
        return {
            "tempo_bpm": 124.0,
            "beat_density": 0.62,
            "rms_mean": 0.31,
            "rms_p90": 0.53,
            "bass_ratio": 0.58,
            "centroid_mean": 0.28,
            "centroid_std": 0.12,
            "flatness_mean": 0.30,
            "flux_mean": 0.22,
            "boundaries_per_minute": 2.4,
        }

    def generate(self, track, config_state, progress_cb=None):
        if progress_cb:
            progress_cb(100, "ok")
        fake_transition = SimpleNamespace(
            frame_start=30,
            frame_end=60,
            effect_type=SimpleNamespace(name="SWEEP"),
        )
        return GeneratedSequence(
            track=track,
            config_state=config_state,
            effect_config=EffectConfigCompat(),
            led_patterns=[(1, [0, 20, 40, 40, 20, 0])] * 120,
            dominant_leds=[2] * 120,
            duration_s=4.0,
            transitions=[fake_transition],
            playback_source=track.resolved_source,
            tempo_bpm=124.0,
            beat_frames=[0, 15, 30],
            feature_snapshot=self.extract_features(track),
            preview_columns=[[0, 20, 40, 40, 20, 0]] * 20,
            temp_files=[],
        )


class FailingSourceResolver:
    def load(self, _source_mode: str, _source_value: str):
        raise RuntimeError()


class FailingHardwareController(FakeHardwareController):
    def connect_port(self, _port: str, baudrate: int = 115200) -> None:
        raise RuntimeError()


class KeywordOnlyPeakPick:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, onset_env, *, pre_max, post_max, pre_avg, post_avg, delta, wait):
        self.calls.append(
            {
                "onset_env": onset_env,
                "pre_max": pre_max,
                "post_max": post_max,
                "pre_avg": pre_avg,
                "post_avg": post_avg,
                "delta": delta,
                "wait": wait,
            }
        )
        return [1, 4]


class PositionalOnlyPeakPick:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, onset_env, pre_max, post_max, pre_avg, post_avg, delta, wait, /):
        self.calls.append((onset_env, pre_max, post_max, pre_avg, post_avg, delta, wait))
        return [2, 5, 8]


class VisionAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def test_ui_config_to_effect_config_clamps_and_casts(self):
        config_state = UIConfigState(
            transition_profile="high",
            effect_profile="low",
            middle_speed_multiplier=2.0,
            values={
                "ACTIVE_MAX_PWM": 999.0,
                "TRANS_BLEND_FRAMES": 7.8,
                "SEG_NOVELTY_THRESHOLD": -2.0,
            },
        )

        effect_config = config_state.to_effect_config()

        self.assertIsInstance(effect_config, EffectConfigCompat)
        self.assertEqual(effect_config.ACTIVE_MAX_PWM, CONTROL_SPECS["ACTIVE_MAX_PWM"].maximum)
        self.assertEqual(effect_config.TRANS_BLEND_FRAMES, 8)
        self.assertEqual(
            effect_config.SEG_NOVELTY_THRESHOLD,
            CONTROL_SPECS["SEG_NOVELTY_THRESHOLD"].minimum,
        )
        self.assertEqual(effect_config.SEG_K_SEGMENTS, 74)

    def test_recommendation_service_returns_valid_ranges(self):
        service = ParameterRecommendationService()
        result = service.recommend(
            {
                "tempo_bpm": 126.0,
                "beat_density": 0.72,
                "rms_mean": 0.36,
                "rms_p90": 0.61,
                "bass_ratio": 0.64,
                "centroid_mean": 0.30,
                "centroid_std": 0.10,
                "flatness_mean": 0.32,
                "flux_mean": 0.27,
                "boundaries_per_minute": 2.8,
            }
        )

        self.assertIsInstance(result, RecommendationResult)
        for key, value in result.numeric_values.items():
            spec = CONTROL_SPECS[key]
            self.assertGreaterEqual(value, spec.minimum)
            self.assertLessEqual(value, spec.maximum)
        self.assertIn(result.categorical_values["transition_profile"], {"low", "medium", "high"})
        self.assertIn(result.categorical_values["effect_profile"], {"low", "medium", "high"})
        self.assertIn(result.categorical_values["middle_speed_multiplier"], {1.0, 2.0})

    def test_peak_pick_helper_supports_keyword_only_signature(self):
        peak_pick = KeywordOnlyPeakPick()
        librosa_module = SimpleNamespace(util=SimpleNamespace(peak_pick=peak_pick))
        service = GenerationService()

        result = service._peak_pick_onsets(librosa_module, [0.1, 0.4, 0.8])

        self.assertEqual(result, [1, 4])
        self.assertEqual(len(peak_pick.calls), 1)
        self.assertEqual(peak_pick.calls[0]["pre_max"], 3)
        self.assertEqual(peak_pick.calls[0]["wait"], 5)

    def test_peak_pick_helper_falls_back_to_positional_signature(self):
        peak_pick = PositionalOnlyPeakPick()
        librosa_module = SimpleNamespace(util=SimpleNamespace(peak_pick=peak_pick))
        service = GenerationService()

        result = service._peak_pick_onsets(librosa_module, [0.2, 0.6, 0.9])

        self.assertEqual(result, [2, 5, 8])
        self.assertEqual(len(peak_pick.calls), 1)
        self.assertEqual(peak_pick.calls[0][1:], (3, 3, 3, 5, 0.5, 5))
        self.assertGreater(float(len(result)) / (30.0 / 60.0), 0.0)

    def test_window_flow_loaded_recommended_generated_and_playing(self):
        playback = FakePlaybackController()
        hardware = FakeHardwareController()
        window = VisionAudioWindow(
            source_resolver=FakeSourceResolver(),
            generation_service=FakeGenerationService(),
            recommendation_service=ParameterRecommendationService(),
            playback_controller=playback,
            hardware_controller=hardware,
        )

        track = LoadedTrack(
            source_mode="local",
            requested_source="/tmp/demo.wav",
            resolved_source="/tmp/demo.wav",
            title="Demo",
            display_source="/tmp/demo.wav",
            file_size_bytes=1024,
        )
        window.handle_loaded_track(track)
        self.assertEqual(window.current_track.title, "Demo")
        self.assertTrue(window.suggest_button.isEnabled())
        self.assertTrue(window.generate_button.isEnabled())

        recommendation = window.recommendation_service.recommend(
            window.generation_service.extract_features(track)
        )
        window.handle_recommendation(recommendation)
        self.assertIsNotNone(window.current_recommendation)
        self.assertIn("Modelo:", window.feature_summary_label.text())

        sequence = window.generation_service.generate(track, window.current_ui_state())
        window.handle_generated_sequence(sequence)
        self.assertIsNotNone(window.current_sequence)
        self.assertEqual(window.transition_list.count(), 1)
        self.assertTrue(window.play_button.isEnabled())

        window.toggle_playback()
        self.assertEqual(playback.play_calls, 1)
        self.assertEqual(window.player_state_label.text(), "Tocando")
        self.assertEqual(window.play_button.text(), "Pause")

        window.stop_playback()
        self.assertEqual(playback.stop_calls, 1)
        self.assertEqual(window.player_state_label.text(), "Idle")
        window.close()

    def test_layout_uses_splitters_and_preview_panel_is_retractable(self):
        window = VisionAudioWindow(
            source_resolver=FakeSourceResolver(),
            generation_service=FakeGenerationService(),
            recommendation_service=ParameterRecommendationService(),
            playback_controller=FakePlaybackController(),
            hardware_controller=FakeHardwareController(),
        )
        window.show()
        window._configure_initial_splitters()
        self.app.processEvents()

        self.assertEqual(window.content_splitter.objectName(), "MainContentSplitter")
        self.assertEqual(window.monitor_splitter.objectName(), "MonitorSplitter")
        self.assertFalse(hasattr(window, "preview_toggle_button"))
        self.assertEqual(window.preview_panel.minimumWidth(), 0)
        initial_sizes = window.content_splitter.sizes()
        self.assertGreater(initial_sizes[2], 0)

        window.content_splitter.setSizes([260, 920, 0])
        self.app.processEvents()
        collapsed_sizes = window.content_splitter.sizes()
        self.assertEqual(collapsed_sizes[2], 0)

        window.content_splitter.setSizes([260, 700, 320])
        self.app.processEvents()
        expanded_sizes = window.content_splitter.sizes()
        self.assertGreater(expanded_sizes[2], 0)
        window.close()

    def test_logs_panel_has_taller_defaults(self):
        window = VisionAudioWindow(
            source_resolver=FakeSourceResolver(),
            generation_service=FakeGenerationService(),
            recommendation_service=ParameterRecommendationService(),
            playback_controller=FakePlaybackController(),
            hardware_controller=FakeHardwareController(),
        )
        window.show()
        window._configure_initial_splitters()
        self.app.processEvents()

        self.assertGreaterEqual(window.log_output.minimumHeight(), 120)
        self.assertEqual(window.log_output.maximumHeight(), 220)
        self.assertGreaterEqual(window.log_output.parentWidget().minimumHeight(), 180)
        self.assertEqual(window.log_output.parentWidget().maximumHeight(), 280)
        splitter_sizes = window.monitor_splitter.sizes()
        self.assertGreater(splitter_sizes[1], splitter_sizes[0])
        window.close()

    def test_load_worker_error_emits_message_and_details(self):
        worker = LoadWorker(FailingSourceResolver(), "local", "/tmp/invalido.wav")
        received = []
        worker.failed.connect(lambda message, details: received.append((message, details)))

        worker.run()

        self.assertEqual(len(received), 1)
        message, details = received[0]
        self.assertTrue(message)
        self.assertIn("RuntimeError", message)
        self.assertTrue(details)

    def test_worker_error_popup_receives_non_empty_message_and_details(self):
        window = VisionAudioWindow(
            source_resolver=FakeSourceResolver(),
            generation_service=FakeGenerationService(),
            recommendation_service=ParameterRecommendationService(),
            playback_controller=FakePlaybackController(),
            hardware_controller=FakeHardwareController(),
        )

        with mock.patch("visionaudio7.qt_message") as qt_message_mock:
            window._handle_worker_error("", "traceback line")

        args = qt_message_mock.call_args.args
        self.assertEqual(args[1], "VisionAudio7")
        self.assertTrue(args[2])
        self.assertEqual(args[4], "traceback line")
        window.close()

    def test_connection_and_playback_errors_use_visible_messages(self):
        window = VisionAudioWindow(
            source_resolver=FakeSourceResolver(),
            generation_service=FakeGenerationService(),
            recommendation_service=ParameterRecommendationService(),
            playback_controller=FakePlaybackController(),
            hardware_controller=FailingHardwareController(),
        )
        window.port_combo.setEditText("/dev/ttyACM0")

        with mock.patch("visionaudio7.qt_message") as qt_message_mock:
            window.toggle_connection()
            window._on_playback_error("")

        first_call = qt_message_mock.call_args_list[0].args
        second_call = qt_message_mock.call_args_list[1].args
        self.assertEqual(first_call[1], "Arduino")
        self.assertTrue(first_call[2])
        self.assertIn("RuntimeError", first_call[2])
        self.assertEqual(second_call[1], "Player")
        self.assertTrue(second_call[2])
        self.assertIn("PlaybackError", second_call[2])
        window.close()


if __name__ == "__main__":
    unittest.main()
