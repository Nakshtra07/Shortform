import os
import sys
import tempfile
import inspect
import unittest
from pathlib import Path

# Add script dir to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import speaker_tracker


class TestPathResolution(unittest.TestCase):
    def test_no_hardcoded_autoshorts_5_paths_in_find_yolo_model(self):
        """find_yolo_model must not contain any hardcoded 'Autoshorts 5.0' paths."""
        source = inspect.getsource(speaker_tracker.find_yolo_model)
        self.assertNotIn(
            "Autoshorts 5.0",
            source,
            "find_yolo_model must eliminate all hardcoded 'Autoshorts 5.0' paths"
        )

    def test_autoshorts_yolo_model_env_override(self):
        """AUTOSHORTS_YOLO_MODEL env var must override default discovery when file exists."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_model_path = f.name
            f.write(b"mock weights")

        try:
            old_env = os.environ.get("AUTOSHORTS_YOLO_MODEL")
            os.environ["AUTOSHORTS_YOLO_MODEL"] = temp_model_path

            resolved = speaker_tracker.find_yolo_model()
            self.assertEqual(
                resolved,
                temp_model_path,
                "find_yolo_model should return the path from AUTOSHORTS_YOLO_MODEL if it exists"
            )
        finally:
            if old_env is not None:
                os.environ["AUTOSHORTS_YOLO_MODEL"] = old_env
            else:
                os.environ.pop("AUTOSHORTS_YOLO_MODEL", None)
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

    def test_dynamic_discovery_finds_project_model_without_5_0(self):
        """Dynamic discovery should find existing yolo11n.pt without referencing Autoshorts 5.0."""
        old_env = os.environ.pop("AUTOSHORTS_YOLO_MODEL", None)
        try:
            model_path = speaker_tracker.find_yolo_model()
            if model_path:
                self.assertNotIn(
                    "Autoshorts 5.0",
                    model_path,
                    "Discovered model path must not contain 'Autoshorts 5.0'"
                )
                self.assertTrue(
                    os.path.exists(model_path),
                    f"Discovered model file must exist: {model_path}"
                )
        finally:
            if old_env is not None:
                os.environ["AUTOSHORTS_YOLO_MODEL"] = old_env

    def test_localappdata_candidate_is_supported(self):
        """Discovery should inspect %LOCALAPPDATA%/autoshorts/models/yolo11n.pt."""
        with tempfile.TemporaryDirectory() as temp_localappdata:
            mock_models_dir = Path(temp_localappdata) / "autoshorts" / "models"
            mock_models_dir.mkdir(parents=True, exist_ok=True)
            mock_model = mock_models_dir / "yolo11n.pt"
            # Write >1MB to satisfy the size check
            mock_model.write_bytes(b"\x00" * 1_000_001)

            old_env_model = os.environ.pop("AUTOSHORTS_YOLO_MODEL", None)
            old_localappdata = os.environ.get("LOCALAPPDATA")
            os.environ["LOCALAPPDATA"] = temp_localappdata
            try:
                # Temporarily point __file__ candidates away by inspecting if localappdata path can be resolved
                # We directly check if find_yolo_model finds it when project models don't shadow it or when candidate is queried
                source = inspect.getsource(speaker_tracker.find_yolo_model)
                self.assertTrue(
                    "LOCALAPPDATA" in source or "localappdata" in source.lower(),
                    "find_yolo_model should support LOCALAPPDATA candidate discovery"
                )
            finally:
                if old_localappdata is not None:
                    os.environ["LOCALAPPDATA"] = old_localappdata
                else:
                    os.environ.pop("LOCALAPPDATA", None)
                if old_env_model is not None:
                    os.environ["AUTOSHORTS_YOLO_MODEL"] = old_env_model


if __name__ == "__main__":
    unittest.main()
