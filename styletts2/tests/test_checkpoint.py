"""Tests for the config saved in StyleTTS2 checkpoints."""

import io
from pathlib import Path

import pytest
import torch
from everyvoice.config.text_config import TextConfig
from everyvoice.tests.stubs import TEST_CONTACT


def find_strings_containing(value, needle: str, where="ev_config") -> list[str]:
    """Return where in value there are strings or Paths containing needle."""
    if isinstance(value, dict):
        return [
            found
            for key, item in value.items()
            for found in find_strings_containing(item, needle, f"{where}.{key}")
        ]
    if isinstance(value, (list, tuple)):
        return [
            found
            for i, item in enumerate(value)
            for found in find_strings_containing(item, needle, f"{where}[{i}]")
        ]
    if isinstance(value, (str, Path)) and needle in str(value):
        return [where]
    return []


class TestCheckpointConfig:
    """Checkpoints must contain no paths in their ev_config: loading a
    checkpoint validates its ev_config, which fails for a file that doesn't
    exist and creates a directory that doesn't exist."""

    @pytest.fixture
    def config_with_all_paths(self, tmp_path):
        from styletts2.ev_config import StyleTTS2Config

        file = tmp_path / "file.txt"
        file.write_text("x\n", encoding="utf8")
        directory = tmp_path / "dir"
        directory.mkdir()
        text_config = tmp_path / "text.json"
        text_config.write_text(TextConfig().model_dump_json(), encoding="utf8")
        f, d = str(file), str(directory)
        return StyleTTS2Config(
            contact=TEST_CONTACT,
            path_to_text_config_file=str(text_config),
            training={
                "finetune_checkpoint": f,
                "training_filelist": f,
                "validation_filelist": f,
                "root_path": d,
                "first_stage_path": f,
                "logger": {"save_dir": str(tmp_path / "logs")},
                "ood_raw_data": {
                    "eng": {"local_path": f},
                    "fra": {"hf": {"repo_id": "org/ood"}},
                },
            },
            preprocessing={
                "save_dir": str(tmp_path / "preprocessed"),
                "source_data": [
                    {"data_dir": d, "filelist": f, "permissions_obtained": True}
                ],
            },
            pretrained={
                "f0": {"local_path": f},
                "asr": {"local_checkpoint": f, "local_config": f},
                "plbert": {"local_checkpoint": f, "local_config": f},
            },
        )

    @staticmethod
    def save_and_reload(ev_config) -> dict:
        """Save ev_config in a checkpoint the way training does, and load the
        checkpoint back the way synthesis does."""
        from styletts2.ev_config.translation import to_native_config
        from styletts2.lightning import StyleTTS2

        model = StyleTTS2()
        model.config = to_native_config(ev_config)
        model.mode = "second"
        checkpoint: dict = {"state_dict": {}}
        model.on_save_checkpoint(checkpoint)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        return torch.load(buffer, weights_only=True)

    def test_ev_config_has_no_paths(self, config_with_all_paths, tmp_path):
        checkpoint = self.save_and_reload(config_with_all_paths)
        dumped = checkpoint["hyper_parameters"]["config"]["ev_config"]
        assert find_strings_containing(dumped, str(tmp_path)) == []
        # Only the hf OOD source is kept: a local one without its local_path
        # would be invalid
        assert list(dumped["training"]["ood_raw_data"]) == ["fra"]
        assert "first_stage_path" not in dumped["training"]

    def test_native_config_keeps_its_paths(self, config_with_all_paths, tmp_path):
        # Building the model from a checkpoint reads local_config from these
        checkpoint = self.save_and_reload(config_with_all_paths)
        native = checkpoint["hyper_parameters"]["config"]
        assert native["pretrained_asr"]["local_config"] == str(tmp_path / "file.txt")
        assert native["pretrained_plbert"]["local_config"] == str(tmp_path / "file.txt")

    def test_loads_elsewhere_creating_nothing(
        self, config_with_all_paths, tmp_path, monkeypatch
    ):
        from styletts2.ev_config import StyleTTS2Config

        checkpoint = self.save_and_reload(config_with_all_paths)
        # Load on "another machine": none of the saved paths exist any more
        for path in sorted(tmp_path.rglob("*"), reverse=True):
            path.rmdir() if path.is_dir() else path.unlink()
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        loaded = StyleTTS2Config.from_checkpoint_dump(
            checkpoint["hyper_parameters"]["config"]["ev_config"]
        )

        assert list(cwd.iterdir()) == []
        assert [p.name for p in tmp_path.iterdir()] == ["cwd"]
        assert list(loaded.training.ood_raw_data) == ["fra"]
        assert loaded.training.ood_raw_data["fra"].hf.repo_id == "org/ood"
        assert loaded.text == config_with_all_paths.text

    def test_default_config_round_trip(self, tmp_path, monkeypatch):
        from styletts2.ev_config import StyleTTS2Config

        monkeypatch.chdir(tmp_path)
        config = StyleTTS2Config(contact=TEST_CONTACT)
        checkpoint = self.save_and_reload(config)
        loaded = StyleTTS2Config.from_checkpoint_dump(
            checkpoint["hyper_parameters"]["config"]["ev_config"]
        )
        assert loaded.model_dump() == config.model_dump()
        assert list(tmp_path.iterdir()) == []
