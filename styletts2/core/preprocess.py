from pathlib import Path
from typing import TYPE_CHECKING

from everyvoice.config.type_definitions import DatasetTextRepresentation

if TYPE_CHECKING:
    from ..ev_config import StyleTTS2Config


def load_config(
    config_file: Path,
    config_args: list[str] = [],
) -> "StyleTTS2Config":
    """Load StyleTTS2 configuration from config_file, possibly overriding some parameters"""
    from everyvoice.utils import spinner

    with spinner():
        from everyvoice.base_cli.helpers import load_config_base_command

        from ..ev_config import StyleTTS2Config

    config = load_config_base_command(
        model_config=StyleTTS2Config,
        config_file=config_file,
        config_args=config_args,
    )
    assert isinstance(config, StyleTTS2Config)
    return config


PREPROCESS_CATEGORIES = ["audio", "text"]


def preprocess(
    config: "StyleTTS2Config",
    steps: list[str],
    cpus: int,
    overwrite: bool,
    debug: bool,
):
    """Preprocess audio and text data for StyleTTS2 training."""
    from everyvoice.base_cli.helpers import preprocess_base_command

    preprocessor, _ = preprocess_base_command(
        config=config, steps=steps, cpus=cpus, overwrite=overwrite, debug=debug
    )

    if not config.training.ood_raw_data:
        return

    resolved: dict[str, tuple[Path, DatasetTextRepresentation]] = {}
    for lang, source in config.training.ood_raw_data.items():
        if source.hf is not None:
            from huggingface_hub import hf_hub_download

            local_path = Path(
                hf_hub_download(
                    source.hf.repo_id,
                    repo_type="dataset",
                    filename=source.hf.filename,
                    revision=source.hf.revision,
                )
            )
        else:
            assert source.local_path is not None  # guaranteed by "after" validator
            local_path = source.local_path
        resolved[lang] = (local_path, source.text_representation)

    preprocessor.preprocess_ood(resolved)
