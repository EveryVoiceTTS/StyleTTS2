import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from everyvoice.base_cli.interfaces import (
    ConfigArgsOption,
    ConfigFileArgument,
    CPUsOption,
    DebugFlag,
    OverwriteFlag,
)


class PreprocessCategories(str, Enum):
    audio = "audio"
    text = "text"


StepsOption = typer.Option(
    "-s",
    "--steps",
    help="Which preprocessing steps to run. If none are provided, text and audio processing steps are performed.",
)


def preprocess(
    config_file: Annotated[Path, ConfigFileArgument],
    steps: Annotated[list[PreprocessCategories], StepsOption] = list(
        PreprocessCategories
    ),
    config_args: Annotated[list[str], ConfigArgsOption] = [],
    cpus: Annotated[int, CPUsOption] = min(4, mp.cpu_count()),
    overwrite: Annotated[bool, OverwriteFlag] = False,
    debug: Annotated[bool, DebugFlag] = False,
):
    """Preprocess audio and text data for StyleTTS2 training."""
    from everyvoice.utils import spinner

    with spinner():
        from everyvoice.base_cli.helpers import preprocess_base_command

        from ..ev_config import StyleTTS2Config

    preprocessor, config, _ = preprocess_base_command(
        model_config=StyleTTS2Config,
        steps=[step.name for step in steps],
        config_file=config_file,
        config_args=config_args,
        cpus=cpus,
        overwrite=overwrite,
        debug=debug,
    )

    if not config.training.ood_raw_data:
        return

    resolved: dict[str, tuple[Path, object]] = {}
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
            local_path = source.local_path
        resolved[lang] = (local_path, source.text_representation)

    preprocessor.preprocess_ood(resolved)
