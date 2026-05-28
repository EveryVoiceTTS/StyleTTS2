from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from everyvoice.base_cli.interfaces import preprocess_base_command_interface
from merge_args import merge_args


class PreprocessCategories(str, Enum):
    audio = "audio"
    text = "text"


@merge_args(preprocess_base_command_interface)
def preprocess(
    steps: list[PreprocessCategories] = typer.Option(
        [cat.value for cat in PreprocessCategories],
        "-s",
        "--steps",
        help="Which preprocessing steps to run. If none are provided, text and audio processing steps are performed.",
    ),
    ood_data_file: Optional[Path] = typer.Option(
        None,
        "--ood-data-file",
        exists=True,
        help="Path to a plain-text OOD file (one utterance per line) to preprocess alongside the main data. "
        "Produces ood.psv in the preprocessed output directory, which is used automatically during training.",
    ),
    **kwargs,
):
    """Preprocess audio and text data for StyleTTS2 training."""
    from everyvoice.utils import spinner

    with spinner():
        from everyvoice.base_cli.helpers import preprocess_base_command

        from ..ev_config import (
            StyleTTS2Config,
        )

    preprocess_base_command(
        model_config=StyleTTS2Config,
        steps=[step.name for step in steps],
        ood_data_file=ood_data_file,
        **kwargs,
    )
