from enum import Enum

import typer
from everyvoice.base_cli.interfaces import preprocess_base_command_interface
from merge_args import merge_args

from .. import core

PreprocessCategories = Enum(  # type: ignore[misc]
    "PreprocessCategories",
    {category: category for category in core.PREPROCESS_CATEGORIES},
    type=str,
)


@merge_args(preprocess_base_command_interface)
def preprocess(
    steps: list[PreprocessCategories] = typer.Option(
        [cat.value for cat in PreprocessCategories],
        "-s",
        "--steps",
        help="Which preprocessing steps to run. If none are provided, text and audio processing steps are performed.",
    ),
    **kwargs,
):
    """Preprocess data for text-to-wav (StyleTTS2) training

    **styletts2 preprocess config/everyvoice-text-to-wav.yaml**
    """
    config = core.load_config(
        config_file=kwargs.pop("config_file"), config_args=kwargs.pop("config_args")
    )
    core.preprocess(config, steps=[step.name for step in steps], **kwargs)
