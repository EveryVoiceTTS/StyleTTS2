import sys
from typing import Annotated

import typer
from everyvoice.base_cli import make_enum
from everyvoice.base_cli.interfaces import preprocess_base_command_interface
from loguru import logger
from merge_args import merge_args

from .. import core


@merge_args(preprocess_base_command_interface)
def preprocess(
    steps: Annotated[
        list[make_enum(core.PREPROCESS_CATEGORIES)],  # type: ignore[valid-type]
        typer.Option(
            "-s",
            "--steps",
            help="Which preprocessing steps to run. If none are provided, text and audio processing steps are performed.",
        ),
    ] = core.PREPROCESS_CATEGORIES,
    **kwargs,
):
    """Preprocess data for text-to-wav (StyleTTS2) training

    **styletts2 preprocess config/everyvoice-text-to-wav.yaml**
    """
    try:
        config = core.load_config(
            config_file=kwargs.pop("config_file"), config_args=kwargs.pop("config_args")
        )
        core.preprocess(config, steps=[step.name for step in steps], **kwargs)
    except typer.BadParameter as e:
        raise e
    except Exception as e:
        logger.error(f"{e.__class__.__name__}: {e}")
        if kwargs.get("debug", False):
            raise e
        else:
            logger.info("Rerun with --debug to see the full stack trace.")
            sys.exit(1)
