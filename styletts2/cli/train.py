from typing import Annotated

import typer
from everyvoice.base_cli.interfaces import train_base_command_interface
from merge_args import merge_args

from .. import core
from ..core.train import Mode


@merge_args(train_base_command_interface)
def train(
    mode: Annotated[
        Mode,
        typer.Option(
            "-m",
            "--mode",
            help="Training mode: 'first' (acoustic pre-training with TMA), 'second' (joint diffusion+adversarial), or 'finetune'.",
        ),
    ] = Mode.first,
    precision: Annotated[
        str,
        typer.Option(
            help="Floating-point precision passed to Lightning Trainer (e.g. '32', '16-mixed', 'bf16-mixed').",
        ),
    ] = "32",
    **kwargs,
):
    """Train an end-to-end (StyleTTS2) model

    For example:

    **styletts2 train config/everyvoice-text-to-wav.yaml --mode first**
    """
    ev_config = core.load_config(
        config_file=kwargs["config_file"],  # don't pop this one, train() needs it
        config_args=kwargs.pop("config_args"),
    )
    core.train(config=ev_config, mode=mode, precision=precision, **kwargs)
