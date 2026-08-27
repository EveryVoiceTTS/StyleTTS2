import typer
from everyvoice.base_cli import command, default_typer_args
from everyvoice.wizard import TEXT_TO_WAV_CONFIG_FILENAME_PREFIX

from .preprocess import preprocess as app_preprocess
from .train import train as app_train

app = typer.Typer(
    **default_typer_args,
    help="A StyleTTS2 end-to-end text-to-speech model configured via EveryVoice.",
)

command(
    app,
    name="preprocess",
    short_help="Preprocess your data",
    help=f"""Preprocess your data for StyleTTS2 training. For example:

    **everyvoice preprocess text-to-wav config/{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml**
    """,
)(app_preprocess)

command(
    app,
    name="train",
    short_help="Train your StyleTTS2 model",
    help=f"""Train a StyleTTS2 end-to-end model. For example:

    **everyvoice train text-to-wav config/{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml --mode first**
    """,
)(app_train)
