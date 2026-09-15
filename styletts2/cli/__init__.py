import typer
from everyvoice.base_cli import command, default_typer_args

from .fetch_pretrained import fetch_pretrained
from .preprocess import preprocess
from .synthesize import synthesize
from .train import train

app = typer.Typer(
    **default_typer_args,
    help="A StyleTTS2 end-to-end text-to-speech model configured via EveryVoice.",
)

command(app)(preprocess)
command(app)(fetch_pretrained)
command(app)(train)
command(app)(synthesize)
