import shutil
from pathlib import Path


def copy_template(path: Path) -> None:
    """Create pixi.toml, jupyterbook, etc for an experiment."""
    template = Path(__file__).parent / "experiment_template"
    for filename in ["pixi.toml", ".gitignore", ".gitattributes", "analysis.ipynb"]:
        shutil.copy(str(template / filename), str(path))


async def prepare_experiment(folder: Path) -> None:
    """Set up folder structure and make API calls for a new experiment.

    If the experiment is already set up, no action will be taken.

    If parts of the experiment are set up, only those missing will be
    added.

    """
    folder.mkdir(parents=True, exist_ok=True)
    copy_template(folder)
