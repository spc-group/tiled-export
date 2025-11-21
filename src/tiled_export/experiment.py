import shutil
from pathlib import Path

from tiled_export.notebook import prepare_notebook
from tiled_export.protocols import Experiment


def copy_template(path: Path) -> None:
    """Create pixi.toml, jupyterbook, etc for an experiment."""
    template = Path(__file__).parent / "experiment_template"
    for filename in [
        "pixi.toml",
        ".gitignore",
        ".gitattributes",
        "rois.toml",
        "analysis.ipynb",
        "xraytools/__init__.py",
        "xraytools/xrf_analysis.py",
    ]:
        src = template / filename
        dest = path / filename
        dest.parent.mkdir(parents=False, exist_ok=True)
        if not dest.exists():
            shutil.copy(str(src), str(dest))


async def prepare_experiment(folder: Path, name: str) -> Experiment:
    """Set up folder structure and make API calls for a new experiment.

    If the experiment is already set up, no action will be taken.

    If parts of the experiment are set up, only those missing will be
    added.

    """
    folder.mkdir(parents=True, exist_ok=True)
    copy_template(folder)
    notebook_file = folder / "analysis.ipynb"
    experiment = Experiment(notebook=notebook_file, name=name)
    prepare_notebook(notebook_file, experiment=experiment)
    return experiment
