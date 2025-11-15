import shutil
from pathlib import Path

import nbformat
import pytest
import pytest_asyncio

from tiled_export.notebook import add_run, prepare_notebook, role


@pytest.fixture()
def notebook(tmp_path):
    src_file = (
        Path(__file__).parent.parent
        / "src/tiled_export/experiment_template/analysis.ipynb"
    )
    new_file = tmp_path / "analysis.ipynb"
    shutil.copy(src_file, new_file)
    return new_file


@pytest_asyncio.fixture()
async def run(mocker):
    run = mocker.AsyncMock()
    run.metadata = {"start": {"uid": "12345"}}
    return run


@pytest.mark.asyncio
async def test_add_run_cells(notebook, run):
    await add_run(run, notebook)
    nb = nbformat.read(notebook, as_version=4)
    run_cells = [cell for cell in nb.cells if role(cell) == "run"]
    assert len(run_cells) == 2
    assert run_cells[0].metadata["tiled_export"]["run"]["uid"] == "12345"
    assert run_cells[1].metadata["tiled_export"]["run"]["uid"] == "12345"


@pytest.mark.asyncio
async def test_remove_instructions(notebook, run):
    """Make sure the tutorial instructions, etc get removed."""
    prepare_notebook(notebook)
    nb = nbformat.read(notebook, as_version=4)
    template_instructions = [cell for cell in nb.cells if role(cell) == "delete"]
    assert len(template_instructions) == 0
