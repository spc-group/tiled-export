import shutil
from pathlib import Path

import nbformat
import pytest
import pytest_asyncio

from tiled_export.notebook import add_run, execute_notebook, prepare_notebook, role


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
async def test_add_run_cells_idempotence(notebook, run):
    """Adding a run a second time should not do anything."""
    await add_run(run, notebook)
    nb = nbformat.read(notebook, as_version=4)
    run_cells = [cell for cell in nb.cells if role(cell) == "run"]
    assert len(run_cells) == 2
    # Do it a second time
    await add_run(run, notebook)
    nb = nbformat.read(notebook, as_version=4)
    run_cells = [cell for cell in nb.cells if role(cell) == "run"]
    assert len(run_cells) == 2


@pytest.mark.asyncio
async def test_add_run_data_file_paths(notebook, run):
    """Check that the XDI/NeXus file paths are in the template."""
    await add_run(
        run,
        notebook,
        xdi_file=notebook.parent / "data.xdi",
        hdf_file=notebook.parent / "data.h5",
    )
    nb = nbformat.read(notebook, as_version=4)
    run_cells = [cell for cell in nb.cells if role(cell) == "run"]
    code_cell = run_cells[1]
    print(code_cell)
    assert '.from_aps_25idc_nexus("data.h5")' in code_cell.source
    assert '.update_hdf_file("data.h5")' in code_cell.source
    assert '.update_xdi_file("data.xdi")' in code_cell.source


@pytest.mark.asyncio
async def test_remove_instructions(notebook, run):
    """Make sure the tutorial instructions, etc get removed."""
    prepare_notebook(notebook)
    nb = nbformat.read(notebook, as_version=4)
    template_instructions = [cell for cell in nb.cells if role(cell) == "delete"]
    assert len(template_instructions) == 0


@pytest.mark.asyncio
async def test_execute_notebook(tmp_path, mocker):
    mock = mocker.patch("tiled_export.notebook.asyncio.create_subprocess_exec")
    # Make a simple hello world notebook to execute
    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(nbformat.v4.new_code_cell(source="print('hello, world!')"))
    nb_file = tmp_path / "hello_world.ipynb"
    nbformat.write(notebook, nb_file)
    # Execute the notebook
    await execute_notebook(nb_file)
    # Check that an attempt was made at calling jupyter
    assert mock.called
    call_args, call_kwargs = mock.call_args
    cmd = call_args[0].split(" ")
    assert "pixi" in cmd[0]
    assert cmd[1:4] == ["run", "jupyter", "execute"]
    assert cmd[4] == str(nb_file)
