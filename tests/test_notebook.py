import nbformat
import pytest
import pytest_asyncio

from tiled_export.experiment import copy_template
from tiled_export.notebook import add_run, execute_notebook, prepare_notebook, role
from tiled_export.protocols import Experiment


@pytest.fixture()
def notebook(tmp_path):
    copy_template(tmp_path)
    return tmp_path / "analysis.ipynb"


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
    xdi_path = notebook.parent / "data.xdi"
    xdi_path.touch()
    hdf_path = notebook.parent / "data.h5"
    hdf_path.touch()
    await add_run(
        run,
        notebook,
        xdi_file=xdi_path,
        hdf_file=hdf_path,
    )
    nb = nbformat.read(notebook, as_version=4)
    run_cells = [cell for cell in nb.cells if role(cell) == "run"]
    code_cell = run_cells[1]
    assert '.from_hdf_file("data.h5")' in code_cell.source
    assert ".from_tiled" not in code_cell.source
    assert ".update_hdf_files()" in code_cell.source
    assert "# .update_hdf_files()" not in code_cell.source
    assert ".update_xdi_files()" in code_cell.source
    assert "# .update_xdi_files()" not in code_cell.source


@pytest.mark.asyncio
async def test_add_run_data_no_files(notebook, run):
    """Check that the right modifications are made if no files are available."""
    await add_run(
        run,
        notebook,
        xdi_file=notebook.parent / "no_file.xdi",
        hdf_file=notebook.parent / "no_file.h5",
    )
    nb = nbformat.read(notebook, as_version=4)
    run_cells = [cell for cell in nb.cells if role(cell) == "run"]
    code_cell = run_cells[1]
    assert '.from_tiled("12345")' in code_cell.source
    assert "# .update_hdf_files()" in code_cell.source
    assert "# .update_xdi_files()" in code_cell.source


@pytest.mark.asyncio
async def test_remove_instructions(notebook, run):
    """Make sure the tutorial instructions, etc get removed."""
    prepare_notebook(
        notebook, experiment=Experiment(name="test", notebook="analysis.ipynb")
    )
    nb = nbformat.read(notebook, as_version=4)
    template_instructions = [cell for cell in nb.cells if role(cell) == "delete"]
    assert len(template_instructions) == 0


@pytest.mark.asyncio
async def test_notebook_templates(notebook, run):
    """Make sure the notebook template cells get formatted."""
    prepare_notebook(
        notebook, experiment=Experiment(name="test", notebook="analysis.ipynb")
    )
    nb = nbformat.read(notebook, as_version=4)
    cell = nb.cells[0]
    assert cell.metadata.tiled_export.role == "notebook"


@pytest.mark.asyncio
async def test_execute_notebook(tmp_path, mocker):
    exec_mock = mocker.patch("tiled_export.notebook.asyncio.create_subprocess_exec")
    process_mock = exec_mock.return_value
    # process_mock = mocker.AsyncMock()
    # # process_mock.returncode = 0
    # # exec_mock.return_value = process_mock
    process_mock.returncode = 0
    process_mock.communicate.return_value = (b"", b"")
    # Make a simple hello world notebook to execute
    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(nbformat.v4.new_code_cell(source="print('hello, world!')"))
    nb_file = tmp_path / "hello_world.ipynb"
    nbformat.write(notebook, nb_file)
    # Execute the notebook
    await execute_notebook(nb_file)
    # Check that an attempt was made at calling jupyter
    assert exec_mock.called
    call_args, call_kwargs = exec_mock.call_args
    cmd = call_args[0]
    cmd_args = call_args[1:5]
    assert "pixi" in cmd
    assert cmd_args == ("run", "papermill", str(nb_file), str(nb_file))
