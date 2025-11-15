import asyncio
import shutil
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import nbformat
from jinja2 import Template
from nbformat import NotebookNode
from tiled.client.container import AsyncContainer


def role(cell: NotebookNode) -> str | None:
    return cell.metadata.get("tiled_export", {}).get("role")


def cell_uid(cell) -> str | None:
    return cell.metadata.get("tiled_export", {}).get("run", {}).get("uid")


def prepare_notebook(notebook: str | Path) -> None:
    """Parse a jupyter notebook to fix any global template artifacts."""
    # Remove cells with template instructions, etc
    nb = nbformat.read(notebook, as_version=4)
    nb.cells = [cell for cell in nb.cells if role(cell) != "delete"]
    nbformat.write(nb, notebook)


def render_template_cell(
    cell: NotebookNode, values: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Take a template cell defined in the jupyter notebook and create
    a new cell with run metadata inserted.

    """
    # Update the cell metadata
    new_metadata = deepcopy(cell.metadata)
    cell_md = new_metadata.setdefault("tiled_export", {})
    cell_md["role"] = "run"
    uid = values["run"]["metadata"].get("start", {}).get("uid")
    cell_md.setdefault("run", {"uid": uid})
    # Create the new jupyter notebook cell
    new_source = Template(cell.source).render(**values)
    new_cell = nbformat.NotebookNode(
        {
            "cell_type": cell.cell_type,
            "id": str(uuid.uuid4()),
            "metadata": new_metadata,
            "source": new_source,
        }
    )
    # Add metadata for specific cell types
    if "execution_count" in cell:
        new_cell["execution_count"] = None
    if "outputs" in cell:
        new_cell["outputs"] = []
    return new_cell


async def add_run(
    run: AsyncContainer,
    notebook: Path,
    xdi_file: Path | None = None,
    hdf_file: Path | None = None,
) -> None:
    """Add cells to a *notebook* for analysing a bluesky *run*."""
    nb = nbformat.read(notebook, as_version=4)
    # Check if cells have already been rendered for this run.
    existing_cells = [
        cell for cell in nb.cells if cell_uid(cell) == run.metadata["start"]["uid"]
    ]
    if len(existing_cells) > 0:
        return
    # Parse cell templates
    run = {"metadata": run.metadata}
    if xdi_file is not None:
        run["xdi_file"] = str(xdi_file.relative_to(notebook.parent))
    if hdf_file is not None:
        run["hdf_file"] = str(hdf_file.relative_to(notebook.parent))
    template_cells = [cell for cell in nb.cells if role(cell) == "run_template"]
    run_cells = [
        render_template_cell(cell, values={"run": run}) for cell in template_cells
    ]
    # Add cells to notebook
    nb.cells.extend(run_cells)
    nbformat.write(nb, notebook)


async def execute_notebook(notebook: Path) -> None:
    """Execute the python cells in the jupyter notebook."""
    cmd = " ".join([shutil.which("pixi"), "run", "jupyter", "execute", str(notebook)])
    proc = await asyncio.create_subprocess_exec(cmd, cwd=notebook.parent)
    await proc.wait()
