import asyncio
import importlib.util
import logging
import shutil
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import nbformat
from jinja2 import Template
from nbformat import NotebookNode
from tiled.client.container import AsyncContainer

from tiled_export.protocols import Experiment

log = logging.getLogger(__name__)


def role(cell: NotebookNode) -> str | None:
    return cell.metadata.get("tiled_export", {}).get("role")


def cell_uid(cell) -> str | None:
    return cell.metadata.get("tiled_export", {}).get("run", {}).get("uid")


def render_notebook_template_cell(
    cell: NotebookNode, values: Mapping[str, Any]
) -> NotebookNode:
    if role(cell) != "notebook_template":
        return cell
    cell.metadata.tiled_export.role = "notebook"
    cell.source = Template(cell.source).render(**values)
    return cell


def prepare_notebook(notebook: str | Path, experiment: Experiment) -> None:
    """Parse a jupyter notebook to fix any global template artifacts."""
    # Remove cells with template instructions, etc
    nb = nbformat.read(notebook, as_version=4)
    nb.cells = [cell for cell in nb.cells if role(cell) != "delete"]
    # Format cells for the entire experiment
    values = {"experiment": experiment}
    nb.cells = [render_notebook_template_cell(cell, values=values) for cell in nb.cells]
    nbformat.write(nb, notebook)


RUN_MARKDOWN_TEMPLATE = """
### {{ run.metadata.start.scan_name }}

**UID:** ``{{ run.metadata.start.uid }}``
**Start:** {{ run.start_time }}
"""


class CellType(str, Enum):
    MARKDOWN = "markdown"
    CODE = "code"


def render_run_cell(
    usage_template: str, values: Mapping[str, Any], cell_type: CellType
) -> NotebookNode:
    """Take a Jinja template for markdown and create a new notebook code
    cell.

    """
    uid = values["run"]["metadata"].get("start", {}).get("uid")
    source = Template(usage_template).render(**values)
    cell_md = {"tiled_export": {"role": "run", "run": {"uid": uid}}}
    new_cell = {
        CellType.MARKDOWN: nbformat.v4.new_markdown_cell,
        CellType.CODE: nbformat.v4.new_code_cell,
    }[cell_type](source, metadata=cell_md)
    return new_cell


def usage_template(folder: Path, module_name: str = "xrf_analysis"):
    """Return the string template for how to use a particular xraytools module.

    *folder* should point to the user's data folder (not the
    tiled_export experiment template folder), in case users have made
    local changes.

    """
    module_path = folder / "xraytools" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(module_path)
    spec.loader.exec_module(module)
    return module.USAGE_TEMPLATE.strip()


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
        xdi_path = Path(xdi_file)
        run["xdi_file"] = str(xdi_path.relative_to(notebook.parent))
        run["xdi_file_exists"] = xdi_path.exists()
    if hdf_file is not None:
        hdf_path = Path(hdf_file)
        run["hdf_file"] = str(hdf_path.relative_to(notebook.parent))
        run["hdf_file_exists"] = xdi_path.exists()
    usage = usage_template(notebook.parent)
    run_cells = [
        render_run_cell(
            RUN_MARKDOWN_TEMPLATE, values={"run": run}, cell_type=CellType.MARKDOWN
        ),
        render_run_cell(usage, values={"run": run}, cell_type=CellType.CODE),
    ]
    # Add cells to notebook
    nb.cells.extend(run_cells)
    nbformat.write(nb, notebook)


async def execute_notebook(notebook: Path) -> None:
    """Execute the python cells in the jupyter notebook."""
    cmd = shutil.which("pixi")
    cmd_args = ["run", "papermill", str(notebook), str(notebook)]
    if cmd is not None:
        proc = await asyncio.create_subprocess_exec(
            cmd,
            *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=notebook.parent,
        )
    else:
        raise RuntimeError("Can not find pixi binary to execute notebook")
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        log.error(stderr.decode())
        raise RuntimeError("Failure executing notebook")
