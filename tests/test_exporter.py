from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest import mock

import h5py
import numpy as np
import pytest

from tiled_export.export import (
    QuerySet,
    export_run,
    fetch_experiments,
    harden_external_links,
    parse_args,
)


def test_empty_queryset_queries():
    qs = QuerySet(exit_status=None)
    assert len(qs.queries()) == 0


def test_queryset_queries_with_filters():
    qs = QuerySet(
        before="2025-10-05T08:00:00",
        after="2025-10-03T09:00:00",
        exit_status="success",
        plan_name="xafs_scan",
        sample_name="NMC-833",
        sample_formula="NiMnCo",
        scan_name="pristine",
        edge="Ni-K",
        proposal="22348",
        beamline="25-ID-C",
        esaf="549301",
        uid="a1b2c3d4-e5f6",
    )
    queries = qs.queries()
    assert len(queries) == 12
    assert queries[0].key == "stop.exit_status"


@pytest.mark.asyncio
async def test_export_run():
    run = mock.AsyncMock()
    run.metadata = {"start": {}}
    with TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        await export_run(run, base_dir=base_dir, use_xdi=True)
    assert run.export.called


@pytest.fixture()
def temp_h5_file():
    fd = NamedTemporaryFile(delete=False, suffix=".h5")
    try:
        yield fd
    except:
        fd.delete()
        raise


@pytest.mark.asyncio
async def test_apply_queryset(mocker):
    catalog = mocker.MagicMock()
    catalog.search.return_value = catalog
    qs = QuerySet(before="2025-10-05T08:00:00", after="2025-10-03T09:00:00")
    catalog = qs.apply(catalog)
    assert catalog.search.call_count == 2


@pytest.mark.skip(reason="Needs to be re-written to work with bluesky run schema.")
@pytest.mark.asyncio
async def test_harden_link(temp_h5_file):
    src_file = NamedTemporaryFile(delete_on_close=False, suffix=".h5")
    link_file = NamedTemporaryFile(delete_on_close=False, suffix=".h5")
    with src_file, link_file:
        # Create source data to copy
        with h5py.File(src_file, mode="w") as src_h5fd:
            src_h5fd["src_data"] = np.random.random((10, 20, 30))
        # Create a link to the source data
        link_file.close()
        with h5py.File(link_file.name, mode="w") as target_file:
            target_file["target_link"] = h5py.ExternalLink(src_file.name, "/src_data")
            target_file["target_link"].attrs["spam"] = "eggs"
            assert isinstance(
                target_file.get("target_link", getlink=True), h5py.ExternalLink
            )
            harden_external_links(parent=target_file, link_path="target_link")
            assert isinstance(
                target_file.get("target_link", getlink=True), h5py.HardLink
            )
            assert len(target_file.keys()) == 1


@pytest.mark.asyncio
async def test_fetch_experiments(tiled_async_client):
    queries = QuerySet()
    exps = fetch_experiments(catalog=tiled_async_client, queries=queries)
    exps = [exp async for exp in exps]


def test_parse_args():
    args = parse_args(
        [
            "-p",
            "default",
        ]
    )
    assert args.tiled_profile == "default"


def test_parse_datetime_args():
    args = parse_args(
        [
            "-A",
            "2025-12-31T08:45:00-05:00",
            "--before",
            "2025-12-31T08:45:03-05:00",
        ]
    )
    assert args.after == 1767188700
    assert args.before == 1767188703
