import io
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest import mock

import pandas as pd
import pytest

from tiled_export.export import build_queries, export_run, update_summary_spreadsheet


def test_build_quries_empty():
    qs = build_queries(exit_status=None)
    assert len(qs) == 0


def test_build_quries_with_filters():
    qs = build_queries(
        before="2025-10-05T08:00:00",
        after="2025-10-03T09:00:00",
        esaf="549301",
        proposal="22348",
        sample_name="NMC-833",
        plan_name="xafs_scan",
        sample_formula="NiMnCo",
        scan_name="pristine",
        edge="Ni-K",
        uid="a1b2c3d4-e5f6",
    )
    assert len(qs) == 11
    assert qs[0].key == "stop.exit_status"


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


# @pytest.mark.asyncio
# async def test_harden_link(temp_h5_file):
#     src_file = NamedTemporaryFile(delete_on_close=False, suffix=".h5")
#     link_file = NamedTemporaryFile(delete_on_close=False, suffix=".h5")
#     with src_file, link_file:
#         # Create source data to copy
#         with h5py.File(src_file, mode='w') as src_h5fd:
#             src_h5fd['src_data'] = np.random.random((10, 20, 30))
#         # Create a link to the source data
#         link_file.close()
#         with h5py.File(link_file.name, mode='w') as target_file:
#             target_file['target_link'] = h5py.ExternalLink(src_file.name, '/src_data')
#             target_file['target_link'].attrs['spam'] = 'eggs'
#             assert isinstance(target_file.get('target_link', getlink=True), h5py.ExternalLink)
#             harden_link(parent=target_file, link_path="target_link")
#             assert isinstance(target_file.get('target_link', getlink=True), h5py.HardLink)
#             assert len(target_file.keys()) == 1


@pytest.fixture()
def runs_df():
    df = pd.DataFrame(
        {
            "uid": [
                "34431b85-84a9-4655-8b1f-6f23e50f8754",
                "8c6846fb-7515-4209-a3e7-e94823d0e6cc",
            ],
            "esaf_id": [
                123456,
                789053,
            ],  # to_excel() converts these strings to ints anyway...
            "start_time": [1756398780.1790738, 1756398791.0785813],
            "exit_status": ["success", None],
            "sample_name": ["Ni foil", "xenonite"],
            "scan_name": ["calibration", "XAFS"],
            "plan_name": ["scan", "xafs_scan"],
            "experiment_name": ["rocky_2025-05_123456", "rocky_2025-05_123456"],
            "filename": [
                "202505091343-Ni_foil-calibration-scan-34431b85",
                "202505091431-xenonite-XAFS-xafs_scan-8c6846fb",
            ],
        },
    )
    df.set_index("uid", inplace=True, drop=True)
    return df


def test_write_new_spreadsheet(runs_df):
    """Can we write a new summary-of-runs spreadsheet?"""
    output = io.BytesIO()
    update_summary_spreadsheet(runs=runs_df, fd=output)
    new_df = pd.read_excel(output, engine="odf")
    new_df.set_index("uid", inplace=True, drop=True)
    pd.testing.assert_frame_equal(new_df, runs_df)


def test_update_existing_spreadsheet(runs_df):
    """Can we write a new summary-of-runs spreadsheet?"""
    output = io.BytesIO()
    # Make sure we can do it in steps, too
    update_summary_spreadsheet(runs=runs_df.iloc[:1], fd=output)
    update_summary_spreadsheet(runs=runs_df.iloc[1:], fd=output)
    new_df = pd.read_excel(output, engine="odf")
    new_df.set_index("uid", inplace=True, drop=True)
    pd.testing.assert_frame_equal(new_df, runs_df)
