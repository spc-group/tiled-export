import io

import pandas as pd
import pytest

from tiled_export.spreadsheets import update_summary_spreadsheet


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
    return df


def test_write_new_spreadsheet(runs_df, tmp_path):
    """Can we write a new summary-of-runs spreadsheet?"""
    fp = tmp_path / "spreadsheet.ods"
    update_summary_spreadsheet(runs=runs_df, fp=fp)
    new_df = pd.read_excel(fp, engine="odf")
    pd.testing.assert_frame_equal(new_df, runs_df)


@pytest.mark.skip(reason="appending spreadsheets is broken, excel's typing is bad")
def test_update_existing_spreadsheet(runs_df):
    """Can we write a new summary-of-runs spreadsheet?"""
    output = io.BytesIO()
    # Make sure we can do it in steps, too
    update_summary_spreadsheet(runs=runs_df.iloc[:1], fd=output)
    update_summary_spreadsheet(runs=runs_df.iloc[1:], fd=output)
    new_df = pd.read_excel(output, engine="odf")
    new_df.set_index("uid", inplace=True, drop=True)
    pd.testing.assert_frame_equal(new_df, runs_df)
