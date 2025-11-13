import numpy as np
import pandas as pd
import pytest

from tiled_export.serializers import update_summary_files


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


def test_write_new_summary(runs_df, tmp_path):
    """Can we write a new summary-of-runs spreadsheet?"""
    fp = tmp_path / "spreadsheet.parquet"
    excel_fp = tmp_path / "spreadsheet.ods"
    update_summary_files(runs=runs_df, parquet_file=fp)
    # Check the parquet file
    new_df = pd.read_parquet(fp)
    pd.testing.assert_frame_equal(new_df, runs_df)
    # Check the spreadsheet
    new_df = pd.read_excel(excel_fp)
    # Excel doesn't have None, so we need NaN for comparison
    runs_df = runs_df.fillna(np.nan)
    pd.testing.assert_frame_equal(new_df, runs_df)


def test_update_existing_summary(runs_df, tmp_path):
    """Can we write a new summary-of-runs spreadsheet?"""
    fp = tmp_path / "spreadsheet.parquet"
    # Make sure we can do it in steps, too
    update_summary_files(runs=runs_df.iloc[:1], parquet_file=fp)
    update_summary_files(runs=runs_df.iloc[1:], parquet_file=fp)
    new_df = pd.read_parquet(fp)
    pd.testing.assert_frame_equal(new_df, runs_df)


def test_update_existing_summary_duplicates(runs_df, tmp_path):
    """Can we write a new summary-of-runs spreadsheet?"""
    fp = tmp_path / "spreadsheet.parquet"
    # Make sure we can do it in steps, too
    update_summary_files(runs=runs_df, parquet_file=fp)
    update_summary_files(runs=runs_df, parquet_file=fp)
    new_df = pd.read_parquet(fp)
    pd.testing.assert_frame_equal(new_df, runs_df)
