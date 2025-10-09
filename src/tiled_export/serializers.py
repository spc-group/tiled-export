import os
from pathlib import Path
from typing import BinaryIO

import pandas as pd


def _buffer_size(buff: BinaryIO) -> int:
    current = buff.tell()
    buff.seek(0, os.SEEK_END)
    size = buff.tell()
    buff.seek(current)
    return size


def update_summary_files(runs: pd.DataFrame, parquet_file: Path):
    """Write a summary of the runs as a spreadsheet into *fp*."""
    engine = "odf"
    # Ideally we could read previous sheets, but excel has no typing system
    try:
        # We need to merge the existing and new dataframes
        existing_df = pd.read_parquet(parquet_file)
    except FileNotFoundError:
        pass
    else:
        # runs = # runs.set_index('uid').merge(existing_df.set_index('uid'), how="cross")
        runs = pd.concat([runs, existing_df]).drop_duplicates("uid")
    # Write to disk (or whatever *fp* is)
    runs = runs.sort_values("start_time")
    runs.to_parquet(parquet_file, index=False)
    excel_file = parquet_file.parent / f"{parquet_file.stem}.ods"
    runs.to_excel(excel_file, engine="odf", index=False)
