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


def update_summary_spreadsheet(runs: pd.DataFrame, fp: Path):
    """Write a summary of the runs as a spreadsheet into *fd*."""
    engine = "odf"
    # Ideally we could read previous sheets, but excel has no typing system
    # try:
    #     # We need to merge the existing and new dataframes
    #     existing_df = pd.read_excel(fp, engine=engine)
    #     print(f"{existing_df.esaf_id.dtype=}")
    # except FileNotFoundError:
    #     pass
    # else:
    #     runs = runs.set_index('uid').merge(existing_df.set_index('uid'))
    # Write to disk (or whatever *fp* is)
    runs = runs.sort_values("start_time")
    runs.to_excel(fp, engine=engine, index=False)
