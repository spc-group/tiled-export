import os
from typing import BinaryIO

import pandas as pd


def _buffer_size(buff: BinaryIO) -> int:
    current = buff.tell()
    buff.seek(0, os.SEEK_END)
    size = buff.tell()
    buff.seek(current)
    return size


def update_summary_spreadsheet(runs: pd.DataFrame, fd: BinaryIO):
    """Write a summary of the runs as a spreadsheet into *fd*."""
    dataframes = [runs]
    if _buffer_size(fd) > 0:
        # We need to merge the existing and new dataframes
        existing_df = pd.read_excel(fd, engine="odf", index_col="uid")
        dataframes.append(existing_df)
    # Write to disk (or whatever *fd* is)
    new_df = pd.concat(dataframes)
    new_df = new_df.sort_values("start_time")
    new_df.to_excel(fd, engine="odf", index=True, index_label="uid")
