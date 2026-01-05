import datetime as dt
import math
from collections.abc import Awaitable, MutableMapping

from rich.console import Group
from rich.live import Live, VerticalOverflowMethod
from rich.progress import Progress, TaskID
from rich.table import Table


class LiveTable(Live):
    """A rich table for showing runs with progress bars, etc."""

    table_headers = [
        "#",
        "UID",
        "Start",
        "Status",
        "Beamline",
        "Sample",
        "Scan",
        "Plan",
    ]
    _progress_tasks: MutableMapping[str, TaskID]
    count = 0

    def __init__(
        self, *args, vertical_overflow: VerticalOverflowMethod = "visible", **kwargs
    ):
        super().__init__(*args, vertical_overflow=vertical_overflow, **kwargs)
        self.progress = Progress()
        self._progress_tasks = {}
        self._md_task = self.progress.add_task("Reading metadata", total=None)
        self.table = Table()
        for col in self.table_headers:
            self.table.add_column(col)
        # The various UI elements need to be added to the live display as one group
        self.update(Group(self.table, self.progress))

    def __exit__(self, *args, **kwargs):
        # Make the metadata progress bar look "done"
        self.progress.update(
            self._md_task,
            completed=self.count,
            total=self.count,
            refresh=True,
        )
        super().__exit__(*args, **kwargs)

    def add_run(
        self,
        idx: int,
        uid: str = "",
        start_time: int | float = float("nan"),
        exit_status: str = "",
        beamline: str = "",
        sample_name: str = "",
        scan_name: str = "",
        plan_name: str = "",
    ):
        """Add metadata from *run* to the table and update progress bars."""
        # Turn the start time into a suitable string
        if math.isfinite(start_time):
            start = str(
                dt.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            start = ""
        # Add the actual row to the table
        row = [
            idx,
            uid,
            start,
            exit_status,
            beamline,
            sample_name,
            scan_name,
            plan_name,
        ]
        self.table.add_row(*[str(item) for item in row])
        self.progress.update(self._md_task, advance=1)
        self.count += 1
        # self.refresh()

    async def track(self, coro: Awaitable, name: str, total: int):
        """Update the progress bar when an awaitable is complete."""
        # Create a new progress bar if one doesn't exist
        if name not in self._progress_tasks:
            self._progress_tasks[name] = self.progress.add_task(
                name, total=total, start=True, transient=True
            )
        # Update the progress bar once the coroutine is done
        await coro
        self.progress.update(self._progress_tasks[name], advance=1)
