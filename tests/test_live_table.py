import pytest
from rich.progress import Progress

from tiled_export.live_table import LiveTable


def test_progress():
    lt = LiveTable()
    assert isinstance(lt.progress, Progress)


def test_add_run_to_table(mocker):
    lt = LiveTable()
    run = mocker.MagicMock()
    lt.add_run(run)
    assert len(lt.table.rows) == 1


@pytest.mark.asyncio
async def test_track_coroutine(mocker):
    lt = LiveTable()
    coro = mocker.AsyncMock().method()
    await lt.track(coro, name="hello", total=1)
