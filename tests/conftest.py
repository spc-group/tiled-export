import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from tiled.adapters.dataframe import DataFrameAdapter
from tiled.adapters.mapping import MapAdapter
from tiled.client import from_context, from_context_async
from tiled.client.context import Context
from tiled.server.app import build_app

tree = MapAdapter(
    {
        "scan0": MapAdapter(
            {
                "streams": MapAdapter(
                    {
                        "primary": DataFrameAdapter.from_pandas(
                            pd.DataFrame(
                                {
                                    "x": 1 * np.ones(10),
                                    "y": 2 * np.ones(10),
                                    "z": 3 * np.ones(10),
                                }
                            ),
                            npartitions=3,
                        ),
                    }
                ),
            },
            metadata={
                "start": {"uid": "scan0"},
            },
        ),
    }
)


@pytest_asyncio.fixture()
async def tiled_async_client():
    async with Context.from_app(build_app(tree), awaitable=True) as context:
        client = await from_context_async(context)
        yield client


@pytest.fixture()
def tiled_client():
    with Context.from_app(build_app(tree), awaitable=False) as context:
        client = from_context(context)
        yield client
