import numpy as np
import pandas as pd
import pytest
from tiled.adapters.dataframe import DataFrameAdapter
from tiled.adapters.mapping import MapAdapter
from tiled.client import from_context
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
            metadata={},
        ),
    }
)


@pytest.fixture()
def tiled_client():
    with Context.from_app(build_app(tree)) as context:
        client = from_context(context)
        yield client
