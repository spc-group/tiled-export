import pytest

from tiled_export.experiment import prepare_experiment


@pytest.mark.asyncio
async def test_create_dir(tmp_path):
    exp_path = tmp_path / "experiment"
    assert not exp_path.exists()
    await prepare_experiment(exp_path)
    assert exp_path.exists()


@pytest.mark.asyncio
async def test_create_pixi_toml(tmp_path):
    exp_path = tmp_path
    toml_file = exp_path / "pixi.toml"
    assert not toml_file.exists()
    await prepare_experiment(exp_path)
    assert toml_file.exists()
    with open(toml_file, mode="r") as toml_fp:
        assert "[dependencies]" in toml_fp.read()


@pytest.mark.asyncio
async def test_creates_analysis_notebook(tmp_path):
    exp_path = tmp_path
    nb_file = exp_path / "analysis.ipynb"
    assert not nb_file.exists()
    await prepare_experiment(exp_path)
    assert nb_file.exists()
