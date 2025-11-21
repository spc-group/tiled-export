import h5py
import numpy as np
import pandas as pd
import pytest
from hollowfoot import Group

from tiled_export.experiment_template.xraytools import (
    XRFAnalysis,
    read_roi_sources,
    read_rois,
)
from tiled_export.experiment_template.xraytools.xrf_analysis import HDFGroup


@pytest.fixture()
def xdi_file(tmp_path):
    xdi_path = tmp_path / "data.xdi"
    with open(xdi_path, mode="w") as fp:
        fp.write("# XDI/1.0\n# -----")
    return xdi_path


@pytest.fixture()
def hdf_file(tmp_path):
    hdf_path = tmp_path / "data.h5"
    with h5py.File(hdf_path, mode="w") as h5fp:
        entry = h5fp.create_group("scan0")
        primary = (
            entry.create_group("instrument")
            .create_group("bluesky")
            .create_group("streams")
            .create_group("primary")
        )
        It_net_count = primary.create_group("It-net_count")
        It_net_count.create_dataset("value", data=np.linspace(9000, 10000, num=201))
        vortex = primary.create_group("vortex_me4")
        vortex.create_dataset("value", data=np.random.rand(201, 4, 1024))
    return hdf_path


def test_from_hdf_file(hdf_file):
    analysis = XRFAnalysis.from_hdf_file(hdf_file)
    assert len(analysis.groups) == 1
    (group,) = analysis.groups
    # assert "It-net_count" in group
    assert group["primary/It-net_count"].shape == (201,)


def test_from_tiled(tiled_client):
    anl = XRFAnalysis.from_tiled("scan0", client=tiled_client)
    expected = tiled_client["scan0/streams/primary/x"].read()
    np.testing.assert_array_equal(anl.groups[0]["primary/x"], expected)


def test_hdf_save_signals(hdf_file):
    group = HDFGroup(hdf_file, entry_path="scan0", dataset_path="")
    group["vortex_me4-Ni-K"] = np.linspace(0, 100, num=101)
    group.save_signals(["vortex_me4-Ni-K"])
    # Check that the file was actually updated
    with h5py.File(hdf_file, mode="r") as fp:
        saved_data = fp["/scan0/data/vortex_me4-Ni-K"][()]
    np.testing.assert_array_equal(saved_data, np.linspace(0, 100, num=101))


def test_correct_live_times():
    s_per_tick = 12.5e-9
    data = np.linspace(0, 100, num=21 * 4 * 128).reshape(21, 4, 128)
    dt_factors = np.linspace(1, 1.5, num=21 * 4).reshape(21, 4)
    clock_ticks = np.linspace(
        int(0.8 / s_per_tick), int(1.2 / s_per_tick), num=21 * 4
    ).reshape(21, 4)
    group = Group(vortex_me4=data)
    group["vortex_me4-element0-deadtime_factor"] = dt_factors[:, 0]
    group["vortex_me4-element1-deadtime_factor"] = dt_factors[:, 1]
    group["vortex_me4-element2-deadtime_factor"] = dt_factors[:, 2]
    group["vortex_me4-element3-deadtime_factor"] = dt_factors[:, 3]
    group["vortex_me4-element0-clock_ticks"] = clock_ticks[:, 0]
    group["vortex_me4-element1-clock_ticks"] = clock_ticks[:, 1]
    group["vortex_me4-element2-clock_ticks"] = clock_ticks[:, 2]
    group["vortex_me4-element3-clock_ticks"] = clock_ticks[:, 3]
    analysis = XRFAnalysis([group])
    analysis = analysis.correct_live_times(["vortex_me4"]).calculate()
    group = analysis.groups[0]
    group.vortex_me4.shape == (21, 4, 128)
    expected = (
        data
        * dt_factors.reshape(21, 4, 1)
        / (clock_ticks * s_per_tick).reshape(21, 4, 1)
    )
    np.testing.assert_array_equal(group.vortex_me4, expected)


def test_apply_rois():
    data = np.linspace(0, 100, num=21 * 4 * 128).reshape(21, 4, 128)
    group = Group()
    group["primary/vortex_me4"] = data
    analysis = XRFAnalysis([group])
    rois = {
        "vortex_me4-Ni-K": {
            "source": "primary/vortex_me4",
            "slices": [slice(None), slice(104, 110)],
        }
    }
    analysis = analysis.apply_rois(rois).calculate()
    group = analysis.groups[0]
    np.testing.assert_array_equal(
        group["vortex_me4-Ni-K"], np.sum(data[:, :, 104:110], axis=(1, 2))
    )


def test_plot_rois(mocker):
    mock_ax = mocker.MagicMock()
    data = np.linspace(0, 100, num=21 * 4 * 128).reshape(21, 4, 128)
    group = Group()
    group["primary/vortex_me4"] = data
    analysis = XRFAnalysis([group])
    rois = {
        "vortex_me4-Ni-K": {
            "source": "primary/vortex_me4",
            "slices": [slice(None), slice(104, 110)],
        }
    }
    analysis = analysis.apply_rois(rois)
    assert not mock_ax.axvspan.called
    analysis.plot_rois(ax=mock_ax)
    assert mock_ax.axvspan.called


def test_read_full_roi(tmp_path):
    """An empty ROI table should use all values in array."""
    # Create a fake yaml file
    roi_path = tmp_path / "rois.toml"
    toml_text = "[ vortex_me4-total ]\nsource = 'vortex_me4'"
    with open(roi_path, "w") as fp:
        fp.write(toml_text)
    # Read the yaml back in and check
    new_rois = read_rois(roi_path)
    assert new_rois == {
        "vortex_me4-total": {"source": "vortex_me4", "slices": [slice(None)]}
    }


def test_read_full_roi(tmp_path):
    """An empty ROI table should use all values in array."""
    # Create a fake yaml file
    roi_path = tmp_path / "rois.toml"
    toml_text = "[ vortex_me4-total ]\nsource = 'vortex_me4'"
    with open(roi_path, "w") as fp:
        fp.write(toml_text)
    # Read the yaml back in and check
    new_rois = read_rois(roi_path)
    assert new_rois == {
        "vortex_me4-total": {"source": "vortex_me4", "slices": [slice(None)]}
    }


def test_read_roi_slice(tmp_path):
    """An empty ROI table should use all values in array."""
    # Create a fake yaml file
    roi_path = tmp_path / "rois.toml"
    toml_text = """
        [ vortex_me4-total ]
        source = 'vortex_me4'
        slices = [{}, {start=150, stop=80, step=3}]
    """
    with open(roi_path, "w") as fp:
        fp.write(toml_text)
    # Read the yaml back in and check
    new_rois = read_rois(roi_path)
    assert new_rois == {
        "vortex_me4-total": {
            "source": "vortex_me4",
            "slices": [slice(None), slice(150, 80, 3)],
        }
    }


def test_read_roi_array(tmp_path):
    """An empty ROI table should use all values in array."""
    # Create a fake yaml file
    roi_path = tmp_path / "rois.toml"
    toml_text = """
        [ vortex_me4-total ]
        source = 'vortex_me4'
        slices = [[0, 1, 3], {start=150, stop=80, step=3}]
    """
    with open(roi_path, "w") as fp:
        fp.write(toml_text)
    # Read the yaml back in and check
    new_rois = read_rois(roi_path)
    assert new_rois == {
        "vortex_me4-total": {
            "source": "vortex_me4",
            "slices": [[0, 1, 3], slice(150, 80, 3)],
        }
    }


def test_read_roi_sources(tmp_path):
    """An empty ROI table should use all values in array."""
    # Create a fake yaml file
    roi_path = tmp_path / "rois.toml"
    toml_text = """
        [ vortex_me4-total ]
        source = 'vortex_me4'

        [ ge_8element-total ]
        source = 'ge_8element'
    """
    with open(roi_path, "w") as fp:
        fp.write(toml_text)
    # Read the yaml back in and check
    sources = read_roi_sources(roi_path)
    assert sources == ["vortex_me4", "ge_8element"]


def test_update_hdf_rois(hdf_file):
    analysis = XRFAnalysis.from_hdf_file(hdf_file)
    rois = {
        "vortex_me4-Ni-K": {
            "source": "primary/vortex_me4",
            "slices": [slice(None), slice(104, 110)],
        }
    }
    analysis.apply_rois(rois).update_hdf_files()
    # Check that the new dataset was saved
    with h5py.File(hdf_file, mode="r") as fp:
        ds = fp["scan0/data/vortex_me4-Ni-K"][()]


def test_update_hdf_replaces_rois(hdf_file):
    # Set a dummy dataset to be overwritten
    with h5py.File(hdf_file, mode="a") as h5fp:
        h5fp["/scan0/"].create_group("data")
        h5fp["/scan0/data/"].create_dataset(
            "vortex_me4-Ni-K", np.linspace(0, 20, num=21)
        )
    analysis = XRFAnalysis.from_hdf_file(hdf_file)
    rois = {
        "vortex_me4-Ni-K": {
            "source": "primary/vortex_me4",
            "slices": [slice(None), slice(104, 110)],
        }
    }
    analysis.apply_rois(rois).update_hdf_files()
    # Check that the new dataset was saved
    with h5py.File(hdf_file, mode="r") as fp:
        ds = fp["scan0/data/vortex_me4-Ni-K"][()]
        assert ds.shape == (201,)


@pytest.mark.skip(reason="Need to write the parser first.")
def test_update_xdi_rois(hdf_file, xdi_file):
    analysis = XRFAnalysis.from_hdf_file(hdf_file)
    rois = {
        "vortex_me4-Ni-K": {
            "source": "primary/vortex_me4",
            "slices": [slice(None), slice(104, 110)],
        }
    }
    analysis.apply_rois(rois).update_xdi_files()
    # Check that the new dataset was saved
    df = pd.read_csv(xdi_file)
    assert False
