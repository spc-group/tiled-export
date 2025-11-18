import h5py
import numpy as np
import pytest
from hollowfoot import Group

from tiled_export import XRFAnalysis


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
    return hdf_path


def test_from_hdf_file(hdf_file):
    analysis = XRFAnalysis.from_hdf_file(hdf_file)
    assert len(analysis.groups) == 1
    (group,) = analysis.groups
    # assert "It-net_count" in group
    assert group["primary/It-net_count"].shape == (201,)


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
    group = Group(vortex_me4=data)
    analysis = XRFAnalysis([group])
    rois = {
        "vortex_me4": {
            "Ni-K": [slice(None), slice(104, 110)],
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
    group = Group(vortex_me4=data)
    analysis = XRFAnalysis([group])
    rois = {
        "vortex_me4": {
            "Ni-K": [slice(None), slice(104, 110)],
        }
    }
    analysis = analysis.apply_rois(rois)
    assert not mock_ax.axvspan.called
    analysis.plot_rois(ax=mock_ax)
    assert mock_ax.axvspan.called
