from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from hollowfoot import Analysis, Group, operation
from mergedeep import merge
from numpy.typing import NDArray

USAGE_TEMPLATE = """
(xrt.XRFAnalysis
    .from_hdf_file("{{ run.hdf_file }}")
    .correct_live_times(signals=DEFAULT_ROIS.keys())
    .apply_rois(DEFAULT_ROIS)
    # .update_hdf_file("{{ run.hdf_file }}")
    # .update_xdi_file("{{ run.xdi_file }}")
    .plot_rois()
)
"""


class HDFGroup(Group):
    def __init__(self, hdf_file: str, dataset_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hdf_file = hdf_file
        self._dataset_path = dataset_path

    def __getattr__(self, name: str):
        with h5py.File(self._hdf_file, mode="r") as h5fp:
            path = self._dataset_path.format(name=name)
            ds = h5fp[path][()]
        return ds


def entry_group(name: str, entry: h5py.Group) -> Group:
    """Creates a data group from an h5py NXEntry, like those for an
    individual bluesky run.

    """
    dataset_path = f"{entry.name}/instrument/bluesky/streams/{{name}}/value"
    group = HDFGroup(name=name, hdf_file=entry.file.filename, dataset_path=dataset_path)
    return group


def _ndattributes(
    group: Group, signal: str, attr: str, data_shape: tuple[int, ...]
) -> NDArray[int | float]:
    """Read *attr* NDAttributes for a given area detector *signal*
    from *group*, combining elements from an array of shape
    *data_shape* into 1 array.

    """
    n_elements = data_shape[1]
    paths = [f"{signal}-element{idx}-{attr}" for idx in range(n_elements)]
    arr = np.asarray([group[path] for path in paths])
    # Convert the array into the correct shape
    arr = arr.swapaxes(0, 1)
    full_shape = arr.shape + tuple([1] * (len(data_shape) - arr.ndim))
    arr = arr.reshape(full_shape)
    return arr


def _correct_live_time(group: Group, signals: Sequence[str]) -> Group:
    """Apply live-time correction for a fluorescence detector signals."""
    s_per_tick = 12.5e-9
    for signal in signals:
        data = group[signal]
        dt_factor = _ndattributes(group, signal, "deadtime_factor", data.shape)
        clock_ticks = _ndattributes(group, signal, "clock_ticks", data.shape)
        acquire_time = clock_ticks * s_per_tick
        group[signal] = data * dt_factor / acquire_time
    return group


class XRFAnalysis(Analysis):
    """For doing analysis of fluorescence area detector data."""

    @classmethod
    def from_hdf_file(cls: type[Analysis], hdf_file: str | Path) -> "XRFAnalysis":
        groups = []
        with h5py.File(hdf_file, mode="r") as fd:

            groups = [entry_group(name=name, entry=entry) for name, entry in fd.items()]
        return cls(groups)

    @operation("apply live time correction")
    def correct_live_times(groups, signals: Sequence[str]):
        """Apply corrections for detector count time and dead-time.

        Parameters
        ==========
        signals
          Names for the signals to read. These should have suitable
          dead time and live time signals in the group.

        """
        groups_ = [_correct_live_time(group, signals=signals) for group in groups]
        return groups_

    @operation("apply ROIs")
    def apply_rois(
        groups: Sequence[Group], rois: Mapping[str, Mapping[str, list[slice]]]
    ) -> Sequence[Group]:
        for group in groups:
            for signal_name, signal_rois in rois.items():
                for roi_name, roi in signal_rois.items():
                    arr = group[signal_name][:, *roi]
                    arr = np.sum(arr, axis=tuple(range(1, arr.ndim)))
                    group[f"{signal_name}-{roi_name}"] = arr
        return groups

    def plot_rois(self, ax=None):
        """Plot fluorescence spectra for previously applied ROIs.

        Depends on having called ``.apply_rois()`` first.

        """
        analysis = self.calculate()
        for group in analysis.groups:
            # Figure out which operations have been previously applied
            op_args = [op.bound_arguments.arguments for op in group.past_operations]
            roi_args = [args["rois"] for args in op_args if "rois" in args]
            rois = merge({}, *roi_args)
            for ysignal, signal_rois in rois.items():
                if ax is None:
                    fig = plt.figure()
                    ax = plt.gca()
                ydata = np.sum(group[ysignal], axis=(0, 1))
                ax.plot(ydata)
                for roi_name, roi in signal_rois.items():
                    ax.axvspan(roi[1].start, roi[1].stop, alpha=0.5)
        return analysis
