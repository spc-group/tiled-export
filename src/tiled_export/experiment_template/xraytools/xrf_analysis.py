import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tomlkit
import xarray as xr
from hollowfoot import Analysis, Group, operation, xdi
from mergedeep import merge
from numpy.typing import NDArray
from tiled.client import from_profile
from tiled.client.container import Container
from tiled.profiles import get_default_profile_name

# For some reason, jupyter lab does not show warnings without this
#   Fix taken from https://github.com/microsoft/vscode-jupyter/issues/1312
warnings.simplefilter(action="default")

USAGE_TEMPLATE = """
(xrt.XRFAnalysis
    {% if run.hdf_file_exists is true %}.from_hdf_file("{{ run.hdf_file }}")
    {% else %}.from_tiled("{{ run.metadata.start.uid }}"){% endif %}
    .correct_live_times(xrt.read_roi_sources("rois.toml"))
    .apply_rois(xrt.read_rois("rois.toml"))
    {% if run.hdf_file_exists is false %}# {% endif %}.update_hdf_files()
    {% if run.xdi_file_exists is false %}# {% endif %}.update_xdi_files()
    .plot_rois()
)
"""


def read_roi_sources(toml_file: str | Path):
    rois = read_rois(toml_file)
    sources = [roi["source"] for roi in rois.values()]
    return list(dict.fromkeys(sources))


def read_rois(toml_file: str | Path):
    """Read a file containing defintions for regions-of-interest (ROI).

    Parameters
    ==========
    toml_file
      Location of the file containing ROI defintion.

    Returns
    =======
    rois

    """

    def fix_slice(slc):
        """Convert mappings to slice() objects."""
        if isinstance(slc, Mapping):
            return slice(slc.get("start"), slc.get("stop"), slc.get("step"))
        return slc

    with open(toml_file, mode="r") as fp:
        rois = tomlkit.parse(fp.read()).unwrap()
        for name, roi in rois.items():
            slices = roi.get("slices", [slice(None)])
            roi["slices"] = [fix_slice(slc) for slc in slices]
    if len(rois) == 0:
        warnings.warn(f"No ROI's are specified in '{toml_file}'.")
    return rois


class HDFGroup(Group):
    def __init__(
        self, hdf_file: str, entry_path: str, dataset_path: str, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._hdf_file = hdf_file
        self._entry_path = entry_path
        self._dataset_path = dataset_path

    def __getattr__(self, name: str):
        with h5py.File(self._hdf_file, mode="r") as h5fp:
            path = f"{self._entry_path}/{self._dataset_path.format(name=name)}"
            try:
                ds = h5fp[path][()]
            except (KeyError, TypeError) as exc:
                msg = f"'{type(self).__name__}' has no attribute '{name}'"
                raise AttributeError(msg) from exc
        return ds

    def save_signals(self, signals: Sequence[str]):
        """Save signals to the original NeXus file as NXData nodes."""
        data_to_save = {signal: self[signal] for signal in signals}
        with h5py.File(self._hdf_file, mode="a") as h5fp:
            entry = h5fp[self._entry_path]
            if "data" not in entry.keys():
                data_group = entry.create_group("data")
                data_group.attrs["NX_class"] = "NXdata"
            data_group = entry["data"]
            for signal, ds in data_to_save.items():
                if signal in data_group:
                    del data_group[signal]
                data_group.create_dataset(signal, data=ds)


class TiledGroup(Group):
    def __init__(self, path: str, *args, client: Container, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = PurePosixPath(path)
        self._client = client

    def __getattr__(self, name: str):
        data = self._client[str(self.path / "streams" / name)].read()
        return data


def entry_group(name: str, entry: h5py.Group) -> Group:
    """Creates a data group from an h5py NXEntry, like those for an
    individual bluesky run.

    """
    dataset_path = "instrument/bluesky/streams/{name}/value"
    group = HDFGroup(
        name=name,
        hdf_file=entry.file.filename,
        entry_path=entry.name,
        dataset_path=dataset_path,
    )
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

    @classmethod
    def from_tiled(
        cls: type[Analysis],
        path: str,
        client: Container | None = None,
        profile: str | None = None,
    ):
        """Load an analysis object from a Bluesky Tiled server."""
        if client is None:
            profile_name = get_default_profile_name() if profile is None else profile
            client = from_profile(profile_name)
        group = TiledGroup(path=path, client=client)
        return cls([group])

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
            for roi_name, roi in rois.items():
                arr = group[roi["source"]][:, *roi["slices"]]
                arr = np.sum(arr, axis=tuple(range(1, arr.ndim)))
                group[roi_name] = arr
        return groups

    def plot_rois(self, ax=None) -> "XRFAnalysis":
        """Plot fluorescence spectra for previously applied ROIs.

        Depends on having called ``.apply_rois()`` first.

        """
        analysis = self.calculate()
        for group in analysis.groups:
            # Figure out which operations have been previously applied
            op_args = [op.bound_arguments.arguments for op in group.past_operations]
            roi_args = [args["rois"] for args in op_args if "rois" in args]
            rois = merge({}, *roi_args)
            external_datasets: dict[str, h5py.Group] = {}
            for roi_name, roi in rois.items():
                external_datasets.setdefault(roi["source"], {})[roi_name] = roi
            # Plot each flourescence signal with its ROIs
            for signal, rois in external_datasets.items():
                if ax is None:
                    fig = plt.figure()
                    ax = plt.gca()
                ydata = np.sum(group[signal], axis=(0, 1))
                ax.plot(ydata)
                # Plot the ROIs (only those defined as slices, though)
                for roi_name, roi in rois.items():
                    if len(roi["slices"]) < 2 or not isinstance(
                        roi["slices"][1], slice
                    ):
                        continue
                    slc = roi["slices"][1]
                    ax.axvspan(slc.start, slc.stop, alpha=0.5)
        return analysis

    @operation("Write new data back to HDF files.", defer=False)
    def update_hdf_files(groups: Sequence[Group]) -> Sequence[Group]:
        for group in groups:
            op_args = [op.bound_arguments.arguments for op in group.past_operations]
            roi_args = [args["rois"] for args in op_args if "rois" in args]
            rois = merge({}, *roi_args)
            group.save_signals(rois.keys())
        return groups

    @operation("Write new data back to XDI files.", defer=False)
    def update_xdi_files(groups: Sequence[Group]) -> Sequence[Group]:

        for group in groups:
            op_args = [op.bound_arguments.arguments for op in group.past_operations]
            roi_args = [args["rois"] for args in op_args if "rois" in args]
            rois = merge({}, *roi_args)
            # Read existing XDI file
            hdf_path = Path(group._hdf_file)
            xdi_path = hdf_path.parent / f"{hdf_path.stem}.xdi"
            with open(xdi_path, mode="r") as fp:
                ds = xdi.load(fp.read())
            # Add the new ROI signals to the dataset
            abscissa = list(set(ds.dims))[0]
            new_data_vars = {
                roi_name: (abscissa, group[roi_name]) for roi_name in rois.keys()
            }
            new_ds = xr.Dataset(
                coords=ds.coords,
                data_vars={**ds.data_vars, **new_data_vars},
                attrs=ds.attrs,
            )
            # Write the new dataset back to disk
            with open(xdi_path, mode="w") as fp:
                fp.write(xdi.dump(new_ds))
        return groups
