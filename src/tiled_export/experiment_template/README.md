# Spectroscopy Experiment Exports

This folder contains the exported data from your experiment, plus some
tools to aid in analysis. All the files in this folder are yours to
edit as you see fit.

## Getting Started

The recommended way to get started is to install Pixi, then launch the
analysis notebook in your browser by running:

```sh
pixi run notebook
```

This will install all the necessary dependencies for the built-in
analysis. Additional dependencies can be added with

```sh
pixi add <package-name>
```

## Setting Regions of Interest (ROIs)

The file `rois.toml` in this folder contains definitions for the
various ROIs that can be extracted from multi-dimensional signals
(e.g. area detectors).

Edit this file, defining new ROIs as needed, then re-run the analysis
code in the Jupyter notebook.
