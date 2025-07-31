import argparse
import asyncio
import datetime as dt
import logging
import re
import textwrap
from collections.abc import AsyncIterable, Generator
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
from httpx import HTTPStatusError
from tabulate import tabulate
from tiled import queries
from tiled.profiles import ProfileNotFound, get_default_profile_name, load_profiles
from tqdm.asyncio import tqdm

from tiled_export.catalog import Catalog, CatalogScan

NEXUS_MIMETYPE = "application/x-nexus"
XDI_MIMETYPE = "text/x-xdi"
TSV_MIMETYPE = "text/tab-separated-values"

extensions = {
    NEXUS_MIMETYPE: ".hdf",
    TSV_MIMETYPE: ".tab",
    XDI_MIMETYPE: ".xdi",
}


log = logging.getLogger("haven")


class IncompleteRun(ValueError):
    pass


def load_catalog(tiled_profile: str, catalog: str):
    profiles = load_profiles()
    try:
        filepath, profile_content = profiles[tiled_profile]
    except KeyError as err:
        raise ProfileNotFound(
            f"Profile {tiled_profile!r} not found. Found profiles {list(profiles)}."
        ) from err
    return Catalog(catalog, uri=profile_content["uri"])


async def export_run(
    run: CatalogScan,
    *,
    base_dir: Path,
    use_xdi: bool = False,
    use_nexus: bool = False,
    rewrite_hdf_links: bool = False,
):
    # Decide on export formats
    valid_formats = await run.formats()
    target_formats = []
    if use_nexus:
        target_formats.append(NEXUS_MIMETYPE)
    if use_xdi:
        target_formats.append(
            XDI_MIMETYPE if XDI_MIMETYPE in valid_formats else TSV_MIMETYPE
        )
    # Retrieve needed metadata
    md = await run.metadata
    start_doc = md.get("start", {})
    esaf = start_doc.get("esaf_id", "noesaf")
    pi_name = "nopi"
    start_time = dt.datetime.fromtimestamp(start_doc.get("time", 0))
    # Decide on how to structure the file storage
    esaf_dir = base_dir / f"{pi_name}_{start_time.strftime('%Y-%m')}_{esaf}"
    sample_name = start_doc.get("sample_name")
    scan_name = start_doc.get("scan_name")
    plan_name = start_doc.get("plan_name")
    uid_base = start_doc.get("uid", "").split("-")[0]
    bits = [
        start_time.strftime("%Y%m%d%H%M"),
        sample_name,
        scan_name,
        plan_name,
        uid_base,
    ]
    bits = [bit for bit in bits if bit not in ["", None]]
    base_name = "-".join(bits)
    base_name = re.sub(r"[ ]", "_", base_name)
    base_name = re.sub(r"[/]", "", base_name)
    # Write to disk
    esaf_dir.mkdir(parents=True, exist_ok=True)
    for fmt in target_formats:
        ext = extensions[fmt]
        fp = esaf_dir / f"{base_name}{ext}"
        if fp.exists():
            continue
        # Export files
        try:
            await run.export(fp, format=fmt)
        except HTTPStatusError as exc:
            print(start_doc["uid"], exc)
        else:
            if fmt == NEXUS_MIMETYPE and rewrite_hdf_links:
                with h5py.File(fp, mode="a") as fd:
                    harden_external_links(fd[start_doc["uid"]])
    # Add an entry to the spreadsheet for this run
    spreadsheet_path = esaf_dir / "runs_summary.ods"
    if spreadsheet_path.exists():
        df = pd.read_excel(spreadsheet_path, engine="odf")
    else:
        df = pd.DataFrame(
            columns=[
                "uid",
                "start_timestamp",
                "start_datetime",
                "exit_status",
                "sample",
                "scan",
                "plan",
                "filebase",
            ]
        )
    if start_doc.get("uid") not in df.uid.values:
        # Add the row to the spreadsheet
        df.loc[len(df)] = [
            start_doc.get("uid", ""),
            start_doc.get("time", ""),
            start_time.isoformat(),
            md.get("stop", {}).get("exit_status", ""),
            start_doc.get("sample_name", ""),
            start_doc.get("scan_name", ""),
            start_doc.get("plan_name", ""),
            base_name,
        ]
        df.to_excel(spreadsheet_path, engine="odf", index=False)


def build_queries(
    *,
    exit_status: str | None = "success",
    before: str | None = None,
    after: str | None = None,
    esaf: str | None = None,
    proposal: str | None = None,
    beamline: str | None = None,
    sample_name: str | None = None,
    plan_name: str | None = None,
    sample_formula: str | None = None,
    scan_name: str | None = None,
    edge: str | None = None,
    uid: str | None = None,
) -> list[queries.NoBool]:
    # Parse datestrings
    before_ts: float | None = None
    after_ts: float | None = None
    if before is not None:
        before_ts = dt.datetime.fromisoformat(before).timestamp()
    if after is not None:
        after_ts = dt.datetime.fromisoformat(after).timestamp()
    # Process the queries parameters into actual queries
    qs = []
    query_params = [
        # filter_name: (query type, metadata key)
        (exit_status, queries.Eq, "stop.exit_status"),
        (plan_name, queries.Eq, "start.plan_name"),
        (sample_name, queries.Contains, "start.sample_name"),
        (sample_formula, queries.Contains, "start.sample_formula"),
        (scan_name, queries.Contains, "start.scan_name"),
        (edge, queries.Contains, "start.edge"),
        (proposal, queries.Eq, "start.proposal_id"),
        (beamline, queries.Contains, "start.beamline_id"),
        (esaf, queries.Eq, "start.esaf_id"),
        (before_ts, partial(queries.Comparison, "le"), "stop.time"),
        (after_ts, partial(queries.Comparison, "ge"), "start.time"),
        (uid, queries.Contains, "start.uid"),
    ]
    for arg, query, key in query_params:
        if arg is not None:
            qs.append(query(key, arg))

    return qs


async def table_row(run: CatalogScan) -> list[str]:
    md = await run.metadata
    start_dt = dt.datetime.fromtimestamp(md["start"]["time"])
    start_time = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    return [
        md["start"]["uid"],
        start_time,
        md.get("stop", {}).get("exit_status", ""),
        md["start"].get("beamline_id", ""),
        md["start"].get("sample_name", ""),
        md["start"].get("scan_name", ""),
        md["start"].get("plan_name", ""),
    ]


async def export_runs(
    base_dir: Path,
    runs: AsyncIterable[CatalogScan],
    use_xdi: bool,
    use_nexus: bool,
    rewrite_hdf_links: bool = False,
):
    valid_runs = []
    rows = []
    headers = ["#", "UID", "Start", "Status", "Beamline", "Sample", "Scan", "Plan"]
    # Print a table of runs for approval
    row_num = 0
    async for run in runs:
        rows.append(await table_row(run))
        valid_runs.append(run)
    print(tabulate(rows, headers=headers, tablefmt="fancy_outline", showindex=True))
    # Save a table of runs for
    # Do the exporting
    for run in tqdm(valid_runs, desc="Exporting", unit="runs"):
        await export_run(
            run,
            base_dir=base_dir,
            use_xdi=use_xdi,
            use_nexus=use_nexus,
            rewrite_hdf_links=rewrite_hdf_links,
        )


def main():
    default_profile = get_default_profile_name()
    parser = argparse.ArgumentParser(
        prog="export-runs",
        description="""Export runs from the database as files on disk.

        """,
        epilog=textwrap.dedent(
            """Note: exporting external datasets. It is non-trivial to include
        large area detector datasets in an exported HDF5 (NeXus)
        file. By default, the exported file contains external links to
        the original area detector data. This keeps file sizes
        manageable, but means the resulting file will only be usable
        if the source files pointed to be the external links are
        available. To make these links more portable, consider using
        the --hdf-expand option, at the cost of larger file sizes and
        considerably slower exporting."""
        ),
    )
    parser.add_argument(
        "base_dir", help="The base directory for storing files.", type=str
    )
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
    parser.add_argument("-q", "--quiet", help="Verbose output", action="store_true")
    parser.add_argument(
        "-p",
        "--tiled-profile",
        help="Profile for the Tiled client, or else the default profile will be used. See https://blueskyproject.io/tiled/how-to/profiles.html",
        type=str,
        default=default_profile,
    )
    parser.add_argument(
        "-c",
        "--catalog",
        help="Catalog name in the Tiled server. Default: scans",
        default="scans",
        type=str,
    )
    # Arguments for filtering runs
    parser.add_argument(
        "--failed",
        help="Also include scans that did not complete.",
        action="store_true",
    )
    parser.add_argument("--plan", help="Export runs with plan name.", type=str)
    parser.add_argument("--sample", help="Export runs with this sample name.", type=str)
    parser.add_argument(
        "--formula",
        help="Export runs with samples matching this chemical formula.",
        type=str,
    )
    parser.add_argument("--scan", help="Export runs with this scan name.", type=str)
    parser.add_argument("--edge", help="Export runs with this edge.", type=str)
    parser.add_argument("--esaf", help="Export runs with this ESAF ID.", type=str)
    parser.add_argument(
        "--proposal", help="Export runs with this proposal ID.", type=str
    )
    parser.add_argument(
        "--beamline",
        help="Export runs only on this beamline. Incomplete matches are allowed, so '25-ID' will match both '25-ID-C' and '25-ID-D'.",
    )
    parser.add_argument(
        "--before",
        help="Only include runs before this timestamp. E.g. 2025-04-22T8:00:00.",
        type=str,
    )
    parser.add_argument(
        "--after",
        help="Only include runs after this ISO datetime. E.g. 2025-04-22T8:00:00.",
        type=str,
    )
    parser.add_argument("--uid", help="Export runs with this UID.", type=str)
    # Arguments for export formats
    parser.add_argument(
        "--hdf",
        "--nexus",
        help="Export files to HDF files with the NeXus schema",
        action="store_true",
    )
    parser.add_argument(
        "--hdf-expand",
        help="""Each exported HDF5 file will have external links replaced with the source data. This option will result in portable files with considerably larger file sizes and slower export times.""",
        action="store_true",
    )
    parser.add_argument(
        "--xdi",
        help="Export files to XDI tab-separated value files.",
        action="store_true",
    )
    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    if not args.quiet:
        logging.basicConfig(level=log_level)
    # Get the list of runs we need
    catalog = load_catalog(catalog=args.catalog, tiled_profile=args.tiled_profile)
    exit_status = None if args.failed else "success"
    qs = build_queries(
        before=args.before,
        after=args.after,
        esaf=args.esaf,
        proposal=args.proposal,
        beamline=args.beamline,
        plan_name=args.plan,
        sample_name=args.sample,
        sample_formula=args.formula,
        scan_name=args.scan,
        edge=args.edge,
        uid=args.uid,
        exit_status=exit_status,
    )
    runs = catalog.runs(queries=qs)
    # Save each run to disk
    base_dir = Path(args.base_dir)
    do_export = export_runs(
        base_dir=base_dir,
        runs=runs,
        use_xdi=args.xdi,
        use_nexus=args.hdf,
        rewrite_hdf_links=args.hdf_expand,
    )
    asyncio.run(do_export)


def external_data_links(
    entry: h5py.Group,
) -> Generator[tuple[h5py.Group, h5py.ExternalLink], Any, None]:
    """Generator for all the nxdata groups that are external links."""
    for stream in entry["instrument/bluesky/streams"].values():
        for data in stream.values():
            link = data.get("value", getlink=True)
            if isinstance(link, h5py.ExternalLink):
                yield data, link


def harden_external_links(entry: h5py.Group):
    """Replace external links with copies of the target."""
    src_filename = Path(entry.file.filename)
    for data, link in external_data_links(entry):
        # Sort out where all the paths should point
        target_path, path = Path(link.filename), link.path
        # Copy the dataset into the target HDF5 file
        with h5py.File(target_path, mode="r") as target_fd:
            # Make sure the dataset exists first, then replace it
            target_fd[path]
            del data["value"]
            target_fd.copy(target_fd[path], data, "value")


# -----------------------------------------------------------------------------
# :author:    Mark Wolfman
# :email:     wolfman@anl.gov
# :copyright: Copyright © 2025, UChicago Argonne, LLC
#
# Distributed under the terms of the 3-Clause BSD License
#
# The full license is in the file LICENSE, distributed with this software.
#
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# -----------------------------------------------------------------------------
