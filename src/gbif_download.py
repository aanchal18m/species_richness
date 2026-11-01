"""Part 2: GBIF occurrence acquisition -- winter Aves, India, 2015-2024.

ROUTE (2026-10-05): the predicate download in SIMPLE_CSV is the one used.
The SQL download below (request 0011506-260928105237408) finished SUCCEEDED with
0 records although GBIF's validator accepted the query; its code and tests are
kept as the record of what was tried. eventID, the reason for SQL, is null for
eBird records (99.3% of the data), so SIMPLE_CSV loses nothing the GBIF route
uses; `eventID` is treated as optional everywhere downstream.

Why a record-level SQL download was tried rather than SIMPLE_CSV or DWCA
  * SIMPLE_CSV and SIMPLE_PARQUET do not contain eventID (checked against
    /occurrence/download/describe, 2026-10-05).
  * DWCA does, but measured at ~315 B/record for this dataset it is ~9 GB.
  * A plain SELECT ... WHERE with no GROUP BY keeps one row per occurrence and
    only the columns the sampling-unit logic needs. The earlier failure was the
    GROUP BY, not SQL itself; tests/test_gbif_download.py guards against it.

Subcommands
  estimate   record count and size estimate from the public search API
  validate   ask GBIF to validate the SQL (no credentials, nothing submitted)
  request    submit the download (--format simple_csv | sql; default simple_csv);
             writes the key to data/raw/gbif/download_key.txt and the
             pre-request estimate to data/raw/gbif/request_<key>.json
  count-sql  diagnostic: SQL COUNT(*) for Aves in India only
  fetch      poll until finished, download, verify, record the DOI in data/raw/README.md
  report     summary of the landed file -> outputs/tables/gbif_download_summary.json

Credentials come only from GBIF_USER, GBIF_PWD and GBIF_EMAIL and are never
written to disk.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GBIF_DIR = RAW / "gbif"
KEY_FILE = GBIF_DIR / "download_key.txt"
README = RAW / "README.md"
SUMMARY = ROOT / "outputs" / "tables" / "gbif_download_summary.json"
SUMMARY_MD = ROOT / "outputs" / "tables" / "gbif_download_report.md"

API = "https://api.gbif.org/v1"
CRED_VARS = ("GBIF_USER", "GBIF_PWD", "GBIF_EMAIL")

# --- query specification (Part 2 brief) -------------------------------------
AVES_CLASS_KEY = 212
COUNTRY = "IN"
YEARS = (2015, 2024)
WINTER_MONTHS = (11, 12, 1, 2)
EXCLUDED_BASIS = ("FOSSIL_SPECIMEN", "LIVING_SPECIMEN")
MIN_CENTROID_DISTANCE_M = 2000
EXCLUDED_ISSUES = ("TAXON_MATCH_FUZZY",)

# Fields the sampling-unit logic needs (Darwin Core names).
REQUIRED_FIELDS = (
    "decimalLatitude", "decimalLongitude", "recordedBy", "eventDate", "eventID",
    "occurrenceID", "coordinateUncertaintyInMeters", "speciesKey", "basisOfRecord",
)

# (Darwin Core name, SQL expression). GBIF lowercases identifiers in the output,
# so the reader maps headers back to these names.
COLUMNS = (
    ("gbifID", "gbifid"),
    ("datasetKey", "datasetkey"),
    ("occurrenceID", "occurrenceid"),
    ("eventID", "eventid"),
    # recordedBy is ARRAY<STRING> in GBIF's SQL schema; '|' never occurs in eBird
    # observer ids, so the join is reversible.
    ("recordedBy", "ARRAY_JOIN(recordedby, '|')"),
    ("eventDate", "eventdate"),
    ("year", '"year"'),
    ("month", '"month"'),
    ("day", '"day"'),
    ("decimalLatitude", "decimallatitude"),
    ("decimalLongitude", "decimallongitude"),
    ("coordinateUncertaintyInMeters", "coordinateuncertaintyinmeters"),
    ("speciesKey", "specieskey"),
    ("species", "species"),
    ("taxonRank", "taxonrank"),
    ("basisOfRecord", "basisofrecord"),
    ("occurrenceStatus", "occurrencestatus"),
)
DWC_BY_LOWER = {dwc.lower(): dwc for dwc, _ in COLUMNS}

# --- size estimation --------------------------------------------------------
# Bytes per record, zipped. DWCA and SIMPLE_CSV are medians over 140 past
# downloads of the eBird dataset (GBIF download log, 2026-10-05). SQL is
# measured by zipping 3,000 shuffled search-API rows (2015-2024 winters) laid
# out as exactly the COLUMNS above: 42 B/record. The range allows for GBIF
# using a different compression level.
BYTES_PER_RECORD = {"DWCA": 315, "SIMPLE_CSV": 107, "SQL_TSV_ZIP": (40, 60)}
MAX_DOWNLOAD_BYTES = 5e9


def build_sql() -> str:
    select = ",\n  ".join(
        expr if expr.strip('"') == dwc.lower() else f"{expr} AS {dwc.lower()}"
        for dwc, expr in COLUMNS
    )
    months = ", ".join(str(m) for m in WINTER_MONTHS)
    basis = ", ".join(f"'{b}'" for b in EXCLUDED_BASIS)
    where = [
        f"classkey = {AVES_CLASS_KEY}",
        f"countrycode = '{COUNTRY}'",
        f'"year" >= {YEARS[0]}',
        f'"year" <= {YEARS[1]}',
        f'"month" IN ({months})',
        "hascoordinate = TRUE",
        "hasgeospatialissues = FALSE",
        f"basisofrecord NOT IN ({basis})",
        # NULL-safe: NULL >= 2000 is NULL, and a NULL in WHERE drops the row.
        "(distancefromcentroidinmeters IS NULL"
        f" OR distancefromcentroidinmeters >= {MIN_CENTROID_DISTANCE_M})",
    ]
    # NULL-safe for the same reason: a record with no issues may carry a NULL
    # array, and NOT ARRAY_CONTAINS(NULL, x) is NULL, not TRUE.
    where += [f"(issue IS NULL OR NOT ARRAY_CONTAINS(issue, '{i}'))" for i in EXCLUDED_ISSUES]
    return f"SELECT\n  {select}\nFROM occurrence\nWHERE " + "\n  AND ".join(where)


def build_predicate() -> dict:
    """The same filters as build_sql(), as a GBIF JSON predicate. The month set
    is one `in` predicate; the centroid filter keeps nulls, as in the SQL."""
    return {"type": "and", "predicates": [
        {"type": "equals", "key": "CLASS_KEY", "value": str(AVES_CLASS_KEY)},
        {"type": "equals", "key": "COUNTRY", "value": COUNTRY},
        {"type": "greaterThanOrEquals", "key": "YEAR", "value": str(YEARS[0])},
        {"type": "lessThanOrEquals", "key": "YEAR", "value": str(YEARS[1])},
        {"type": "in", "key": "MONTH", "values": [str(m) for m in WINTER_MONTHS]},
        {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
        {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
        {"type": "not", "predicate": {"type": "in", "key": "BASIS_OF_RECORD",
                                      "values": list(EXCLUDED_BASIS)}},
        {"type": "or", "predicates": [
            {"type": "isNull", "parameter": "DISTANCE_FROM_CENTROID_IN_METERS"},
            {"type": "greaterThanOrEquals", "key": "DISTANCE_FROM_CENTROID_IN_METERS",
             "value": str(MIN_CENTROID_DISTANCE_M)}]},
        {"type": "not", "predicate": {"type": "in", "key": "ISSUE", "values": list(EXCLUDED_ISSUES)}},
    ]}


COUNT_SQL = f"SELECT COUNT(*) AS n FROM occurrence WHERE classkey = {AVES_CLASS_KEY} AND countrycode = '{COUNTRY}'"


def validate(sql: str, timeout: int = 60) -> str:
    """GBIF parses and checks the query without running it. Returns GBIF's normalised SQL."""
    r = requests.post(f"{API}/occurrence/download/request/validate",
                      json={"format": "SQL_TSV_ZIP", "sql": sql}, timeout=timeout)
    if r.status_code != 201:
        raise ValueError(f"GBIF rejected the SQL ({r.status_code}): {r.text}")
    return r.json()["sql"]


def search_count(timeout: int = 120, **params) -> int:
    r = requests.get(f"{API}/occurrence/search", params={**params, "limit": 0}, timeout=timeout)
    r.raise_for_status()
    return r.json()["count"]


def estimate() -> dict:
    """Count from the search API. It cannot express the NOT predicates, so the
    excluded subsets are counted separately; overlaps make `low` a lower bound."""
    base = dict(classKey=AVES_CLASS_KEY, country=COUNTRY, year=f"{YEARS[0]},{YEARS[1]}",
                month=list(WINTER_MONTHS), hasCoordinate="true", hasGeospatialIssue="false")
    n_base = search_count(**base)
    n_centroid = search_count(**base, distanceFromCentroidInMeters=f"0,{MIN_CENTROID_DISTANCE_M - 0.001}")
    n_issue = sum(search_count(**base, issue=i) for i in EXCLUDED_ISSUES)
    n_basis = search_count(**base, basisOfRecord=list(EXCLUDED_BASIS))
    low, high = n_base - n_centroid - n_issue - n_basis, n_base
    lo_b, hi_b = BYTES_PER_RECORD["SQL_TSV_ZIP"]
    return {
        "records_base_filter": n_base,
        "excluded_centroid": n_centroid,
        "excluded_issue": n_issue,
        "excluded_basis": n_basis,
        "records_low": low,
        "records_high": high,
        "sql_bytes_low": low * lo_b,
        "sql_bytes_high": high * hi_b,
        "dwca_bytes": high * BYTES_PER_RECORD["DWCA"],
        "simple_csv_bytes": high * BYTES_PER_RECORD["SIMPLE_CSV"],
    }


def credentials(environ=os.environ) -> tuple[str, str, str]:
    missing = [v for v in CRED_VARS if not environ.get(v)]
    if missing:
        raise RuntimeError(f"Missing environment variable(s): {', '.join(missing)}. "
                           "Export them in your shell; they are never written to disk.")
    return tuple(environ[v] for v in CRED_VARS)


# --- polling ----------------------------------------------------------------
OK_STATUS = {"SUCCEEDED"}
BAD_STATUS = {"FAILED", "KILLED", "CANCELLED", "FILE_ERASED"}


def wait_for(key, get_meta, sleep=time.sleep, interval=60, timeout=8 * 3600, log=print) -> dict:
    """Poll until the download reaches a terminal state. Transient network errors
    count as a wait rather than aborting a multi-hour job."""
    waited = 0
    while True:
        try:
            meta = get_meta(key)
            status = meta.get("status")
        except requests.RequestException as e:
            meta, status = None, f"unreachable ({type(e).__name__})"
        if status in OK_STATUS:
            return meta
        if status in BAD_STATUS:
            raise RuntimeError(f"GBIF download {key} ended with status {status}")
        if waited >= timeout:
            raise TimeoutError(f"GBIF download {key} still {status} after {waited}s")
        log(f"[{datetime.now():%H:%M:%S}] {key}: {status}; next check in {interval}s")
        sleep(interval)
        waited += interval


def check_nonempty(meta: dict) -> None:
    """GBIF reports SUCCEEDED, and mints a DOI, even when the query matched
    nothing. An empty download must never be fetched or cited."""
    if not meta.get("totalRecords"):
        raise ValueError(f"GBIF download {meta.get('key')} SUCCEEDED with 0 records "
                         f"(DOI {meta.get('doi')}): not fetched, DOI not recorded. "
                         "The WHERE clause matched nothing on GBIF's engine.")


def check_record_count(meta: dict, estimate: dict | None, tol: float = 0.05) -> None:
    """Zero is only the extreme case: a predicate that half-applies would also
    pass check_nonempty. Compare with the pre-request search-API estimate."""
    if not estimate:
        return
    n, lo, hi = meta.get("totalRecords") or 0, estimate["records_low"], estimate["records_high"]
    if not (lo * (1 - tol) <= n <= hi * (1 + tol)):
        raise ValueError(f"GBIF download {meta.get('key')} has {n:,} records; the search API "
                         f"estimated {lo:,}-{hi:,}. Not fetched; the filters did not apply as intended.")


def verify_zip(path: Path, expected_size: int) -> None:
    size = path.stat().st_size
    if size != expected_size:
        raise IOError(f"{path.name}: {size} bytes on disk, GBIF reports {expected_size}")
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
    except zipfile.BadZipFile as e:
        raise IOError(f"{path.name}: not a readable zip ({e})") from e
    if bad is not None:
        raise IOError(f"{path.name}: CRC failure in member {bad}")


def download_resumable(url: str, part: Path, expected_size: int, attempts: int = 200,
                       timeout: int = 120, chunk: int = 1 << 16, sleep=time.sleep, log=print) -> Path:
    """Stream `url` into `part`, resuming with an HTTP Range request from the
    bytes already on disk after every drop. A dropped stream may end silently or
    raise IncompleteRead; either way completeness is judged only by the byte
    count, and integrity is checked separately (verify_zip).

    Chunks are small (64 KiB) because a drop raises before the chunk being
    filled is yielded: its bytes are lost, so the chunk size bounds the loss per
    drop and any connection delivering more than one chunk makes progress."""
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if have == expected_size:
            return part
        if have > expected_size:
            raise IOError(f"{part.name}: {have} bytes exceeds the expected {expected_size}")
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                if have and r.status_code != 206:
                    raise IOError(f"server ignored the Range request (HTTP {r.status_code})")
                with open(part, "ab" if have else "wb") as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
        except (requests.RequestException, IOError) as e:
            if isinstance(e, IOError) and "Range" in str(e):
                raise
            log(f"  attempt {attempt + 1}: {type(e).__name__}; resuming from {part.stat().st_size if part.exists() else 0:,} bytes")
        done = part.stat().st_size if part.exists() else 0
        if done < expected_size:
            log(f"  {done / 1e6:,.1f} / {expected_size / 1e6:,.1f} MB; resuming")
            sleep(min(60, 5 * (attempt + 1)) if done == have else 1)
    raise IOError(f"{part.name}: incomplete after {attempts} attempts")


def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


# --- README provenance ------------------------------------------------------
README_HEADER = """# data/raw

Downloaded inputs. **Never edited by hand.** Each section below is written by the
script that produced the file and records what is needed to cite and reproduce it.
"""


def write_section(readme: Path, name: str, body: str) -> None:
    """Insert or replace one marker-delimited section, leaving the others intact."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    block = f"{start}\n{body.strip()}\n{end}"
    text = readme.read_text() if readme.exists() else README_HEADER
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    text = pattern.sub(lambda _: block, text) if pattern.search(text) else text.rstrip() + "\n\n" + block + "\n"
    readme.write_text(text)


def citation(meta: dict) -> str:
    created = datetime.fromisoformat(meta["created"].replace("Z", "+00:00"))
    return (f"GBIF.org ({created.day} {created:%B %Y}) GBIF Occurrence Download "
            f"https://doi.org/{meta['doi']}")


NOT_CITED = ("- **Not used, not cited:** SQL request `0011506-260928105237408` finished SUCCEEDED "
             "with **0 records** (header only) and was still assigned DOI 10.15468/dl.g5uxhg. "
             "**Cause:** GBIF's SQL layer uses alphanumeric taxon keys (Aves is classkey `'V2'`), "
             "so the hand-written `classkey = 212` matched nothing -- not permissions, not nulls "
             "(CLAUDE.md trap 15). Record: `gbif/request_0011506-260928105237408.json`.\n"
             "- Diagnostic SQL `0011565-260928105237408` (DOI 10.15468/dl.6rjtqg, not cited): "
             "`SELECT COUNT(*) FROM occurrence WHERE classkey = 212 AND countrycode = 'IN'` returned "
             "**0**, against 61,519,735 for the same two keys in the search API, confirming the "
             "basic taxon column as the fault.")


def count_note(meta: dict, estimate: dict | None, file_rows: int | None = None) -> str:
    """Record count against the pre-request search-API estimate."""
    if not estimate:
        return ""
    n, lo, hi = meta["totalRecords"], estimate["records_low"], estimate["records_high"]
    line = (f"- Record count: **{n:,}** ({100 * (n - lo) / lo:+.2f}% against the pre-request "
            f"search-API estimate of {lo:,}-{hi:,}")
    line += "; exactly the lower bound)." if n == lo else ")."
    if file_rows is not None:
        line += f" Rows in the file: {file_rows:,}."
    return line


def gbif_readme_body(meta: dict, zip_path: Path, digest: str, count: str = "") -> str:
    req = meta.get("request") or {}
    if req.get("predicate"):
        lang, query, fmt = "json", json.dumps(req["predicate"], indent=2), req.get("format", "SIMPLE_CSV")
    else:
        lang, query, fmt = "sql", req.get("sql") or build_sql(), "SQL_TSV_ZIP"
    return f"""## GBIF occurrence download (Part 2)

- **Citation (required in the report):** {citation(meta)}
- DOI: `{meta['doi']}`
- Download key: `{meta['key']}`
- Created: {meta['created']}
- Records: {meta.get('totalRecords')}
- Datasets contributing: {meta.get('numberDatasets')}
- License: {meta.get('license')}
- File: `gbif/{zip_path.name}` ({zip_path.stat().st_size:,} bytes), SHA-256 `{digest}`
- Format: {fmt}, one row per occurrence (no aggregation)
- Produced by: `python src/gbif_download.py fetch`

{count}
{NOT_CITED}

Query as GBIF executed it:

```{lang}
{query}
```
"""


# --- reading and summarising ------------------------------------------------
def read_chunks(zip_path, chunksize=250_000, usecols=None):
    """Yield DataFrames from the TSV inside a GBIF download (SIMPLE_CSV or SQL).

    GBIF writes tab-separated text with no quoting, so a literal '"' in a field
    is data. QUOTE_NONE stops pandas treating it as the start of a quoted field.

    `usecols` (Darwin Core names) goes to read_csv itself, so unused columns are
    never materialised: with dtype=str, all 50 SIMPLE_CSV columns would be ~50M
    string objects per million rows. Requested columns the file lacks (eventID in
    SIMPLE_CSV) are skipped."""
    import pandas as pd

    with zipfile.ZipFile(zip_path) as z:
        member = max((i for i in z.infolist() if not i.is_dir()), key=lambda i: i.file_size)
        with z.open(member) as f:
            header = f.readline().decode("utf-8").rstrip("\r\n").split("\t")
        to_dwc = {h: DWC_BY_LOWER.get(h.lower(), h) for h in header}
        wanted = None if usecols is None else set(usecols)
        cols = [h for h in header if wanted is None or to_dwc[h] in wanted]
        with z.open(member) as f:
            reader = pd.read_csv(f, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, usecols=cols,
                                 keep_default_na=False, na_values=[""], chunksize=chunksize)
            for chunk in reader:
                chunk.columns = [to_dwc[c] for c in chunk.columns]
                yield chunk if usecols is None else chunk[[c for c in usecols if c in chunk.columns]]


def winter_label(year, month):
    """Nov-Dec belong to the winter that ends the following calendar year."""
    return year.where(month < 11, year + 1).astype("Int64").map(lambda y: f"{y - 1}/{str(y)[2:]}")


def summarise(chunks) -> dict:
    import pandas as pd

    n = 0
    species, no_species = set(), 0
    ym, winters, status, basis, datasets = Counter(), Counter(), Counter(), Counter(), Counter()
    has_eid = has_obs = single_day = observer_day = any_unit = 0
    unc_null = unc_gt = 0
    for c in chunks:
        n += len(c)
        sk = c["speciesKey"].dropna()
        species.update(sk.unique())
        no_species += int(c["speciesKey"].isna().sum())

        year = pd.to_numeric(c["year"], errors="coerce")
        month = pd.to_numeric(c["month"], errors="coerce")
        ym.update(Counter(zip(year.astype("Int64"), month.astype("Int64"))))
        winters.update(winter_label(year, month).value_counts().to_dict())
        status.update(c["occurrenceStatus"].fillna("<null>").value_counts().to_dict())
        basis.update(c["basisOfRecord"].fillna("<null>").value_counts().to_dict())
        datasets.update(c["datasetKey"].value_counts().to_dict())

        # SIMPLE_CSV has no eventID column; on the GBIF route it is null for eBird anyway
        eid = c["eventID"].notna() if "eventID" in c else pd.Series(False, index=c.index)
        obs = c["recordedBy"].notna()
        # A sampling day is identifiable only if eventDate resolves to one day:
        # "2016-01-10" or "2016-01-10T07:30" yes; "2016-01" or a range no.
        ed = c["eventDate"].fillna("")
        start, _, stop = ed.str.partition("/").values.T
        start = pd.Series(start, index=c.index).str[:10]
        stop = pd.Series(stop, index=c.index).str[:10]
        day = start.str.fullmatch(r"\d{4}-\d{2}-\d{2}") & ((stop == "") | (stop == start))
        has_eid += int(eid.sum())
        has_obs += int(obs.sum())
        single_day += int(day.sum())
        observer_day += int((obs & day).sum())
        any_unit += int((eid | (obs & day)).sum())

        unc = pd.to_numeric(c["coordinateUncertaintyInMeters"], errors="coerce")
        unc_null += int(unc.isna().sum())
        unc_gt += int((unc > 10_000).sum())

    def frac(k):
        return round(k / n, 6) if n else None

    return {
        "rows": n,
        "distinct_species": len(species),
        "rows_without_speciesKey": no_species,
        "years": sorted({int(y) for y, _ in ym if y is not pd.NA}),
        "months": sorted({int(m) for _, m in ym if m is not pd.NA}),
        "rows_by_year_month": {f"{y}-{m:02d}": k for (y, m), k in sorted(ym.items())},
        "rows_by_winter": dict(sorted(winters.items())),
        "event_fields": {
            "eventID_present": has_eid, "eventID_frac": frac(has_eid),
            "recordedBy_present": has_obs, "recordedBy_frac": frac(has_obs),
            "eventDate_single_day": single_day, "eventDate_single_day_frac": frac(single_day),
            "recordedBy_and_single_day": observer_day, "recordedBy_and_single_day_frac": frac(observer_day),
            "eventID_or_observer_day": any_unit, "eventID_or_observer_day_frac": frac(any_unit),
        },
        "coordinate_uncertainty": {
            "null": unc_null, "null_frac": frac(unc_null),
            "gt_10km": unc_gt, "gt_10km_frac_of_all": frac(unc_gt),
            "gt_10km_frac_of_reported": round(unc_gt / (n - unc_null), 6) if n > unc_null else None,
        },
        "occurrenceStatus": dict(status),
        "basisOfRecord": dict(basis),
        "top_datasets": dict(datasets.most_common(10)),
    }


# --- commands ---------------------------------------------------------------
def cmd_estimate(_):
    e = estimate()
    print(json.dumps(e, indent=2))
    print(f"\nSQL_TSV_ZIP ~ {e['sql_bytes_low'] / 1e9:.1f}-{e['sql_bytes_high'] / 1e9:.1f} GB "
          f"(DWCA would be ~ {e['dwca_bytes'] / 1e9:.1f} GB)")
    return e


def cmd_validate(_):
    print(validate(build_sql()))


def request_file(key: str) -> Path:
    return GBIF_DIR / f"request_{key}.json"


def cmd_request(args):
    from pygbif import occurrences as occ

    if KEY_FILE.exists():
        sys.exit(f"A download was already requested: {KEY_FILE.read_text().strip()}. "
                 f"Run `fetch`, or delete {KEY_FILE} to request a new one.")
    user, pwd, email = credentials()
    e = estimate()
    size_key = "sql_bytes_high" if args.format == "sql" else "simple_csv_bytes_high"
    e["simple_csv_bytes_high"] = e["records_high"] * 117      # upper quartile of measured B/record
    if e[size_key] > MAX_DOWNLOAD_BYTES:
        sys.exit(f"Estimated up to {e[size_key] / 1e9:.1f} GB, over the "
                 f"{MAX_DOWNLOAD_BYTES / 1e9:.0f} GB limit agreed for Part 2. Not submitting.")
    if args.format == "sql":
        query = build_sql()
        validate(query)
        key = occ.download_sql(query, user=user, pwd=pwd, email=email).strip()
    else:
        # GBIF cannot validate predicate downloads (422 "not implemented"): a malformed
        # predicate fails at submission; one that misbehaves fails check_record_count.
        query = build_predicate()
        key, _ = occ.download(query, format="SIMPLE_CSV", user=user, pwd=pwd, email=email)
        key = key.strip()
    GBIF_DIR.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key + "\n")
    request_file(key).write_text(json.dumps({"key": key, "format": args.format, "query": query,
                                             "estimate": e, "requested": datetime.now().isoformat()},
                                            indent=2))
    print(f"Submitted ({args.format}). Download key {key} saved to {KEY_FILE.relative_to(ROOT)}")


def cmd_count_sql(_):
    """Diagnostic for the empty SQL download: does the engine find Aves in India at all?"""
    from pygbif import occurrences as occ

    user, pwd, email = credentials()
    validate(COUNT_SQL)
    key = occ.download_sql(COUNT_SQL, user=user, pwd=pwd, email=email).strip()
    print(f"Submitted diagnostic COUNT(*) {key}. Check: {API}/occurrence/download/{key}")


def cmd_fetch(args):
    from pygbif import occurrences as occ

    key = args.key or KEY_FILE.read_text().strip()
    meta = wait_for(key, occ.download_meta, interval=args.interval)
    check_nonempty(meta)
    req = request_file(key)
    check_record_count(meta, json.loads(req.read_text())["estimate"] if req.exists() else None)
    dest = GBIF_DIR / f"{key}.zip"
    if dest.exists() and dest.stat().st_size == meta["size"]:
        print(f"{dest.name} already present")
    else:
        # Download beside, verify, then move: an interrupted transfer must not
        # leave a truncated file under the final name.
        partial = GBIF_DIR / ".partial"
        partial.mkdir(parents=True, exist_ok=True)
        # pygbif's download_get restarts from zero and accepts a silently dropped
        # stream (a 3.57 GB download arrived as 5.7 MB). Resume instead.
        tmp = download_resumable(f"{API}/occurrence/download/request/{key}.zip",
                                 partial / f"{key}.zip", meta["size"])
        verify_zip(tmp, meta["size"])
        tmp.replace(dest)
    write_provenance(key, meta, dest)
    print(f"{dest.relative_to(ROOT)} verified; DOI {meta['doi']} recorded in {README.relative_to(ROOT)}")


def write_provenance(key: str, meta: dict, dest: Path, file_rows: int | None = None):
    req = request_file(key)
    est = json.loads(req.read_text())["estimate"] if req.exists() else None
    note = count_note(meta, est, file_rows)
    write_section(README, "gbif", gbif_readme_body(meta, dest, sha256(dest), note))


def report_text(s: dict) -> str:
    """The Part 2 numbers, led by the one that bounds everything downstream."""
    e, u = s["event_fields"], s["coordinate_uncertainty"]
    lost = s["rows"] - e["recordedBy_and_single_day"]
    lines = [
        "# GBIF download: what it can support",
        "",
        f"**Sampling-unit fields: {e['recordedBy_and_single_day']:,} of {s['rows']:,} records "
        f"({100 * e['recordedBy_and_single_day_frac']:.2f}%) carry BOTH a usable recordedBy and a "
        f"single-day eventDate.** On the GBIF route that pair (with the cell) defines every sampling "
        f"unit: eventID is present on {e['eventID_present']:,} records. The other {lost:,} records "
        f"cannot be placed in any unit and do not enter the coverage estimates, so coverage and "
        f"rarefied richness describe only the observer-day sample.",
        "",
        f"- Rows: {s['rows']:,}; distinct species (speciesKey): {s['distinct_species']:,}; "
        f"rows without speciesKey: {s['rows_without_speciesKey']:,}",
        f"- recordedBy present: {100 * e['recordedBy_frac']:.2f}%; single-day eventDate: "
        f"{100 * e['eventDate_single_day_frac']:.2f}%",
        f"- Years: {s['years'][0]}-{s['years'][-1]}; months: {s['months']}",
        f"- Records by winter: " + ", ".join(f"{k}: {v:,}" for k, v in s["rows_by_winter"].items()),
        f"- coordinateUncertaintyInMeters > 10 km: {u['gt_10km']:,} = "
        f"{100 * u['gt_10km_frac_of_all']:.3f}% of all records, "
        f"{100 * (u['gt_10km_frac_of_reported'] or 0):.2f}% of the {s['rows'] - u['null']:,} that report it "
        f"(null on {100 * u['null_frac']:.1f}%; trap 7)",
        f"- occurrenceStatus: {s['occurrenceStatus']}",
        f"- basisOfRecord: {s['basisOfRecord']}",
    ]
    return "\n".join(lines) + "\n"


def cmd_report(args):
    path = Path(args.zip) if args.zip else GBIF_DIR / f"{KEY_FILE.read_text().strip()}.zip"
    # eventID is absent from SIMPLE_CSV; read_chunks skips requested columns a format lacks
    cols = ["datasetKey", "eventID", "recordedBy", "eventDate", "year", "month", "speciesKey",
            "coordinateUncertaintyInMeters", "basisOfRecord", "occurrenceStatus"]
    s = summarise(read_chunks(path, usecols=cols))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(s, indent=2))
    SUMMARY_MD.write_text(report_text(s))
    print(report_text(s))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estimate").set_defaults(fn=cmd_estimate)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    rq = sub.add_parser("request")
    rq.add_argument("--format", choices=["simple_csv", "sql"], default="simple_csv")
    rq.set_defaults(fn=cmd_request)
    sub.add_parser("count-sql").set_defaults(fn=cmd_count_sql)
    f = sub.add_parser("fetch")
    f.add_argument("key", nargs="?")
    f.add_argument("--interval", type=int, default=120)
    f.set_defaults(fn=cmd_fetch)
    r = sub.add_parser("report")
    r.add_argument("zip", nargs="?")
    r.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
