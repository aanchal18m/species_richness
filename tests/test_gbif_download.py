"""Tests for src/gbif_download.py.

Run: python tests/test_gbif_download.py   (or pytest)

The query tests execute the generated SQL in SQLite against simulated rows whose
expected fate is known, rather than inspecting the string for keywords.
"""
import io
import json
import os
import re
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gbif_download as g  # noqa: E402

# --- helpers ----------------------------------------------------------------
TABLE_COLS = ["gbifid", "datasetkey", "occurrenceid", "eventid", "recordedby", "eventdate",
              "year", "month", "day", "decimallatitude", "decimallongitude",
              "coordinateuncertaintyinmeters", "specieskey", "species", "taxonrank",
              "basisofrecord", "occurrencestatus", "classkey", "countrycode", "hascoordinate",
              "hasgeospatialissues", "distancefromcentroidinmeters", "issue"]

GOOD = dict(datasetkey="ds", occurrenceid="occ", eventid=None, recordedby=json.dumps(["obsr1"]),
            eventdate="2016-01-10", year=2016, month=1, day=10, decimallatitude=20.0,
            decimallongitude=78.0, coordinateuncertaintyinmeters=None, specieskey=1,
            species="Sp a", taxonrank="SPECIES", basisofrecord="HUMAN_OBSERVATION",
            occurrencestatus="PRESENT", classkey=212, countrycode="IN", hascoordinate=1,
            hasgeospatialissues=0, distancefromcentroidinmeters=50_000.0,
            issue=json.dumps(["COORDINATE_ROUNDED"]))

# gbifid -> (overrides, should_be_kept)
CASES = {
    "keep_baseline": ({}, True),
    "keep_null_issue": ({"issue": None}, True),
    "keep_empty_issue": ({"issue": "[]"}, True),
    "keep_null_centroid": ({"distancefromcentroidinmeters": None}, True),
    "keep_centroid_exactly_2km": ({"distancefromcentroidinmeters": 2000.0}, True),
    "keep_nov_2015": ({"year": 2015, "month": 11}, True),
    "keep_dec_2024": ({"year": 2024, "month": 12}, True),
    "keep_feb": ({"month": 2}, True),
    "keep_machine_obs": ({"basisofrecord": "MACHINE_OBSERVATION"}, True),
    "drop_march": ({"month": 3}, False),
    "drop_october": ({"month": 10}, False),
    "drop_null_month": ({"month": None}, False),
    "drop_2014": ({"year": 2014, "month": 12}, False),
    "drop_2025": ({"year": 2025, "month": 1}, False),
    "drop_not_aves": ({"classkey": 359}, False),
    "drop_not_india": ({"countrycode": "NP"}, False),
    "drop_no_coords": ({"hascoordinate": 0}, False),
    "drop_geo_issue": ({"hasgeospatialissues": 1}, False),
    "drop_fossil": ({"basisofrecord": "FOSSIL_SPECIMEN"}, False),
    "drop_living": ({"basisofrecord": "LIVING_SPECIMEN"}, False),
    "drop_centroid_1999m": ({"distancefromcentroidinmeters": 1999.0}, False),
    "drop_fuzzy": ({"issue": json.dumps(["COORDINATE_ROUNDED", "TAXON_MATCH_FUZZY"])}, False),
}


def run_in_sqlite(sql, cases=CASES):
    con = sqlite3.connect(":memory:")
    # SQL NULL semantics: a NULL array gives NULL, exactly as in GBIF's engine.
    con.create_function("ARRAY_CONTAINS", 2,
                        lambda a, v: None if a is None else int(v in json.loads(a)))
    con.create_function("ARRAY_JOIN", 2,
                        lambda a, s: None if a is None else s.join(json.loads(a)))
    con.execute(f"CREATE TABLE occurrence ({', '.join(TABLE_COLS)})")
    for gid, (over, _) in cases.items():
        row = {**GOOD, **over, "gbifid": gid}
        con.execute(f"INSERT INTO occurrence VALUES ({', '.join('?' * len(TABLE_COLS))})",
                    [row[c] for c in TABLE_COLS])
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    return names, [dict(zip(names, r)) for r in cur.fetchall()]


def selected_names(sql):
    body = re.search(r"SELECT(.*?)\bFROM\b", sql, re.S | re.I).group(1)
    parts, depth, cur = [], 0, ""
    for ch in body:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    out = []
    for p in parts:
        m = re.search(r"\bAS\s+(\w+)\s*$", p.strip(), re.I)
        out.append((m.group(1) if m else p.strip()).strip('"').lower())
    return out


def write_zip(path, header, rows):
    text = "\t".join(header) + "\n" + "".join("\t".join(r) + "\n" for r in rows)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("0000000-000000000000000.csv", text)


# --- query ------------------------------------------------------------------
def test_select_contains_every_required_field():
    names = selected_names(g.build_sql())
    missing = [f for f in g.REQUIRED_FIELDS if f.lower() not in names]
    assert not missing, f"required fields not selected: {missing}"


def test_sql_is_record_level():
    """Trap 5: an aggregate download destroys the sampling units."""
    sql = g.build_sql().upper()
    for bad in ("GROUP BY", "DISTINCT", "COUNT(", "SUM(", "MIN(", "MAX(", "AVG("):
        assert bad not in sql, f"aggregate construct {bad!r} in download SQL"
    names, rows = run_in_sqlite(g.build_sql())
    # one output row per kept input row: nothing collapsed
    assert len(rows) == sum(keep for _, keep in CASES.values())


def test_filters_keep_and_drop_exactly_the_right_rows():
    _, rows = run_in_sqlite(g.build_sql())
    kept = {r["gbifid"] for r in rows}
    expected = {gid for gid, (_, keep) in CASES.items() if keep}
    assert kept == expected, (f"wrongly dropped: {sorted(expected - kept)}; "
                              f"wrongly kept: {sorted(kept - expected)}")


def test_harness_detects_null_unsafe_predicates():
    """The previous test is only meaningful if the naive form fails it."""
    naive = re.sub(r"\(issue IS NULL OR (NOT ARRAY_CONTAINS\(issue, '\w+'\))\)", r"\1", g.build_sql())
    naive = re.sub(r"\(distancefromcentroidinmeters IS NULL OR (distancefromcentroidinmeters >= \d+)\)",
                   r"\1", naive)
    assert naive != g.build_sql(), "mutation did not apply; harness check is vacuous"
    _, rows = run_in_sqlite(naive)
    kept = {r["gbifid"] for r in rows}
    assert "keep_null_issue" not in kept and "keep_null_centroid" not in kept


def test_recordedby_array_is_joined_reversibly():
    cases = {"two_observers": ({"recordedby": json.dumps(["obsr1", "obsr2"])}, True),
             "no_observer": ({"recordedby": None}, True)}
    _, rows = run_in_sqlite(g.build_sql(), cases)
    by_id = {r["gbifid"]: r["recordedby"] for r in rows}
    assert by_id == {"two_observers": "obsr1|obsr2", "no_observer": None}


def test_gbif_accepts_the_sql():
    try:
        normalised = g.validate(g.build_sql(), timeout=30)
    except requests.RequestException as e:
        raise unittest.SkipTest(f"GBIF unreachable: {e}")
    assert "GROUP BY" not in normalised.upper()
    assert "ARRAY_CONTAINS" in normalised.upper()


# --- predicate (SIMPLE_CSV) route ---------------------------------------------
PRED_COLUMN = {"CLASS_KEY": "classkey", "COUNTRY": "countrycode", "YEAR": "year", "MONTH": "month",
               "HAS_COORDINATE": "hascoordinate", "HAS_GEOSPATIAL_ISSUE": "hasgeospatialissues",
               "BASIS_OF_RECORD": "basisofrecord", "ISSUE": "issue",
               "DISTANCE_FROM_CENTROID_IN_METERS": "distancefromcentroidinmeters"}


def _value(row, key):
    v = row[PRED_COLUMN[key]]
    if key in ("HAS_COORDINATE", "HAS_GEOSPATIAL_ISSUE"):
        return None if v is None else ("true" if v else "false")
    return v


def eval_predicate(p, row):
    """GBIF predicate semantics on one fixture row. ISSUE is an array: equals/in
    mean 'contains', and a record with no issues contains nothing."""
    t = p["type"]
    if t == "and":
        return all(eval_predicate(q, row) for q in p["predicates"])
    if t == "or":
        return any(eval_predicate(q, row) for q in p["predicates"])
    if t == "not":
        return not eval_predicate(p["predicate"], row)
    if t == "isNull":
        return row[PRED_COLUMN[p["parameter"]]] is None
    v = _value(row, p["key"])
    if p["key"] == "ISSUE":
        issues = json.loads(v) if v else []
        wanted = p["values"] if t == "in" else [p["value"]]
        return any(w in issues for w in wanted)
    if v is None:
        return False
    if t == "equals":
        return str(v) == p["value"]
    if t == "in":
        return str(v) in p["values"]
    num = float(v)
    return {"greaterThanOrEquals": num >= float(p["value"]),
            "lessThanOrEquals": num <= float(p["value"])}[t]


def test_predicate_keeps_and_drops_exactly_what_the_sql_does():
    """Both routes must select the same records from the same edge-case fixture."""
    pred = g.build_predicate()
    kept_pred = {gid for gid, (over, _) in CASES.items()
                 if eval_predicate(pred, {**GOOD, **over, "gbifid": gid})}
    _, rows = run_in_sqlite(g.build_sql())
    assert kept_pred == {r["gbifid"] for r in rows}, kept_pred ^ {r["gbifid"] for r in rows}
    months = next(q for q in pred["predicates"] if q.get("key") == "MONTH")
    assert months["type"] == "in" and sorted(map(int, months["values"])) == [1, 2, 11, 12]


def test_record_count_guard():
    est = {"records_low": 28_449_817, "records_high": 28_465_059}
    g.check_record_count({**FAKE_META, "totalRecords": 28_455_000}, est)
    g.check_record_count({**FAKE_META, "totalRecords": 28_455_000}, None)     # no estimate: no check
    for n in (0, 14_000_000, 40_000_000):
        try:
            g.check_record_count({**FAKE_META, "totalRecords": n}, est)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{n:,} records accepted against an estimate of ~28.45M")


SIMPLE_CSV_HEADER = ["gbifID", "datasetKey", "occurrenceID", "kingdom", "phylum", "class", "order",
                     "family", "genus", "species", "infraspecificEpithet", "taxonRank", "scientificName",
                     "verbatimScientificName", "verbatimScientificNameAuthorship", "countryCode",
                     "locality", "stateProvince", "occurrenceStatus", "individualCount",
                     "publishingOrgKey", "decimalLatitude", "decimalLongitude",
                     "coordinateUncertaintyInMeters", "coordinatePrecision", "elevation",
                     "elevationAccuracy", "depth", "depthAccuracy", "eventDate", "day", "month", "year",
                     "taxonKey", "speciesKey", "basisOfRecord", "institutionCode", "collectionCode",
                     "catalogNumber", "recordNumber", "identifiedBy", "dateIdentified", "license",
                     "rightsHolder", "recordedBy", "typeStatus", "establishmentMeans", "lastInterpreted",
                     "mediaType", "issue"]   # GBIF /occurrence/download/describe/simpleCsv, 2026-10-05


def test_read_chunks_passes_usecols_to_read_csv():
    """Output alone cannot tell column selection inside read_csv from selection
    afterwards; the second builds every column first. Assert on the call."""
    import pandas as pd
    calls = []
    real = pd.read_csv

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)
    header = SIMPLE_CSV_HEADER
    with tempfile.TemporaryDirectory() as d:
        z = Path(d) / "simple.zip"
        write_zip(z, header, [[f"v{i}" for i in range(len(header))]] * 5)
        pd.read_csv = spy
        try:
            want = ["speciesKey", "eventID", "recordedBy"]          # eventID absent from SIMPLE_CSV
            out = pd.concat(g.read_chunks(z, usecols=want))
            full = pd.concat(g.read_chunks(z))
        finally:
            pd.read_csv = real
    assert sorted(calls[0]["usecols"]) == ["recordedBy", "speciesKey"], calls[0]["usecols"]
    assert calls[0]["chunksize"] == 250_000
    assert list(out.columns) == ["speciesKey", "recordedBy"] and len(out) == 5
    assert len(calls[1]["usecols"]) == len(header) and full.shape == (5, len(header))


def test_read_chunks_maps_sql_lowercase_headers():
    with tempfile.TemporaryDirectory() as d:
        z = Path(d) / "sql.zip"
        write_zip(z, ["gbifid", "specieskey", "decimallatitude", "eventid"], [["1", "10", "20.5", ""]])
        out = next(g.read_chunks(z, usecols=["speciesKey", "decimalLatitude", "eventID"]))
    assert list(out.columns) == ["speciesKey", "decimalLatitude", "eventID"]
    assert out.iloc[0].tolist()[:2] == ["10", "20.5"]


def test_simple_csv_reads_and_summarises_without_eventid():
    def row(**kv):
        base = {c: "" for c in SIMPLE_CSV_HEADER}
        base.update(datasetKey="e", occurrenceStatus="PRESENT", basisOfRecord="HUMAN_OBSERVATION",
                    taxonRank="SPECIES", recordedBy="obsr1", eventDate="2016-01-05", year="2016",
                    month="1", speciesKey="10", decimalLatitude="20.0", decimalLongitude="78.0",
                    locality='5" north of the "lake"')                      # stray quotes, as in real data
        base.update(kv)
        return [base[c] for c in SIMPLE_CSV_HEADER]
    with tempfile.TemporaryDirectory() as d:
        z = Path(d) / "simple.zip"
        write_zip(z, SIMPLE_CSV_HEADER, [row(), row(speciesKey="11", coordinateUncertaintyInMeters="25000"),
                                         row(recordedBy="", eventDate="2016-02")])
        cols = ["datasetKey", "eventID", "recordedBy", "eventDate", "year", "month", "speciesKey",
                "coordinateUncertaintyInMeters", "basisOfRecord", "occurrenceStatus"]
        chunks = list(g.read_chunks(z, chunksize=2, usecols=cols))
        assert all("eventID" not in c for c in chunks) and sum(map(len, chunks)) == 3
        s = g.summarise(g.read_chunks(z, chunksize=2, usecols=cols))
    assert s["rows"] == 3 and s["distinct_species"] == 2
    assert s["event_fields"]["eventID_present"] == 0
    assert s["event_fields"]["recordedBy_and_single_day"] == 2
    assert s["coordinate_uncertainty"]["gt_10km"] == 1


# --- credentials ------------------------------------------------------------
def test_missing_credentials_named_and_not_echoed():
    env = {"GBIF_USER": "u", "GBIF_PWD": "SENTINEL-PWD-91x"}
    try:
        g.credentials(env)
    except RuntimeError as e:
        msg = str(e)
        assert re.search(r"variable\(s\): ([^.]*)\.", msg).group(1) == "GBIF_EMAIL", msg
        assert "SENTINEL-PWD-91x" not in msg
    else:
        raise AssertionError("missing GBIF_EMAIL did not raise")


def test_credentials_never_reach_written_files():
    secret = "SENTINEL-PWD-91x"
    env = {"GBIF_USER": "sentinel-user-91x", "GBIF_PWD": secret, "GBIF_EMAIL": "s91x@example.org"}
    assert g.credentials(env) == ("sentinel-user-91x", secret, "s91x@example.org")
    with tempfile.TemporaryDirectory() as d:
        z = Path(d) / "k.zip"
        write_zip(z, ["gbifid"], [["1"]])
        readme = Path(d) / "README.md"
        g.write_section(readme, "gbif", g.gbif_readme_body(FAKE_META, z, "abc"))
        text = readme.read_text()
    for v in env.values():
        assert v not in text


# --- polling ----------------------------------------------------------------
def _meta_sequence(statuses):
    it = iter(statuses)

    def get_meta(_key):
        s = next(it)
        if isinstance(s, Exception):
            raise s
        return {"status": s, "key": "k"}
    return get_meta


def test_wait_returns_on_success_after_running():
    sleeps = []
    meta = g.wait_for("k", _meta_sequence(["PREPARING", "RUNNING", "RUNNING", "SUCCEEDED"]),
                      sleep=sleeps.append, interval=5, log=lambda *_: None)
    assert meta["status"] == "SUCCEEDED" and sleeps == [5, 5, 5]


def test_wait_raises_on_failure_status():
    for bad in ("FAILED", "KILLED", "CANCELLED"):
        try:
            g.wait_for("k", _meta_sequence(["RUNNING", bad]), sleep=lambda _: None,
                       log=lambda *_: None)
        except RuntimeError as e:
            assert bad in str(e)
        else:
            raise AssertionError(f"{bad} not treated as terminal failure")


def test_wait_survives_transient_network_error():
    meta = g.wait_for("k", _meta_sequence(["RUNNING", requests.ConnectionError("x"), "SUCCEEDED"]),
                      sleep=lambda _: None, log=lambda *_: None)
    assert meta["status"] == "SUCCEEDED"


def test_wait_times_out():
    try:
        g.wait_for("k", _meta_sequence(["RUNNING"] * 10), sleep=lambda _: None,
                   interval=10, timeout=30, log=lambda *_: None)
    except TimeoutError:
        pass
    else:
        raise AssertionError("no timeout")


# --- file integrity and provenance ------------------------------------------
FAKE_META = {"key": "0001234-261005120000000", "doi": "10.15468/dl.abc123",
             "created": "2026-10-05T11:30:00.000+00:00", "totalRecords": 3,
             "numberDatasets": 2, "license": "http://creativecommons.org/licenses/by-nc/4.0/legalcode",
             "request": {"sql": "SELECT gbifid FROM occurrence"}, "status": "SUCCEEDED"}


def test_empty_download_is_refused():
    """Download 0011506-260928105237408 came back SUCCEEDED, 0 records, with a DOI."""
    try:
        g.check_nonempty({**FAKE_META, "totalRecords": 0})
    except ValueError as e:
        assert "0 records" in str(e)
    else:
        raise AssertionError("empty download accepted")
    g.check_nonempty(FAKE_META)  # 3 records: fine


def _flaky_server(payload: bytes, drop_after: int):
    """Local HTTP server with Range support whose responses are cut off after
    `drop_after` bytes -- each attempt delivers at most that much."""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            start = 0
            rng = self.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(payload) - start))
            self.end_headers()
            self.wfile.write(payload[start:start + drop_after])   # then hang up early
            self.close_connection = True

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_resumable_download_survives_dropped_connections():
    payload = np.random.default_rng(0).bytes(1_000_000)
    srv = _flaky_server(payload, drop_after=150_000)          # 7 drops before completion
    try:
        with tempfile.TemporaryDirectory() as d:
            part = Path(d) / "x.zip"
            part.write_bytes(payload[:40_000])                # a partial left by an earlier attempt
            g.download_resumable(f"http://127.0.0.1:{srv.server_port}/x.zip", part, len(payload),
                                 sleep=lambda _: None, log=lambda *_: None)
            assert part.read_bytes() == payload
    finally:
        srv.shutdown()


def test_verify_zip_rejects_truncated_and_accepts_whole():
    with tempfile.TemporaryDirectory() as d:
        z = Path(d) / "a.zip"
        write_zip(z, ["gbifid"], [[str(i)] for i in range(5000)])
        size = z.stat().st_size
        g.verify_zip(z, size)
        try:
            g.verify_zip(z, size + 1)
        except IOError:
            pass
        else:
            raise AssertionError("size mismatch accepted")
        z.write_bytes(z.read_bytes()[: size // 2])
        try:
            g.verify_zip(z, size // 2)
        except IOError:
            pass
        else:
            raise AssertionError("truncated zip accepted")


def test_readme_for_predicate_download_shows_predicate_and_uncited_request():
    meta = {**FAKE_META, "request": {"format": "SIMPLE_CSV", "predicate": g.build_predicate()}}
    with tempfile.TemporaryDirectory() as d:
        z = Path(d) / "k.zip"
        write_zip(z, ["gbifID"], [["1"]])
        body = g.gbif_readme_body(meta, z, "abc")
    assert "SIMPLE_CSV" in body and '"DISTANCE_FROM_CENTROID_IN_METERS"' in body
    assert "10.15468/dl.g5uxhg" in body and "0 records" in body and "not cited" in body
    assert "GBIF.org (5 October 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.abc123" in body


def test_count_note_states_count_against_estimate():
    est = {"records_low": 28_449_817, "records_high": 28_465_059}
    note = g.count_note({"totalRecords": 28_449_817}, est, file_rows=28_449_817)
    assert "28,449,817" in note and "+0.00%" in note and "exactly the lower bound" in note
    assert "Rows in the file: 28,449,817" in note
    assert "-0.10%" in g.count_note({"totalRecords": 28_421_309}, est)
    assert g.count_note({"totalRecords": 5}, None) == ""


def test_readme_records_doi_and_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        z = Path(d) / "k.zip"
        write_zip(z, ["gbifid"], [["1"]])
        readme = Path(d) / "README.md"
        g.write_section(readme, "boundary", "## Boundary\nkeep me")
        for _ in range(2):
            g.write_section(readme, "gbif", g.gbif_readme_body(FAKE_META, z, "abc"))
        text = readme.read_text()
    assert text.count("<!-- gbif:start -->") == 1
    assert "keep me" in text
    assert "GBIF.org (5 October 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.abc123" in text


# --- summary: known answers -------------------------------------------------
def _summary_fixture(path):
    header = ["gbifid", "datasetkey", "occurrenceid", "eventid", "recordedby", "eventdate",
              "year", "month", "specieskey", "coordinateuncertaintyinmeters", "basisofrecord",
              "occurrencestatus"]
    H, P = "HUMAN_OBSERVATION", "PRESENT"
    rows = [
        # id ds  occ   eventid recordedBy eventDate              y      m    sp   unc      basis status
        ["1", "e", "a", "", "obsr1", "2015-01-05", "2015", "1", "10", "", H, P],
        ["2", "e", "b", "", "obsr1", "2015-01-05", "2015", "1", "11", "10000", H, P],       # exactly 10 km: not >
        ["3", "e", 'c"q', "", "obsr2", "2015-11-30T06:15", "2015", "11", "10", "10000.5", H, P],  # stray quote
        ["4", "i", "d", "ev9", "", "2016-02", "2016", "2", "12", "25000", H, P],          # eventID only
        ["5", "i", "e", "", "obsr3", "2016-02-01/2016-02-03", "2016", "2", "", "50", H, "ABSENT"],  # range
        ["6", "i", "f", "", "", "2024-12-31", "2024", "12", "13", "", "MACHINE_OBSERVATION", P],
    ]
    write_zip(path, header, rows)


def test_summarise_known_answers():
    with tempfile.TemporaryDirectory() as d:
        z = Path(d) / "s.zip"
        _summary_fixture(z)
        s = g.summarise(g.read_chunks(z, chunksize=2))  # tiny chunks: cross-chunk accumulation
    assert s["rows"] == 6, s["rows"]
    assert s["distinct_species"] == 4                      # 10, 11, 12, 13
    assert s["rows_without_speciesKey"] == 1
    assert s["years"] == [2015, 2016, 2024] and s["months"] == [1, 2, 11, 12]
    assert s["rows_by_winter"] == {"2014/15": 2, "2015/16": 3, "2024/25": 1}, s["rows_by_winter"]
    e = s["event_fields"]
    assert e["eventID_present"] == 1
    assert e["recordedBy_present"] == 4
    assert e["eventDate_single_day"] == 4                  # rows 1,2,3,6 (not '2016-02', not a range)
    assert e["recordedBy_and_single_day"] == 3             # rows 1,2,3
    assert e["eventID_or_observer_day"] == 4               # + row 4
    u = s["coordinate_uncertainty"]
    assert u["null"] == 2 and u["gt_10km"] == 2            # 10000.5 and 25000; 10000 is not >
    assert abs(u["gt_10km_frac_of_all"] - 2 / 6) < 1e-6
    assert abs(u["gt_10km_frac_of_reported"] - 2 / 4) < 1e-6
    assert s["occurrenceStatus"] == {"PRESENT": 5, "ABSENT": 1}


def test_report_text_leads_with_unit_fields():
    with tempfile.TemporaryDirectory() as d:
        z = Path(d) / "s.zip"
        _summary_fixture(z)
        s = g.summarise(g.read_chunks(z, chunksize=2))
    first = g.report_text(s).split("\n")[2]
    assert first.startswith("**Sampling-unit fields: 3 of 6 records (50.00%) carry BOTH"), first
    assert "The other 3 records" in first


def test_reader_does_not_merge_rows_on_stray_quote():
    with tempfile.TemporaryDirectory() as d:
        z = Path(d) / "s.zip"
        _summary_fixture(z)
        df = next(g.read_chunks(z, chunksize=100))
    assert len(df) == 6 and df.loc[2, "occurrenceID"] == 'c"q'
    assert "decimalLatitude" not in df.columns and "eventID" in df.columns  # DwC names restored


# --- runner -----------------------------------------------------------------
if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS  {name}")
        except unittest.SkipTest as e:
            skipped += 1
            print(f"SKIP  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed + failed} checks passed" + (f", {skipped} skipped" if skipped else ""))
    sys.exit(1 if failed else 0)
