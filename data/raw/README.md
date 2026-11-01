# data/raw

Downloaded inputs. **Never edited by hand.** Each section below is written by the
script that produced the file and records what is needed to cite and reproduce it.

<!-- boundary:start -->
## India national boundary (Part 2)

- File: `india_boundary.gpkg` (layer `india`, EPSG:4326, one MultiPolygon, 35 parts)
- Source: Natural Earth 1:10m Admin-0, India point-of-view edition, version 5.1.1
  <https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries_ind.zip>; raw zip kept at `naturalearth/ne_10m_admin_0_countries_ind.zip`, SHA-256 `36a54ee97a509325e98eb44fd91bc7e53b0c5139cf10039918042bbd64102192`
- License: public domain (Natural Earth terms of use)
- Depiction: boundary as claimed by the Government of India, including all of Jammu &
  Kashmir, Ladakh (with Aksai Chin) and Arunachal Pradesh
- Ellipsoidal area: 3,266,360 km² (official figure 3,287,263 km²)
- Bounds: 68.143–97.362 °E, 6.746–37.054 °N
- Produced by: `python src/boundary.py`

Citation: Natural Earth. Free vector and raster map data @ naturalearthdata.com.
<!-- boundary:end -->

<!-- gbif:start -->
## GBIF occurrence download (Part 2)

- **Citation (required in the report):** GBIF.org (5 October 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.3k5dhd
- DOI: `10.15468/dl.3k5dhd`
- Download key: `0011564-260928105237408`
- Created: 2026-10-05T18:11:30.923+00:00
- Records: 28449817
- Datasets contributing: 47
- License: http://creativecommons.org/licenses/by-nc/4.0/legalcode
- File: `gbif/0011564-260928105237408.zip` (3,565,253,185 bytes), SHA-256 `acb3848d405f5faae441fdb8667580e877876422992b4865a4ad292aab78edc5`
- Format: SIMPLE_CSV, one row per occurrence (no aggregation)
- Produced by: `python src/gbif_download.py fetch`

- Record count: **28,449,817** (+0.00% against the pre-request search-API estimate of 28,449,817-28,465,059; exactly the lower bound). Rows in the file: 28,449,817.
- **Not used, not cited:** SQL request `0011506-260928105237408` finished SUCCEEDED with **0 records** (header only) and was still assigned DOI 10.15468/dl.g5uxhg. **Cause:** GBIF's SQL layer uses alphanumeric taxon keys (Aves is classkey `'V2'`), so the hand-written `classkey = 212` matched nothing -- not permissions, not nulls (CLAUDE.md trap 15). Record: `gbif/request_0011506-260928105237408.json`.
- Diagnostic SQL `0011565-260928105237408` (DOI 10.15468/dl.6rjtqg, not cited): `SELECT COUNT(*) FROM occurrence WHERE classkey = 212 AND countrycode = 'IN'` returned **0**, against 61,519,735 for the same two keys in the search API, confirming the basic taxon column as the fault.

Query as GBIF executed it:

```json
{
  "type": "and",
  "predicates": [
    {
      "type": "equals",
      "key": "CLASS_KEY",
      "value": "212"
    },
    {
      "type": "equals",
      "key": "COUNTRY",
      "value": "IN"
    },
    {
      "type": "greaterThanOrEquals",
      "key": "YEAR",
      "value": "2015"
    },
    {
      "type": "lessThanOrEquals",
      "key": "YEAR",
      "value": "2024"
    },
    {
      "type": "in",
      "key": "MONTH",
      "values": [
        "11",
        "12",
        "1",
        "2"
      ]
    },
    {
      "type": "equals",
      "key": "HAS_COORDINATE",
      "value": "true"
    },
    {
      "type": "equals",
      "key": "HAS_GEOSPATIAL_ISSUE",
      "value": "false"
    },
    {
      "type": "not",
      "predicate": {
        "type": "in",
        "key": "BASIS_OF_RECORD",
        "values": [
          "FOSSIL_SPECIMEN",
          "LIVING_SPECIMEN"
        ]
      }
    },
    {
      "type": "or",
      "predicates": [
        {
          "type": "isNull",
          "parameter": "DISTANCE_FROM_CENTROID_IN_METERS"
        },
        {
          "type": "greaterThanOrEquals",
          "key": "DISTANCE_FROM_CENTROID_IN_METERS",
          "value": "2000"
        }
      ]
    },
    {
      "type": "not",
      "predicate": {
        "type": "in",
        "key": "ISSUE",
        "values": [
          "TAXON_MATCH_FUZZY"
        ]
      }
    }
  ]
}
```
<!-- gbif:end -->

<!-- chelsa:start -->
## CHELSA V2.1 bioclimatic variables (Part 3)

- Used for all 19 bioclim predictors (bio1-bio19), climatology 1981-2010, 30 arc-sec (~1 km)
- Files: `chelsa/CHELSA_bio{1..19}_1981-2010_V.2.1_india.tif` -- exact pixel subsets
  (India bounds + 0.1 deg) of the global files, read by HTTP range request; raw
  integer values unchanged, GeoTIFF scale/offset carried over (e.g. bio1: 0.1, -273.15)
- Source: <https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio/CHELSA_bioN_1981-2010_V.2.1.tif>
- Note: CHELSA has modelled values over the sea and declares no nodata
- Corrections applied in `src/env_covariates.py` (raw files untouched):
  bio3 x100 (stored /10 yet tagged scale 0.1) and bio4 x0.01 (SD x100) to physical units;
  bio16/bio18 uint16 wraparound undone (`fix-chelsa` -> `data/interim/chelsa_fixed/`);
  bio12 clipping at 6553.5 mm cannot be undone and is flagged per cell as
  `bio12_saturated_frac`. See `outputs/limitations.md`.
- Citation: Karger, D.N. et al. (2017) Climatologies at high resolution for the earth's land
  surface areas. *Scientific Data* 4, 170122. Data: https://doi.org/10.16904/envidat.228
- Produced by: `python src/env_covariates.py fetch-chelsa`
<!-- chelsa:end -->

<!-- copdem:start -->
## Copernicus DEM GLO-90 (Part 3)

- 389 1x1 degree COG tiles intersecting India, 3 arc-sec (~90 m), in `copdem90/`
- Source: <https://copernicus-dem-90m.s3.amazonaws.com/> (AWS Open Data registry)
- Note: sea is filled with ~0 m, not nodata
- Citation: Copernicus DEM GLO-90, (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space
  GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA.
  https://doi.org/10.5270/ESA-c5d3d65
- Produced by: `python src/env_covariates.py fetch-dem`
<!-- copdem:end -->

<!-- worldcover:start -->
## ESA WorldCover 2021 v200 (Part 3) -- streamed, not stored

- 10 m tiles (64 3x3 degree tiles intersecting India) downloaded one at a time,
  aggregated to per-cell class coverage, then deleted. Only the aggregate is kept:
  `data/interim/worldcover_cell_coverage.parquet`.
- The internal overviews were not used: they are majority-resampled (checked on tile
  N27E075: overview followed the block majority in 99% of disagreeing blocks), which
  understated grassland 0.43% -> 0.17% in a test window.
- Source: <https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_<tile>_Map.tif>
- Citation: Zanaga, D. et al. (2022) ESA WorldCover 10 m 2021 v200. Zenodo.
  https://doi.org/10.5281/zenodo.7254221 (CC BY 4.0)
- Produced by: `python src/env_covariates.py worldcover`
<!-- worldcover:end -->
