# Limitations register

Running list for the report's Limitations section. Each entry gives the
evidence and the number of grid cells affected, so it can be carried into the
write-up without rerunning anything. Numbers come from
`outputs/tables/env_report.json` unless stated.

## Grid (Part 3)

**Lakshadweep falls out of the analysis entirely.** Every one of its islands is
smaller than half a 10 km cell, so none passes the 0.5 land-fraction threshold.
That threshold is fixed for good reason: below it, a cell's richness is divided
by an area that is mostly sea. The same rule removes other small islands. In
total, 21 of the boundary's 35 polygon parts (391 km²) have no grid cell:

| Region | Parts lost | Area (km²) |
|---|---|---|
| Lakshadweep | 11 (the whole archipelago, including Minicoy) | 34 |
| Nicobar Islands | 3 | 148 |
| Andaman Islands (smaller islands) | 5 | 105 |
| Sundarbans delta islands | 2 | 103 |

The Nicobars are a centre of bird endemism, so the loss there is ecologically
larger than its area suggests. No conclusions are drawn for any of these
islands. Coastal and border trimming removes a further 22,750 km² (0.70% of
India's land) from cells that fail the threshold.

## Climate predictors (CHELSA V2.1)

**The Andaman & Nicobar Islands get marine temperature-range values.** On these
narrow islands CHELSA's diurnal range (bio2) and annual range (bio7) can't be
told apart from the open sea next to them. Port Blair reads 1.50 / 3.30 °C; the
Andaman Sea 0.3° east reads 1.50 / 3.20 °C. The real diurnal range at Port Blair
is about 7 °C. This affects 65 cells (43 with bio2 < 2 °C). bio3, the ratio of
those two layers, is affected too, and bio4 probably is. CHELSA itself can't
correct it. These cells will probably also fall outside the area of
applicability (Part 7); they should be named, not quietly modelled.

**Annual precipitation (bio12) is clipped at 6,553.5 mm.** CHELSA stores
precipitation as uint16, and bio12 saturates at the ceiling across 13,909 land
pixels in the Western Ghats and Meghalaya. Real totals there reach about
7,500–12,000 mm. 245 cells contain clipped pixels; in 104 of them more than half
the land is clipped. In those cells bio12 is a lower bound. The table carries
`bio12_saturated_frac` so Part 4 can account for it. bio16 (wettest quarter) is
not clipped after the fix below and is the cleaner wet-extreme variable.

**Corrected in the pipeline (recorded here for transparency):**
- bio16 and bio18 *wrap around* past the uint16 ceiling instead of clipping:
  Mawsynram read 34.7 mm beside 6,552 mm. This is undone exactly (140 and 96
  pixels) using the rule that a quarter cannot hold less rain than its wettest
  month. After the fix, every precipitation identity holds on 100% of land pixels.
- bio3 is stored already divided by 10 but still carries a 0.1 scale tag, and
  bio4 is SD × 100. Both are converted to physical units (% and °C), each checked
  against an independent identity.

**Temperature range is damped along mainland coasts.** Diurnal range drops
toward the sea (Kochi 3.0 °C, Mumbai 4.9 °C, against about 5–8 °C observed),
because CHELSA's coastal pixels mix land with marine air.

## Terrain (Copernicus DEM GLO-90)

The Copernicus DEM is a surface model, so it includes canopy and buildings. It
also records open-pit mines: the cell minimum of −93 m is the Neyveli lignite
mine (79.5°E, 11.5°N), and −64 m is a quarry pit near Chennai (80.1°E, 13.0°N).
They inflate `elev_range` in those two cells. Mean elevation and slope are barely
affected.

## Temporal mismatch

The occurrence records cover winters 2015/16–2023/24, but the predictors come
from three other periods: the 1981–2010 climatology (CHELSA), DEM acquisitions
from 2010–2015, and land cover for the single year 2021. Climate and terrain
change slowly, so the mismatch matters mostly for land cover, where
conversion within the decade (for example urban growth or wetland loss) shows up
as unexplained error.

## Occurrence data (Part 2, GBIF route)

eBird records on GBIF have no checklist ID and no "all species reported" flag,
so sampling units are rebuilt from observer + date + cell and checklist
completeness can't be known. This is why the minimum-species-per-unit threshold
N is run as a sensitivity parameter (1, 3, 5, 10) rather than set to one value.
`coordinateUncertaintyInMeters` is null for eBird records.
