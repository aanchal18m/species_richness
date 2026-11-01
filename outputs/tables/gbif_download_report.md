# GBIF download: what it can support

**Sampling-unit fields: 28,433,294 of 28,449,817 records (99.94%) carry BOTH a usable recordedBy and a single-day eventDate.** On the GBIF route that pair (with the cell) defines every sampling unit: eventID is present on 0 records. The other 16,523 records cannot be placed in any unit and do not enter the coverage estimates, so coverage and rarefied richness describe only the observer-day sample.

- Rows: 28,449,817; distinct species (speciesKey): 1,333; rows without speciesKey: 762,644
- recordedBy present: 99.94%; single-day eventDate: 100.00%
- Years: 2015-2024; months: [1, 2, 11, 12]
- Records by winter: 2014/15: 352,800, 2015/16: 1,041,168, 2016/17: 1,499,235, 2017/18: 1,787,919, 2018/19: 2,263,856, 2019/20: 2,378,524, 2020/21: 3,066,708, 2021/22: 3,543,390, 2022/23: 4,498,558, 2023/24: 5,600,696, 2024/25: 2,416,963
- coordinateUncertaintyInMeters > 10 km: 16,438 = 0.058% of all records, 9.99% of the 164,543 that report it (null on 99.4%; trap 7)
- occurrenceStatus: {'PRESENT': 28449614, 'ABSENT': 203}
- basisOfRecord: {'HUMAN_OBSERVATION': 28446312, 'MATERIAL_SAMPLE': 10, 'MACHINE_OBSERVATION': 3487, 'PRESERVED_SPECIMEN': 8}
