# Can Environmental Variables Predict Bird Species Richness in Poorly Sampled Regions of India?

**Team Name:** Groot

**Team Members:** Mahek Desai (2024101096), Laveena Jain (2024111022), Aanchal Mundhada (2023112016)

---

## 1. Problem statement

Species richness is recorded unevenly. Across India, bird observation effort concentrates around cities, protected areas and established birding sites, while large areas carry few records or none. The practical hope behind environmental richness modelling is that this gap can be closed by inference: if climate, terrain and land cover govern where species occur, a model trained where data exist should be able to describe biodiversity where data do not.

Whether that hope is justified is rarely tested directly. Models are fitted on well-surveyed data, evaluated with cross-validation that randomly withholds rows from the same well-surveyed pool, and then projected into regions that resemble nothing in the training set. The reported accuracy describes performance on data-rich cells, which is not the situation anyone actually cares about.

This project tests the hope on its own terms. It asks whether environmental variables predict bird species richness in poorly sampled parts of India well enough to be useful, defines in advance what "well enough" means, and uses explainable machine learning to establish whether the model's answer rests on ecologically sensible relationships or on artefacts of where people happen to go birdwatching.

## 2. Research questions

**Primary.** Can environmental variables predict bird species richness in poorly sampled regions of India sufficiently well to identify biodiversity patterns there?

**S1.** How far does apparent accuracy fall when evaluation moves from random cross-validation to spatially blocked cross-validation, and further to simulated sparse-sampling conditions?

**S2.** Over what fraction of poorly sampled India do predictions fall inside the model's area of applicability?

**S3.** Which environmental gradients drive the predictions, and are the relationships SHAP recovers consistent with established ecological understanding of bird richness?

**S4.** Which poorly sampled areas are predicted to be richer than their records suggest, and therefore merit survey effort?

## 3. What "sufficiently well" will mean

The brief asks whether prediction is good enough "to identify biodiversity patterns". That is a question about *ordering and pattern*, not about exact counts: a model that systematically under-predicts by fifteen species but ranks cells correctly is useful for finding where biodiversity is concentrated, while one with low error and scrambled ranking is not. We therefore judge the model on pattern recovery rather than on R², and we fix the criteria before seeing any results.

| Criterion | Measure | Threshold | Why this threshold |
|---|---|---|---|
| **C1 Pattern** | Spearman rank correlation between predicted and withheld observed richness, under spatial blocking | ρ ≥ 0.60 | Below this the cell ordering is too noisy to guide any decision |
| **C2 Hotspots** | Share of truly top-quintile cells that appear in the predicted top quintile | recall ≥ 0.70 | A screening tool missing a third of hotspots is still a useful first filter; one missing half is not |
| **C3 Reach** | Fraction of poorly sampled cells inside the area of applicability | ≥ 0.50 | Below this, most of the region of interest is extrapolation rather than prediction |

These are judgement calls. They are stated here so that they cannot be adjusted after the fact, and the report will give the achieved values whether or not they clear the bar. A model meeting C1 and C2 but failing C3 would mean the predictions are good where they apply and inapplicable over most of the area we care about — a different and more interesting answer than simple failure.

## 4. Scope

| Dimension | Decision |
|---|---|
| Taxon | Aves, species rank |
| Extent | India, terrestrial |
| Grain | 10 km equal-area grid (approximately 33,000 cells) |
| Season | Winter window (November–February) |
| Period | 2015–2024 |
| Response | Coverage-based rarefied richness from complete checklists |
| Training set | Cells meeting a sample-coverage threshold |
| Target set | Cells below that threshold |

**Why season is fixed rather than averaged.** India receives a large winter migrant influx, so annual richness confounds residency with migration. The winter window is chosen because eBird participation in India peaks then, maximising the number of cells that qualify as well sampled.

**Out of scope:** species-level distribution models, population trends, climate-change projection, comparison between biogeographic regions, and any attempt to beat existing maps of Indian bird distribution on accuracy.

## 5. Data

| Dataset | Contribution | Resolution | Access |
|---|---|---|---|
| eBird Basic Dataset | Complete checklists with effort metadata; species lists | Checklist | Free; request form; processed with `auk` |
| GBIF | Fallback occurrence source; 61,430,928 georeferenced Indian bird records confirmed available | Point | API; free account; citable DOI |
| CHELSA V2.1 | 19 bioclimatic variables, 1981–2010 | ~1 km | Direct download |
| Copernicus DEM GLO-90 | Mean elevation, elevation range, standard deviation, mean slope | 90 m | AWS open-data mirror |
| ESA WorldCover 2021 v200 | Per-class area fractions; Shannon diversity of classes | 10 m | Zenodo / Terrascope |
| IUCN Red List | Threat category per species, for the optional extension in section 6.8 | Per species | API |
| GHSL population; OSM roads | Effort and accessibility covariates | Varies | Direct download |
| WorldClim 2.1 | Robustness check only | ~1 km | Direct download |

## 6. Method

### 6.1 Response variable

Checklists are filtered to a comparable subset: complete checklists only, stationary and travelling protocols, duration 5–300 minutes, distance under 5 km, at most ten observers. Per cell we compute coverage-based rarefied richness and sample coverage following Chao et al. (2014). Raw counts are not used as the response, because they measure richness multiplied by effort.

### 6.2 Splitting well- from poorly sampled cells

Cells achieving sample coverage at or above the target are **well sampled** and form the training pool. Cells below it are **poorly sampled** and are the prediction targets — they have a richness value, but not one we trust. The threshold is set at Gate 1 from the observed coverage distribution rather than guessed in advance, and the split is reported as a map, because where the data are is itself a finding.

### 6.3 Predictors

Nineteen bioclimatic variables from CHELSA; elevation mean, range, standard deviation and slope from the Copernicus DEM; land-cover class fractions and Shannon diversity from ESA WorldCover. Collinearity is reduced by variance inflation factor screening. Effort covariates — checklist count, observer count, population density, distance to road — are included so the model can account for residual effort structure within the training pool rather than absorbing it into the climate terms.

### 6.4 Models

Random Forest and XGBoost. Algorithm choice is deliberately conservative: the brief asks for ecological interpretation rather than algorithmic complexity, and both methods support SHAP cleanly. Hyperparameters are tuned inside the spatial folds.

### 6.5 Validation at three levels of honesty

This is the core of S1. The same model is evaluated three ways, and the decline between them is itself the result.

**Level 1 — random k-fold.** The standard, optimistic approach. Reported only as a baseline for comparison.

**Level 2 — spatially blocked cross-validation** with nearest-neighbour distance matching (Milà et al., 2022). Withholds spatial blocks rather than random rows, so the model cannot exploit autocorrelation with neighbouring training cells.

**Level 3 — simulated sparse sampling.** Well-sampled cells are downsampled to the checklist depth typical of poorly sampled cells. Because the full-depth richness of those cells is known, this measures performance under exactly the conditions that prevail in the target region, with a known answer to check against. This is the level at which C1 and C2 are judged.

### 6.6 Area of applicability

The area of applicability (Meyer and Pebesma, 2021) is computed for the fitted model and mapped across India, answering S2 and C3. Every predictive map is presented both masked and unmasked. Where a prediction is outside the area of applicability, we report it as unsupported rather than quietly including it.

### 6.7 Ecological interpretation

SHAP values and dependence curves are computed for every retained predictor. The analysis asks three things: which gradients carry the most influence; what shape each relationship takes; and whether those shapes agree with established understanding — a mid-elevation richness peak, positive responses to precipitation and to habitat heterogeneity, negative responses to intensive land use. Agreement supports the claim that the model has learned ecology. Disagreement, particularly importance attaching to effort-correlated variables, indicates it has learned where birdwatchers go.

### 6.8 Robustness and optional extension

The model is refitted with WorldClim 2.1 in place of CHELSA and at 5 km in place of 10 km, to confirm conclusions are not artefacts of predictor source or grain.

If time allows, IUCN Red List categories are joined to the species list and a second response — richness of threatened species — is modelled alongside total richness, with SHAP compared between the two. Threatened-species richness is the quantity conservation planning actually needs, and the drivers need not be the same.

## 7. Limitations

Well-sampled cells are not a random sample of India. They cluster around cities, protected areas and known birding sites, so the training set covers a biased slice of the country's environmental space. This is not a flaw to be concealed but the central difficulty the project measures: C3 and the area-of-applicability analysis exist precisely to quantify it.

Regional variation in environment–richness relationships is not examined. A single national model is fitted, and if relationships differ systematically between, say, the Himalaya and the Deccan, that heterogeneity will appear as unexplained error rather than as structure. This is the natural extension of the work and is noted as such rather than attempted here.

## 8. Deliverables

A verdict on the primary question against the three pre-registered criteria; the three-level validation table showing how accuracy declines from random to spatial to sparse-sampling evaluation; a predicted winter bird richness map for India at 10 km with an area-of-applicability mask; SHAP dependence curves with an ecological reading of each major gradient; a map and district-level list of poorly sampled areas predicted to be richer than their records suggest; a reproducible Python pipeline; and the report and presentation.

## 9. Timeline

| Week | Work |
|---|---|
| 1 | eBird request submitted; GBIF fallback tested; grid built |
| 2 | Checklist filtering; predictor aggregation from CHELSA, DEM, WorldCover |
| 3 | Rarefaction and coverage; **Gate 1** — set the coverage threshold, report the well/poorly sampled split |
| 4 | Collinearity screening; model table frozen; models fitted; random and spatial cross-validation |
| 5 | Downsampling experiment; C1 and C2 evaluated; area of applicability computed |
| 6 | SHAP analysis and ecological reading; under-surveyed area mapping |
| 7 | Robustness checks; threatened-species extension if time allows |
| 8 | Writing, figures, presentation |

## 10. Risks and fallbacks

| Risk | Response |
|---|---|
| eBird request delayed | Proceed on GBIF; state the effort-metadata limitation explicitly |
| Too few cells clear the coverage threshold | Lower the threshold and report the trade-off; the split itself becomes a reported result |
| Model fails all three criteria | This is a complete answer to the assigned question and is reported as such. Do not tune until it passes. |
| Model passes C1 and C2 but fails C3 | The more likely and more interesting outcome: good predictions over a small supported area. Report the supported fraction prominently. |
| SHAP shows effort variables dominating | A direct finding about the limits of open occurrence data; report rather than remove the variables |

## 11. What counts as success

The project succeeds if it returns a defensible verdict on the assigned question — whether environmental variables predict bird species richness in poorly sampled regions of India well enough to identify biodiversity patterns — supported by validation that does not flatter itself, an explicit statement of where the answer applies, and an ecological account of what the model learned. A clear negative verdict, properly evidenced, is a successful outcome.

## 12. References

Chao, A., Gotelli, N. J., Hsieh, T. C., Sander, E. L., Ma, K. H., Colwell, R. K., & Ellison, A. M. (2014). Rarefaction and extrapolation with Hill numbers: A framework for sampling and estimation in species diversity studies. *Ecological Monographs*, 84(1), 45–67.

Fick, S. E., & Hijmans, R. J. (2017). WorldClim 2: New 1-km spatial resolution climate surfaces for global land areas. *International Journal of Climatology*, 37(12), 4302–4315.

Jung, M. (2022). Predictability and transferability of local biodiversity environment relationships. *PeerJ*, 10, e13872.

Karger, D. N., Conrad, O., Böhner, J., Kawohl, T., Kreft, H., Soria-Auza, R. W., Zimmermann, N. E., Linder, H. P., & Kessler, M. (2017). Climatologies at high resolution for the Earth's land surface areas. *Scientific Data*, 4, 170122.

Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution*, 12(9), 1620–1633.

Milà, C., Mateu, J., Pebesma, E., & Meyer, H. (2022). Nearest neighbour distance matching leave-one-out cross-validation for map validation. *Methods in Ecology and Evolution*, 13(6), 1304–1316.

Moudrý, V., Bazzichetto, M., Remelgado, R., Devillers, R., Lenoir, J., Mateo, R. G., ... & Šímová, P. (2024). Optimising occurrence data in species distribution models: Sample size, positional uncertainty, and sampling bias matter. *Ecography*, 2024(12), e07294.

Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N., Picard, N., Rossi, V., ... & Pélissier, R. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11, 4540.

Ryo, M., Angelov, B., Mammola, S., Kass, J. M., Benito, B. M., & Hartig, F. (2021). Explainable artificial intelligence enhances the ecological interpretability of black-box species distribution models. *Ecography*, 44(2), 199–205.

Stein, A., Gerstner, K., & Kreft, H. (2014). Environmental heterogeneity as a universal driver of species richness across taxa, biomes and spatial scales. *Ecology Letters*, 17(7), 866–880.