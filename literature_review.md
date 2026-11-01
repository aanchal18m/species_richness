Team Name: Groot
Team Members: Mahek Desai (2024101096), Laveena Jain (2024111022), Aanchal Mundhada
(2023112016)
1. Introduction
The number of species present within a given area is known as species richness. However, the
observation of these species has an extremely skewed geographic distribution across the globe.
There are many publicly available biological occurrence databases such as the Global
Biodiversity Information Facility (GBIF) and the IUCN Red List that have high volume data, but
this data is aggregated across highly studied regions. As a result, regions with high mapped
species richness may have high richness due to high sampling rates, not true ecological presence.
On the other hand, it is known that climate, topography, and land use factors drive species
distributions, and with high resolution global data available, such as WorldClim and CHELSA
for climate, the Copernicus DEM for topography, and ESA WorldCover for land cover,
predictions on the biodiversity of unexplored regions are possible. While machine learning
techniques are well suited for detecting complex nonlinearity between the environment and
species richness, these methods' ability to generalize to completely novel regions is unknown.
This literature review explores environmental predictors for species richness, machine learning
prediction models, database biases, and the importance of biological interpretability of predictive
models. In particular, emphasis is given on the ability for machine learning models to extrapolate
learned relationships in regions of low sampling effort.
2. Review of existing studies
Environmental determinants of species richness
Environmental heterogeneity has been found to be a strong determinant of species richness but
displays varying relationships at different spatial extents. Across nearly two hundred studies, a
positive correlation was found between biodiversity and the diversity of vegetation, climate, and
topography (Stein et al., 2014). As such, these factors can be used as predictors of species
richness, but caution must be applied when extrapolating these relationships beyond a particular
spatial extent. Additionally, each aspect of the environment encodes unique ecological
information: topographic heterogeneity drives biodiversity through increased habitat diversity
(Rahbek et al., 2019), while land use can be used as a proxy for human impact on the landscape,
with negative impacts on local biodiversity (Newbold et al., 2015). In essence, climate and
topography act as broad regulators of natural biodiversity, while land cover acts as a proxy for
structural modification and human influence.
From statistical models to machine learning
Due to the complex, nonlinear relationships between species distribution and the environment,
proper selection and validation of statistical models is critical. Earlier research has seen the rise
of boosted regression trees and similar machine learning algorithms that allow better modeling of
these distribution-climate relationships (Elith et al., 2006). More recent studies have utilized
machine learning methods in modeling species richness directly. In particular, researchers have
found that machine learning algorithms, such as random forests and extreme gradient boosting,
have proven superior in describing the distribution of global plant diversity compared to
statistical models (Cai et al., 2023). Some studies seek to combine machine learning with
interpretability, finding that such methods allow for better determination of variables affecting
tree species richness (Brugere et al., 2023). These studies demonstrate a shift from producing
general-purpose biodiversity maps to seeking biological insights into how the environment
influences species richness. After all, a complex predictive model is only useful if its
relationships are valid under a given ecological context.
Biodiversity data and sampling bias
Biases in occurrence data are a severe limitation when attempting to build predictive models.
Due to the highly aggregated nature of GBIF and similar databases, predictive models are
strongly biased towards regions of high sampling effort (Beck et al., 2014). Naturally, a region
that has been intensively surveyed will display higher species richness than one that has not,
regardless of the biological realities of either region. Ecologists have developed rarefaction and
extrapolation techniques to account for differences in sampling effort between regions (Chao et
al., 2014). However, it is now well established that sampling bias is an intrinsic property of
biodiversity models. Researchers find that both the overall observation rate in a region and the
positional accuracy of observations encode significant bias in predictive models (Moudrý et al.,
2024). Correcting for spatial bias becomes nontrivial when the assumptions of a model cannot be
validated on an independent test set (Baker et al., 2024). This becomes a critical issue for this
project, as making predictions about unsampled regions can be highly unreliable.
Validation, transferability and explainability
The literature reflects an increased interest in validation and transferability of predictive models.
Traditional practices of random subsampling can severely overestimate the accuracy of a model
on independent data. In particular, spatial cross validation performs significantly worse than
traditional validation practices due to the clustered nature of spatial data (Ploth et al., 2020).
When attempting to make predictions in a novel region, spatial cross-validation is necessary for
ensuring that a model's predictions are not inflated due to random chance. Predictive models
must be validated on independent test sets that follow the same spatial structure as the data a
model will see in the wild. Transferability to new regions is highly nontrivial, as the ecological
relationships a model learns may not be applicable beyond the region of model training (Jung,
2022). The area of applicability framework offers a solution, explicitly defining the spatial extent
within which a model's predictions are reliable without requiring additional validation (Meyer &
Pebesma, 2021). Finally, explainable machine learning is crucial for ensuring that a model's
internal relationships are valid in an ecological context. Particular methods such as SHAP allow
investigation of a model's inner workings, detecting if the model relies on spurious relationships
between the environment and species richness (Ryo et al., 2021).
3. Critical discussion and research gap
It is generally agreed upon that climate, topography, and land cover are critical factors in
determining species richness. Modern machine learning algorithms allow for excellent
approximation of the complex, nonlinear relationships between species and the environment.
However, several critical challenges prevent these methods from being readily applied to the task
of estimating species richness:
First, the richness values stored in the GBIF database reflect observed values, not true richness
values, and the latter may be significantly different from the former due to sampling bias. In
other words, a region with few observations will be predicted to have low species richness, even
if it truly has high species richness.
Second, standard validation practices greatly overestimate the accuracy of a predictive model by
using randomly selected test sets. Spatial correlation between adjacent regions introduces strong
positive autocorrelation between test and training data, inflating a model's accuracy by chance
alone. There exist methods, such as spatial validation and area of applicability, that explicitly
account for this issue, but they are not yet standard practice.
Finally, while many papers apply explainable AI to determine important drivers of species
richness, researchers rarely test if these relationships hold for novel environments.
This review highlights the need for an overarching framework that relates biological data from
the GBIF and IUCN Red List databases to environmental predictors using a combination of
explainable machine learning algorithms, spatial validation practices, and area of applicability
constraints.
4. Conclusion
In summary, the physical environment contains a wealth of information that can be utilized to
map large-scale biodiversity patterns. In particular, it has been found that climate, topography,
and land cover are positively correlated with species richness. Moreover, machine learning
algorithms are particularly well suited for modeling the complex, nonlinear relationships
between the environment and species distributions. However, species distributions are
challenging to predict in regions of low observation due to sampling bias and issues with model
validation. In particular, raw richness values from the GBIF database tend to reflect human
biases rather than true species richness values. Finally, standard model validation techniques
greatly overestimate the accuracy of species richness models. By adopting spatial model
validation, area of applicability constraints, and explainable AI, these pitfalls can be avoided. In
the future, the field of spatial ecology should strive not to simply improve model accuracy, but to
ensure that a model's assumptions hold true for the region of interest.
5. References
Baker, D. J., Maclean, I. M. D., & Gaston, K. J. (2024). Effective strategies for correcting spatial
sampling bias in species distribution models without independent test data. Diversity and
Distributions, 30(3), e13802.
Beck, J., Böller, M., Erhardt, A., & Schwanghart, W. (2014). Spatial bias in the GBIF database
and its effect on modeling species' geographic distributions. Ecological Informatics, 19, 10–15.
Brugere, L., Kwon, Y., Frazier, A. E., & Kedron, P. (2023). Improved prediction of tree species
richness and interpretability of environmental drivers using a machine learning approach. Forest
Ecology and Management, 539, 120972.
Cai, L., Kreft, H., Taylor, A., Denelle, P., Schrader, J., Essl, F., van Kleunen, M., Pergl, J., Pyšek,
P., Stein, A., Winter, M., Barcelona, J. F., Fuentes, N., Inderjit, Karger, D. N., Kartesz, J.,
Kuprijanov, A., Nishino, M., Nickrent, D., Nowak, A., Patzelt, A., Pelser, P. B., Singh, P.,
Wieringa, J. J., & Weigelt, P. (2023). Global models and predictions of plant diversity based on
advanced machine learning techniques. New Phytologist, 237(4), 1432–1445.
Chao, A., Gotelli, N. J., Hsieh, T. C., Sander, E. L., Ma, K. H., Colwell, R. K., & Ellison, A. M.
(2014). Rarefaction and extrapolation with Hill numbers: A framework for sampling and
estimation in species diversity studies. Ecological Monographs, 84(1), 45–67.
Elith, J., Graham, C. H., Anderson, R. P., Dudík, M., Ferrier, S., Guisan, A., Hijmans, R. J.,
Huettmann, F., Leathwick, J. R., Lehmann, A., Li, J., Lohmann, L. G., Loiselle, B. A., Manion,
G., Moritz, C., Nakamura, M., Nakazawa, Y., Overton, J. M., Peterson, A. T., Phillips, S. J.,
Richardson, K., Scachetti-Pereira, R., Schapire, R. E., Soberón, J., Williams, S., Wisz, M., &
Zimmermann, N. E. (2006). Novel methods improve prediction of species' distributions from
occurrence data. Ecography, 29(2), 129–151.
Jung, M. (2022). Predictability and transferability of local biodiversity–environment
relationships. PeerJ, 10, e13872.
Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of
applicability of spatial prediction models. Methods in Ecology and Evolution, 12(9), 1620–1633.
Moudrý, V., Bazzichetto, M., Remelgado, R., Devillers, R., Lenoir, J., Mateo, R. G., Lembrechts,
J. J., Sillero, N., Lecours, V., Cord, A. F., Barták, V., Balej, P., Rocchini, D., Torresani, M.,
Arenas-Castro, S., Man, M., Prajzlerová, D., Gdulová, K., Prošek, J., Marchetto, E., Zarzo-Arias,
A., Gábor, L., Leroy, F., Martini, M., Malavasi, M., Cazzolla Gatti, R., Wild, J., & Šímová, P.
(2024). Optimising occurrence data in species distribution models: Sample size, positional
uncertainty, and sampling bias matter. Ecography, 2024(12), e07294.
Newbold, T., Hudson, L. N., Hill, S. L. L., Contu, S., Lysenko, I., Senior, R. A., Börger, L.,
Bennett, D. J., Choimes, A., Collen, B., Day, J., De Palma, A., Díaz, S., Echeverria-Londoño, S.,
Edgar, M. J., Feldman, A., Garon, M., Harrison, M. L. K., Alhusseini, T., Ingram, D. J., Itescu,
Y., Kattge, J., Kemp, V., Kirkpatrick, L., Kleyer, M., Correia, D. L. P., Martin, C. D., Meiri, S.,
Novosolov, M., Pan, Y., Phillips, H. R. P., Purves, D. W., Robinson, A., Simpson, J., Tuck, S. L.,
Weiher, E., White, H. J., Ewers, R. M., Mace, G. M., Scharlemann, J. P. W., & Purvis, A. (2015).
Global effects of land use on local terrestrial biodiversity. Nature, 520(7545), 45–50.
Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N., Picard, N., Rossi, V., Dormann, C.,
Cornu, G., Viennois, G., Bayol, N., Lyapustin, A., Gourlet-Fleury, S., & Pélissier, R. (2020).
Spatial validation reveals poor predictive performance of large-scale ecological mapping models.
Nature Communications, 11, 4540.
Rahbek, C., Borregaard, M. K., Antonelli, A., Colwell, R. K., Holt, B. G., Nogués-Bravo, D.,
Rasmussen, C. M. Ø., Richardson, K., Rosing, M. T., Whittaker, R. J., & Fjeldså, J. (2019).
Building mountain biodiversity: Geological and evolutionary processes. Science, 365(6458),
1114–1119.
Ryo, M., Angelov, B., Mammola, S., Kass, J. M., Benito, B. M., & Hartig, F. (2021). Explainable
artificial intelligence enhances the ecological interpretability of black-box species distribution
models. Ecography, 44(2), 199–205.
Stein, A., Gerstner, K., & Kreft, H. (2014). Environmental heterogeneity as a universal driver of
species richness across taxa, biomes and spatial scales. Ecology Letters, 17(7), 866–880.