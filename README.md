# Predicting Insurance Claims With GLMs from Scratch

A motor insurer has a year of policies and wants the expected number of claims for every driver. About 95 percent of policies make no claim, exposures run from a few days to a full year, and the rating factors are a mix of categories and skewed numbers, which is exactly the setting of the worked example in section 12.5 of Murphy's Probabilistic Machine Learning. You build a seeded portfolio that imitates the French motor dataset freMTPL2freq (area, vehicle power and age, driver age, bonus-malus, brand, fuel, density, region, exposure and Poisson claim counts with hidden heterogeneity), engineer quantile bins and one-hot features, and fit three models in plain numpy: a constant frequency, ridge regression on the claim frequency, and a Poisson GLM with an exposure offset fitted by Newton's method and IRLS from the gradient and Hessian of section 12.4. Then you find out which one is better and why the answer depends on the metric: exposure-weighted MSE and MAE barely separate the models, the Poisson deviance of eq. 12.37 does, ridge predicts negative frequencies and is miscalibrated where risk is low, and the GLM matches the observed claim total. You finish with a calibration curve, a Lorenz curve and Gini index, a check of the predicted claim-count distribution against overdispersion, and an extension beyond the book: a Gamma severity GLM with a non-canonical log link, multiplied into a pure premium and scored with the Tweedie deviance.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_policies
- [x] **2.** true_frequency
- [x] **3.** simulate_claims
- [x] **4.** train_test_split
- [x] **5.** claims_summary
- [x] **6.** quantile_bin_edges
- [x] **7.** one_hot
- [x] **8.** fit_preprocessor
- [x] **9.** transform
- [x] **10.** poisson_deviance
- [x] **11.** evaluate
- [x] **12.** fit_dummy
- [x] **13.** fit_ridge
- [x] **14.** poisson_objective
- [x] **15.** irls_step
- [x] **16.** fit_poisson_glm
- [x] **17.** relativities
- [x] **18.** compare_models
- [x] **19.** calibration_curve

---

Built on Deep-ML.
