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
- [x] **20.** lorenz_gini
- [x] **21.** count_distribution
- [x] **22.** simulate_claim_amounts
- [x] **23.** fit_gamma_glm
- [x] **24.** tweedie_deviance
- [x] **25.** pure_premium_report
- [x] **26.** insurance_report

## Results

```
1. The portfolio (Figure 12.1a)
   100000 policies, mean exposure 0.5796 years
   94.7% of policies make no claim; claim frequency 0.1016 per year
   policies by number of claims: [94698, 4829, 390, 64, 13, 2, 2, 1, 0, 1]
   Book (real freMTPL2 test set): 94% zeros, frequency about 10%.

2. Table 12.1 on the held-out 25% (exposure-weighted metrics)
   model      MSE     MAE   deviance  non-positive  predicted claims  observed
   dummy    0.2024  0.1910  0.5771             0            1460.0      1518
   ridge    0.1920  0.1802  0.5239           498            1482.9      1518
   poisson  0.1923  0.1794  0.5153             0            1487.8      1518
   Book: dummy 0.564 / 0.189 / 0.625, ridge 0.560 / 0.177 / 0.601, Poisson 0.560 / 0.186 / 0.594.
   MSE and MAE barely separate ridge from the GLM; the Poisson deviance (eq. 12.37) does,
   even though ridge's deviance skips the policies where it predicts a non-positive rate.

3. Calibration (Figure 12.2): mean predicted vs observed frequency, 10 exposure-equal bins
   dummy    predicted [0.101, 0.101, 0.101, 0.101, 0.101, 0.101, 0.101, 0.101, 0.101, 0.101]
            observed  [0.105, 0.102, 0.104, 0.106, 0.093, 0.115, 0.113, 0.101, 0.104, 0.103]
   ridge    predicted [0.01, 0.034, 0.047, 0.059, 0.07, 0.081, 0.095, 0.113, 0.15, 0.363]
            observed  [0.037, 0.057, 0.054, 0.065, 0.068, 0.07, 0.097, 0.109, 0.133, 0.356]
   poisson  predicted [0.037, 0.048, 0.055, 0.062, 0.069, 0.077, 0.087, 0.101, 0.131, 0.358]
            observed  [0.038, 0.049, 0.054, 0.075, 0.069, 0.072, 0.094, 0.097, 0.147, 0.352]
   Ridge under-predicts the safest decile; the GLM stays on the diagonal at the low end.

4. Ranking power (Gini of the Lorenz curve, higher ranks risks better)
   {'dummy': 0.0043, 'ridge': 0.3727, 'poisson': 0.3741}
   Ridge ranks about as well as the GLM: its problem is the level of its predictions, not their order.

5. The fitted GLM
   Newton iterations: 5
   relativities DrivAge: {0: 1.0, 1: 0.6265, 2: 0.5714, 3: 0.5994, 4: 0.5371, 5: 0.5332, 6: 0.5714, 7: 0.5788, 8: 0.5527, 9: 0.5864}
   relativities VehAge: {0: 1.0, 1: 0.7918, 2: 0.817, 3: 0.8255, 4: 0.697, 5: 0.6409, 6: 0.5879, 7: 0.565, 8: 0.5202}
   relativities Area: {'A': 1.0, 'B': 1.3132, 'C': 1.2026, 'D': 1.4181, 'E': 1.6085, 'F': 2.044}
   expected policies with 0..4+ claims: [23624.2, 1278.4, 85.4, 9.8, 2.1]
   observed policies with 0..4 claims:  [23612, 1274, 100, 12, 2]
   Pearson dispersion: 1.124 (1 for a Poisson; above 1 means overdispersion)

6. Beyond the book: frequency x severity pure premium
   predicted losses 3,144,197 vs observed 3,248,144
   Tweedie deviance (p = 1.9): model 33.7582 vs flat premium 34.7628
   Gini on claim costs: 0.4150
```
