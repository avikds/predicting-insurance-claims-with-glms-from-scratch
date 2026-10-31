"""
Predicting Insurance Claims With GLMs from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_policies
import numpy as np

AREAS = ["A", "B", "C", "D", "E", "F"]
AREA_P = [0.15, 0.11, 0.28, 0.22, 0.20, 0.04]
AREA_LOG_DENSITY = [3.6, 4.6, 5.6, 6.6, 7.6, 9.4]

POWER_P = [0.17, 0.18, 0.22, 0.21, 0.07, 0.05, 0.04, 0.03, 0.01, 0.01, 0.005, 0.005]

BRANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
BRAND_P = [0.24, 0.24, 0.08, 0.04, 0.05, 0.04, 0.026, 0.02, 0.24, 0.018, 0.006]

REGIONS = ["R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31", "R41", "R42", "R43",
           "R52", "R53", "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"]

REGION_P = [0.102, 0.004, 0.012, 0.013, 0.237, 0.016, 0.015, 0.040, 0.019, 0.003, 0.002,
            0.057, 0.062, 0.028, 0.046, 0.025, 0.007, 0.125, 0.008, 0.053, 0.117, 0.009]


def make_policies(n, seed):
    # Initialize the seeded random number generator.
    rng = np.random.default_rng(seed)

    # Normalize probability distributions.
    area_p = np.asarray(AREA_P, dtype=float)
    area_p /= area_p.sum()

    power_p = np.asarray(POWER_P, dtype=float)
    power_p /= power_p.sum()

    brand_p = np.asarray(BRAND_P, dtype=float)
    brand_p /= brand_p.sum()

    region_p = np.asarray(REGION_P, dtype=float)
    region_p /= region_p.sum()

    # 1. Generate geographical area.
    area_idx = rng.choice(len(AREAS), size=n, p=area_p)

    # 2. Generate population density based on area.
    log_density = np.asarray(AREA_LOG_DENSITY)[area_idx] + rng.normal(0.0, 0.5, size=n)
    density = np.clip(np.round(np.exp(log_density)), 1, 27000).astype(int)

    # 3. Generate driver age.
    driv_age = np.clip(np.round(rng.normal(45.0, 14.0, size=n)), 18, 99).astype(int)

    # 4. Generate malus and calculate the bonus-malus coefficient.
    malus = rng.poisson(0.3, size=n)

    bonus_malus = np.clip(
        np.round(100 * 0.95 ** (driv_age - 18) * 1.25 ** malus),
        50,
        230
    ).astype(int)

    # 5. Generate vehicle age.
    veh_age = np.minimum(
        rng.geometric(0.13, size=n) - 1,
        100
    ).astype(int)

    # 6. Generate vehicle power (4 to 15).
    veh_power = (
        4 + rng.choice(len(POWER_P), size=n, p=power_p)
    ).astype(int)

    # 7. Generate vehicle brand.
    veh_brand = np.asarray(BRANDS)[
        rng.choice(len(BRANDS), size=n, p=brand_p)
    ]

    # 8. Generate fuel type.
    veh_gas = np.asarray(["Diesel", "Regular"])[
        rng.integers(0, 2, size=n)
    ]

    # 9. Generate geographical region.
    region = np.asarray(REGIONS)[
        rng.choice(len(REGIONS), size=n, p=region_p)
    ]

    # 10. Generate policy exposure.
    full_year = rng.random(n) < 0.15

    exposure = np.where(
        full_year,
        1.0,
        np.round(rng.uniform(0.01, 1.0, size=n), 2)
    )

    # Return all policy attributes as NumPy arrays.
    return {
        "IDpol": np.arange(1, n + 1),
        "Exposure": exposure,
        "Area": np.asarray(AREAS)[area_idx],
        "VehPower": veh_power,
        "VehAge": veh_age,
        "DrivAge": driv_age,
        "BonusMalus": bonus_malus,
        "VehBrand": veh_brand,
        "VehGas": veh_gas,
        "Density": density,
        "Region": region
    }

# Step 2 - true_frequency
BRAND_EFFECT = {
    "B11": 0.10,
    "B12": -0.15,
    "B13": 0.05,
    "B14": 0.10
}

REGION_EFFECT = {
    "R11": 0.10,
    "R24": -0.05,
    "R52": -0.06,
    "R53": -0.08,
    "R82": 0.05,
    "R93": 0.12
}


def true_frequency(policies):
    # Convert integer columns to float before performing arithmetic.
    bm = np.asarray(policies["BonusMalus"], dtype=float)
    age = np.asarray(policies["DrivAge"], dtype=float)
    veh_age = np.asarray(policies["VehAge"], dtype=float)
    density = np.asarray(policies["Density"], dtype=float)
    veh_power = np.asarray(policies["VehPower"], dtype=float)

    # Extract categorical columns.
    veh_gas = np.asarray(policies["VehGas"])
    veh_brand = np.asarray(policies["VehBrand"])
    region = np.asarray(policies["Region"])

    # Retrieve brand and region effects.
    # Categories not listed in the dictionaries have an effect of zero.
    brand_effect = np.array(
        [BRAND_EFFECT.get(brand, 0.0) for brand in veh_brand],
        dtype=float
    )

    region_effect = np.array(
        [REGION_EFFECT.get(reg, 0.0) for reg in region],
        dtype=float
    )

    # Calculate the log of the hidden annual claim frequency.
    log_rate = (
        -2.6
        + 1.6 * np.log(bm / 50.0)
        + 0.9 * np.exp(-(age - 18.0) / 6.0)
        + 0.4 * np.maximum(age - 70.0, 0.0) / 20.0
        + 0.25 * (veh_age == 0).astype(float)
        - 0.03 * np.minimum(veh_age, 15.0)
        + 0.15 * (np.log(density) - 6.0)
        + 0.08 * (veh_gas == "Regular").astype(float)
        + 0.03 * (veh_power - 6.0)
        + brand_effect
        + region_effect
    )

    # Convert log frequency into annual claim frequency.
    return np.exp(log_rate)

# Step 3 - simulate_claims
def simulate_claims(policies, seed, frailty_shape=2.0):
    # Initialize the seeded random number generator.
    rng = np.random.default_rng(seed)

    # Get the number of policies.
    n = len(policies["Exposure"])

    # Generate Gamma frailty values with mean 1
    # and variance 1 / frailty_shape.
    u = rng.gamma(
        frailty_shape,
        1.0 / frailty_shape,
        size=n
    )

    # Calculate the expected claim count for each policy.
    claim_rate = (
        policies["Exposure"]
        * true_frequency(policies)
        * u
    )

    # Generate Poisson claim counts and return integer values.
    return rng.poisson(claim_rate).astype(int)


def make_portfolio(n, seed):
    # Generate the motor insurance policies.
    policies = make_policies(n, seed)

    # Generate observed claim counts using a separate seed.
    policies["ClaimNb"] = simulate_claims(policies, seed + 1)

    return policies

# Step 4 - train_test_split
def train_test_split(data, test_frac=0.25, seed=0):
    # Get the total number of policy records.
    n = len(next(iter(data.values())))

    # Generate a reproducible random permutation of row indices.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    # Determine the number of test records.
    n_test = int(round(test_frac * n))

    # Select test indices from the first n_test entries.
    test_idx = np.sort(perm[:n_test])

    # Select training indices from the remaining entries.
    train_idx = np.sort(perm[n_test:])

    # Apply the same indices to every column in the dataset.
    train = {
        key: values[train_idx]
        for key, values in data.items()
    }

    test = {
        key: values[test_idx]
        for key, values in data.items()
    }

    # Return the training and testing dictionaries.
    return train, test

# Step 5 - claims_summary
def claims_summary(claim_nb, exposure):
    # Convert inputs to NumPy arrays with appropriate data types.
    claim_nb = np.asarray(claim_nb, dtype=int)
    exposure = np.asarray(exposure, dtype=float)

    # Calculate the total number of policies.
    n_policies = len(claim_nb)

    # Calculate the fraction of policies with zero claims.
    frac_zero = (
        round(float(np.mean(claim_nb == 0)), 4)
        if n_policies > 0 else 0.0
    )

    # Calculate portfolio claim frequency:
    # Total number of claims divided by total exposure.
    total_exposure = float(np.sum(exposure))
    frequency = (
        round(float(np.sum(claim_nb)) / total_exposure, 4)
        if total_exposure > 0 else 0.0
    )

    # Calculate the mean policy exposure.
    mean_exposure = (
        round(float(np.mean(exposure)), 4)
        if n_policies > 0 else 0.0
    )

    # Calculate the claim-count histogram.
    # Each element represents the number of policies
    # with 0, 1, 2, ... claims, respectively.
    histogram = np.bincount(claim_nb).tolist()

    # Return the portfolio summary.
    return {
        "n_policies": int(n_policies),
        "frac_zero": frac_zero,
        "frequency": frequency,
        "mean_exposure": mean_exposure,
        "histogram": histogram
    }

# Step 6 - quantile_bin_edges
def quantile_bin_edges(x, n_bins):
    # Convert the input into a NumPy floating-point array.
    x = np.asarray(x, dtype=float)

    # Calculate the interior quantile probabilities.
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]

    # Compute the quantile edges.
    edges = np.quantile(x, quantiles).astype(float)

    # Remove duplicate edges and sort them.
    edges = np.unique(edges)

    # Remove edges equal to or below the minimum value.
    edges = edges[edges > np.min(x)]

    return edges


def bin_codes(x, edges):
    # Assign each value to its corresponding quantile bin.
    # side="right" ensures that a value equal to an edge
    # belongs to the bin immediately to its right.
    return np.searchsorted(edges, x, side="right")

# Step 7 - one_hot
def one_hot(values, levels):
    # Convert inputs into NumPy arrays.
    values = np.asarray(values)
    levels = np.asarray(levels)

    # Compare every value against every possible level.
    # The resulting Boolean matrix has shape (n, K).
    encoded = values[:, None] == levels[None, :]

    # Convert Boolean values into floating-point 1.0 and 0.0.
    return encoded.astype(float)

# Step 8 - fit_preprocessor
CATEGORICAL = ["VehBrand", "VehPower", "VehGas", "Region", "Area"]
BINNED = ["VehAge", "DrivAge"]


def fit_preprocessor(data, n_bins=10):
    # Learn sorted unique categorical levels from the training data.
    levels = {
        col: np.unique(data[col]).tolist()
        for col in CATEGORICAL
    }

    # Calculate quantile bin edges for continuous age features.
    edges = {
        col: quantile_bin_edges(data[col], n_bins)
        for col in BINNED
    }

    # Convert numerical features to floating-point arrays.
    bonus_malus = np.asarray(data["BonusMalus"], dtype=float)
    density = np.asarray(data["Density"], dtype=float)

    # Apply the natural logarithm to population density.
    log_density = np.log(density)

    # Calculate means and population standard deviations (ddof=0).
    scale = {
        "BonusMalus": (
            float(np.mean(bonus_malus)),
            float(np.std(bonus_malus, ddof=0))
        ),
        "LogDensity": (
            float(np.mean(log_density)),
            float(np.std(log_density, ddof=0))
        )
    }

    # Return all learned preprocessing parameters.
    return {
        "levels": levels,
        "edges": edges,
        "scale": scale
    }

# Step 9 - transform
def transform(data, prep):
    # Store the individual feature blocks.
    blocks = []

    # 1. One-hot encode categorical features.
    # Use the category levels learned from the training dataset.
    for col in CATEGORICAL:
        encoded = one_hot(
            data[col],
            prep["levels"][col]
        )
        blocks.append(encoded)

    # 2. Quantile-bin and one-hot encode age features.
    for col in BINNED:
        edges = prep["edges"][col]

        # Assign each observation to its quantile bin.
        codes = bin_codes(data[col], edges)

        # Create a fixed list of bin levels.
        levels = list(range(len(edges) + 1))

        # One-hot encode the bin assignments.
        encoded = one_hot(codes, levels)
        blocks.append(encoded)

    # 3. Standardize BonusMalus using training statistics.
    bm_mean, bm_std = prep["scale"]["BonusMalus"]

    bonus_malus = np.asarray(
        data["BonusMalus"],
        dtype=float
    )

    standardized_bm = (
        (bonus_malus - bm_mean) / bm_std
    ).reshape(-1, 1)

    blocks.append(standardized_bm)

    # 4. Log-transform and standardize Density.
    density = np.asarray(
        data["Density"],
        dtype=float
    )

    log_density = np.log(density)

    ld_mean, ld_std = prep["scale"]["LogDensity"]

    standardized_density = (
        (log_density - ld_mean) / ld_std
    ).reshape(-1, 1)

    blocks.append(standardized_density)

    # Concatenate all feature blocks column-wise.
    # Return a floating-point design matrix without an intercept.
    return np.concatenate(blocks, axis=1).astype(float)

# Step 10 - poisson_deviance
def poisson_deviance(y, mu, sample_weight=None):
    # Convert inputs to floating-point NumPy arrays.
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # Support compatible broadcastable input shapes.
    y, mu = np.broadcast_arrays(y, mu)

    # Predicted Poisson means must be strictly positive.
    if np.any(mu <= 0):
        raise ValueError("Predicted means must be strictly positive.")

    # Observed claim values must be non-negative.
    if np.any(y < 0):
        raise ValueError("Observed values must be non-negative.")

    # Calculate y * log(y / mu).
    # By convention, this term equals zero when y == 0.
    log_term = np.zeros_like(y, dtype=float)

    positive = y > 0

    log_term[positive] = (
        y[positive] * np.log(y[positive] / mu[positive])
    )

    # Calculate the unit Poisson deviance.
    unit_deviance = 2.0 * (
        log_term - y + mu
    )

    # Use equal weights when sample_weight is None.
    if sample_weight is None:
        weights = np.ones_like(unit_deviance, dtype=float)
    else:
        weights = np.broadcast_to(
            np.asarray(sample_weight, dtype=float),
            unit_deviance.shape
        )

    # Validate the sample weights.
    if np.any(weights < 0):
        raise ValueError("Sample weights must be non-negative.")

    weight_sum = np.sum(weights)

    if weight_sum <= 0:
        raise ValueError("The sum of sample weights must be positive.")

    # Return the weighted mean Poisson deviance as a Python float.
    return float(
        np.sum(weights * unit_deviance) / weight_sum
    )

# Step 11 - evaluate
def evaluate(freq, pred, exposure):
    # Convert inputs into floating-point NumPy arrays.
    freq = np.asarray(freq, dtype=float)
    pred = np.asarray(pred, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    # Calculate the total exposure.
    total_exposure = np.sum(exposure)

    # Calculate the differences between observed and predicted frequencies.
    errors = freq - pred

    # 1. Exposure-weighted mean squared error.
    mse = float(
        np.sum(exposure * errors**2) / total_exposure
    )

    # 2. Exposure-weighted mean absolute error.
    mae = float(
        np.sum(exposure * np.abs(errors)) / total_exposure
    )

    # 3. Calculate Poisson deviance only for positive predictions.
    # Non-positive predicted frequencies are excluded because
    # the Poisson deviance requires strictly positive predictions.
    valid = pred > 0

    if np.any(valid):
        deviance = poisson_deviance(
            freq[valid],
            pred[valid],
            sample_weight=exposure[valid]
        )
    else:
        # Deviance is undefined when there are no valid predictions.
        deviance = float("nan")

    # 4. Count predictions that are zero or negative.
    n_nonpositive = int(np.sum(pred <= 0))

    # 5. Calculate the total predicted number of claims.
    pred_claims = float(np.sum(pred * exposure))

    # 6. Calculate the total observed number of claims.
    obs_claims = float(np.sum(freq * exposure))

    # Return all evaluation metrics.
    return {
        "mse": mse,
        "mae": mae,
        "deviance": float(deviance),
        "n_nonpositive": n_nonpositive,
        "pred_claims": pred_claims,
        "obs_claims": obs_claims
    }

# Step 12 - fit_dummy
def fit_dummy(freq, exposure):
    # Convert inputs into floating-point NumPy arrays.
    freq = np.asarray(freq, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    # Calculate the exposure-weighted portfolio claim frequency.
    intercept = float(
        np.sum(freq * exposure) / np.sum(exposure)
    )

    # Return the constant-frequency model.
    return {
        "intercept": intercept,
        "coef": None,
        "link": "identity"
    }


def predict(model, X):
    # Convert the design matrix into a NumPy array.
    X = np.asarray(X, dtype=float)

    # Calculate the linear predictor.
    if model["coef"] is None:
        # Constant model: use the intercept for every observation.
        eta = np.full(
            X.shape[0],
            model["intercept"],
            dtype=float
        )
    else:
        # Regression model: intercept plus matrix multiplication.
        eta = model["intercept"] + X @ model["coef"]

    # Apply the specified link function.
    if model["link"] == "identity":
        return eta

    elif model["link"] == "log":
        return np.exp(eta)

    else:
        raise ValueError("Unsupported link function.")

# Step 13 - fit_ridge
def fit_ridge(X, y, sample_weight, alpha):
    # Convert inputs into floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(sample_weight, dtype=float).reshape(-1)

    # Normalize sample weights so that they sum to one.
    w = w / np.sum(w)

    # Calculate weighted means of features and target.
    x_bar = np.sum(w[:, None] * X, axis=0)
    y_bar = np.sum(w * y)

    # Center the feature matrix and target.
    Xc = X - x_bar
    yc = y - y_bar

    # Number of features.
    n_features = X.shape[1]

    # Construct the weighted Ridge normal equations:
    # (Xc.T @ diag(w) @ Xc + alpha * I) @ coef
    #     = Xc.T @ (w * yc)
    lhs = (
        Xc.T @ (w[:, None] * Xc)
        + alpha * np.eye(n_features)
    )

    rhs = Xc.T @ (w * yc)

    # Solve the regularized normal equations.
    coef = np.linalg.solve(lhs, rhs)

    # Recover the unpenalized intercept.
    intercept = y_bar - x_bar @ coef

    # Return the fitted Ridge model.
    return {
        "intercept": float(intercept),
        "coef": coef,
        "link": "identity"
    }

# Step 14 - poisson_objective
def poisson_objective(params, X, counts, exposure, alpha):
    # Convert inputs into floating-point NumPy arrays.
    params = np.asarray(params, dtype=float)
    X = np.asarray(X, dtype=float)
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    # Total exposure across all policies.
    E = np.sum(exposure)

    # Construct the augmented design matrix with an intercept.
    # The first column consists entirely of ones.
    X_tilde = np.column_stack([
        np.ones(X.shape[0]),
        X
    ])

    # Calculate the linear predictor with the exposure offset.
    eta = X_tilde @ params + np.log(exposure)

    # Apply the exponential link function.
    mu = np.exp(eta)

    # Create the penalty matrix.
    # The intercept is not penalized.
    p = X.shape[1]
    P = alpha * np.diag(np.r_[0.0, np.ones(p)])

    # 1. Calculate the penalized Poisson negative log-likelihood.
    objective = (
        np.sum(mu - counts * eta) / E
        + (alpha / 2.0) * np.sum(params[1:] ** 2)
    )

    # 2. Calculate the gradient.
    gradient = (
        X_tilde.T @ (mu - counts) / E
        + P @ params
    )

    # 3. Calculate the Hessian.
    hessian = (
        X_tilde.T @ (mu[:, None] * X_tilde) / E
        + P
    )

    # Return the objective, gradient and Hessian.
    return (
        float(objective),
        gradient,
        hessian
    )

# Step 15 - irls_step
def irls_step(params, X, counts, exposure, alpha):
    # Convert inputs into floating-point NumPy arrays.
    params = np.asarray(params, dtype=float)
    X = np.asarray(X, dtype=float)
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    # Construct the augmented design matrix with an intercept.
    X_tilde = np.column_stack([
        np.ones(X.shape[0]),
        X
    ])

    # Calculate the linear predictor without the exposure offset.
    s = X_tilde @ params

    # Calculate the expected claim counts using the exposure offset.
    mu = exposure * np.exp(s)

    # Calculate the IRLS working response.
    z = s + (counts - mu) / mu

    # Calculate total exposure.
    E = np.sum(exposure)

    # Construct the penalty matrix.
    # The intercept is not penalized.
    p = X.shape[1]
    P = alpha * np.diag(np.r_[0.0, np.ones(p)])

    # Construct the weighted least-squares left-hand side.
    lhs = (
        X_tilde.T @ (mu[:, None] * X_tilde) / E
        + P
    )

    # Construct the weighted least-squares right-hand side.
    rhs = X_tilde.T @ (mu * z) / E

    # Solve the weighted least-squares system for new parameters.
    new_params = np.linalg.solve(lhs, rhs)

    # Return the updated parameter vector.
    return new_params

# Step 16 - fit_poisson_glm
def fit_poisson_glm(X, counts, exposure, alpha, max_iter=25, tol=1e-10):
    # Convert inputs into floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    # Number of features.
    p = X.shape[1]

    # Initialize all coefficients to zero.
    params = np.zeros(p + 1, dtype=float)

    # Initialize the intercept using the logarithm of
    # the portfolio claim frequency.
    params[0] = np.log(
        np.sum(counts) / np.sum(exposure)
    )

    # Evaluate the initial objective, gradient and Hessian.
    obj, grad, hess = poisson_objective(
        params, X, counts, exposure, alpha
    )

    # Store the objective value at every accepted iteration.
    history = [obj]

    # Perform damped Newton optimization.
    for _ in range(max_iter):

        # Calculate the Newton step.
        step = np.linalg.solve(hess, grad)

        # Calculate the Newton decrement.
        decrement = 0.5 * (grad @ step)

        # Stop when convergence is reached.
        if decrement < tol:
            break

        # Begin with a full Newton step.
        t = 1.0

        # Calculate the candidate parameter vector.
        candidate_params = params - t * step

        # Evaluate the candidate objective, gradient and Hessian.
        candidate_obj, candidate_grad, candidate_hess = (
            poisson_objective(
                candidate_params, X, counts, exposure, alpha
            )
        )

        # Apply step halving while the objective increases.
        while candidate_obj > obj and t > 1e-4:

            # Reduce the step size by half.
            t *= 0.5

            # Recalculate the candidate parameters.
            candidate_params = params - t * step

            # Reevaluate the candidate solution.
            candidate_obj, candidate_grad, candidate_hess = (
                poisson_objective(
                    candidate_params, X, counts, exposure, alpha
                )
            )

        # Accept the last candidate solution.
        params = candidate_params
        obj = candidate_obj
        grad = candidate_grad
        hess = candidate_hess

        # Record the accepted objective value.
        history.append(obj)

    # Return the fitted Poisson GLM.
    return {
        "intercept": float(params[0]),
        "coef": params[1:],
        "link": "log",
        "history": history
    }

# Step 17 - relativities
def relativities(model, prep, column):
    # Check whether the requested column is categorical or binned.
    if column not in CATEGORICAL and column not in BINNED:
        raise KeyError(column)

    # Initialize the starting coefficient position.
    start = 0

    # Reconstruct the exact feature-column layout used by transform().
    all_columns = CATEGORICAL + BINNED

    for col in all_columns:
        if col in CATEGORICAL:
            n_levels = len(prep["levels"][col])
        else:
            n_levels = len(prep["edges"][col]) + 1

        if col == column:
            break

        start += n_levels

    # Get the levels belonging to the requested feature.
    if column in CATEGORICAL:
        levels = prep["levels"][column]
    else:
        levels = list(range(len(prep["edges"][column]) + 1))

    # Extract the corresponding coefficients.
    coef = np.asarray(model["coef"], dtype=float)
    column_coef = coef[start:start + len(levels)]

    # Calculate relativities relative to the first level.
    reference_coef = column_coef[0]

    # Return a dictionary with relativities rounded to four decimals.
    return {
        level: float(np.round(np.exp(value - reference_coef), 4))
        for level, value in zip(levels, column_coef)
    }

# Step 18 - compare_models
def compare_models(train, test, alpha_ridge=1e-6, alpha_glm=1e-6):
    # 1. Fit the preprocessor using training data only.
    prep = fit_preprocessor(train)

    # Transform training and test datasets using the same preprocessor.
    X_tr = transform(train, prep)
    X_te = transform(test, prep)

    # Extract training and test exposures.
    exposure_tr = np.asarray(train["Exposure"], dtype=float)
    exposure_te = np.asarray(test["Exposure"], dtype=float)

    # 2. Calculate observed claim frequencies.
    f_tr = (
        np.asarray(train["ClaimNb"], dtype=float) / exposure_tr
    )

    f_te = (
        np.asarray(test["ClaimNb"], dtype=float) / exposure_te
    )

    # Extract training claim counts.
    counts_tr = np.asarray(train["ClaimNb"], dtype=float)

    # 3. Fit the constant-frequency baseline.
    dummy_model = fit_dummy(f_tr, exposure_tr)

    # Fit weighted Ridge regression.
    ridge_model = fit_ridge(
        X_tr,
        f_tr,
        exposure_tr,
        alpha_ridge
    )

    # Fit the Poisson GLM using claim counts and exposure.
    poisson_model = fit_poisson_glm(
        X_tr,
        counts_tr,
        exposure_tr,
        alpha_glm
    )

    # Store all fitted models in the required order.
    models = {
        "dummy": dummy_model,
        "ridge": ridge_model,
        "poisson": poisson_model
    }

    # 4. Generate predicted annual claim frequencies.
    predictions = {
        name: predict(model, X_te)
        for name, model in models.items()
    }

    # Evaluate each model on the held-out test portfolio.
    table = {
        name: evaluate(f_te, pred, exposure_te)
        for name, pred in predictions.items()
    }

    # Return evaluation results, fitted models and preprocessor.
    return table, models, prep

# Step 19 - calibration_curve
def calibration_curve(pred, freq, exposure, n_bins=10):
    # Convert inputs into floating-point NumPy arrays.
    pred = np.asarray(pred, dtype=float)
    freq = np.asarray(freq, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    # Calculate total exposure.
    total_exposure = np.sum(exposure)

    # 1. Sort policies by predicted frequency.
    # Stable sorting preserves the original order of tied predictions.
    order = np.argsort(pred, kind="stable")

    sorted_pred = pred[order]
    sorted_freq = freq[order]
    sorted_exposure = exposure[order]

    # 2. Calculate cumulative exposure shares.
    cum = np.cumsum(sorted_exposure) / total_exposure

    # 3. Assign policies to exposure-equal bins.
    # Using ceil(cum * n_bins) - 1 implements the intervals
    # (b / n_bins, (b + 1) / n_bins] correctly.
    bin_idx = np.ceil(cum * n_bins).astype(int) - 1

    # Keep bin indices within the valid range.
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Initialize output arrays.
    mean_pred = np.empty(n_bins, dtype=float)
    mean_obs = np.empty(n_bins, dtype=float)
    weight = np.empty(n_bins, dtype=float)

    # 4. Calculate exposure-weighted statistics for each bin.
    for b in range(n_bins):
        mask = bin_idx == b

        # Total exposure within the current bin.
        bin_exposure = np.sum(sorted_exposure[mask])

        # Exposure-weighted mean predicted frequency.
        mean_pred[b] = (
            np.sum(sorted_pred[mask] * sorted_exposure[mask])
            / bin_exposure
        )

        # Exposure-weighted mean observed frequency.
        mean_obs[b] = (
            np.sum(sorted_freq[mask] * sorted_exposure[mask])
            / bin_exposure
        )

        # Fraction of total portfolio exposure in this bin.
        weight[b] = bin_exposure / total_exposure

    # Return the three arrays.
    return mean_pred, mean_obs, weight

