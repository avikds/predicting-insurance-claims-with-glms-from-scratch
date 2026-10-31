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

