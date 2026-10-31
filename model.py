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

