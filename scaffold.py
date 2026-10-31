"""
Predicting Insurance Claims With GLMs from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Predicting insurance claims with GLMs from scratch.

Story (section 12.5 of Murphy, Probabilistic Machine Learning): a motor insurer wants the
expected number of claims per policy-year. Most policies never claim, so the target is a pile
of zeros with a thin tail. A constant frequency is a strong baseline, ridge regression on the
frequency looks fine by MSE and MAE but predicts negative claim rates and misprices the safest
drivers, and a Poisson GLM with an exposure offset, fitted by Newton's method, wins on the
deviance and reproduces the observed number of claims. The portfolio is a seeded imitation of
freMTPL2freq, so the true claim rate is known and every model can be held against it.
"""
import numpy as np


def main() -> None:
    rep = insurance_report(n=100000, seed=0)
    s = rep["summary"]

    print("1. The portfolio (Figure 12.1a)")
    print(f"   {s['n_policies']} policies, mean exposure {s['mean_exposure']} years")
    print(f"   {100 * s['frac_zero']:.1f}% of policies make no claim; claim frequency {s['frequency']:.4f} per year")
    print(f"   policies by number of claims: {s['histogram']}")
    print("   Book (real freMTPL2 test set): 94% zeros, frequency about 10%.")

    print("\n2. Table 12.1 on the held-out 25% (exposure-weighted metrics)")
    print("   model      MSE     MAE   deviance  non-positive  predicted claims  observed")
    for name, row in rep["table"].items():
        print(f"   {name:8s} {row['mse']:.4f}  {row['mae']:.4f}  {row['deviance']:.4f}  {row['n_nonpositive']:>12d}  {row['pred_claims']:>16.1f}  {row['obs_claims']:>8.0f}")
    print("   Book: dummy 0.564 / 0.189 / 0.625, ridge 0.560 / 0.177 / 0.601, Poisson 0.560 / 0.186 / 0.594.")
    print("   MSE and MAE barely separate ridge from the GLM; the Poisson deviance (eq. 12.37) does,")
    print("   even though ridge's deviance skips the policies where it predicts a non-positive rate.")

    print("\n3. Calibration (Figure 12.2): mean predicted vs observed frequency, 10 exposure-equal bins")
    for name, (pred, obs) in rep["calibration"].items():
        print(f"   {name:8s} predicted {pred}")
        print(f"   {'':8s} observed  {obs}")
    print("   Ridge under-predicts the safest decile; the GLM stays on the diagonal at the low end.")

    print("\n4. Ranking power (Gini of the Lorenz curve, higher ranks risks better)")
    print(f"   {rep['gini']}")
    print("   Ridge ranks about as well as the GLM: its problem is the level of its predictions, not their order.")

    print("\n5. The fitted GLM")
    print(f"   Newton iterations: {rep['newton_iterations']}")
    for col, rel in rep["relativities"].items():
        print(f"   relativities {col}: {rel}")
    print(f"   expected policies with 0..4+ claims: {rep['counts_expected']}")
    print(f"   observed policies with 0..4 claims:  {rep['counts_observed']}")
    print(f"   Pearson dispersion: {rep['dispersion']} (1 for a Poisson; above 1 means overdispersion)")

    print("\n6. Beyond the book: frequency x severity pure premium")
    pp = rep["pure_premium"]
    print(f"   predicted losses {pp['pred_losses']:,.0f} vs observed {pp['obs_losses']:,.0f}")
    print(f"   Tweedie deviance (p = 1.9): model {pp['tweedie_model']:.4f} vs flat premium {pp['tweedie_flat']:.4f}")
    print(f"   Gini on claim costs: {pp['gini']:.4f}")


if __name__ == "__main__":
    main()

