"""
Budget vs. Actual Variance Analytics — Synthetic Dataset Generator v2
======================================================================
Based on: Eskwelabs Data Modelling Process Documentation v1.0 (latest)

Produces TWO sets of relational tables written to CSV:

  FULL (with latent variables) — for internal debugging and model tracing
  ├── full/dim_cost_center.csv
  ├── full/dim_gl_account.csv
  ├── full/dim_month.csv
  ├── full/fact_budget.csv
  └── full/fact_variance.csv

  CLEAN (production, no latent variables) — mirrors what a real ERP exports
  ├── clean/dim_cost_center.csv
  ├── clean/dim_gl_account.csv
  ├── clean/dim_month.csv
  ├── clean/fact_budget.csv
  └── clean/fact_variance.csv

Dimensions
----------
  d = 1 ... 40   cost centers (explicitly named, fixed Size_d)
  g = 1 ... 15   GL accounts
  t = 1 ... 60   months (5 years, starting Jan 2020)

Equations
---------
  L1  Macro_t        = rho * Macro_{t-1} + n_t,   n_t ~ N(0, sigma_m^2)
  L2  M_d            ~ N(0, 1)
      B_d            = alpha * M_d + mu_d,          mu_d ~ N(0, sigma_B^2)
  L3  n_{d,g}        = exp(u_{d,g}),               u_{d,g} ~ N(0, sigma_n^2)
      Base_{d,g}     = Scale_g * Size_d * n_{d,g}
      Budget_{d,g}   = Base_{d,g} * (1 + B_d)
  L4  epsilon_{d,g,t} ~ t_nu(0, sigma_g^2)
      TrueSpend_{d,g,t} = Budget_{d,g} * (1 + beta_m * Macro_t) * exp(epsilon)
  L5  Actual_{d,g,t} = (1 - gamma) * TrueSpend_{d,g,t} + gamma * TrueSpend_{d,g,t-1}
      TrueSpend_{d,g,0} = Budget_{d,g}  [initial condition]
  L6  Variance_{d,g,t} = Actual_{d,g,t} - Budget_{d,g}

Dependencies
------------
  pip install numpy pandas scipy
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from datetime import date
from pathlib import Path

DOWNLOADS = Path.home() / "Downloads" / "output"

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — SIMULATION DIMENSIONS
# ══════════════════════════════════════════════════════════════════════════════

N_MONTHS       = 60
SIMULATION_START = date(2020, 1, 1)   # month 1 = January 2020


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SCALAR PARAMETERS
#  All scalar constants in one place. Change values here only.
# ══════════════════════════════════════════════════════════════════════════════

# L1 — Macro Process
RHO     = 0.85    # AR(1) persistence
SIGMA_M = 0.10    # Macro shock std dev
MACRO_0 = 0.0     # Initial condition

# L2 — Manager Latent Variables
ALPHA   = -0.30   # Discipline-to-bias sensitivity
SIGMA_B = 0.10    # Idiosyncratic bias noise std dev

# L3 — Budget Model
SIGMA_N = 0.20    # Base cost cross-sectional noise (log-normal)

# L4 — True Spend
BETA_M  = 0.05    # Macro sensitivity
NU      = 5       # Student-t degrees of freedom

# L5 — Delayed Recognition
GAMMA   = 0.20    # Accrual lag weight


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DEPARTMENT CATALOGUE
#  (name, size_d) — 40 entries, exactly as specified in the document
# ══════════════════════════════════════════════════════════════════════════════

DEPARTMENTS = [
    ("Engineering_1",         1.5),
    ("Engineering_2",         1.5),
    ("Engineering_3",         1.4),
    ("DevOps_1",              1.3),
    ("DevOps_2",              1.2),
    ("Product_1",             1.2),
    ("Product_2",             1.1),
    ("QA_1",                  1.0),
    ("QA_2",                  1.0),
    ("QA_3",                  1.0),
    ("IT_Support_1",          0.9),
    ("IT_Support_2",          0.9),
    ("IT_Support_3",          0.8),
    ("HR_1",                  0.7),
    ("HR_2",                  0.7),
    ("Finance_1",             0.8),
    ("Finance_2",             0.8),
    ("Finance_3",             0.9),
    ("Sales_1",               1.1),
    ("Sales_2",               1.1),
    ("Sales_3",               1.0),
    ("Marketing_1",           1.0),
    ("Marketing_2",           0.9),
    ("Marketing_3",           0.9),
    ("Customer_Success_1",    1.0),
    ("Customer_Success_2",    0.9),
    ("Customer_Success_3",    0.8),
    ("Legal_1",               0.7),
    ("Legal_2",               0.7),
    ("Security_1",            1.0),
    ("Security_2",            0.9),
    ("Analytics_1",           1.2),
    ("Analytics_2",           1.1),
    ("Analytics_3",           1.1),
    ("Data_Engineering_1",    1.3),
    ("Data_Engineering_2",    1.2),
    ("Operations_1",          1.0),
    ("Operations_2",          1.0),
    ("Operations_3",          0.9),
    ("Customer_Experience_1", 0.9),
]

N_COST_CENTERS = len(DEPARTMENTS)
assert N_COST_CENTERS == 40, "Department list must have exactly 40 entries."


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — GL ACCOUNT CATALOGUE
#  (name, type, scale_g_usd, sigma_g) — 15 entries
# ══════════════════════════════════════════════════════════════════════════════

GL_ACCOUNTS = [
    # name                        type           scale_g    sigma_g
    ("Rent / Lease",             "Fixed",        35_000,    0.05),
    ("Software Licenses",        "Fixed",        15_000,    0.05),
    ("Salaries",                 "Fixed",       120_000,    0.03),
    ("Insurance Premiums",       "Fixed",         4_000,    0.03),
    ("Cloud Hosting / IT Svcs",  "Semi-fixed",   40_000,    0.08),
    ("Utilities",                "Semi-fixed",    5_000,    0.10),
    ("Office Supplies",          "Variable",      2_500,    0.20),
    ("Training & Development",   "Variable",      6_000,    0.20),
    ("Miscellaneous",            "Variable",      3_000,    0.10),
    ("Legal / Professional",     "Variable",      8_000,    0.20),
    ("Recruitment / Hiring",     "Variable",     12_000,    0.20),
    ("Contractor / Freelancer",  "Variable",     30_000,    0.15),
    ("Equipment / Hardware",     "Variable",     10_000,    0.20),
    ("Travel & Transportation",  "Variable",      7_000,    0.15),
    ("Marketing / Advertising",  "Variable",     20_000,    0.20),
]

N_GL_ACCOUNTS = len(GL_ACCOUNTS)
assert N_GL_ACCOUNTS == 15, "GL account list must have exactly 15 entries."


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — LEVEL FUNCTIONS
#  One function per model level. Each is self-contained and documented.
# ══════════════════════════════════════════════════════════════════════════════

def simulate_macro(rng: np.random.Generator) -> np.ndarray:
    """
    L1 — Macro Process
    Macro_t = rho * Macro_{t-1} + n_t,  n_t ~ N(0, sigma_m^2)
    Initial condition: Macro_0 = 0

    Returns
    -------
    macro : ndarray (N_MONTHS,)
        macro[t-1] = Macro_t for t = 1..60
    """
    macro    = np.zeros(N_MONTHS)
    prev     = MACRO_0
    for t in range(N_MONTHS):
        n_t     = rng.normal(0.0, SIGMA_M)
        macro[t] = RHO * prev + n_t
        prev     = macro[t]
    return macro


def simulate_manager_variables(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    L2 — Manager Latent Variables
    M_d  ~ N(0, 1)
    B_d  = alpha * M_d + mu_d,  mu_d ~ N(0, sigma_B^2)

    Returns
    -------
    M : ndarray (N_COST_CENTERS,)   discipline scores
    B : ndarray (N_COST_CENTERS,)   realized budget bias
    """
    M  = rng.standard_normal(N_COST_CENTERS)
    mu = rng.normal(0.0, SIGMA_B, N_COST_CENTERS)
    B  = ALPHA * M + mu
    return M, B


def simulate_budgets(
    B: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    L3 — Budget Model
    n_{d,g} = exp(u_{d,g}),  u_{d,g} ~ N(0, sigma_n^2)
    Base_{d,g}   = Scale_g * Size_d * n_{d,g}
    Budget_{d,g} = Base_{d,g} * (1 + B_d)    [constant across all t]

    Returns
    -------
    base   : ndarray (N_COST_CENTERS, N_GL_ACCOUNTS)
    budget : ndarray (N_COST_CENTERS, N_GL_ACCOUNTS)
    """
    base   = np.zeros((N_COST_CENTERS, N_GL_ACCOUNTS))
    budget = np.zeros((N_COST_CENTERS, N_GL_ACCOUNTS))

    sizes  = np.array([size for _, size in DEPARTMENTS])   # shape (D,)

    for g, (_, _, scale_g, _) in enumerate(GL_ACCOUNTS):
        u          = rng.normal(0.0, SIGMA_N, N_COST_CENTERS)
        n_dg       = np.exp(u)
        base_g     = scale_g * sizes * n_dg
        base[:, g]   = base_g
        budget[:, g] = base_g * (1.0 + B)

    return base, budget


def simulate_true_spend(
    budget: np.ndarray,
    macro:  np.ndarray,
    rng:    np.random.Generator,
) -> np.ndarray:
    """
    L4 — True Spend Model
    epsilon_{d,g,t} ~ t_nu(0, sigma_g^2)
    TrueSpend_{d,g,t} = Budget_{d,g} * (1 + beta_m * Macro_t) * exp(epsilon)

    Returns
    -------
    true_spend : ndarray (N_COST_CENTERS, N_GL_ACCOUNTS, N_MONTHS)
    """
    true_spend = np.zeros((N_COST_CENTERS, N_GL_ACCOUNTS, N_MONTHS))

    for g, (_, _, _, sigma_g) in enumerate(GL_ACCOUNTS):
        for t in range(N_MONTHS):
            raw        = stats.t.rvs(df=NU, size=N_COST_CENTERS, random_state=rng)
            epsilon    = sigma_g * raw
            macro_mult = 1.0 + BETA_M * macro[t]
            true_spend[:, g, t] = budget[:, g] * macro_mult * np.exp(epsilon)

    return true_spend


def simulate_actual(
    true_spend: np.ndarray,
    budget:     np.ndarray,
) -> np.ndarray:
    """
    L5 — Delayed Recognition
    Actual_{d,g,t} = (1 - gamma) * TrueSpend_t + gamma * TrueSpend_{t-1}
    Initial condition: TrueSpend_{d,g,0} = Budget_{d,g}

    Returns
    -------
    actual : ndarray (N_COST_CENTERS, N_GL_ACCOUNTS, N_MONTHS)
    """
    actual  = np.zeros_like(true_spend)
    ts_prev = budget.copy()   # TrueSpend_{d,g,0} = Budget_{d,g}

    for t in range(N_MONTHS):
        ts_curr       = true_spend[:, :, t]
        actual[:, :, t] = (1.0 - GAMMA) * ts_curr + GAMMA * ts_prev
        ts_prev        = ts_curr

    return actual


def compute_variance(
    actual: np.ndarray,
    budget: np.ndarray,
) -> np.ndarray:
    """
    L6 — Observed Variance
    Variance_{d,g,t} = Actual_{d,g,t} - Budget_{d,g}
    Positive = overspend (unfavorable). Negative = underspend (favorable).

    Returns
    -------
    variance : ndarray (N_COST_CENTERS, N_GL_ACCOUNTS, N_MONTHS)
    """
    return actual - budget[:, :, np.newaxis]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — TABLE BUILDERS
#  One function per table. Returns a clean DataFrame ready for CSV export.
# ══════════════════════════════════════════════════════════════════════════════

def build_dim_cost_center(M: np.ndarray, B: np.ndarray) -> pd.DataFrame:
    """
    40 rows — one per department.
    Full version includes latent columns (size_d, manager_discipline, manager_bias).
    Clean version drops those columns.
    """
    rows = []
    for d, (dept_name, size_d) in enumerate(DEPARTMENTS):
        rows.append({
            "cost_center_id":      d + 1,
            "department_name":     dept_name,
            "size_d":              round(size_d, 2),
            "manager_discipline":  round(float(M[d]), 6),
            "manager_bias":        round(float(B[d]), 6),
        })
    return pd.DataFrame(rows)


def build_dim_gl_account() -> pd.DataFrame:
    """
    15 rows — one per GL account.
    Full version includes latent columns (scale_g, sigma_g).
    Clean version drops those columns.
    """
    rows = []
    for g, (name, acct_type, scale_g, sigma_g) in enumerate(GL_ACCOUNTS):
        rows.append({
            "gl_account_id":   g + 1,
            "gl_account_name": name,
            "gl_account_type": acct_type,
            "scale_g":         scale_g,
            "sigma_g":         sigma_g,
        })
    return pd.DataFrame(rows)


def build_dim_month(macro: np.ndarray) -> pd.DataFrame:
    """
    60 rows — one per month.
    Full version includes latent column (macro_t).
    Clean version drops that column.
    Month 1 = January 2020, month 60 = December 2024.
    """
    rows = []
    from calendar import month_abbr

    year  = SIMULATION_START.year
    month = SIMULATION_START.month

    for t in range(N_MONTHS):
        rows.append({
            "month_id":     t + 1,
            "year_number":  year,
            "month_number": month,
            "month_label":  f"{month_abbr[month]} {year}",
            "macro_t":      round(float(macro[t]), 6),
        })
        month += 1
        if month > 12:
            month = 1
            year += 1

    return pd.DataFrame(rows)


def build_fact_budget(
    base:   np.ndarray,
    budget: np.ndarray,
) -> pd.DataFrame:
    """
    600 rows — one per (cost_center, GL account) pair.
    Budget is fixed across all 60 months so no month dimension here.
    Full version includes latent column (base_cost).
    Clean version drops that column.
    """
    rows = []
    for d in range(N_COST_CENTERS):
        for g in range(N_GL_ACCOUNTS):
            rows.append({
                "cost_center_id": d + 1,
                "gl_account_id":  g + 1,
                "base_cost":      float(base[d, g]),
                "budget":         float(budget[d, g]),
            })
    return pd.DataFrame(rows)


def build_fact_variance(
    true_spend: np.ndarray,
    actual:     np.ndarray,
    variance:   np.ndarray,
    budget:     np.ndarray,
) -> pd.DataFrame:
    """
    36,000 rows — one per (cost_center, GL account, month).
    Full version includes latent column (true_spend).
    Clean version drops that column.
    """
    rows = []
    for d in range(N_COST_CENTERS):
        for g in range(N_GL_ACCOUNTS):
            bgt = budget[d, g]
            for t in range(N_MONTHS):
                act = actual[d, g, t]
                var = variance[d, g, t]
                rows.append({
                    "cost_center_id": d + 1,
                    "gl_account_id":  g + 1,
                    "month_id":       t + 1,
                    "true_spend":     float(true_spend[d, g, t]),
                    "actual":         float(act),
                    "variance":       float(var),
                    "variance_pct":   round((var / bgt) * 100, 4) if bgt != 0 else None,
                })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — EXPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

LATENT_COLUMNS = {
    "dim_cost_center": ["size_d", "manager_discipline", "manager_bias"],
    "dim_gl_account":  ["scale_g", "sigma_g"],
    "dim_month":       ["macro_t"],
    "fact_budget":     ["base_cost"],
    "fact_variance":   ["true_spend"],
}


def save_tables(tables: dict[str, pd.DataFrame], folder: str, full: bool) -> None:
    """
    Write all five tables to CSV under the given folder.
    If full=False, strips latent columns before writing.
    """
    os.makedirs(folder, exist_ok=True)
    for name, df in tables.items():
        out = df.copy()
        if not full:
            drop = [c for c in LATENT_COLUMNS.get(name, []) if c in out.columns]
            out  = out.drop(columns=drop)
        path = os.path.join(folder, f"{name}.csv")
        out.to_csv(path, index=False)
        print(f"    {name}.csv  ({len(out):,} rows, {len(out.columns)} columns)")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — SANITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def run_sanity_checks(
    tables: dict[str, pd.DataFrame],
    macro:  np.ndarray,
    B:      np.ndarray,
) -> None:
    fb = tables["fact_budget"]
    fv = tables["fact_variance"]

    print("\n── Sanity checks ──────────────────────────────────────────────")
    print(f"  fact_variance rows    : {len(fv):,}   (expected 36,000)")
    print(f"  Negative budgets      : {(fb['budget'] < 0).sum()}   (expected 0)")
    print(f"  Negative actual       : {(fv['actual'] < 0).sum()}   (expected 0)")
    print(f"  Negative true spend   : {(fv['true_spend'] < 0).sum()}   (expected 0)")
    print(f"  Macro range           : [{macro.min():.4f}, {macro.max():.4f}]")
    print(f"  Manager bias range    : [{B.min():.4f}, {B.max():.4f}]")
    print(f"  Variance_pct range    : [{fv['variance_pct'].min():.2f}%, {fv['variance_pct'].max():.2f}%]")
    print()
    print("  Variance_pct by GL account type:")
    gl   = tables["dim_gl_account"][["gl_account_id", "gl_account_type"]]
    joined = fv.merge(gl, on="gl_account_id")
    summary = (
        joined.groupby("gl_account_type")["variance_pct"]
        .agg(mean="mean", std="std", min="min", max="max")
        .round(2)
    )
    print(summary.to_string())


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate(seed=42, output_dir=r"C:\Users\Acer\Downloads\output") -> dict[str, pd.DataFrame]:
    """
    Run the full six-level simulation and write two sets of relational tables.

    Parameters
    ----------
    seed       : int   Random seed for reproducibility.
    output_dir : str   Root folder. Subfolders 'full/' and 'clean/' are created.

    Returns
    -------
    tables : dict mapping table name → full DataFrame (with latent columns).
    """
    rng = np.random.default_rng(seed)

    print("=" * 62)
    print("  Budget vs. Actual Variance Analytics — Generator v2")
    print("=" * 62)
    print(f"  Seed            : {seed}")
    print(f"  Cost centers    : {N_COST_CENTERS}")
    print(f"  GL accounts     : {N_GL_ACCOUNTS}")
    print(f"  Months          : {N_MONTHS}  (Jan 2020 – Dec 2024)")
    print(f"  Total fact rows : {N_COST_CENTERS * N_GL_ACCOUNTS * N_MONTHS:,}")
    print()

    # ── Run simulation levels ────────────────────────────────────────────────
    print("  [L1] Macro process...")
    macro = simulate_macro(rng)

    print("  [L2] Manager latent variables...")
    M, B = simulate_manager_variables(rng)

    print("  [L3] Budget model...")
    base, budget = simulate_budgets(B, rng)

    print("  [L4] True spend model...")
    true_spend = simulate_true_spend(budget, macro, rng)

    print("  [L5] Delayed recognition...")
    actual = simulate_actual(true_spend, budget)

    print("  [L6] Observed variance...")
    variance = compute_variance(actual, budget)

    # ── Build tables ─────────────────────────────────────────────────────────
    print("\n  Building relational tables...")
    tables = {
        "dim_cost_center": build_dim_cost_center(M, B),
        "dim_gl_account":  build_dim_gl_account(),
        "dim_month":       build_dim_month(macro),
        "fact_budget":     build_fact_budget(base, budget),
        "fact_variance":   build_fact_variance(true_spend, actual, variance, budget),
    }

    # ── Export full set (with latent variables) ───────────────────────────────
    full_dir = os.path.join(output_dir, "full")
    print(f"\n  Writing FULL tables → {full_dir}/")
    save_tables(tables, full_dir, full=True)

    # ── Export clean set (no latent variables) ────────────────────────────────
    clean_dir = os.path.join(output_dir, "clean")
    print(f"\n  Writing CLEAN tables → {clean_dir}/")
    save_tables(tables, clean_dir, full=False)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    run_sanity_checks(tables, macro, B)

    print("\n  Done.")
    return tables


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate(seed=42, output_dir=str(DOWNLOADS))



print("OUTPUT FOLDER LOCATION:")
print(os.path.abspath("output"))
