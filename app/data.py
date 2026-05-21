"""
data.py
-------
Shared loaders, normalization, and forecast engine.
"""

from __future__ import annotations

import io

import boto3
import pandas as pd
import streamlit as st
from pandas.tseries.offsets import MonthEnd


# ============================================================================
# S3 loaders
# ============================================================================

def _s3_client():
    aws = st.secrets["aws"]
    return boto3.client(
        "s3",
        aws_access_key_id=aws["access_key_id"],
        aws_secret_access_key=aws["secret_access_key"],
        region_name=aws["region"],
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_actuals() -> pd.DataFrame:
    s3 = _s3_client()
    obj = s3.get_object(
        Bucket=st.secrets["s3"]["bucket"],
        Key=st.secrets["s3"]["key"],
    )
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Value" in df.columns:
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_budgets() -> pd.DataFrame:
    s3 = _s3_client()
    obj = s3.get_object(
        Bucket=st.secrets["s3"]["bucket"],
        Key="financial_budgets.csv",
    )
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "BudgetAmount" in df.columns:
        df = df.rename(columns={"BudgetAmount": "Value"})
    if "Value" in df.columns:
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_amortization() -> pd.DataFrame:
    """Read amortization_schedules.csv from S3.

    Schema (per loan/month):
      Property | PropertyKey | Month | Year | Date | End of Month
      Interest Amount | Principal Amount | Payment Amount | Principal Balance

    Currency columns arrive as '$' + comma-formatted strings - we parse them
    to floats. 'End of Month' is parsed to datetime and is the join key for
    looking up balance at a property's exit month-end.
    """
    s3 = _s3_client()
    obj = s3.get_object(
        Bucket=st.secrets["s3"]["bucket"],
        Key="amortization_schedules.csv",
    )
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    if "End of Month" in df.columns:
        # MM/DD/YYYY in the source - be explicit so we don't misread
        df["End of Month"] = pd.to_datetime(
            df["End of Month"], errors="coerce", format="%m/%d/%Y"
        )
        # Fallback for any rows where the strict format didn't parse
        if df["End of Month"].isna().any():
            mask = df["End of Month"].isna()
            df.loc[mask, "End of Month"] = pd.to_datetime(
                df.loc[mask, "End of Month"], errors="coerce"
            )
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="%m/%d/%Y")
        if df["Date"].isna().any():
            mask = df["Date"].isna()
            df.loc[mask, "Date"] = pd.to_datetime(df.loc[mask, "Date"], errors="coerce")
    for col in ["Interest Amount", "Principal Amount", "Payment Amount", "Principal Balance"]:
        if col in df.columns:
            cleaned = (df[col].astype(str)
                       .str.replace("$", "", regex=False)
                       .str.replace(",", "", regex=False)
                       .str.strip())
            df[col] = pd.to_numeric(cleaned, errors="coerce").fillna(0.0)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Alias decisions from the data-review phase."""
    out = df.copy()
    out["Main Bucket"] = out["Main Bucket"].replace({"Capex": "CAPEX"})
    aliases = [
        ("CAPEX",          {"Capex": "CAPEX"}),
        ("Asset Mgmt Fee", {"Asset Management Fee": "Asset Mgmt Fee"}),
        ("Monitoring Fee", {"Monitoring Fees": "Monitoring Fee"}),
    ]
    for main, mapping in aliases:
        mask = out["Main Bucket"] == main
        out.loc[mask, "RST Sub Bucket"] = (
            out.loc[mask, "RST Sub Bucket"].replace(mapping)
        )
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def combine_actuals_budget(actuals: pd.DataFrame,
                           budgets: pd.DataFrame) -> pd.DataFrame:
    """Per property: keep all actuals; append budget rows for months past the
    property's actuals cutoff. Adds a 'Source' column."""
    a = actuals.copy()
    a["Source"] = "Actual"

    cutoffs = a.groupby("Property")["Date"].max().to_dict()

    parts = []
    for prop, group in budgets.groupby("Property"):
        cutoff = cutoffs.get(prop)
        if cutoff is None or pd.isna(cutoff):
            parts.append(group)
        else:
            parts.append(group[group["Date"] > cutoff])
    if parts:
        b = pd.concat(parts, ignore_index=True)
        b["Source"] = "Budget"
        return pd.concat([a, b], ignore_index=True)
    return a


# ============================================================================
# Proforma layout
# ============================================================================

PROFORMA_STRUCTURE = [
    ("Net Rental Income", [
        "Gross Potential Rent",
        "Vacancy",
        "Financial Revenue",
        "Other Income",
        "Rental Concessions",
        "Bad Debt",
        "Former Resident Collections",
        "Interest Income",
    ], "detailed"),
    ("Operating Expenses", [
        "Administrative Expenses",
        "Utilities",
        "Less Utility Reimbursements",
        "Operations & Maintenance",
        "Less Trash Reimbursements",
        "Insurance & Taxes",
        "Real Estate Taxes",
    ], "detailed"),
    ("Debt Costs", [
        "Mortgage Interest",
        "Mortgage Principal",
        "Financing & Loan Admin Fees",
        "MRTG INS PREM & RATE CAP ESC",
        "Mortgage Insurance Premium",
        "Asset Mgmt Fee",
    ], "detailed"),
    ("Asset Mgmt Fee",       None, "single_total"),
    ("Replacement Reserves", None, "single_total"),
    ("Monitoring Fee",       None, "single_total"),
    ("CAPEX",                None, "single_total"),
]

# Main buckets excluded from the Assumptions UI - debt is amortized separately
DEBT_MAIN_BUCKETS = {"Debt Costs"}

# Default forecast method per Sub Bucket. Format: (method, ratio_base_sub or None)
# method: "growth" or "percent_of"
DEFAULT_FORECAST_METHODS = {
    # Net Rental Income
    "Gross Potential Rent":        ("growth", None),
    "Vacancy":                     ("percent_of", "Gross Potential Rent"),
    "Financial Revenue":           ("percent_of", "Gross Potential Rent"),
    "Other Income":                ("percent_of", "Gross Potential Rent"),
    "Rental Concessions":          ("percent_of", "Gross Potential Rent"),
    "Bad Debt":                    ("percent_of", "Gross Potential Rent"),
    "Former Resident Collections": ("percent_of", "Bad Debt"),
    "Interest Income":             ("growth", None),
    # Operating Expenses
    "Administrative Expenses":     ("growth", None),
    "Utilities":                   ("growth", None),
    "Less Utility Reimbursements": ("percent_of", "Utilities"),
    "Operations & Maintenance":    ("growth", None),
    "Less Trash Reimbursements":   ("percent_of", "Operations & Maintenance"),
    "Insurance & Taxes":           ("growth", None),
    "Real Estate Taxes":           ("growth", None),
    # Single-totals
    "Asset Mgmt Fee":              ("growth", None),
    "Replacement Reserves":        ("growth", None),
    "Monitoring Fee":              ("growth", None),
    "CAPEX":                       ("growth", None),
}

# Sub Buckets that can serve as a ratio base in the assumptions UI
RATIO_BASE_CANDIDATES = [
    "Gross Potential Rent",
    "Bad Debt",
    "Utilities",
    "Operations & Maintenance",
    "Effective Gross Income",
]


def sub_buckets_in_proforma_order(df: pd.DataFrame, property_name: str,
                                   exclude_debt: bool = False) -> list[tuple[str, str]]:
    """Return (main_bucket, sub_bucket) pairs in the proforma display order, for
    sub-buckets that exist in the data for this property."""
    if property_name == "All Properties":
        scope = df
    else:
        scope = df[df["Property"] == property_name]
    existing = set(zip(scope["Main Bucket"], scope["RST Sub Bucket"]))

    out = []
    for main, subs, kind in PROFORMA_STRUCTURE:
        if exclude_debt and main in DEBT_MAIN_BUCKETS:
            continue
        if kind == "detailed":
            for sb in subs:
                if (main, sb) in existing:
                    out.append((main, sb))
            scope_subs = scope.loc[scope["Main Bucket"] == main, "RST Sub Bucket"].dropna().unique()
            for sb in scope_subs:
                if sb not in subs and (main, sb) in existing:
                    out.append((main, sb))
        else:
            scope_subs = scope.loc[scope["Main Bucket"] == main, "RST Sub Bucket"].dropna().unique()
            for sb in scope_subs:
                if (main, sb) in existing:
                    out.append((main, sb))
    return out


# ============================================================================
# Historical metrics
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def compute_t12_growth(combined_df: pd.DataFrame) -> dict:
    """T-12 vs prior-T-12 growth per (property, main, sub). Actuals only.
    Returns dict keyed by (property, main, sub) -> growth (float, e.g. 0.03)."""
    df = combined_df[combined_df["Source"] == "Actual"]
    if df.empty:
        return {}

    rates: dict = {}
    for prop, prop_df in df.groupby("Property"):
        anchor = prop_df["Date"].max()
        t12_start = (anchor - pd.DateOffset(months=11)).replace(day=1)
        prior_end = (t12_start - pd.DateOffset(days=1))
        prior_start = (prior_end - pd.DateOffset(months=11)).replace(day=1)

        for (main, sub), grp in prop_df.groupby(["Main Bucket", "RST Sub Bucket"]):
            t12 = grp.loc[(grp["Date"] >= t12_start) & (grp["Date"] <= anchor), "Value"].sum()
            prior = grp.loc[(grp["Date"] >= prior_start) & (grp["Date"] <= prior_end), "Value"].sum()
            if prior != 0:
                rates[(prop, main, sub)] = (t12 - prior) / abs(prior)
            else:
                rates[(prop, main, sub)] = 0.0
    return rates


@st.cache_data(ttl=3600, show_spinner=False)
def compute_3yr_cagr(combined_df: pd.DataFrame) -> dict:
    """3-year CAGR per (property, main, sub) using T-12 ending at anchor vs
    T-12 ending 36 months prior. Actuals only."""
    df = combined_df[combined_df["Source"] == "Actual"]
    if df.empty:
        return {}

    out: dict = {}
    for prop, prop_df in df.groupby("Property"):
        anchor = prop_df["Date"].max()
        latest_start = (anchor - pd.DateOffset(months=11)).replace(day=1)
        early_end = (anchor - pd.DateOffset(months=36))
        early_start = (early_end - pd.DateOffset(months=11)).replace(day=1)

        for (main, sub), grp in prop_df.groupby(["Main Bucket", "RST Sub Bucket"]):
            latest = grp.loc[(grp["Date"] >= latest_start) & (grp["Date"] <= anchor), "Value"].sum()
            early = grp.loc[(grp["Date"] >= early_start) & (grp["Date"] <= early_end), "Value"].sum()
            if early != 0 and (latest * early) > 0:  # same sign
                try:
                    out[(prop, main, sub)] = (latest / early) ** (1 / 3) - 1
                except (ValueError, ZeroDivisionError):
                    out[(prop, main, sub)] = 0.0
            else:
                out[(prop, main, sub)] = 0.0
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def compute_3yr_ratios(combined_df: pd.DataFrame) -> dict:
    """3-year aggregate ratio: total of (property, main, sub) over the last 36
    months divided by total of (property, base_sub) over the same window.
    Returns dict keyed by (property, main, sub, base_sub) -> ratio."""
    df = combined_df[combined_df["Source"] == "Actual"]
    if df.empty:
        return {}

    out: dict = {}
    for prop, prop_df in df.groupby("Property"):
        anchor = prop_df["Date"].max()
        start = (anchor - pd.DateOffset(months=35)).replace(day=1)

        window = prop_df[(prop_df["Date"] >= start) & (prop_df["Date"] <= anchor)]
        sums_main_sub = window.groupby(["Main Bucket", "RST Sub Bucket"])["Value"].sum()
        base_totals = window.groupby("RST Sub Bucket")["Value"].sum()

        for (main, sub), v in sums_main_sub.items():
            for base in RATIO_BASE_CANDIDATES:
                base_total = base_totals.get(base, 0.0)
                if base_total != 0:
                    out[(prop, main, sub, base)] = v / base_total
                else:
                    out[(prop, main, sub, base)] = 0.0
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def compute_t12_ratios(combined_df: pd.DataFrame) -> dict:
    """T-12 historical ratio of (property, main, sub) over (property, base_sub).
    Returns dict keyed by (property, main, sub, base_sub) -> ratio."""
    df = combined_df[combined_df["Source"] == "Actual"]
    if df.empty:
        return {}

    out: dict = {}
    for prop, prop_df in df.groupby("Property"):
        anchor = prop_df["Date"].max()
        start = (anchor - pd.DateOffset(months=11)).replace(day=1)

        # T-12 totals per (main, sub) and per sub-only-for-ratio-base
        t12 = (
            prop_df[(prop_df["Date"] >= start) & (prop_df["Date"] <= anchor)]
            .groupby(["Main Bucket", "RST Sub Bucket"])["Value"].sum()
        )
        base_totals = (
            prop_df[(prop_df["Date"] >= start) & (prop_df["Date"] <= anchor)]
            .groupby(["RST Sub Bucket"])["Value"].sum()
        )

        for (main, sub), v in t12.items():
            for base in RATIO_BASE_CANDIDATES:
                base_total = base_totals.get(base, 0.0)
                if base_total != 0:
                    out[(prop, main, sub, base)] = v / base_total
                else:
                    out[(prop, main, sub, base)] = 0.0
    return out


# ============================================================================
# Forecast engine
# ============================================================================

def _flat_repeat_forecast(prior_year_values, growth=0.0):
    return {m: v * (1.0 + growth) for m, v in prior_year_values.items()}


def apply_forecast(combined_df: pd.DataFrame,
                   assumptions: dict,
                   horizon_years: int) -> pd.DataFrame:
    """Extend combined_df with Forecast-source rows.

    Sub-buckets without an explicit user assumption fall back to historical
    T-12 defaults (growth or % ratio per DEFAULT_FORECAST_METHODS).
    Debt Costs sub-buckets are forecast by repeating the last actual year flat.
    """
    if horizon_years <= 0:
        return combined_df

    last_date = combined_df["Date"].max()
    if pd.isna(last_date):
        return combined_df
    last_year = int(last_date.year)
    forecast_years = list(range(last_year + 1, last_year + 1 + horizon_years))

    # Historical T-12 defaults used as fallback when user has no explicit assumption
    hist_growth = compute_t12_growth(combined_df)
    hist_ratios = compute_t12_ratios(combined_df)

    rows = []

    for prop, prop_df in combined_df.groupby("Property"):
        # Anchor monthly values per (main, sub) = last_year monthly distribution
        # We carry this forward and update it each forecast year.
        # subkey = (main, sub) -> {month: value}
        anchor_year_rows = prop_df[prop_df["Date"].dt.year == last_year]
        prior_values: dict = {}
        for (main, sub), grp in anchor_year_rows.groupby(["Main Bucket", "RST Sub Bucket"]):
            monthly = {m: 0.0 for m in range(1, 13)}
            for _, r in grp.iterrows():
                monthly[int(r["Date"].month)] += float(r["Value"])
            prior_values[(main, sub)] = monthly

        # Make sure every (main, sub) in prop_df has an entry (even if zero)
        all_subs = list(prop_df.groupby(["Main Bucket", "RST Sub Bucket"]).groups.keys())
        for k in all_subs:
            prior_values.setdefault(k, {m: 0.0 for m in range(1, 13)})

        # Row template lookup
        templates: dict = {}
        for k, grp in prop_df.groupby(["Main Bucket", "RST Sub Bucket"]):
            sample = grp.iloc[0]
            templates[k] = {
                "PropertyKey": sample.get("PropertyKey") if "PropertyKey" in sample.index else None,
                "Code": sample.get("Code") if "Code" in sample.index else None,
                "Bucket": sample.get("Bucket") if "Bucket" in sample.index else None,
            }

        for fy in forecast_years:
            year_values: dict = {}
            unresolved = []  # subs whose forecast depends on a base not yet computed

            # ----- Pass 1: growth methods + debt placeholder + overrides -----
            for (main, sub) in all_subs:
                key = (prop, main, sub)
                ass = assumptions.get(key, {})
                method = ass.get("method")
                # Auto-pick a default method from DEFAULT_FORECAST_METHODS
                if not method:
                    default = DEFAULT_FORECAST_METHODS.get(sub, ("growth", None))
                    method = default[0]
                overrides = ass.get("overrides", {}) or {}

                # Debt placeholder: flat repeat of prior year
                if main in DEBT_MAIN_BUCKETS:
                    year_values[(main, sub)] = dict(prior_values[(main, sub)])
                    continue

                # Year-level override wins for any method
                if fy in overrides and overrides[fy] is not None:
                    total = float(overrides[fy])
                    prior_total = sum(prior_values[(main, sub)].values())
                    if prior_total != 0:
                        proportions = {m: v / prior_total for m, v in prior_values[(main, sub)].items()}
                        year_values[(main, sub)] = {m: total * p for m, p in proportions.items()}
                    else:
                        year_values[(main, sub)] = {m: total / 12.0 for m in range(1, 13)}
                    continue

                if method == "growth":
                    if "growth_rate" in ass and ass.get("method") == "growth":
                        growth = float(ass.get("growth_rate", 0.0) or 0.0)
                    else:
                        growth = float(hist_growth.get((prop, main, sub), 0.0))
                    year_values[(main, sub)] = _flat_repeat_forecast(prior_values[(main, sub)], growth)
                else:
                    # percent_of -> defer to pass 2
                    unresolved.append((main, sub))

            # ----- Pass 2: percent_of methods, iterate until all resolved -----
            max_iterations = len(unresolved) + 2
            iteration = 0
            while unresolved and iteration < max_iterations:
                iteration += 1
                still_unresolved = []
                for (main, sub) in unresolved:
                    key = (prop, main, sub)
                    ass = assumptions.get(key, {})
                    default = DEFAULT_FORECAST_METHODS.get(sub, ("growth", None))
                    base_sub = ass.get("ratio_base_sub") or default[1]
                    if "ratio" in ass and ass.get("method") == "percent_of":
                        ratio = float(ass.get("ratio", 0.0) or 0.0)
                    else:
                        ratio = float(hist_ratios.get((prop, main, sub, base_sub), 0.0))
                    # Find base value (search across all main buckets for this property)
                    base_year_vals = None
                    for (m_, s_), vals in year_values.items():
                        if s_ == base_sub:
                            base_year_vals = vals
                            break
                    if base_year_vals is None:
                        # Base not yet computed - try again
                        still_unresolved.append((main, sub))
                        continue
                    year_values[(main, sub)] = {m: v * ratio for m, v in base_year_vals.items()}
                unresolved = still_unresolved

            # Any remaining unresolved subs - fall back to flat repeat with 0% growth
            for (main, sub) in unresolved:
                year_values[(main, sub)] = dict(prior_values[(main, sub)])

            # ----- Emit rows for the year -----
            for (main, sub), vals in year_values.items():
                tmpl = templates.get((main, sub), {})
                for month, value in vals.items():
                    me = pd.Timestamp(year=fy, month=month, day=1) + MonthEnd(0)
                    rows.append({
                        "Property": prop,
                        "PropertyKey": tmpl.get("PropertyKey"),
                        "Main Bucket": main,
                        "RST Sub Bucket": sub,
                        "Code": tmpl.get("Code"),
                        "Bucket": tmpl.get("Bucket"),
                        "Date": me,
                        "Year": fy,
                        "Value": value,
                        "Source": "Forecast",
                    })

            # Roll forward
            prior_values = year_values

    if not rows:
        return combined_df
    forecast_df = pd.DataFrame(rows)
    return pd.concat([combined_df, forecast_df], ignore_index=True)


# ============================================================================
# Period columns + source detection
# ============================================================================

def build_period_columns(years_to_show, years_to_expand):
    cols = []
    for year in sorted(years_to_show):
        if year in years_to_expand:
            for month in range(1, 13):
                me = pd.Timestamp(year=year, month=month, day=1) + MonthEnd(0)
                label = f"{me.month}/{me.day}/{me.year}"
                cols.append((label, year, month))
            cols.append((f"{year} Actuals", year, None))
        else:
            cols.append((f"{year} Actuals", year, None))
    return cols


def detect_source_columns(df, period_columns, property_name, source_value):
    if property_name == "All Properties":
        scope = df
    else:
        scope = df[df["Property"] == property_name]
    sub = scope[scope["Source"] == source_value]
    if sub.empty:
        return set()
    y = sub["Date"].dt.year
    m = sub["Date"].dt.month
    out = set()
    for label, year, month in period_columns:
        mask = y == year
        if month is not None:
            mask &= m == month
        if mask.any():
            out.add(label)
    return out


def build_proforma_wide(df, property_name, period_columns,
                       collapse_debt: bool = True):
    """When collapse_debt=True, Debt Costs is shown as a single total line."""
    if property_name == "All Properties":
        base = df.copy()
    else:
        base = df[df["Property"] == property_name].copy()

    base["Y"] = base["Date"].dt.year
    base["M"] = base["Date"].dt.month
    agg = (
        base.groupby(["Main Bucket", "RST Sub Bucket", "Y", "M"], dropna=False)["Value"]
            .sum()
            .reset_index()
    )

    def value_for(main, sub, year, month):
        m = (agg["Main Bucket"] == main) & (agg["RST Sub Bucket"] == sub) & (agg["Y"] == year)
        if month is not None:
            m &= agg["M"] == month
        return float(agg.loc[m, "Value"].sum())

    def bucket_total_value(main, year, month):
        m = (base["Main Bucket"] == main) & (base["Y"] == year)
        if month is not None:
            m &= base["M"] == month
        return float(base.loc[m, "Value"].sum())

    col_names = [c[0] for c in period_columns]
    rows = []
    totals_by_col = {c: {} for c in col_names}

    def empty_row(bucket, sub, row_type):
        r = {"Bucket": bucket, "Sub Bucket": sub, "RowType": row_type}
        for c in col_names:
            r[c] = None
        return r

    for main, subs, kind in PROFORMA_STRUCTURE:
        if collapse_debt and main in DEBT_MAIN_BUCKETS:
            kind = "single_total"
            subs = None
        if kind == "detailed":
            rows.append(empty_row(main, "", "header"))

            bucket_totals = {c: 0.0 for c in col_names}
            for sb in subs:
                r = {"Bucket": "", "Sub Bucket": sb, "RowType": "subbucket"}
                for label, year, month in period_columns:
                    v = value_for(main, sb, year, month)
                    r[label] = v
                    bucket_totals[label] += v
                rows.append(r)

            actual = base.loc[base["Main Bucket"] == main, "RST Sub Bucket"].dropna().unique()
            for sb in actual:
                if sb not in subs:
                    r = {"Bucket": "", "Sub Bucket": f"{sb} (unmapped)",
                         "RowType": "subbucket"}
                    for label, year, month in period_columns:
                        v = value_for(main, sb, year, month)
                        r[label] = v
                        bucket_totals[label] += v
                    rows.append(r)

            r = {"Bucket": "", "Sub Bucket": f"Total {main}", "RowType": "total"}
            for label in col_names:
                r[label] = bucket_totals[label]
                totals_by_col[label][main] = bucket_totals[label]
            rows.append(r)

            if main == "Operating Expenses":
                r = {"Bucket": "Net Operating Income", "Sub Bucket": "", "RowType": "noi"}
                for label in col_names:
                    nri = totals_by_col[label].get("Net Rental Income", 0)
                    opex = totals_by_col[label].get("Operating Expenses", 0)
                    noi = nri - abs(opex) if opex > 0 else nri + opex
                    r[label] = noi
                    totals_by_col[label]["NOI"] = noi
                rows.append(r)
        else:
            r = {"Bucket": main, "Sub Bucket": f"Total {main}", "RowType": "total"}
            for label, year, month in period_columns:
                v = bucket_total_value(main, year, month)
                r[label] = v
                totals_by_col[label][main] = v
            rows.append(r)

    def cost(x):
        return abs(x) if x > 0 else -x

    r = {"Bucket": "Results from Operations", "Sub Bucket": "", "RowType": "results"}
    for label in col_names:
        t = totals_by_col[label]
        rfo = (
            t.get("NOI", 0)
            - cost(t.get("Debt Costs", 0))
            - cost(t.get("Asset Mgmt Fee", 0))
            - cost(t.get("Replacement Reserves", 0))
            - cost(t.get("Monitoring Fee", 0))
            - cost(t.get("CAPEX", 0))
        )
        r[label] = rfo
    rows.append(r)

    return pd.DataFrame(rows), col_names


def apply_exit_dates(df: pd.DataFrame, exit_assumptions: dict) -> pd.DataFrame:
    """Remove rows whose Date is after each property's exit date.
    After a property is sold, it should stop contributing cash flow / NOI.

    exit_assumptions schema:
        {property_name: {"exit_year": int, "exit_month": int,
                         "exit_cap_rate": float, "exit_method": str}}
    """
    if not exit_assumptions:
        return df

    keep = pd.Series(True, index=df.index)
    for prop, info in exit_assumptions.items():
        if not info or "exit_year" not in info or "exit_month" not in info:
            continue
        try:
            exit_date = pd.Timestamp(
                year=int(info["exit_year"]),
                month=int(info["exit_month"]),
                day=1,
            ) + MonthEnd(0)
        except (ValueError, TypeError):
            continue
        drop_mask = (df["Property"] == prop) & (df["Date"] > exit_date)
        keep &= ~drop_mask
    return df[keep]


def build_cash_flow_wide(proforma_df: pd.DataFrame,
                         df: pd.DataFrame,
                         property_name: str,
                         period_columns,
                         exit_assumptions: dict,
                         amortization_df: pd.DataFrame | None = None,
                         selling_cost_pct: float = 0.02) -> pd.DataFrame:
    """Cash-flow table with the same columns as the proforma.

    Rows:
      Operating Cash Flow  (= Results from Operations from the proforma)
      Sales Proceeds (header)
        Gross Sale Price            = T-12 NOI ending at exit / Exit Cap Rate
        Less: Selling Costs (2%)
        Less: Mortgage Payoff       (placeholder $0 until amortization arrives)
        Less: Tax on Sale           (placeholder $0)
        Net Sale Proceeds
      Total Cash Flow      (= Operating + Net Sale Proceeds)

    Sale values appear in the column(s) containing each property's exit month.
    For monthly columns, the exit month gets the value; for year-total columns,
    the year containing the exit gets the value.
    """
    col_names = [c[0] for c in period_columns]

    # ---- Operating Cash Flow = Results from Operations row from the proforma
    rfo = proforma_df.loc[proforma_df["RowType"] == "results"]
    op_cf_by_col = {c: 0.0 for c in col_names}
    if not rfo.empty:
        rfo = rfo.iloc[0]
        for c in col_names:
            v = rfo.get(c)
            if isinstance(v, (int, float)) and not pd.isna(v):
                op_cf_by_col[c] = float(v)

    # ---- Sales Proceeds per period column
    properties_in_view = (
        sorted(df["Property"].dropna().unique().tolist())
        if property_name == "All Properties" else [property_name]
    )

    bucket_keys = ["gross", "selling_costs", "mortgage", "tax", "net"]
    sales_by_col = {c: {k: 0.0 for k in bucket_keys} for c in col_names}

    for prop in properties_in_view:
        info = exit_assumptions.get(prop, {}) or {}
        if not info or "exit_year" not in info or "exit_month" not in info:
            continue
        try:
            exit_year = int(info["exit_year"])
            exit_month = int(info["exit_month"])
        except (TypeError, ValueError):
            continue
        cap_rate = float(info.get("exit_cap_rate", 0.0) or 0.0)
        if cap_rate <= 0:
            continue

        exit_date = pd.Timestamp(year=exit_year, month=exit_month, day=1) + MonthEnd(0)
        t12_start = (exit_date - pd.DateOffset(months=11)).replace(day=1)

        prop_data = df[df["Property"] == prop]
        window = prop_data[(prop_data["Date"] >= t12_start) & (prop_data["Date"] <= exit_date)]
        nri_t12 = float(window.loc[window["Main Bucket"] == "Net Rental Income", "Value"].sum())
        opex_t12 = float(window.loc[window["Main Bucket"] == "Operating Expenses", "Value"].sum())
        noi_t12 = nri_t12 - abs(opex_t12) if opex_t12 > 0 else nri_t12 + opex_t12

        sale_price = noi_t12 / cap_rate
        sell_costs = sale_price * selling_cost_pct
        # Mortgage payoff = Principal Balance at exit month-end. Match on
        # PropertyKey (stable numeric id) rather than Property name to avoid
        # name-spelling mismatches between data sources. If the schedule
        # doesn't reach the exit date, fall back to the most recent month-end
        # on or before exit_date.
        mortgage = 0.0
        if amortization_df is not None and not amortization_df.empty:
            prop_rows = df[df["Property"] == prop]
            prop_key = None
            if not prop_rows.empty and "PropertyKey" in prop_rows.columns:
                pk = prop_rows["PropertyKey"].dropna().iloc[0] if not prop_rows["PropertyKey"].dropna().empty else None
                prop_key = pk
            if prop_key is not None:
                prop_sched = amortization_df[amortization_df["PropertyKey"] == prop_key]
                if not prop_sched.empty:
                    exact = prop_sched[prop_sched["End of Month"] == exit_date]
                    if not exact.empty:
                        mortgage = float(exact.iloc[0]["Principal Balance"])
                    else:
                        on_or_before = prop_sched[prop_sched["End of Month"] <= exit_date]
                        if not on_or_before.empty:
                            latest = on_or_before.sort_values(
                                "End of Month", ascending=False
                            ).iloc[0]
                            mortgage = float(latest["Principal Balance"])
        tax = 0.0        # placeholder; exit method not yet wired up
        net = sale_price - sell_costs - mortgage - tax

        for col_label, col_year, col_month in period_columns:
            hit = False
            if col_month is None:
                hit = (col_year == exit_year)
            else:
                hit = (col_year == exit_year and col_month == exit_month)
            if hit:
                sales_by_col[col_label]["gross"] += sale_price
                sales_by_col[col_label]["selling_costs"] += sell_costs
                sales_by_col[col_label]["mortgage"] += mortgage
                sales_by_col[col_label]["tax"] += tax
                sales_by_col[col_label]["net"] += net

    # ---- Build rows
    def row(bucket, sub, rtype, values_by_col):
        r = {"Bucket": bucket, "Sub Bucket": sub, "RowType": rtype}
        for c in col_names:
            r[c] = values_by_col.get(c)
        return r

    rows = []
    rows.append(row("Operating Cash Flow", "Results from Operations",
                    "total", op_cf_by_col))
    rows.append(row("Sales Proceeds", "", "header", {c: None for c in col_names}))
    rows.append(row("", "Gross Sale Price",
                    "subbucket", {c: sales_by_col[c]["gross"] for c in col_names}))
    rows.append(row("", "Less: Selling Costs (2%)",
                    "subbucket", {c: -sales_by_col[c]["selling_costs"] for c in col_names}))
    rows.append(row("", "Less: Mortgage Payoff",
                    "subbucket", {c: -sales_by_col[c]["mortgage"] for c in col_names}))
    rows.append(row("", "Less: Tax on Sale (placeholder)",
                    "subbucket", {c: -sales_by_col[c]["tax"] for c in col_names}))
    rows.append(row("", "Net Sale Proceeds",
                    "total", {c: sales_by_col[c]["net"] for c in col_names}))

    total_cf = {c: op_cf_by_col[c] + sales_by_col[c]["net"] for c in col_names}
    rows.append(row("Total Cash Flow", "", "results", total_cf))

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def load_property_metadata() -> pd.DataFrame:
    """Load property_metadata.csv if it exists in S3, else return an empty
    placeholder DataFrame so the rest of the app can keep functioning."""
    expected_cols = ["PropertyKey", "Property", "AcquisitionDate", "InitialEquity",
                     "PurchasePrice", "Notes"]
    try:
        s3 = _s3_client()
        obj = s3.get_object(
            Bucket=st.secrets["s3"]["bucket"],
            Key="property_metadata.csv",
        )
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception:
        return pd.DataFrame(columns=expected_cols)

    if "AcquisitionDate" in df.columns:
        df["AcquisitionDate"] = pd.to_datetime(df["AcquisitionDate"], errors="coerce")
    for currency_col in ["InitialEquity", "PurchasePrice"]:
        if currency_col in df.columns:
            cleaned = (df[currency_col].astype(str)
                       .str.replace("$", "", regex=False)
                       .str.replace(",", "", regex=False)
                       .str.strip())
            df[currency_col] = pd.to_numeric(cleaned, errors="coerce")
    return df


def compute_irr(cash_flows, periods_per_year: int = 12,
                max_iter: int = 300, tol: float = 1e-7):
    """Newton-Raphson IRR for a list of equal-interval cash flows.
    Returns annualized IRR as a decimal, or None if it doesn't converge or the
    series is degenerate."""
    if not cash_flows or len(cash_flows) < 2:
        return None
    has_pos = any(c > 0 for c in cash_flows)
    has_neg = any(c < 0 for c in cash_flows)
    if not (has_pos and has_neg):
        return None

    rate = 0.01  # per-period start guess
    for _ in range(max_iter):
        npv = 0.0
        dnpv = 0.0
        for i, c in enumerate(cash_flows):
            d = (1 + rate) ** i
            npv += c / d
            if i > 0:
                dnpv += -i * c / (d * (1 + rate))
        if abs(dnpv) < 1e-15:
            return None
        delta = npv / dnpv
        new_rate = rate - delta
        if abs(delta) < tol:
            return (1 + new_rate) ** periods_per_year - 1
        rate = max(-0.99, new_rate)
    return None


def compute_property_returns(property_name: str,
                             combined_df: pd.DataFrame,
                             proforma_df_property: pd.DataFrame | None,
                             exit_assumptions: dict,
                             amortization_df: pd.DataFrame | None,
                             metadata_df: pd.DataFrame | None,
                             selling_cost_pct: float = 0.02) -> dict:
    """Return a dict of return metrics for one property. Any metric that
    can't be computed (e.g., because metadata is missing) is set to None;
    callers should render those as 'Needs metadata'.

    Always computed (don't need metadata):
        net_sale_proceeds, sale_price, gross_sale_price, mortgage_payoff,
        selling_costs, t12_noi_at_exit
    Needs property metadata (AcquisitionDate + InitialEquity):
        irr, equity_multiple, hold_months, initial_equity, acquisition_date
    """
    out = {
        "property": property_name,
        "exit_year": None, "exit_month": None, "exit_method": None,
        "exit_cap_rate": None,
        "t12_noi_at_exit": None,
        "gross_sale_price": None,
        "selling_costs": None,
        "mortgage_payoff": None,
        "tax_on_sale": 0.0,  # placeholder
        "net_sale_proceeds": None,
        "acquisition_date": None,
        "initial_equity": None,
        "hold_months": None,
        "irr": None,
        "equity_multiple": None,
        "total_operating_cf_during_hold": None,
        "needs_metadata": True,
    }

    exit_info = exit_assumptions.get(property_name, {}) or {}
    if not exit_info or "exit_year" not in exit_info or "exit_month" not in exit_info:
        return out
    try:
        exit_year = int(exit_info["exit_year"])
        exit_month = int(exit_info["exit_month"])
    except (TypeError, ValueError):
        return out
    cap_rate = float(exit_info.get("exit_cap_rate", 0.0) or 0.0)

    out["exit_year"] = exit_year
    out["exit_month"] = exit_month
    out["exit_method"] = exit_info.get("exit_method")
    out["exit_cap_rate"] = cap_rate

    exit_date = pd.Timestamp(year=exit_year, month=exit_month, day=1) + MonthEnd(0)

    # ---- Trailing 12 NOI ending at exit ----
    t12_start = (exit_date - pd.DateOffset(months=11)).replace(day=1)
    prop_data = combined_df[combined_df["Property"] == property_name]
    window = prop_data[(prop_data["Date"] >= t12_start) & (prop_data["Date"] <= exit_date)]
    nri = float(window.loc[window["Main Bucket"] == "Net Rental Income", "Value"].sum())
    opex = float(window.loc[window["Main Bucket"] == "Operating Expenses", "Value"].sum())
    noi = nri - abs(opex) if opex > 0 else nri + opex
    out["t12_noi_at_exit"] = noi

    if cap_rate > 0:
        gross = noi / cap_rate
        out["gross_sale_price"] = gross
        out["selling_costs"] = gross * selling_cost_pct

        # Mortgage payoff via amortization (PropertyKey match)
        mortgage = 0.0
        if amortization_df is not None and not amortization_df.empty:
            prop_rows = combined_df[combined_df["Property"] == property_name]
            prop_key = None
            if not prop_rows.empty and "PropertyKey" in prop_rows.columns:
                pk_series = prop_rows["PropertyKey"].dropna()
                if not pk_series.empty:
                    prop_key = pk_series.iloc[0]
            if prop_key is not None:
                sched = amortization_df[amortization_df["PropertyKey"] == prop_key]
                if not sched.empty:
                    on_or_before = sched[sched["End of Month"] <= exit_date]
                    if not on_or_before.empty:
                        latest = on_or_before.sort_values("End of Month", ascending=False).iloc[0]
                        mortgage = float(latest["Principal Balance"])
        out["mortgage_payoff"] = mortgage
        out["net_sale_proceeds"] = gross - out["selling_costs"] - mortgage - out["tax_on_sale"]

    # ---- Metadata-dependent ----
    if metadata_df is None or metadata_df.empty:
        return out
    meta = metadata_df[metadata_df["Property"] == property_name]
    if meta.empty:
        return out
    acq = meta.iloc[0].get("AcquisitionDate")
    init_eq = meta.iloc[0].get("InitialEquity")
    if pd.isna(acq) or pd.isna(init_eq) or float(init_eq) == 0:
        return out

    out["acquisition_date"] = acq
    out["initial_equity"] = float(init_eq)
    out["needs_metadata"] = False

    # Hold period in months
    hold_months = (exit_date.year - acq.year) * 12 + (exit_date.month - acq.month)
    out["hold_months"] = max(0, hold_months)

    # Operating cash flow per month from acquisition to exit
    # Use Results-from-Operations equivalent: sum of all Main Buckets,
    # signed so expenses are negative.
    cf_window = prop_data[(prop_data["Date"] >= acq) & (prop_data["Date"] <= exit_date)].copy()
    cf_window["Y"] = cf_window["Date"].dt.year
    cf_window["M"] = cf_window["Date"].dt.month
    monthly_by_main = cf_window.groupby(["Y", "M", "Main Bucket"])["Value"].sum().reset_index()

    # Build month timeline
    months = pd.date_range(
        start=(acq.replace(day=1)),
        end=exit_date,
        freq="MS",
    )
    cf_series = []
    total_op_cf = 0.0
    for ts in months:
        y, m = ts.year, ts.month
        slice_ = monthly_by_main[(monthly_by_main["Y"] == y) & (monthly_by_main["M"] == m)]
        nri_m = float(slice_.loc[slice_["Main Bucket"] == "Net Rental Income", "Value"].sum())
        opex_m = float(slice_.loc[slice_["Main Bucket"] == "Operating Expenses", "Value"].sum())
        noi_m = nri_m - abs(opex_m) if opex_m > 0 else nri_m + opex_m
        debt_m = float(slice_.loc[slice_["Main Bucket"] == "Debt Costs", "Value"].sum())
        amf_m = float(slice_.loc[slice_["Main Bucket"] == "Asset Mgmt Fee", "Value"].sum())
        repl_m = float(slice_.loc[slice_["Main Bucket"] == "Replacement Reserves", "Value"].sum())
        mon_m = float(slice_.loc[slice_["Main Bucket"] == "Monitoring Fee", "Value"].sum())
        capex_m = float(slice_.loc[slice_["Main Bucket"] == "CAPEX", "Value"].sum())

        def cost(x):
            return abs(x) if x > 0 else -x

        op_cf = noi_m - cost(debt_m) - cost(amf_m) - cost(repl_m) - cost(mon_m) - cost(capex_m)
        cf_series.append(op_cf)
        total_op_cf += op_cf

    out["total_operating_cf_during_hold"] = total_op_cf

    # Build full IRR series: t=0 is -initial_equity, then monthly op CF,
    # final month adds net sale proceeds on top of that month's op CF.
    irr_series = [-float(init_eq)] + cf_series
    if out["net_sale_proceeds"] is not None and len(irr_series) >= 2:
        irr_series[-1] += float(out["net_sale_proceeds"])

    out["irr"] = compute_irr(irr_series, periods_per_year=12)
    total_cash_in = sum(c for c in irr_series[1:] if c > 0)
    # Equity Multiple = (positive cash returned) / initial equity
    if init_eq:
        out["equity_multiple"] = total_cash_in / float(init_eq)

    return out


def fmt_value(v) -> str:
    if v is None or pd.isna(v):
        return ""
    if v == 0:
        return "$0"
    if v >= 0:
        return f"${v:,.0f}"
    return f"(${-v:,.0f})"
