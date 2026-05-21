"""
Forecast Assumptions page.

Per property, set forecast rates for each Sub Bucket (grouped by Main Bucket).
The forecast method per Sub Bucket is fixed (read-only) per the
DEFAULT_FORECAST_METHODS map; the value is editable.
Debt Costs is excluded - amortization handled separately.
Saved in session state only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from data import (
    load_actuals, load_budgets, load_amortization, load_property_metadata,
    normalize, combine_actuals_budget,
    compute_t12_growth, compute_3yr_cagr, compute_t12_ratios,
    compute_3yr_ratios, compute_property_returns,
    sub_buckets_in_proforma_order,
    apply_forecast,
    DEFAULT_FORECAST_METHODS, PROFORMA_STRUCTURE, DEBT_MAIN_BUCKETS,
)

st.set_page_config(page_title="Assumptions", layout="wide")

# Bump font size for the assumption tables
st.markdown(
    """
    <style>
    [data-testid="stMarkdownContainer"] p { font-size: 19px !important; }
    [data-testid="stMarkdownContainer"] code { font-size: 18px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Forecast Assumptions")

# Placeholder for the metric cards strip - rendered after data + assumptions
# are loaded further down. We anchor it here visually by reserving a container.
_metric_cards_placeholder = st.container()

st.caption(
    "Per-property forecast rates by Sub Bucket. Method per Sub Bucket is fixed; "
    "the rate is editable. Debt Costs are excluded (amortization is handled "
    "separately). Saved in this browser session only."
)

# ---- Load data ----
with st.spinner("Loading data..."):
    actuals = normalize(load_actuals())
    budgets = normalize(load_budgets())
    amortization = load_amortization()
    metadata = load_property_metadata()
    df = combine_actuals_budget(actuals, budgets)

if "assumptions" not in st.session_state:
    st.session_state.assumptions = {}

# ---- Sidebar: property selector ----
with st.sidebar:
    st.header("Property")
    properties = sorted(df["Property"].dropna().unique().tolist())
    selected_property = st.selectbox("Property", properties)

# ---- Exit Assumptions ----
if "exit_assumptions" not in st.session_state:
    st.session_state.exit_assumptions = {}

st.subheader("Exit Assumptions")

EXIT_METHODS = ["Pay Tax", "1031", "1031 into OZ", "Fund OZ I", "Fund OZ II"]
MONTHS_1_12 = list(range(1, 13))

_last_data_year = int(df["Date"].max().year)
_exit_year_options = list(range(_last_data_year, _last_data_year + 6))

_current_exit = st.session_state.exit_assumptions.get(selected_property, {})

_ec1, _ec2, _ec3, _ec4 = st.columns(4)

_exit_cap_pct = _ec1.number_input(
    "Exit Cap Rate (%)",
    value=float(_current_exit.get("exit_cap_rate", 0.055)) * 100.0,
    step=0.25,
    format="%.2f",
    key=f"exit_cap_{selected_property}",
)

_default_month = int(_current_exit.get("exit_month", 12))
_exit_month = _ec2.selectbox(
    "Exit Month",
    MONTHS_1_12,
    index=MONTHS_1_12.index(_default_month) if _default_month in MONTHS_1_12 else 11,
    format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
    key=f"exit_month_{selected_property}",
)

_default_year = int(_current_exit.get("exit_year", _exit_year_options[-1]))
_exit_year = _ec3.selectbox(
    "Exit Year",
    _exit_year_options,
    index=(_exit_year_options.index(_default_year)
           if _default_year in _exit_year_options else len(_exit_year_options) - 1),
    key=f"exit_year_{selected_property}",
)

_default_method = _current_exit.get("exit_method", "Pay Tax")
_exit_method = _ec4.selectbox(
    "Exit Method",
    EXIT_METHODS,
    index=(EXIT_METHODS.index(_default_method)
           if _default_method in EXIT_METHODS else 0),
    key=f"exit_method_{selected_property}",
)

st.session_state.exit_assumptions[selected_property] = {
    "exit_cap_rate": _exit_cap_pct / 100.0,
    "exit_month": int(_exit_month),
    "exit_year": int(_exit_year),
    "exit_method": _exit_method,
}

st.caption(
    f"Property exits at end of {pd.Timestamp(2000, _exit_month, 1).strftime('%B')} "
    f"{_exit_year} via {_exit_method}. After that month, the property stops "
    "contributing cash flow / NOI on the Proforma. The Exit Cap Rate is "
    "stored but not yet used in calculations."
)

st.divider()

# ---- Render the metric cards strip at the top of the page ----
def _fmt_money(v):
    if v is None:
        return "—"
    if v == 0:
        return "$0"
    return f"${v:,.0f}" if v >= 0 else f"-${-v:,.0f}"

def _fmt_pct(v):
    if v is None:
        return "—"
    return f"{v * 100:+.1f}%"

def _fmt_x(v):
    if v is None:
        return "—"
    return f"{v:.2f}x"

# Build property-level returns using the just-saved exit assumption
_returns = compute_property_returns(
    selected_property, df, None,
    st.session_state.exit_assumptions, amortization, metadata,
)
with _metric_cards_placeholder:
    _cards = st.columns(5)
    _cards[0].metric("T-12 NOI (at exit)", _fmt_money(_returns.get("t12_noi_at_exit")))
    _cards[1].metric("Gross Sale Price",   _fmt_money(_returns.get("gross_sale_price")))
    _cards[2].metric("Net Sale Proceeds",  _fmt_money(_returns.get("net_sale_proceeds")))
    _cards[3].metric("IRR (hold period)",  _fmt_pct(_returns.get("irr")))
    _cards[4].metric("Equity Multiple",    _fmt_x(_returns.get("equity_multiple")))
    if _returns.get("needs_metadata"):
        st.caption(
            "IRR and Equity Multiple show '—' until property metadata "
            "(AcquisitionDate + InitialEquity) is loaded. "
            "Expected file: s3://testbucketcomproject/property_metadata.csv "
            "with columns: PropertyKey, Property, AcquisitionDate, InitialEquity, PurchasePrice (optional)."
        )

# ---- Historical metrics ----
t12_growth = compute_t12_growth(df)
cagr_3yr = compute_3yr_cagr(df)
t12_ratios = compute_t12_ratios(df)
ratios_3yr = compute_3yr_ratios(df)

last_year = int(df["Date"].max().year)
forecast_years = list(range(last_year + 1, last_year + 1 + 5))


def method_for(sub: str) -> tuple[str, str | None]:
    return DEFAULT_FORECAST_METHODS.get(sub, ("growth", None))


def method_label(sub: str) -> str:
    method, base = method_for(sub)
    return "% Growth" if method == "growth" else f"% of {base}"


def default_value_for(prop, main, sub):
    """Default rate: T-12 growth for growth method, T-12 ratio for % of."""
    method, base = method_for(sub)
    if method == "growth":
        return float(t12_growth.get((prop, main, sub), 0.0))
    return float(t12_ratios.get((prop, main, sub, base), 0.0))


def current_value(prop, main, sub) -> float:
    """Get the saved rate (growth or ratio) or the historical default."""
    key = (prop, main, sub)
    ass = st.session_state.assumptions.get(key, {})
    if "method" not in ass:
        return default_value_for(prop, main, sub)
    if ass["method"] == "growth":
        return float(ass.get("growth_rate", 0.0) or 0.0)
    return float(ass.get("ratio", 0.0) or 0.0)


def save_value(prop, main, sub, value_pct):
    """Save the value with the fixed method for this sub-bucket."""
    key = (prop, main, sub)
    method, base = method_for(sub)
    existing = st.session_state.assumptions.get(key, {})
    overrides = existing.get("overrides", {}) or {}
    if method == "growth":
        st.session_state.assumptions[key] = {
            "method": "growth",
            "growth_rate": float(value_pct) / 100.0,
            "ratio": 0.0,
            "ratio_base_sub": None,
            "overrides": overrides,
        }
    else:
        st.session_state.assumptions[key] = {
            "method": "percent_of",
            "growth_rate": 0.0,
            "ratio": float(value_pct) / 100.0,
            "ratio_base_sub": base,
            "overrides": overrides,
        }


# ---- Compute projected Year-1 growth per Main Bucket ----
# (Uses session_state assumptions from last render; for unset subs the engine
# falls back to historical T-12 defaults internally.)
prop_only = df[df["Property"] == selected_property]
fy1 = last_year + 1

prior_by_main = (
    prop_only[prop_only["Date"].dt.year == last_year]
    .groupby("Main Bucket")["Value"].sum().to_dict()
)
try:
    with_forecast = apply_forecast(prop_only, st.session_state.assumptions, 1)
    fy1_rows = with_forecast[
        (with_forecast["Source"] == "Forecast") &
        (with_forecast["Date"].dt.year == fy1)
    ]
    fy1_by_main = fy1_rows.groupby("Main Bucket")["Value"].sum().to_dict()
except Exception:
    fy1_by_main = {}

def fmt_growth_pct(main: str) -> str:
    prior = prior_by_main.get(main, 0.0)
    fc = fy1_by_main.get(main, 0.0)
    if prior == 0:
        return "n/a"
    g = (fc - prior) / abs(prior)
    return f"{g * 100:+.2f}%"


# ---- Subs grouped by Main Bucket ----
subs = sub_buckets_in_proforma_order(df, selected_property, exclude_debt=True)
subs_by_main: dict[str, list[str]] = {}
for main, sub in subs:
    subs_by_main.setdefault(main, []).append(sub)

main_order = [m for m, _, _ in PROFORMA_STRUCTURE
              if m not in DEBT_MAIN_BUCKETS and m in subs_by_main]

# ---- Render expanders ----
for main in main_order:
    label = f"**{main}**  ·  Yr 1 projected growth: **{fmt_growth_pct(main)}**"
    with st.expander(label, expanded=(main == "Net Rental Income")):
        # Header
        cols = st.columns([2.4, 2.0, 0.9, 0.9, 1.1])
        cols[0].markdown("**Sub Bucket**")
        cols[1].markdown("**Method**")
        cols[2].markdown("**3-yr**")
        cols[3].markdown("**T-12**")
        cols[4].markdown("**Rate %**")

        for sub in subs_by_main[main]:
            method, base = method_for(sub)
            # Pick the historical metric that matches the method
            if method == "growth":
                hist_3yr = cagr_3yr.get((selected_property, main, sub), 0.0)
                hist_t12 = t12_growth.get((selected_property, main, sub), 0.0)
            else:
                hist_3yr = ratios_3yr.get((selected_property, main, sub, base), 0.0)
                hist_t12 = t12_ratios.get((selected_property, main, sub, base), 0.0)
            value = current_value(selected_property, main, sub)

            c = st.columns([2.4, 2.0, 0.9, 0.9, 1.1])
            c[0].markdown(sub)
            c[1].markdown(f"`{method_label(sub)}`")
            c[2].markdown(f"{hist_3yr * 100:.2f}%")
            c[3].markdown(f"{hist_t12 * 100:.2f}%")
            new_value = c[4].number_input(
                f"rate_{selected_property}_{main}_{sub}",
                value=float(value) * 100.0,
                step=0.25,
                format="%.2f",
                label_visibility="collapsed",
                key=f"rate_{selected_property}_{main}_{sub}",
            )
            save_value(selected_property, main, sub, new_value)

# ---- Year-level overrides ----
st.divider()
with st.expander("Year-level absolute overrides (optional)", expanded=False):
    st.caption(
        "If set (non-zero), the value replaces the calculated forecast total "
        "for that year. Distributed across months using the prior year's mix."
    )
    for main in main_order:
        st.markdown(f"**{main}**")
        for sub in subs_by_main[main]:
            key = (selected_property, main, sub)
            existing = st.session_state.assumptions.get(key, {}).get("overrides", {}) or {}
            row = st.columns([2.4] + [1.0] * 5)
            row[0].markdown(sub)
            new_overrides = {}
            for i, year in enumerate(forecast_years):
                val = row[i + 1].number_input(
                    f"{year}",
                    value=float(existing.get(year, 0.0)),
                    step=1000.0,
                    format="%.0f",
                    key=f"ov_{selected_property}_{main}_{sub}_{year}",
                    label_visibility="visible" if i == 0 else "collapsed",
                )
                if val != 0:
                    new_overrides[year] = val
            if key not in st.session_state.assumptions:
                method, base = method_for(sub)
                default_value = default_value_for(selected_property, main, sub)
                if method == "growth":
                    st.session_state.assumptions[key] = {
                        "method": "growth",
                        "growth_rate": default_value,
                        "ratio": 0.0,
                        "ratio_base_sub": None,
                        "overrides": {},
                    }
                else:
                    st.session_state.assumptions[key] = {
                        "method": "percent_of",
                        "growth_rate": 0.0,
                        "ratio": default_value,
                        "ratio_base_sub": base,
                        "overrides": {},
                    }
            st.session_state.assumptions[key]["overrides"] = new_overrides

# ---- Reset / summary ----
st.divider()
c1, _ = st.columns([1, 8])
if c1.button("Reset this property to defaults"):
    for main in main_order:
        for sub in subs_by_main.get(main, []):
            key = (selected_property, main, sub)
            st.session_state.assumptions.pop(key, None)
    st.success(f"Reset {selected_property} to defaults.")
    st.rerun()

n_props_set = len({k[0] for k in st.session_state.assumptions.keys()})
n_overrides = sum(
    len(v.get("overrides", {}) or {})
    for v in st.session_state.assumptions.values()
)
st.caption(
    f"In session: {len(st.session_state.assumptions)} sub-bucket assumptions "
    f"across {n_props_set} properties, with {n_overrides} year-level overrides."
)
