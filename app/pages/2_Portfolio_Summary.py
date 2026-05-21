"""
Portfolio Summary page.

A sortable table of all properties with the current session's exit assumptions
applied. Shows T-12 NOI at exit, Sale Price, Net Sale Proceeds, IRR and
Equity Multiple (both require property metadata).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from data import (
    load_actuals, load_budgets, load_amortization, load_property_metadata,
    normalize, combine_actuals_budget,
    compute_property_returns,
)

st.set_page_config(page_title="Portfolio Summary", layout="wide")

# Match the global font bump used elsewhere
st.markdown(
    """
    <style>
    [data-testid="stMarkdownContainer"] p { font-size: 18px !important; }
    h1 { font-size: 36px !important; }
    h2 { font-size: 28px !important; }
    h3 { font-size: 22px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Portfolio Summary")
st.caption(
    "All properties with the current session's exit assumptions applied. "
    "IRR and Equity Multiple require property metadata "
    "(s3://testbucketcomproject/property_metadata.csv) - until that file "
    "is provided, those columns will be empty."
)

# ---- Load data ----
with st.spinner("Loading data..."):
    actuals = normalize(load_actuals())
    budgets = normalize(load_budgets())
    amortization = load_amortization()
    metadata = load_property_metadata()
    df = combine_actuals_budget(actuals, budgets)

# Pull current session assumptions
exit_assumptions = st.session_state.get("exit_assumptions", {})

# ---- Compute returns for every property ----
properties = sorted(df["Property"].dropna().unique().tolist())
rows = []
for prop in properties:
    r = compute_property_returns(
        prop, df, None, exit_assumptions, amortization, metadata,
    )
    rows.append({
        "Property": prop,
        "Exit Year": r.get("exit_year"),
        "Exit Month": r.get("exit_month"),
        "Exit Method": r.get("exit_method"),
        "Exit Cap Rate": r.get("exit_cap_rate"),
        "T-12 NOI (at exit)": r.get("t12_noi_at_exit"),
        "Gross Sale Price": r.get("gross_sale_price"),
        "Mortgage Payoff": r.get("mortgage_payoff"),
        "Net Sale Proceeds": r.get("net_sale_proceeds"),
        "Initial Equity": r.get("initial_equity"),
        "Hold (months)": r.get("hold_months"),
        "IRR": r.get("irr"),
        "Equity Multiple": r.get("equity_multiple"),
    })

table = pd.DataFrame(rows)

# ---- Quick portfolio totals at the top ----
total_noi = table["T-12 NOI (at exit)"].dropna().sum()
total_gross = table["Gross Sale Price"].dropna().sum()
total_payoff = table["Mortgage Payoff"].dropna().sum()
total_net = table["Net Sale Proceeds"].dropna().sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Properties", f"{len(table):,}")
c2.metric("Total T-12 NOI", f"${total_noi:,.0f}")
c3.metric("Total Gross Sale Price", f"${total_gross:,.0f}")
c4.metric("Total Mortgage Payoff", f"${total_payoff:,.0f}")
c5.metric("Total Net Sale Proceeds", f"${total_net:,.0f}")

st.divider()

# ---- Sortable table ----
st.subheader("All properties")

# Render with column_config for clean formatting
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Exit Cap Rate":      st.column_config.NumberColumn(format="%.3f%%", help="Stored as decimal (e.g., 0.055 = 5.50%)"),
        "T-12 NOI (at exit)": st.column_config.NumberColumn(format="$%.0f"),
        "Gross Sale Price":   st.column_config.NumberColumn(format="$%.0f"),
        "Mortgage Payoff":    st.column_config.NumberColumn(format="$%.0f"),
        "Net Sale Proceeds":  st.column_config.NumberColumn(format="$%.0f"),
        "Initial Equity":     st.column_config.NumberColumn(format="$%.0f"),
        "Hold (months)":      st.column_config.NumberColumn(format="%d"),
        "IRR":                st.column_config.NumberColumn(format="%.2f%%"),
        "Equity Multiple":    st.column_config.NumberColumn(format="%.2fx"),
    },
)

# Note: Streamlit displays Exit Cap Rate and IRR as decimals; multiply for percent
# Adjusting display by rescaling
st.caption(
    "Click any column header to sort. IRR and Equity Multiple require "
    "AcquisitionDate + InitialEquity per property in property_metadata.csv."
)

# ---- Metadata loaded? ----
if metadata is None or metadata.empty:
    st.warning(
        "No property metadata loaded. To populate IRR + Equity Multiple, "
        "upload property_metadata.csv to s3://testbucketcomproject/ with "
        "columns: `PropertyKey, Property, AcquisitionDate, InitialEquity` "
        "(plus optional `PurchasePrice` and `Notes`)."
    )
else:
    with_meta = metadata.dropna(subset=["AcquisitionDate", "InitialEquity"])
    st.success(
        f"Property metadata: {len(with_meta)} of {len(properties)} properties "
        "have both AcquisitionDate and InitialEquity set."
    )
