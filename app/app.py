"""
RST Portfolio - Proforma page (Streamlit entry).

Run with:
    cd app
    streamlit run app.py

Loads actuals + budgets from S3, blends them per property, optionally extends
with a 5-year forecast based on assumptions set on the Assumptions page, and
renders the internal RST proforma view.
"""

import pandas as pd
import streamlit as st

from data import (
    load_actuals, load_budgets, load_amortization,
    normalize, combine_actuals_budget,
    apply_forecast, apply_exit_dates,
    build_period_columns, build_proforma_wide, build_cash_flow_wide,
    detect_source_columns, fmt_value,
)

st.set_page_config(page_title="Proforma", layout="wide")

# Bump font sizes globally on this page
st.markdown(
    """
    <style>
    [data-testid="stMarkdownContainer"] p { font-size: 18px !important; }
    [data-testid="stMarkdownContainer"] code { font-size: 17px !important; }
    [data-testid="stMetricValue"] { font-size: 28px !important; }
    [data-testid="stMetricLabel"] { font-size: 15px !important; }
    h1 { font-size: 36px !important; }
    h2 { font-size: 28px !important; }
    h3 { font-size: 22px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Rendering
# ============================================================================

def render_proforma(df, period_columns, budget_cols, forecast_cols):
    """Render the proforma as a styled HTML table.

    - budget_cols: column labels that include any budget data
    - forecast_cols: column labels that include any forecast data
    """
    css = """
    <style>
      .proforma-wrap { overflow-x: auto; max-width: 100%; }
      table.proforma {
        border-collapse: collapse;
        font-family: 'Segoe UI', 'Calibri', Arial, sans-serif;
        font-size: 16px;
        white-space: nowrap;
        margin-top: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      }
      table.proforma th {
        background: #1f3a5c;
        color: #ffffff;
        font-weight: 700;
        font-style: italic;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 2px solid #15293f;
      }
      table.proforma th.col-val { text-align: right; }
      table.proforma td {
        padding: 7px 14px;
        border-bottom: 1px solid #e8eaed;
      }
      table.proforma td.col-val {
        text-align: right;
        font-variant-numeric: tabular-nums;
      }
      table.proforma td.col-val.neg { color: #b3261e; }
      table.proforma tr.row-header td {
        background: #f1f3f4; font-weight: 700; color: #1f3a5c;
      }
      table.proforma tr.row-subbucket td.col-sub { padding-left: 28px; color: #444; }
      table.proforma tr.row-total td {
        background: #fafbfc; font-weight: 700; border-top: 1px solid #d0d4d9;
      }
      table.proforma tr.row-noi td {
        background: #e3edff; font-weight: 700; font-size: 17px;
        border-top: 2px solid #1f3a5c; border-bottom: 2px solid #1f3a5c;
      }
      table.proforma tr.row-results td {
        background: #e6f4ea; font-weight: 700; font-size: 17px;
        border-top: 2px solid #1f3a5c;
      }
      table.proforma td.col-bucket { font-weight: 600; min-width: 180px; }
      table.proforma td.col-sub    { min-width: 240px; }
      table.proforma td.col-val,
      table.proforma th.col-val    { min-width: 110px; }
      /* Monthly columns: light blue cells, black text */
      table.proforma td.col-month { background: #eef4fb; color: #000; }
      table.proforma td.col-month.neg { color: #000; }
      /* Budget cells (monthly) get italics */
      table.proforma td.col-month.col-budget { font-style: italic; }
      /* Forecast cells: dotted underline, italic on monthly */
      table.proforma td.col-forecast { border-bottom: 1px dotted #888; }
      table.proforma td.col-month.col-forecast { font-style: italic; }
    </style>
    """

    year_total_cols = {c[0] for c in period_columns if c[2] is None}
    monthly_cols    = {c[0] for c in period_columns if c[2] is not None}

    def col_extra(label):
        classes = []
        if label in year_total_cols:
            classes.append("col-year-total")
        if label in monthly_cols:
            classes.append("col-month")
        if label in budget_cols:
            classes.append("col-budget")
        if label in forecast_cols:
            classes.append("col-forecast")
        return (" " + " ".join(classes)) if classes else ""

    def header_label(label):
        is_budget = label in budget_cols
        is_forecast = label in forecast_cols
        if not is_budget and not is_forecast:
            return label
        if label in year_total_cols:
            return label.replace(" Actuals", " Forecast")
        # Monthly columns: add a suffix indicating the source
        suffix = ""
        if is_forecast:
            suffix = " (F)"
        elif is_budget:
            suffix = " (B)"
        return f"{label}{suffix}"

    html = ['<div class="proforma-wrap"><table class="proforma">']
    html.append('<thead><tr>')
    html.append('<th class="col-bucket">Bucket</th>')
    html.append('<th class="col-sub">Sub Bucket</th>')
    for col in period_columns:
        label = col[0]
        html.append(f'<th class="col-val{col_extra(label)}">{header_label(label)}</th>')
    html.append('</tr></thead><tbody>')

    for _, row in df.iterrows():
        html.append(f'<tr class="row-{row["RowType"]}">')
        html.append(f'<td class="col-bucket">{row["Bucket"]}</td>')
        html.append(f'<td class="col-sub">{row["Sub Bucket"]}</td>')
        for col in period_columns:
            label = col[0]
            v = row[label]
            is_num = isinstance(v, (int, float)) and not pd.isna(v)
            neg_class = " neg" if is_num and v < 0 else ""
            cell_text = fmt_value(v) if is_num else ""
            html.append(f'<td class="col-val{neg_class}{col_extra(label)}">{cell_text}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')

    st.markdown(css + "".join(html), unsafe_allow_html=True)


# ============================================================================
# App
# ============================================================================

st.title("RST Portfolio - Proforma")

with st.spinner("Loading actuals + budgets from S3..."):
    try:
        actuals_raw = load_actuals()
        budgets_raw = load_budgets()
        amortization = load_amortization()
    except KeyError as e:
        st.error(f"Missing key in `.streamlit/secrets.toml`: {e}. "
                 "Expected sections [aws] and [s3].")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load data from S3: {e}")
        st.stop()

actuals = normalize(actuals_raw)
budgets = normalize(budgets_raw)
df_base = combine_actuals_budget(actuals, budgets)

# ---- Apply forecast based on session assumptions ----
assumptions = st.session_state.get("assumptions", {})

# ---- Sidebar selectors ----
with st.sidebar:
    st.header("Filters")
    properties = ["All Properties"] + sorted(df_base["Property"].dropna().unique().tolist())
    selected_property = st.selectbox("Property", properties)

    horizon = st.slider(
        "Years to forecast",
        min_value=0, max_value=5, value=5,
        help="Extends the data with forecast years past the last actual/budget month. "
             "Set assumptions on the Assumptions page.",
    )
    show_debt_detail = st.checkbox(
        "Show Debt Costs detail",
        value=False,
        help="When unchecked, Debt Costs is collapsed to a single total line.",
    )

df = apply_forecast(df_base, assumptions, horizon)
df = apply_exit_dates(df, st.session_state.get("exit_assumptions", {}))

# Available years from the (possibly forecasted) data
if selected_property == "All Properties":
    year_source = df
else:
    year_source = df[df["Property"] == selected_property]

available_years = sorted(
    year_source["Date"].dt.year.dropna().unique().astype(int).tolist(),
    reverse=True,
)
default_years = available_years[:8] if len(available_years) >= 8 else available_years

with st.sidebar:
    years_to_show = st.multiselect(
        "Years to include",
        available_years,
        default=default_years,
    )
    if not years_to_show:
        st.warning("Pick at least one year.")
        st.stop()

    years_to_expand = st.multiselect(
        "Show monthly detail for",
        sorted(years_to_show, reverse=True),
        default=[],
        help="Years checked here are broken out month-by-month, with a year-total column at the end.",
    )

# ---- Header ----
st.subheader(selected_property)
caption_bits = [f"Years: {', '.join(str(y) for y in sorted(years_to_show))}"]
if years_to_expand:
    caption_bits.append(f"Monthly detail: {', '.join(str(y) for y in sorted(years_to_expand))}")
if horizon > 0:
    n_with = sum(1 for v in assumptions.values() if v.get("growth_rate", 0) or v.get("overrides"))
    caption_bits.append(f"Forecast horizon: {horizon} year(s)  |  {n_with} sub-buckets with custom assumptions")
st.caption("  |  ".join(caption_bits))
st.caption(
    "(B) = month uses budget data (italic).  "
    "(F) = month is forecast (italic + dotted underline).  "
    "Year totals that include any budget or forecast are labeled 'Forecast'."
)

# ---- Proforma ----
period_columns = build_period_columns(years_to_show, years_to_expand)
budget_cols    = detect_source_columns(df, period_columns, selected_property, "Budget")
forecast_cols  = detect_source_columns(df, period_columns, selected_property, "Forecast")
proforma_df, _ = build_proforma_wide(
    df, selected_property, period_columns,
    collapse_debt=not show_debt_detail,
)
render_proforma(proforma_df, period_columns, budget_cols, forecast_cols)

# ---- Cash Flow table (Operating Cash Flow + Sales Proceeds) ----
st.subheader("Cash Flow")
st.caption(
    "Operating Cash Flow restates the Results from Operations from the proforma above. "
    "Sales Proceeds appear in the period containing each property's exit month. "
    "Sale Price = T-12 NOI ending at exit / Exit Cap Rate. Selling Costs 2%. "
    "Mortgage Payoff comes from the amortization schedule (Principal Balance at "
    "exit month-end). Tax on Sale is still a $0 placeholder until exit method "
    "logic is wired up."
)
cash_flow_df = build_cash_flow_wide(
    proforma_df, df, selected_property, period_columns,
    st.session_state.get("exit_assumptions", {}),
    amortization_df=amortization,
)
render_proforma(cash_flow_df, period_columns, budget_cols, forecast_cols)

# ---- Raw data sanity check (collapsible) ----
with st.expander("Raw data sanity check"):
    bucket = st.secrets["s3"]["bucket"]
    st.caption(
        f"Source files: s3://{bucket}/{st.secrets['s3']['key']}, "
        "financial_budgets.csv, amortization_schedules.csv"
    )

    # Amortization diagnostic - matches on PropertyKey (stable numeric id)
    st.markdown("**Amortization schedule diagnostic**")
    actuals_keys = set(df["PropertyKey"].dropna().astype(int).unique())
    amort_keys   = set(amortization["PropertyKey"].dropna().astype(int).unique())
    key_to_name = (
        df.dropna(subset=["PropertyKey"]).drop_duplicates("PropertyKey")
          .set_index(df.dropna(subset=["PropertyKey"]).drop_duplicates("PropertyKey")["PropertyKey"].astype(int))
          ["Property"].to_dict()
    )
    missing_keys = sorted(actuals_keys - amort_keys)
    extra_keys   = sorted(amort_keys - actuals_keys)

    if selected_property == "All Properties":
        st.write(
            f"Amortization data covers **{len(amort_keys)}** PropertyKeys; "
            f"date range "
            f"{amortization['End of Month'].min().date()} - "
            f"{amortization['End of Month'].max().date()}."
        )
        if missing_keys:
            st.warning(
                f"**{len(missing_keys)}** properties in the data have NO amortization "
                "rows (mortgage payoff will be $0 for them):"
            )
            st.write([f"{k}: {key_to_name.get(k, '?')}" for k in missing_keys])
        if extra_keys:
            st.info(
                f"**{len(extra_keys)}** PropertyKeys exist in the amortization file "
                "but not in the actuals/budget data:"
            )
            st.write(extra_keys)
    else:
        # Find PropertyKey for selected property
        rows = df[df["Property"] == selected_property]
        if rows.empty or rows["PropertyKey"].dropna().empty:
            st.warning(f"No PropertyKey found for `{selected_property}`.")
        else:
            prop_key = int(rows["PropertyKey"].dropna().iloc[0])
            prop_sched = amortization[amortization["PropertyKey"] == prop_key]
            st.caption(f"PropertyKey: {prop_key}")
            if prop_sched.empty:
                st.warning(
                    f"No amortization rows for PropertyKey {prop_key} "
                    f"(`{selected_property}`). Mortgage payoff will be $0."
                )
            else:
                exit_info = st.session_state.get("exit_assumptions", {}).get(selected_property, {})
                ey = exit_info.get("exit_year")
                em = exit_info.get("exit_month")
                cnt = len(prop_sched)
                first_d = prop_sched["End of Month"].min().date()
                last_d = prop_sched["End of Month"].max().date()
                st.write(
                    f"PropertyKey {prop_key} (`{selected_property}`): "
                    f"**{cnt}** schedule rows from {first_d} through {last_d}."
                )
                if ey and em:
                    exit_d = (pd.Timestamp(year=int(ey), month=int(em), day=1)
                              + pd.tseries.offsets.MonthEnd(0)).date()
                    exact = prop_sched[prop_sched["End of Month"].dt.date == exit_d]
                    if not exact.empty:
                        bal = float(exact.iloc[0]["Principal Balance"])
                        st.success(
                            f"Exit on {exit_d}: exact-month match found. "
                            f"Principal Balance = ${bal:,.2f}"
                        )
                    else:
                        on_or_before = prop_sched[prop_sched["End of Month"].dt.date <= exit_d]
                        if not on_or_before.empty:
                            latest = on_or_before.sort_values(
                                "End of Month", ascending=False
                            ).iloc[0]
                            st.info(
                                f"Exit on {exit_d} is past the schedule. "
                                f"Falling back to the latest available row "
                                f"({latest['End of Month'].date()}): "
                                f"Principal Balance = ${float(latest['Principal Balance']):,.2f}"
                            )
                        else:
                            st.warning(
                                f"Exit on {exit_d} is before the schedule starts. "
                                "Mortgage payoff will be $0."
                            )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total rows", f"{len(df):,}")
    c2.metric("Actual rows", f"{(df['Source'] == 'Actual').sum():,}")
    c3.metric("Budget rows", f"{(df['Source'] == 'Budget').sum():,}")
    c4.metric("Forecast rows", f"{(df['Source'] == 'Forecast').sum():,}")
    c5.metric("Unique properties", df["Property"].nunique())
    st.caption(
        f"Combined date range: {df['Date'].min().date()} - {df['Date'].max().date()}"
    )
