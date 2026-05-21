"""
RST Portfolio - Streamlit app.

Run with:
    cd app
    streamlit run app.py

Loads historic actuals + financial budgets from S3, blends them per property
(actuals where they exist, budget after the cutoff), and renders the internal
RST proforma view.
"""

import io

import boto3
import pandas as pd
import streamlit as st
from pandas.tseries.offsets import MonthEnd

st.set_page_config(page_title="RST Portfolio", layout="wide")


# ============================================================================
# Data loading
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
    """Read historic_actuals.csv from S3."""
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
    """Read financial_budgets.csv from S3. Renames BudgetAmount -> Value
    so the schema matches actuals."""
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
    """Per property: keep all actuals; append budget rows for months that fall
    after the property's latest actuals month. Adds a 'Source' column."""
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


# ============================================================================
# Period columns + budget detection
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


def detect_budget_columns(df, period_columns, property_name):
    """Return the set of column labels whose contributing data includes any
    Budget-source rows."""
    if property_name == "All Properties":
        scope = df
    else:
        scope = df[df["Property"] == property_name]
    bdg = scope[scope["Source"] == "Budget"]
    if bdg.empty:
        return set()

    bdg_y = bdg["Date"].dt.year
    bdg_m = bdg["Date"].dt.month
    out = set()
    for label, year, month in period_columns:
        mask = bdg_y == year
        if month is not None:
            mask &= bdg_m == month
        if mask.any():
            out.add(label)
    return out


def build_proforma_wide(df, property_name, period_columns):
    """Return wide DataFrame: bucket/subbucket rows + one column per period."""
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


# ============================================================================
# Rendering
# ============================================================================

def fmt_value(v) -> str:
    if v is None or pd.isna(v):
        return ""
    if v == 0:
        return "$0"
    if v >= 0:
        return f"${v:,.0f}"
    return f"(${-v:,.0f})"


def render_proforma(df, period_columns, budget_cols):
    css = """
    <style>
      .proforma-wrap { overflow-x: auto; max-width: 100%; }
      table.proforma {
        border-collapse: collapse;
        font-family: 'Segoe UI', 'Calibri', Arial, sans-serif;
        font-size: 13px;
        white-space: nowrap;
        margin-top: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      }
      table.proforma th {
        background: #1f3a5c;
        color: #ffffff;
        font-weight: 700;
        font-style: italic;
        padding: 8px 12px;
        text-align: left;
        border-bottom: 2px solid #15293f;
      }
      table.proforma th.col-val { text-align: right; }
      table.proforma td {
        padding: 5px 12px;
        border-bottom: 1px solid #e8eaed;
      }
      table.proforma td.col-val {
        text-align: right;
        font-variant-numeric: tabular-nums;
      }
      table.proforma td.col-val.neg { color: #b3261e; }
      table.proforma tr.row-header td {
        background: #f1f3f4;
        font-weight: 700;
        color: #1f3a5c;
      }
      table.proforma tr.row-subbucket td.col-sub { padding-left: 28px; color: #444; }
      table.proforma tr.row-total td {
        background: #fafbfc;
        font-weight: 700;
        border-top: 1px solid #d0d4d9;
      }
      table.proforma tr.row-noi td {
        background: #e3edff;
        font-weight: 700;
        border-top: 2px solid #1f3a5c;
        border-bottom: 2px solid #1f3a5c;
      }
      table.proforma tr.row-results td {
        background: #e6f4ea;
        font-weight: 700;
        border-top: 2px solid #1f3a5c;
      }
      table.proforma td.col-bucket { font-weight: 600; min-width: 180px; }
      table.proforma td.col-sub    { min-width: 240px; }
      table.proforma td.col-val,
      table.proforma th.col-val    { min-width: 110px; }
      /* Monthly columns: very light blue cells, black text */
      table.proforma td.col-month { background: #eef4fb; color: #000; }
      table.proforma td.col-month.neg { color: #000; }
      /* Budget cells (only on monthly columns) get italics */
      table.proforma td.col-month.col-budget { font-style: italic; }
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
        return (" " + " ".join(classes)) if classes else ""

    def header_label(label):
        if label not in budget_cols:
            return label
        # Year-total columns mixing actuals + budget read as a forecast
        if label in year_total_cols:
            return label.replace(" Actuals", " Forecast")
        # Monthly budget columns keep the (B) suffix
        return f"{label} (B)"

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

st.title("RST Portfolio")

with st.spinner("Loading actuals + budgets from S3..."):
    try:
        actuals_raw = load_actuals()
        budgets_raw = load_budgets()
    except KeyError as e:
        st.error(f"Missing key in `.streamlit/secrets.toml`: {e}. "
                 "Expected sections [aws] and [s3].")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load data from S3: {e}")
        st.stop()

actuals = normalize(actuals_raw)
budgets = normalize(budgets_raw)
df = combine_actuals_budget(actuals, budgets)

# ---- Sidebar selectors ----
with st.sidebar:
    st.header("Filters")
    properties = ["All Properties"] + sorted(df["Property"].dropna().unique().tolist())
    selected_property = st.selectbox("Property", properties)

    if selected_property == "All Properties":
        year_source = df
    else:
        year_source = df[df["Property"] == selected_property]

    available_years = sorted(
        year_source["Date"].dt.year.dropna().unique().astype(int).tolist(),
        reverse=True,
    )
    default_years = available_years[:3] if len(available_years) >= 3 else available_years

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
st.caption("  |  ".join(caption_bits))
st.caption("Cells/columns marked **(B)** include budget data (italic monthly cells). Actuals are shown wherever they exist; budget fills in after the property's actuals cutoff.")

# ---- Proforma ----
period_columns = build_period_columns(years_to_show, years_to_expand)
budget_cols    = detect_budget_columns(df, period_columns, selected_property)
proforma_df, _ = build_proforma_wide(df, selected_property, period_columns)
render_proforma(proforma_df, period_columns, budget_cols)

# ---- Raw data sanity check (collapsible) ----
with st.expander("Raw data sanity check"):
    bucket = st.secrets["s3"]["bucket"]
    st.caption(f"Source files: s3://{bucket}/{st.secrets['s3']['key']} + financial_budgets.csv")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows (combined)", f"{len(df):,}")
    c2.metric("Actual rows", f"{(df['Source'] == 'Actual').sum():,}")
    c3.metric("Budget rows", f"{(df['Source'] == 'Budget').sum():,}")
    c4.metric("Unique properties", df["Property"].nunique())
    st.caption(
        f"Combined date range: {df['Date'].min().date()} - {df['Date'].max().date()}"
    )
    st.subheader("Sub Buckets per Main Bucket (after normalization)")
    pivot = (
        df.groupby(["Main Bucket", "RST Sub Bucket"], dropna=False)
          .size()
          .reset_index(name="rows")
          .sort_values(["Main Bucket", "RST Sub Bucket"])
    )
    st.dataframe(pivot, use_container_width=True)
