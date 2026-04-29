"""
Electrolyzer Whisker Plot App
SS316L Corrosion Study — Interactive Histogram + Box-Whisker
"""

import io, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Electrolyzer Whisker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  COLOUR SCHEME
# ─────────────────────────────────────────────────────────────
LC_COLOR  = "#C0392B"   # dark red
WJ_COLOR  = "#1F618D"   # dark navy
LC_FILL   = "rgba(245,190,185,0.45)"
WJ_FILL   = "rgba(185,215,240,0.45)"
LC_FILL_S = "rgba(245,190,185,0.15)"
WJ_FILL_S = "rgba(185,215,240,0.15)"

CUT_COLORS = {"LC": LC_COLOR, "WJ": WJ_COLOR}
CUT_FILLS  = {"LC": LC_FILL,  "WJ": WJ_FILL}
CUT_FILLS_S= {"LC": LC_FILL_S,"WJ": WJ_FILL_S}

# Condition order
COND_ORDER  = ["AC", "Brushed", "Pickled", "B&P", "BPP"]
PARAM_UNITS = {
    "OCP1":"V vs. RHE", "OCP2":"V vs. RHE", "OCP3":"V vs. RHE",
    "Ecorr":"V vs. RHE", "Icorr":"A dm⁻²", "Epp":"V vs. RHE",
}

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
SAMPLE_META = {
    "50": ("LC","AC"),   "51": ("LC","Brushed"), "52": ("LC","Pickled"), "53": ("LC","B&P"),
    "60": ("LC","AC"),   "61": ("LC","Brushed"), "62": ("LC","Pickled"),
    "63": ("LC","B&P"),  "64": ("LC","BPP"),
    "70": ("WJ","AC"),   "71": ("WJ","Brushed"), "72": ("WJ","Pickled"),
    "73": ("WJ","B&P"),  "74": ("WJ","BPP"),
}
PH_MAP = {
    "50": {"01":4,"05":1}, "51": {"02":4}, "52": {"03":1,"10":4}, "53": {"01":4,"04":1},
    "60": {"02":1,"10":4}, "61": {"02":4}, "62": {"03":4},
    "63": {"01":4,"03":1,"05":4}, "64": {"01":4,"05":4,"09":1},
    "70": {"02":4,"03":1}, "71": {"02":4}, "72": {"07":1,"10":4},
    "73": {"01":4,"03":1,"09":4}, "74": {"01":4,"03":4,"05":4,"06":1},
}


def parse_lsv(f):
    s   = re.search(r"Sample (\d+)", str(f))
    fol = re.search(r"Sample \d+\\(\d+)\\", str(f))
    t   = re.search(r"Test (\d+)", str(f))
    p   = re.search(r"Point (\d+)", str(f), re.IGNORECASE)
    return (s.group(1) if s else None, fol.group(1) if fol else None,
            t.group(1) if t else None, p.group(1) if p else None)


def parse_ocp(row):
    fp  = str(row["file_path"]); fn = str(row["file_name"])
    s   = re.search(r"Sample (\d+)", fp)
    fol = re.search(r"Sample \d+\\(\d+)\\", fp)
    t   = re.search(r"Test (\d+)", fp)
    p   = re.search(r"Point (\d+)", fp, re.IGNORECASE)
    ocp_n = re.search(r"ocp(\d*)", fn, re.IGNORECASE)
    n   = ocp_n.group(1) if ocp_n and ocp_n.group(1) else "1"
    dir_ = "Anodic" if "Anodic" in fp else ("Cathodic" if "Cathodic" in fp else "Both")
    return (s.group(1) if s else None, fol.group(1) if fol else None,
            t.group(1) if t else None, p.group(1) if p else None, n, dir_)


@st.cache_data
def load_from_uploaded(lsv_bytes, ocp_bytes):
    df      = pd.read_excel(io.BytesIO(lsv_bytes), sheet_name="LSV")
    ocp_raw = pd.read_excel(io.BytesIO(ocp_bytes), sheet_name="OCP_Summary")

    df[["sample","folder","test","point"]] = df["File"].apply(
        lambda x: pd.Series(parse_lsv(x)))
    df["sample"] = df["sample"].astype(str)
    df["folder"] = df["folder"].astype(str)

    ocp_raw[["sample","folder","test","point","ocp_num","direction"]] = ocp_raw.apply(
        parse_ocp, axis=1, result_type="expand")
    ocp_raw["sample"] = ocp_raw["sample"].astype(str)
    ocp_raw["folder"] = ocp_raw["folder"].astype(str)

    for frame in [df, ocp_raw]:
        frame["pH"]       = frame.apply(lambda r: PH_MAP.get(r["sample"],{}).get(r["folder"], None), axis=1)
        frame["cut"]      = frame["sample"].map(lambda s: SAMPLE_META.get(s,("",""))[0])
        frame["condition"]= frame["sample"].map(lambda s: SAMPLE_META.get(s,("",""))[1])

    # Build combined long dataframe
    rows = []
    param_map = {"Ecorr_fitted_V":"Ecorr","Icorr_abs":"Icorr","Epp_V":"Epp"}
    main_lsv  = df[df["sample"].isin(SAMPLE_META) & df["pH"].notna()].copy()

    for _, r in main_lsv.iterrows():
        for pcol, pname in param_map.items():
            if pd.isna(r[pcol]): continue
            rows.append({
                "sample": f"S{r['sample']}", "cut": r["cut"], "condition": r["condition"],
                "pH": int(r["pH"]), "direction": str(r["scan_direction"]),
                "parameter": pname, "value": float(r[pcol]),
                "r2": float(r["R2_log"]) if not pd.isna(r["R2_log"]) else None,
                "folder": r["folder"], "test": r["test"],
                "row_id": f"lsv_{r.name}_{pcol}",
            })

    main_ocp = ocp_raw[ocp_raw["sample"].isin(SAMPLE_META) & ocp_raw["pH"].notna()].copy()
    for _, r in main_ocp.iterrows():
        if pd.isna(r["last_voltage_v"]): continue
        n = str(r["ocp_num"]).strip()
        ocp_lbl = f"OCP{n}" if n else "OCP1"
        rows.append({
            "sample": f"S{r['sample']}", "cut": r["cut"], "condition": r["condition"],
            "pH": int(r["pH"]), "direction": str(r["direction"]),
            "parameter": ocp_lbl, "value": float(r["last_voltage_v"]),
            "r2": None, "folder": r["folder"], "test": r["test"],
            "row_id": f"ocp_{r.name}_{n}",
        })

    return pd.DataFrame(rows)


@st.cache_data
def load_prefilled():
    """Load from the pre-filled Excel if user hasn't uploaded."""
    try:
        df = pd.read_excel("ElectrolyzerWhisker_Final.xlsx", sheet_name="Data",
                           header=3, skiprows=[])
        # Parse the pre-filled workbook
        df.columns = ["include","sample_id","cut","condition","pH","direction",
                      "parameter","value","unit","folder","test","r2"] + list(df.columns[12:])
        df = df[df["include"]=="YES"].copy()
        df["row_id"] = [f"pre_{i}" for i in range(len(df))]
        df = df.rename(columns={"sample_id":"sample"})
        df["pH"] = pd.to_numeric(df["pH"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[df["value"].notna()].copy()
        return df[["sample","cut","condition","pH","direction","parameter","value","r2","row_id"]]
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────────────────────
def compute_stats(vals: np.ndarray, sd_mult: float):
    if len(vals) == 0:
        return None
    s   = np.sort(vals)
    n   = len(s)
    q1  = np.percentile(s, 25)
    med = np.percentile(s, 50)
    q3  = np.percentile(s, 75)
    mu  = s.mean()
    sd  = s.std(ddof=1) if n > 1 else 0
    lo  = mu - sd_mult * sd
    hi  = mu + sd_mult * sd
    clean = s[(s >= lo) & (s <= hi)]
    if len(clean) == 0:
        clean = s
    lo_whisk  = clean.min()
    hi_whisk  = clean.max()
    mean_clean= clean.mean()
    outliers  = s[(s < lo_whisk) | (s > hi_whisk)]
    return {
        "n": n, "q1": q1, "med": med, "q3": q3,
        "lo": lo_whisk, "hi": hi_whisk,
        "mean": mean_clean, "sd": sd,
        "outliers": outliers, "clean": clean,
    }


# ─────────────────────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────────────────────
def make_figure(df: pd.DataFrame, param: str, pH_filter, dir_filter: str,
                r2_min: float, sd_mult: float, log_icorr: bool,
                deleted_ids: set, sample_filter: str) -> go.Figure:

    # Filter
    sub = df[df["parameter"] == param].copy()

    is_ocp = param.startswith("OCP")
    if not is_ocp:
        if dir_filter != "Both":
            sub = sub[sub["direction"].str.upper() == dir_filter.upper()]
        if sub["r2"].notna().any():
            sub = sub[(sub["r2"].isna()) | (sub["r2"] >= r2_min)]

    if pH_filter != "Both":
        sub = sub[sub["pH"] == int(pH_filter)]

    if sample_filter:
        sub = sub[sub["sample"].str.upper() == sample_filter.upper()]

    # Remove deleted
    sub = sub[~sub["row_id"].isin(deleted_ids)].copy()

    # Log transform
    if param == "Icorr" and log_icorr:
        sub = sub[sub["value"] > 0].copy()
        sub["value"] = np.log10(sub["value"])

    # Y axis label
    unit = PARAM_UNITS.get(param, "")
    if param == "Icorr" and log_icorr:
        ylabel = f"log₁₀(icorr)  [{unit}]"
    else:
        ylabel = f"{param}  [{unit}]"

    # Conditions present
    conds = [c for c in COND_ORDER if c in sub["condition"].unique()]
    cuts  = ["LC", "WJ"]

    n_conds = len(conds)
    if n_conds == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected filters",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font_size=16)
        return fig

    # Layout: (n_conds) columns, each with histogram+box side by side
    # We use a single plot with manual x positioning
    fig = go.Figure()

    # Global Y range
    all_vals = sub["value"].dropna().values
    if len(all_vals) == 0:
        fig.add_annotation(text="No data after filters",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font_size=16)
        return fig

    y_min = all_vals.min(); y_max = all_vals.max()
    y_pad = (y_max - y_min) * 0.12 if y_max > y_min else 0.5
    y_lo  = y_min - y_pad; y_hi = y_max + y_pad

    # X positioning
    # Each condition gets a slot of width 1
    # Within each slot: LC histogram (left), WJ histogram (right), shared box (far right)
    slot_w  = 1.0
    hist_w  = 0.18   # each cut's histogram half-width
    box_w   = 0.12   # box half-width

    tick_vals, tick_text = [], []
    added_legend = set()

    for ci, cond in enumerate(conds):
        cx = ci * slot_w   # slot centre

        # Condition label (via annotation — added outside this loop below)
        tick_vals.append(cx)
        tick_text.append(cond)

        # Draw per-cut
        for ki, cut in enumerate(cuts):
            offset = -0.25 if ki == 0 else 0.25  # LC left, WJ right in histogram zone
            hx     = cx + offset
            bx     = cx + 0.70 + ki * 0.25       # box X position

            colour = CUT_COLORS[cut]
            cfill  = CUT_FILLS[cut]
            cfill_s= CUT_FILLS_S[cut]

            csub = sub[(sub["condition"] == cond) & (sub["cut"] == cut)]
            vals = csub["value"].dropna().values

            if len(vals) == 0:
                continue

            st_  = compute_stats(vals, sd_mult)
            if st_ is None:
                continue

            show_leg = cut not in added_legend
            added_legend.add(cut)

            # ── Histogram (horizontal bars) ────────────────────
            bins  = np.linspace(y_lo, y_hi, 11)
            counts, edges = np.histogram(vals, bins=bins)
            max_c = counts.max() if counts.max() > 0 else 1
            scale = hist_w * 0.9

            for j, c in enumerate(counts):
                if c == 0: continue
                bar_y_lo = edges[j]; bar_y_hi = edges[j+1]
                bar_len  = (c / max_c) * scale
                bar_x_lo = hx - bar_len; bar_x_hi = hx

                fig.add_shape(type="rect",
                    x0=bar_x_lo, x1=bar_x_hi,
                    y0=bar_y_lo, y1=bar_y_hi,
                    fillcolor=cfill, line=dict(color=colour, width=0.6),
                    layer="below")

            # Histogram baseline
            fig.add_shape(type="line",
                x0=hx, x1=hx, y0=y_lo, y1=y_hi,
                line=dict(color=colour, width=0.7, dash="dot"))

            # ── X marks (raw data points) ──────────────────────
            n = len(vals)
            jitter_x = np.linspace(hx - scale*0.6, hx - scale*0.05, n)
            np.random.shuffle(jitter_x)

            fig.add_trace(go.Scatter(
                x=jitter_x, y=vals,
                mode="markers",
                marker=dict(symbol="x", size=5, color=colour,
                            line=dict(color=colour, width=1.2)),
                name=cut if show_leg else None,
                legendgroup=cut,
                showlegend=show_leg,
                customdata=csub["row_id"].values,
                hovertemplate=(
                    f"<b>{cut} — {cond}</b><br>"
                    f"Value: %{{y:.4f}}<br>"
                    "Click to delete<extra></extra>"
                ),
            ))

            # ── Box-whisker ────────────────────────────────────
            # Upper whisker
            fig.add_shape(type="line",
                x0=bx, x1=bx, y0=st_["q3"], y1=st_["hi"],
                line=dict(color=colour, width=1.4))
            fig.add_shape(type="line",
                x0=bx-box_w*0.5, x1=bx+box_w*0.5, y0=st_["hi"], y1=st_["hi"],
                line=dict(color=colour, width=1.6))

            # Lower whisker
            fig.add_shape(type="line",
                x0=bx, x1=bx, y0=st_["lo"], y1=st_["q1"],
                line=dict(color=colour, width=1.4))
            fig.add_shape(type="line",
                x0=bx-box_w*0.5, x1=bx+box_w*0.5, y0=st_["lo"], y1=st_["lo"],
                line=dict(color=colour, width=1.6))

            # IQR box
            fig.add_shape(type="rect",
                x0=bx-box_w, x1=bx+box_w,
                y0=st_["q1"], y1=st_["q3"],
                fillcolor=cfill, line=dict(color=colour, width=1.6))

            # Median line
            fig.add_shape(type="line",
                x0=bx-box_w, x1=bx+box_w, y0=st_["med"], y1=st_["med"],
                line=dict(color=colour, width=2.5))

            # Mean marker
            fig.add_trace(go.Scatter(
                x=[bx], y=[st_["mean"]],
                mode="markers",
                marker=dict(symbol="circle-open", size=9, color=colour,
                            line=dict(color=colour, width=1.8)),
                legendgroup=cut, showlegend=False,
                hovertemplate=f"<b>Mean ({cut})</b>: %{{y:.4f}}<extra></extra>",
            ))

            # Outlier dots
            if len(st_["outliers"]) > 0:
                fig.add_trace(go.Scatter(
                    x=[bx] * len(st_["outliers"]),
                    y=st_["outliers"],
                    mode="markers",
                    marker=dict(symbol="circle-open", size=8, color=colour,
                                line=dict(color=colour, width=1.4)),
                    legendgroup=cut, showlegend=False,
                    hovertemplate=f"<b>Outlier ({cut})</b>: %{{y:.4f}}<extra></extra>",
                ))

            # n= annotation
            fig.add_annotation(
                x=bx, y=y_lo - y_pad * 0.4,
                text=f"n={n}", showarrow=False,
                font=dict(size=9, color=colour),
                xanchor="center",
            )

    # pH label in subtitle
    pH_str = f"pH {pH_filter}" if pH_filter != "Both" else "pH 1 & 4"
    dir_str = f" | {dir_filter}" if not is_ocp and dir_filter != "Both" else ""
    sample_str = f" | {sample_filter}" if sample_filter else ""

    fig.update_layout(
        title=dict(
            text=f"<b>{param}</b>  |  {pH_str}{dir_str}{sample_str}",
            font=dict(size=15, color="#2E4057"),
            x=0.5,
        ),
        xaxis=dict(
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=11, color="#2E4057"),
            showgrid=False,
            zeroline=False,
            range=[-0.55, n_conds * slot_w - 0.45 + 0.7],
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(size=11)),
            range=[y_lo - y_pad*0.6, y_hi + y_pad*0.2],
            gridcolor="#E8EDF5",
            gridwidth=0.7,
            zeroline=False,
        ),
        legend=dict(
            title="Cut type",
            orientation="v",
            x=1.01, y=0.98,
            bgcolor="rgba(248,250,252,0.9)",
            bordercolor="#C8D0DC",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        margin=dict(l=70, r=140, t=60, b=60),
        clickmode="event",
    )

    return fig


# ─────────────────────────────────────────────────────────────
#  STREAMLIT LAYOUT
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    h1 { color: #2E4057; font-size: 1.6rem; }
    .stSelectbox label, .stSlider label, .stMultiSelect label { font-weight: 600; }
    .deleted-badge { background:#C0392B; color:white; padding:2px 8px;
                     border-radius:12px; font-size:0.8rem; margin:2px; display:inline-block; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 Electrolyzer Whisker — SS316L Corrosion Study")
    st.caption("Histogram + Box-Whisker | Click data points on the plot to exclude them | Stats recalculate instantly")

    # ── Session state ─────────────────────────────────────────
    if "deleted_ids"  not in st.session_state: st.session_state.deleted_ids  = set()
    if "active_param" not in st.session_state: st.session_state.active_param = "OCP1"
    if "df"           not in st.session_state: st.session_state.df           = None

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.header("📂 Data Source")

        data_source = st.radio("Load from:", ["Pre-filled Excel", "Upload raw files"], index=0)

        if data_source == "Upload raw files":
            lsv_file = st.file_uploader("batch_fit_summary.xlsx", type=["xlsx"])
            ocp_file = st.file_uploader("ocp_summary_samples.xlsx", type=["xlsx"])

            if lsv_file and ocp_file:
                with st.spinner("Parsing data..."):
                    df = load_from_uploaded(lsv_file.read(), ocp_file.read())
                st.session_state.df = df
                st.success(f"Loaded {len(df):,} rows")
        else:
            if st.session_state.df is None:
                with st.spinner("Loading pre-filled data..."):
                    df = load_prefilled()
                if df is not None:
                    st.session_state.df = df
                    st.success(f"Loaded {len(df):,} rows")
                else:
                    st.info("Upload files above or place ElectrolyzerWhisker_Final.xlsx in the app folder.")

        if st.session_state.df is None:
            st.stop()

        df = st.session_state.df

        st.divider()
        st.header("⚙ Filters")

        params_available = sorted(df["parameter"].unique())
        param = st.selectbox("Parameter", params_available,
                             index=params_available.index("OCP1") if "OCP1" in params_available else 0)
        st.session_state.active_param = param

        pH_opts = ["Both"] + sorted([str(p) for p in df["pH"].dropna().unique()])
        pH_filter = st.selectbox("pH", pH_opts)

        is_ocp = param.startswith("OCP")
        if is_ocp:
            dir_filter = "Both"
        else:
            dir_filter = st.selectbox("Direction", ["Both", "Anodic", "Cathodic"])

        sample_opts = [""] + sorted(df["sample"].unique())
        sample_filter = st.selectbox("Sample (blank = all)", sample_opts)

        st.divider()
        st.header("📐 Statistics")

        r2_min   = st.slider("R² minimum", 0.0, 1.0, 0.90, 0.01)
        sd_mult  = st.slider("Outlier ×StDev", 1.0, 4.0, 2.0, 0.5)
        log_icorr = st.checkbox("Log₁₀ Icorr", value=True)

        st.divider()
        st.header("🗑 Deleted Points")
        n_del = len(st.session_state.deleted_ids)
        if n_del > 0:
            st.write(f"**{n_del}** point(s) excluded")
            if st.button("↺ Restore all", use_container_width=True):
                st.session_state.deleted_ids = set()
                st.rerun()
        else:
            st.caption("None — click plot points to exclude")

        # Download current data
        st.divider()
        filtered = df[~df["row_id"].isin(st.session_state.deleted_ids)]
        csv = filtered.to_csv(index=False).encode()
        st.download_button("⬇ Download filtered data (CSV)",
                           data=csv, file_name="electrolyzer_filtered.csv",
                           mime="text/csv", use_container_width=True)

    # ── Main area ─────────────────────────────────────────────
    df = st.session_state.df

    # Stats summary bar
    col1, col2, col3, col4 = st.columns(4)
    active_df = df[~df["row_id"].isin(st.session_state.deleted_ids)]
    active_df = active_df[active_df["parameter"] == param]
    col1.metric("Total points (param)", len(df[df["parameter"]==param]))
    col2.metric("Active points", len(active_df))
    col3.metric("Excluded", len(st.session_state.deleted_ids))
    col4.metric("Conditions", active_df["condition"].nunique())

    # Plot
    fig = make_figure(
        df=df, param=param, pH_filter=pH_filter, dir_filter=dir_filter,
        r2_min=r2_min, sd_mult=sd_mult, log_icorr=log_icorr,
        deleted_ids=st.session_state.deleted_ids,
        sample_filter=sample_filter,
    )

    # Render with click events
    event = st.plotly_chart(fig, use_container_width=True,
                            on_select="rerun", key=f"chart_{param}_{pH_filter}_{dir_filter}")

    # Handle click events — delete clicked points
    if event and event.get("selection") and event["selection"].get("points"):
        pts = event["selection"]["points"]
        for pt in pts:
            cdata = pt.get("customdata")
            if cdata is not None:
                rid = str(cdata) if not isinstance(cdata, list) else str(cdata[0])
                if rid and rid not in st.session_state.deleted_ids:
                    st.session_state.deleted_ids.add(rid)
                    st.rerun()

    st.caption("💡 **Click any X mark** on the plot to exclude that data point. "
               "The statistics and box-whisker will recalculate immediately. "
               "Use **Restore all** in the sidebar to bring points back.")

    # ── Statistics table ──────────────────────────────────────
    st.subheader("📋 Statistics Table")

    active = df[~df["row_id"].isin(st.session_state.deleted_ids)]
    active = active[active["parameter"] == param]
    if pH_filter != "Both":
        active = active[active["pH"] == int(pH_filter)]
    if not is_ocp and dir_filter != "Both":
        active = active[active["direction"].str.upper() == dir_filter.upper()]
    if sample_filter:
        active = active[active["sample"].str.upper() == sample_filter.upper()]
    if not is_ocp:
        active = active[(active["r2"].isna()) | (active["r2"] >= r2_min)]
    if param == "Icorr" and log_icorr:
        active = active[active["value"] > 0].copy()
        active["value"] = np.log10(active["value"])

    stat_rows = []
    for cond in COND_ORDER:
        for cut in ["LC","WJ"]:
            vals = active[(active["condition"]==cond)&(active["cut"]==cut)]["value"].dropna().values
            if len(vals) == 0: continue
            st_ = compute_stats(vals, sd_mult)
            if st_ is None: continue
            stat_rows.append({
                "Condition": cond, "Cut": cut, "n": st_["n"],
                "n_outlier": len(st_["outliers"]),
                "Min": round(st_["clean"].min(),5),
                "Q1": round(st_["q1"],5), "Median": round(st_["med"],5),
                "Q3": round(st_["q3"],5), "Max": round(st_["clean"].max(),5),
                "Mean_clean": round(st_["mean"],5), "StDev": round(st_["sd"],5),
                "Lo_Whisker": round(st_["lo"],5), "Hi_Whisker": round(st_["hi"],5),
            })

    if stat_rows:
        stat_df = pd.DataFrame(stat_rows)
        def color_cut(row):
            color = "#FAD7D3" if row["Cut"] == "LC" else "#D0E8F7"
            return [f"background-color:{color}" if i==1 else "" for i in range(len(row))]
        st.dataframe(stat_df.style.apply(color_cut, axis=1), use_container_width=True, height=280)
    else:
        st.info("No data for selected filters.")


if __name__ == "__main__":    
    main()
