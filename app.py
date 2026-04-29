"""
Electrolyzer Whisker Plot App  v3.0
SS316L Corrosion Study
Layout: One subplot per condition — each with independent Y-axis
        LEFT half  = histogram bars (LC + WJ)
        RIGHT half = box-whisker + raw X marks (LC + WJ)
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
#  COLOURS
# ─────────────────────────────────────────────────────────────
LC_COL   = "#C0392B"
WJ_COL   = "#1F618D"
LC_FILL  = "rgba(220,100,80,0.18)"
WJ_FILL  = "rgba(50,130,180,0.18)"
LC_FILL2 = "rgba(220,100,80,0.50)"
WJ_FILL2 = "rgba(50,130,180,0.50)"

CUT_COL   = {"LC": LC_COL,   "WJ": WJ_COL}
CUT_FILL  = {"LC": LC_FILL,  "WJ": WJ_FILL}
CUT_FILL2 = {"LC": LC_FILL2, "WJ": WJ_FILL2}

COND_ORDER  = ["AC", "Brushed", "Pickled", "B&P", "BPP"]
PARAM_UNITS = {
    "OCP1": "V vs. RHE", "OCP2": "V vs. RHE", "OCP3": "V vs. RHE",
    "Ecorr": "V vs. RHE", "Icorr": "A dm⁻²",  "Epp":  "V vs. RHE",
}

# ─────────────────────────────────────────────────────────────
#  METADATA
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

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
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
def load_uploaded(lsv_bytes, ocp_bytes):
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
        frame["pH"]        = frame.apply(
            lambda r: PH_MAP.get(r["sample"],{}).get(r["folder"], None), axis=1)
        frame["cut"]       = frame["sample"].map(
            lambda s: SAMPLE_META.get(s,("",""))[0])
        frame["condition"] = frame["sample"].map(
            lambda s: SAMPLE_META.get(s,("",""))[1])

    rows = []
    param_map = {"Ecorr_fitted_V":"Ecorr","Icorr_abs":"Icorr","Epp_V":"Epp"}
    main_lsv  = df[df["sample"].isin(SAMPLE_META) & df["pH"].notna()].copy()

    for _, r in main_lsv.iterrows():
        for pcol, pname in param_map.items():
            if pd.isna(r[pcol]): continue
            rows.append({
                "sample": f"S{r['sample']}", "cut": r["cut"],
                "condition": r["condition"],
                "pH": int(r["pH"]), "direction": str(r["scan_direction"]),
                "parameter": pname, "value": float(r[pcol]),
                "r2": float(r["R2_log"]) if not pd.isna(r["R2_log"]) else None,
                "row_id": f"lsv_{r.name}_{pcol}",
            })

    main_ocp = ocp_raw[ocp_raw["sample"].isin(SAMPLE_META) & ocp_raw["pH"].notna()].copy()
    for _, r in main_ocp.iterrows():
        if pd.isna(r["last_voltage_v"]): continue
        n = str(r["ocp_num"]).strip()
        ocp_lbl = f"OCP{n}" if n else "OCP1"
        rows.append({
            "sample": f"S{r['sample']}", "cut": r["cut"],
            "condition": r["condition"],
            "pH": int(r["pH"]), "direction": str(r["direction"]),
            "parameter": ocp_lbl, "value": float(r["last_voltage_v"]),
            "r2": None, "row_id": f"ocp_{r.name}_{n}",
        })

    return pd.DataFrame(rows)

@st.cache_data
def load_prefilled():
    try:
        raw = pd.read_excel("ElectrolyzerWhisker_Final.xlsx",
                            sheet_name="Data", header=3)
        raw.columns = (
            ["include","sample","cut","condition","pH","direction",
             "parameter","value","unit","folder","test","r2"]
            + list(raw.columns[12:])
        )
        raw = raw[raw["include"] == "YES"].copy()
        raw["row_id"]  = [f"pre_{i}" for i in range(len(raw))]
        raw["pH"]      = pd.to_numeric(raw["pH"],    errors="coerce")
        raw["value"]   = pd.to_numeric(raw["value"], errors="coerce")
        raw["r2"]      = pd.to_numeric(raw["r2"],    errors="coerce")
        raw            = raw[raw["value"].notna()]
        return raw[["sample","cut","condition","pH","direction",
                    "parameter","value","r2","row_id"]].copy()
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────────────────────
def compute_stats(vals: np.ndarray, sd_mult: float):
    if len(vals) == 0:
        return None
    s    = np.sort(vals)
    n    = len(s)
    q1   = np.percentile(s, 25)
    med  = np.percentile(s, 50)
    q3   = np.percentile(s, 75)
    mu   = s.mean()
    sd   = s.std(ddof=1) if n > 1 else 0.0
    lo   = mu - sd_mult * sd
    hi   = mu + sd_mult * sd
    clean    = s[(s >= lo) & (s <= hi)]
    if len(clean) == 0:
        clean = s
    lo_w     = clean.min()
    hi_w     = clean.max()
    outliers = s[(s < lo_w) | (s > hi_w)]
    return dict(n=n, q1=q1, med=med, q3=q3,
                lo=lo_w, hi=hi_w,
                mean=clean.mean(), sd=sd,
                outliers=outliers, clean=clean, raw=s)

# ─────────────────────────────────────────────────────────────
#  DRAW ONE CONDITION PANEL onto a subplot
#  Each subplot has its own Y-axis (independent range).
#  X layout within one subplot (x range 0..1):
#    [0.00 – 0.48]  LC histogram  (bars grow rightward from 0)
#    [0.52 – 1.00]  WJ histogram  (bars grow rightward from 0.52)
#    Divider at x=0.5
#    Box-whisker drawn OVER histogram with X marks on top
# ─────────────────────────────────────────────────────────────
def draw_condition_panel(fig, row, col,
                         lc_vals, lc_rids,
                         wj_vals, wj_rids,
                         cond, sd_mult, nbins,
                         legend_done, show_ylabel, ylabel):
    """
    Draw histogram + box-whisker for one condition into subplot (row, col).
    Returns updated legend_done set.
    """

    # Combine all vals for this condition to set shared Y range
    all_vals = np.concatenate([
        lc_vals if len(lc_vals) > 0 else np.array([]),
        wj_vals if len(wj_vals) > 0 else np.array([])
    ])

    if len(all_vals) == 0:
        return legend_done

    y_min = all_vals.min(); y_max = all_vals.max()
    y_rng = max(y_max - y_min, 1e-9)
    y_pad = y_rng * 0.14
    y_lo  = y_min - y_pad
    y_hi  = y_max + y_pad

    bin_edges = np.linspace(y_lo, y_hi, nbins + 1)

    # X positions (within 0..1 normalised space per subplot):
    # LC: histogram bars from x=0 rightward, max width 0.40
    # WJ: histogram bars from x=0.50 rightward, max width 0.40
    # LC box centred at x=0.20, WJ box centred at x=0.70
    lc_hist_x0  = 0.00; lc_hist_maxw = 0.40
    wj_hist_x0  = 0.50; wj_hist_maxw = 0.40
    lc_box_cx   = 0.20
    wj_box_cx   = 0.70
    box_hw      = 0.14   # half-width of IQR box

    for cut, vals, rids, hist_x0, hist_maxw, box_cx in [
        ("LC", lc_vals, lc_rids, lc_hist_x0, lc_hist_maxw, lc_box_cx),
        ("WJ", wj_vals, wj_rids, wj_hist_x0, wj_hist_maxw, wj_box_cx),
    ]:
        col_c  = CUT_COL[cut]
        fill   = CUT_FILL[cut]
        fill2  = CUT_FILL2[cut]
        show_leg = cut not in legend_done

        if len(vals) == 0:
            if show_leg:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    name=cut, marker=dict(color=col_c, size=8),
                    legendgroup=cut, showlegend=True,
                ), row=row, col=col)
                legend_done.add(cut)
            continue

        # ── Histogram bars ─────────────────────────────────
        counts, _ = np.histogram(vals, bins=bin_edges)
        max_c = counts.max() if counts.max() > 0 else 1

        for bi in range(nbins):
            c = counts[bi]
            if c == 0:
                continue
            bar_w  = (c / max_c) * hist_maxw
            by_lo  = bin_edges[bi]
            by_hi  = bin_edges[bi + 1]

            # Draw as a filled rectangle using a scatter fill trick
            # We use go.Bar with horizontal orientation mapped to the subplot
            # Actually we use shapes — but shapes need paper/axis ref.
            # Use go.Bar with base for horizontal bars:
            fig.add_trace(go.Bar(
                x=[bar_w],
                y=[(by_lo + by_hi) / 2],
                base=[hist_x0],
                orientation="h",
                width=[(by_hi - by_lo) * 0.88],
                marker=dict(color=fill2, line=dict(color=col_c, width=0.7)),
                legendgroup=cut,
                showlegend=False,
                hoverinfo="skip",
            ), row=row, col=col)

        # Thin baseline for histogram
        fig.add_trace(go.Scatter(
            x=[hist_x0 + hist_maxw * 0.01, hist_x0 + hist_maxw * 0.01],
            y=[y_lo, y_hi],
            mode="lines",
            line=dict(color=col_c, width=0.8, dash="dot"),
            legendgroup=cut, showlegend=False,
            hoverinfo="skip",
        ), row=row, col=col)

        # ── Box-whisker ─────────────────────────────────────
        st_ = compute_stats(vals, sd_mult)
        if st_ is None:
            continue

        # Whisker lines — upper
        fig.add_trace(go.Scatter(
            x=[box_cx, box_cx], y=[st_["q3"], st_["hi"]],
            mode="lines", line=dict(color=col_c, width=1.6),
            legendgroup=cut, showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        # Upper cap
        fig.add_trace(go.Scatter(
            x=[box_cx - box_hw*0.6, box_cx + box_hw*0.6],
            y=[st_["hi"], st_["hi"]],
            mode="lines", line=dict(color=col_c, width=1.9),
            legendgroup=cut, showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        # Whisker lower
        fig.add_trace(go.Scatter(
            x=[box_cx, box_cx], y=[st_["lo"], st_["q1"]],
            mode="lines", line=dict(color=col_c, width=1.6),
            legendgroup=cut, showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        # Lower cap
        fig.add_trace(go.Scatter(
            x=[box_cx - box_hw*0.6, box_cx + box_hw*0.6],
            y=[st_["lo"], st_["lo"]],
            mode="lines", line=dict(color=col_c, width=1.9),
            legendgroup=cut, showlegend=False, hoverinfo="skip",
        ), row=row, col=col)

        # IQR box — draw as two vertical lines + fill using scatter fill
        box_h = st_["q3"] - st_["q1"]
        # Box outline
        box_xs = [box_cx-box_hw, box_cx+box_hw, box_cx+box_hw,
                  box_cx-box_hw, box_cx-box_hw]
        box_ys = [st_["q1"], st_["q1"], st_["q3"], st_["q3"], st_["q1"]]
        fig.add_trace(go.Scatter(
            x=box_xs, y=box_ys,
            mode="lines", fill="toself",
            fillcolor=fill,
            line=dict(color=col_c, width=1.8),
            legendgroup=cut, showlegend=False, hoverinfo="skip",
        ), row=row, col=col)

        # Median line
        fig.add_trace(go.Scatter(
            x=[box_cx - box_hw, box_cx + box_hw],
            y=[st_["med"], st_["med"]],
            mode="lines", line=dict(color=col_c, width=2.8),
            legendgroup=cut, showlegend=False, hoverinfo="skip",
        ), row=row, col=col)

        # Mean marker (open circle)
        fig.add_trace(go.Scatter(
            x=[box_cx], y=[st_["mean"]],
            mode="markers",
            marker=dict(symbol="circle-open", size=10, color=col_c,
                        line=dict(color=col_c, width=2.2)),
            legendgroup=cut, showlegend=False,
            hovertemplate=f"<b>Mean ({cut})</b>: %{{y:.5f}}<extra></extra>",
        ), row=row, col=col)

        # Outlier dots (open circles)
        if len(st_["outliers"]) > 0:
            fig.add_trace(go.Scatter(
                x=[box_cx] * len(st_["outliers"]),
                y=st_["outliers"].tolist(),
                mode="markers",
                marker=dict(symbol="circle-open", size=9, color=col_c,
                            line=dict(color=col_c, width=1.5)),
                legendgroup=cut, showlegend=False,
                hovertemplate=f"<b>Outlier ({cut})</b>: %{{y:.5f}}<extra></extra>",
            ), row=row, col=col)

        # ── Raw X marks overlaid on box ─────────────────────
        n_pts = len(vals)
        np.random.seed(42)
        jitter = np.random.uniform(-box_hw * 0.55, box_hw * 0.55, n_pts)

        fig.add_trace(go.Scatter(
            x=(box_cx + jitter).tolist(),
            y=vals.tolist(),
            mode="markers",
            marker=dict(symbol="x", size=6, color=col_c,
                        line=dict(color=col_c, width=1.5), opacity=0.80),
            name=cut if show_leg else None,
            legendgroup=cut,
            showlegend=show_leg,
            customdata=rids,
            hovertemplate=(
                f"<b>{cut} — {cond}</b><br>"
                "Value: %{y:.5f}<br>"
                "<i>Click to exclude</i><extra></extra>"
            ),
        ), row=row, col=col)

        if show_leg:
            legend_done.add(cut)

        # n= annotation — use subplot annotation
        fig.add_annotation(
            x=box_cx, y=y_lo,
            xref=f"x{col}", yref=f"y{col}",
            text=f"n={st_['n']}",
            showarrow=False,
            font=dict(size=8.5, color=col_c),
            xanchor="center", yanchor="top",
            yshift=-4,
        )

    # Mid divider line
    fig.add_trace(go.Scatter(
        x=[0.48, 0.48], y=[y_lo, y_hi],
        mode="lines",
        line=dict(color="#9AAABB", width=1.0),
        legendgroup="__div__", showlegend=False, hoverinfo="skip",
    ), row=row, col=col)

    # Update this subplot's axes
    axis_idx = col  # 1-based
    xaxis_key = f"xaxis{axis_idx}"
    yaxis_key = f"yaxis{axis_idx}"

    fig.update_xaxes(
        range=[-0.03, 1.03],
        showgrid=False, zeroline=False,
        tickvals=[0.20, 0.70],
        ticktext=["LC", "WJ"],
        tickfont=dict(size=9, color="#445566"),
        title_text=cond,
        title_font=dict(size=10, color="#2E4057"),
        row=row, col=col,
    )
    fig.update_yaxes(
        range=[y_lo - y_pad*0.6, y_hi + y_pad*0.2],
        gridcolor="#E8EDF5", gridwidth=0.8,
        zeroline=True, zerolinecolor="#CCCCCC", zerolinewidth=0.8,
        title_text=ylabel if show_ylabel else "",
        title_font=dict(size=10),
        showticklabels=True,
        tickfont=dict(size=8),
        row=row, col=col,
    )

    return legend_done


# ─────────────────────────────────────────────────────────────
#  MAKE FULL FIGURE  (one subplot per condition)
# ─────────────────────────────────────────────────────────────
def make_figure(df, param, pH_filter, dir_filter,
                r2_min, sd_mult, log_icorr,
                deleted_ids, sample_filter):

    is_ocp = param.startswith("OCP")

    # ── Filter ───────────────────────────────────────────────
    sub = df[df["parameter"] == param].copy()
    if not is_ocp:
        if dir_filter != "Both":
            sub = sub[sub["direction"].str.upper() == dir_filter.upper()]
        sub = sub[(sub["r2"].isna()) | (sub["r2"] >= r2_min)]
    if pH_filter != "Both":
        sub = sub[sub["pH"] == int(pH_filter)]
    if sample_filter:
        sub = sub[sub["sample"].str.upper() == sample_filter.upper()]
    sub = sub[~sub["row_id"].isin(deleted_ids)].copy()

    if param == "Icorr" and log_icorr:
        sub = sub[sub["value"] > 0].copy()
        sub["value"] = np.log10(sub["value"])

    unit   = PARAM_UNITS.get(param, "")
    ylabel = (f"log₁₀(icorr)  [{unit}]"
              if (param == "Icorr" and log_icorr)
              else f"{param}  [{unit}]")

    conds = [c for c in COND_ORDER if c in sub["condition"].unique()]
    n_c   = len(conds)

    if n_c == 0 or sub.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected filters",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font_size=16, font_color="#888")
        fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white")
        return fig

    # ── Create subplots — one per condition, shared row ──────
    fig = make_subplots(
        rows=1, cols=n_c,
        shared_yaxes=False,     # ← INDEPENDENT Y-axis per condition
        horizontal_spacing=0.06,
        subplot_titles=[""] * n_c,  # we set titles via xaxis title
    )

    nbins       = 10
    legend_done = set()

    pH_str     = f"pH {pH_filter}" if pH_filter != "Both" else "pH 1 & 4"
    dir_str    = f" | {dir_filter}" if not is_ocp and dir_filter != "Both" else ""
    sample_str = f" | {sample_filter}" if sample_filter else ""

    for ci, cond in enumerate(conds):
        col_idx = ci + 1   # 1-based

        lc_sub = sub[(sub["condition"] == cond) & (sub["cut"] == "LC")]
        wj_sub = sub[(sub["condition"] == cond) & (sub["cut"] == "WJ")]

        lc_vals = lc_sub["value"].dropna().values
        wj_vals = wj_sub["value"].dropna().values
        lc_rids = lc_sub["row_id"].values
        wj_rids = wj_sub["row_id"].values

        legend_done = draw_condition_panel(
            fig=fig, row=1, col=col_idx,
            lc_vals=lc_vals, lc_rids=lc_rids,
            wj_vals=wj_vals, wj_rids=wj_rids,
            cond=cond, sd_mult=sd_mult, nbins=nbins,
            legend_done=legend_done,
            show_ylabel=(ci == 0),
            ylabel=ylabel,
        )

    # ── Figure-level layout ───────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{param}</b>  |  {pH_str}{dir_str}{sample_str}"
                 "<br><sup style='color:#777;font-size:10px'>"
                 "Each panel has independent Y-axis  ·  "
                 "Left = histogram  ·  Right = box-whisker + X marks</sup>",
            font=dict(size=15, color="#2E4057"),
            x=0.5,
        ),
        legend=dict(
            title=dict(text="Cut type", font=dict(size=10)),
            orientation="v",
            x=1.01, y=0.95,
            bgcolor="rgba(248,250,252,0.92)",
            bordercolor="#C0CAD8",
            borderwidth=1,
            font=dict(size=11),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=480,
        margin=dict(l=70, r=110, t=80, b=70),
        barmode="overlay",
        clickmode="event",
    )

    return fig


# ─────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    h1 { color: #2E4057; font-size: 1.55rem; }
    </style>""", unsafe_allow_html=True)

    st.title("📊 Electrolyzer Whisker — SS316L Corrosion Study")
    st.caption(
        "Each condition = independent subplot with its own Y-axis  ·  "
        "Left = histogram  ·  Right = box-whisker + raw X marks  ·  "
        "Click X marks to exclude points"
    )

    # ── Session state ─────────────────────────────────────────
    if "deleted_ids" not in st.session_state:
        st.session_state.deleted_ids = set()
    if "df" not in st.session_state:
        st.session_state.df = None

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.header("📂 Data source")
        src = st.radio("Load from:", ["Pre-filled Excel", "Upload raw files"])

        if src == "Upload raw files":
            lf = st.file_uploader("batch_fit_summary.xlsx",   type=["xlsx"])
            of = st.file_uploader("ocp_summary_samples.xlsx", type=["xlsx"])
            if lf and of:
                with st.spinner("Parsing…"):
                    st.session_state.df = load_uploaded(lf.read(), of.read())
                st.success(f"{len(st.session_state.df):,} rows loaded")
        else:
            if st.session_state.df is None:
                with st.spinner("Loading…"):
                    st.session_state.df = load_prefilled()
                if st.session_state.df is not None:
                    st.success(f"{len(st.session_state.df):,} rows loaded")
                else:
                    st.warning(
                        "Place ElectrolyzerWhisker_Final.xlsx in the app folder, "
                        "or upload files above."
                    )

        if st.session_state.df is None:
            st.stop()

        df = st.session_state.df

        st.divider()
        st.header("⚙ Filters")

        params_av = sorted(df["parameter"].unique())
        param = st.selectbox(
            "Parameter", params_av,
            index=params_av.index("OCP1") if "OCP1" in params_av else 0
        )

        pH_opts   = ["Both"] + sorted(str(p) for p in df["pH"].dropna().unique())
        pH_filter = st.selectbox("pH", pH_opts)

        is_ocp     = param.startswith("OCP")
        dir_filter = ("Both" if is_ocp
                      else st.selectbox("Direction", ["Both","Anodic","Cathodic"]))

        samp_opts     = [""] + sorted(df["sample"].unique())
        sample_filter = st.selectbox("Sample (blank = all)", samp_opts)

        st.divider()
        st.header("📐 Statistics")
        r2_min    = st.slider("R² minimum",      0.0, 1.0, 0.90, 0.01)
        sd_mult   = st.slider("Outlier × StDev", 1.0, 4.0, 2.0,  0.5)
        log_icorr = st.checkbox("Log₁₀(Icorr)", value=True)

        st.divider()
        st.header("🗑 Excluded points")
        n_del = len(st.session_state.deleted_ids)
        if n_del:
            st.write(f"**{n_del}** point(s) excluded")
            if st.button("↺ Restore all", use_container_width=True):
                st.session_state.deleted_ids = set()
                st.rerun()
        else:
            st.caption("None — click X marks on the plot to exclude")

        st.divider()
        filtered_df = df[~df["row_id"].isin(st.session_state.deleted_ids)]
        st.download_button(
            "⬇ Download filtered CSV",
            data=filtered_df.to_csv(index=False).encode(),
            file_name="electrolyzer_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Metrics ───────────────────────────────────────────────
    df       = st.session_state.df
    param_df = df[df["parameter"] == param]
    active   = param_df[~param_df["row_id"].isin(st.session_state.deleted_ids)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total (this param)", len(param_df))
    c2.metric("Active",             len(active))
    c3.metric("Excluded",           len(st.session_state.deleted_ids))
    c4.metric("Conditions",         active["condition"].nunique())

    # ── Plot ──────────────────────────────────────────────────
    fig = make_figure(
        df=df, param=param, pH_filter=pH_filter, dir_filter=dir_filter,
        r2_min=r2_min, sd_mult=sd_mult, log_icorr=log_icorr,
        deleted_ids=st.session_state.deleted_ids,
        sample_filter=sample_filter,
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        key=(f"chart_{param}_{pH_filter}_{dir_filter}_"
             f"{sample_filter}_{len(st.session_state.deleted_ids)}"),
    )

    # Handle point clicks — exclude selected points
    if event and event.get("selection") and event["selection"].get("points"):
        for pt in event["selection"]["points"]:
            cdata = pt.get("customdata")
            if cdata is not None:
                rid = str(cdata[0]) if isinstance(cdata, list) else str(cdata)
                if rid and rid not in st.session_state.deleted_ids:
                    st.session_state.deleted_ids.add(rid)
                    st.rerun()

    st.caption(
        "💡 Click any **✕ mark** to exclude that point — "
        "each panel's statistics and whiskers update instantly."
    )

    # ── Statistics table ──────────────────────────────────────
    st.subheader("📋 Summary Statistics")

    sub = df[~df["row_id"].isin(st.session_state.deleted_ids)]
    sub = sub[sub["parameter"] == param].copy()
    if pH_filter != "Both":
        sub = sub[sub["pH"] == int(pH_filter)]
    if not is_ocp and dir_filter != "Both":
        sub = sub[sub["direction"].str.upper() == dir_filter.upper()]
    if sample_filter:
        sub = sub[sub["sample"].str.upper() == sample_filter.upper()]
    if not is_ocp:
        sub = sub[(sub["r2"].isna()) | (sub["r2"] >= r2_min)]
    if param == "Icorr" and log_icorr:
        sub = sub[sub["value"] > 0].copy()
        sub["value"] = np.log10(sub["value"])

    stat_rows = []
    for cond in COND_ORDER:
        for cut in ["LC", "WJ"]:
            vals = sub[
                (sub["condition"] == cond) & (sub["cut"] == cut)
            ]["value"].dropna().values
            if len(vals) == 0:
                continue
            st_ = compute_stats(vals, sd_mult)
            if st_ is None:
                continue
            stat_rows.append({
                "Condition":  cond,
                "Cut":        cut,
                "n":          st_["n"],
                "n_outlier":  len(st_["outliers"]),
                "Min":        round(float(st_["clean"].min()), 5),
                "Q1":         round(float(st_["q1"]),          5),
                "Median":     round(float(st_["med"]),         5),
                "Q3":         round(float(st_["q3"]),          5),
                "Max":        round(float(st_["clean"].max()), 5),
                "Mean_clean": round(float(st_["mean"]),        5),
                "StDev":      round(float(st_["sd"]),          5),
                "Lo_Whisker": round(float(st_["lo"]),          5),
                "Hi_Whisker": round(float(st_["hi"]),          5),
            })

    if stat_rows:
        sdf = pd.DataFrame(stat_rows)

        def _color_row(row):
            bg = "#FAD7D3" if row["Cut"] == "LC" else "#D0E8F7"
            return [f"background-color:{bg}" if i == 1 else ""
                    for i in range(len(row))]

        st.dataframe(
            sdf.style.apply(_color_row, axis=1),
            use_container_width=True,
            height=300,
        )
    else:
        st.info("No data for the selected filters.")


if __name__ == "__main__":
    main()
