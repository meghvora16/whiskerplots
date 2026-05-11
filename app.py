"""
Electrolyzer Whisker Plot App  v2.6
Fixes:
  - Legend only shows entries for cuts/folders that actually have data
    after all filters are applied
  - Folder legend labels reflect actual samples present (not all possible)
  - Box legend entry added only when data confirmed present
"""

import io, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Electrolyzer Whisker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  COLOUR ASSIGNMENTS
# ─────────────────────────────────────────────────────────────
FOLDER_COLOURS = {
    ("LC","1","02"): "#E53935",
    ("LC","1","03"): "#B71C1C",
    ("LC","1","04"): "#F06292",
    ("LC","1","05"): "#FF7043",
    ("LC","1","09"): "#7B1FA2",
    ("LC","4","01"): "#FB8C00",
    ("LC","4","02"): "#F9A825",
    ("LC","4","05"): "#FDD835",
    ("LC","4","10"): "#A1887F",
    ("WJ","1","03"): "#1565C0",
    ("WJ","1","06"): "#283593",
    ("WJ","1","07"): "#4527A0",
    ("WJ","4","01"): "#00897B",
    ("WJ","4","02"): "#00ACC1",
    ("WJ","4","03"): "#43A047",
    ("WJ","4","05"): "#7CB342",
    ("WJ","4","09"): "#26A69A",
    ("WJ","4","10"): "#0288D1",
}
FALLBACK_COLOUR = "#9E9E9E"
TEST_SYMBOL     = {"1": "x", "2": "cross", "3": "diamond-x", "4": "square-x"}

CUT_COL   = {"LC": "#C0392B",                "WJ": "#1F618D"}
CUT_FILL  = {"LC": "rgba(220,100,80,0.18)",  "WJ": "rgba(50,130,180,0.18)"}
CUT_FILL2 = {"LC": "rgba(220,100,80,0.45)",  "WJ": "rgba(50,130,180,0.45)"}

COND_ORDER  = ["AC","Brushed","Pickled","B&P","BPP"]
PARAM_UNITS = {
    "OCP1":"V vs. RHE","OCP2":"V vs. RHE","OCP3":"V vs. RHE",
    "Ecorr":"V vs. RHE","Icorr":"A dm⁻²","Epp":"V vs. RHE",
}

SAMPLE_META = {
    "50":("LC","AC"),  "51":("LC","Brushed"),"52":("LC","Pickled"),"53":("LC","B&P"),
    "60":("LC","AC"),  "61":("LC","Brushed"),"62":("LC","Pickled"),
    "63":("LC","B&P"), "64":("LC","BPP"),
    "70":("WJ","AC"),  "71":("WJ","Brushed"),"72":("WJ","Pickled"),
    "73":("WJ","B&P"), "74":("WJ","BPP"),
}
PH_MAP = {
    "50":{"01":4,"05":1},"51":{"02":4},"52":{"03":1,"10":4},"53":{"01":4,"04":1},
    "60":{"02":1,"10":4},"61":{"02":4},"62":{"03":4},
    "63":{"01":4,"03":1,"05":4},"64":{"01":4,"05":4,"09":1},
    "70":{"02":4,"03":1},"71":{"02":4},"72":{"07":1,"10":4},
    "73":{"01":4,"03":1,"09":4},"74":{"01":4,"03":4,"05":4,"06":1},
}

def get_mark_colour(cut, ph, folder):
    return FOLDER_COLOURS.get((cut, str(ph), str(folder)), FALLBACK_COLOUR)

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
    fp = str(row["file_path"]); fn = str(row["file_name"])
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
        frame["pH"] = frame.apply(
            lambda r: PH_MAP.get(r["sample"],{}).get(r["folder"], None), axis=1)
        frame["cut"]       = frame["sample"].map(lambda s: SAMPLE_META.get(s,("",""))[0])
        frame["condition"] = frame["sample"].map(lambda s: SAMPLE_META.get(s,("",""))[1])

    rows = []
    param_map = {"Ecorr_fitted_V":"Ecorr","Icorr_abs":"Icorr","Epp_V":"Epp"}
    main_lsv  = df[df["sample"].isin(SAMPLE_META) & df["pH"].notna()].copy()

    for _, r in main_lsv.iterrows():
        for pcol, pname in param_map.items():
            if pd.isna(r[pcol]): continue
            rows.append({
                "sample":    f"S{r['sample']}",
                "cut":       r["cut"],
                "condition": r["condition"],
                "pH":        str(int(r["pH"])),
                "direction": str(r["scan_direction"]),
                "parameter": pname,
                "value":     float(r[pcol]),
                "r2":        float(r["R2_log"]) if not pd.isna(r["R2_log"]) else None,
                "test":      str(r["test"]) if not pd.isna(r["test"]) else "1",
                "folder":    str(r["folder"]),
                "point":     str(r["point"]) if not pd.isna(r["point"]) else "?",
                "source":    "LSV",
                "row_id":    f"lsv_{r.name}_{pcol}",
            })

    main_ocp = ocp_raw[ocp_raw["sample"].isin(SAMPLE_META) & ocp_raw["pH"].notna()].copy()
    for _, r in main_ocp.iterrows():
        if pd.isna(r["last_voltage_v"]): continue
        n = str(r["ocp_num"]).strip()
        ocp_lbl = f"OCP{n}" if n else "OCP1"
        rows.append({
            "sample":    f"S{r['sample']}",
            "cut":       r["cut"],
            "condition": r["condition"],
            "pH":        str(int(r["pH"])),
            "direction": str(r["direction"]),
            "parameter": ocp_lbl,
            "value":     float(r["last_voltage_v"]),
            "r2":        None,
            "test":      str(r["test"]) if not pd.isna(r["test"]) else "1",
            "folder":    str(r["folder"]),
            "point":     str(r["point"]) if not pd.isna(r["point"]) else "?",
            "source":    "OCP",
            "row_id":    f"ocp_{r.name}_{n}",
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
        raw["test"]    = raw["test"].fillna("1").astype(str)
        raw["folder"]  = raw["folder"].fillna("01").astype(str)
        raw            = raw[raw["value"].notna()]
        raw["pH"]      = raw["pH"].apply(lambda x: str(int(x)) if pd.notna(x) else "1")
        raw["point"]   = "?"
        raw["source"]  = raw["parameter"].apply(
            lambda p: "OCP" if str(p).startswith("OCP") else "LSV")
        return raw[["sample","cut","condition","pH","direction","parameter",
                    "value","r2","test","folder","point","source","row_id"]].copy()
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────────────────────
def compute_stats(vals: np.ndarray, sd_mult: float):
    if len(vals) == 0: return None
    s  = np.sort(vals); n = len(s)
    q1 = np.percentile(s,25); med = np.percentile(s,50); q3 = np.percentile(s,75)
    mu = s.mean()
    sd = s.std(ddof=1) if n > 1 else 0.0
    lo = mu - sd_mult*sd; hi = mu + sd_mult*sd
    clean = s[(s >= lo) & (s <= hi)]
    if len(clean) == 0: clean = s
    lo_w = clean.min(); hi_w = clean.max()
    outliers = s[(s < lo_w) | (s > hi_w)]
    return dict(n=n, q1=q1, med=med, q3=q3, lo=lo_w, hi=hi_w,
                mean=clean.mean(), sd=sd, outliers=outliers, clean=clean, raw=s)

# ─────────────────────────────────────────────────────────────
#  FIGURE
# ─────────────────────────────────────────────────────────────
def make_figure(df, param, pH_filter, dir_filter,
                r2_min, sd_mult, log_icorr, deleted_ids, sample_filter):

    is_ocp = param.startswith("OCP")

    # ── Apply all filters first ───────────────────────────────
    sub = df[df["parameter"] == param].copy()
    if not is_ocp:
        if dir_filter != "Both":
            sub = sub[sub["direction"].str.upper() == dir_filter.upper()]
        sub = sub[(sub["r2"].isna()) | (sub["r2"] >= r2_min)]
    if pH_filter != "Both":
        sub = sub[sub["pH"] == str(pH_filter)]
    if sample_filter:
        sub = sub[sub["sample"].str.upper() == sample_filter.upper()]
    sub = sub[~sub["row_id"].isin(deleted_ids)].copy()

    for col_name, default in [("test","1"),("folder","01"),("pH","1"),("point","?")]:
        if col_name not in sub.columns: sub[col_name] = default
        sub[col_name] = sub[col_name].fillna(default).astype(str)

    if param == "Icorr" and log_icorr:
        sub = sub[sub["value"] > 0].copy()
        sub["value"] = np.log10(sub["value"])

    unit   = PARAM_UNITS.get(param, "")
    ylabel = (f"log₁₀(icorr)  [{unit}]"
              if (param == "Icorr" and log_icorr) else f"{param}  [{unit}]")

    conds = [c for c in COND_ORDER if c in sub["condition"].unique()]
    n_c   = len(conds)
    fig   = go.Figure()

    if n_c == 0 or sub.empty:
        fig.add_annotation(text="No data for selected filters",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font_size=16, font_color="#888")
        fig.update_layout(height=500, plot_bgcolor="white", paper_bgcolor="white")
        return fig

    # ── Pre-scan: which cuts actually have data after filtering? ──
    # This determines which box legend entries to show — no ghosts
    cuts_with_data = set(sub["cut"].unique())

    all_v = sub["value"].dropna().values
    y_min = all_v.min(); y_max = all_v.max()
    y_rng = max(y_max - y_min, 1e-9)
    y_pad = y_rng * 0.13
    y_lo  = y_min - y_pad; y_hi = y_max + y_pad

    slot       = 1.0; cut_half = slot / 2
    box_gap    = 0.15; box_slot = 0.65; box_box_w = 0.13
    hist_max_w = cut_half * 0.80
    nbins      = 10
    bin_edges  = np.linspace(y_lo, y_hi, nbins + 1)

    added_box_legend  = set()
    added_mark_legend = set()
    tick_hist_vals, tick_hist_text = [], []
    tick_box_vals,  tick_box_text  = [], []

    for ci, cond in enumerate(conds):
        hist_cx = ci * slot
        box_cx  = n_c * slot + box_gap + ci * box_slot

        tick_hist_vals.append(hist_cx + cut_half); tick_hist_text.append(cond)
        tick_box_vals.append(box_cx + box_slot/2); tick_box_text.append(cond)

        if ci > 0:
            fig.add_shape(type="line",
                x0=hist_cx, x1=hist_cx, y0=y_lo, y1=y_hi,
                line=dict(color="#D0D8E8", width=1, dash="dash"))

        for ki, cut in enumerate(["LC","WJ"]):
            # ── Skip entirely if this cut has zero data after filtering ──
            if cut not in cuts_with_data:
                continue

            col   = CUT_COL[cut]
            fill  = CUT_FILL[cut]
            fill2 = CUT_FILL2[cut]

            csub = sub[(sub["condition"]==cond) & (sub["cut"]==cut)]
            vals = csub["value"].dropna().values

            # Histogram
            bar_right = hist_cx + (ki+1)*cut_half - 0.02
            if len(vals) > 0:
                counts, _ = np.histogram(vals, bins=bin_edges)
                max_c = counts.max() if counts.max() > 0 else 1
                for bi in range(nbins):
                    c = counts[bi]
                    if c == 0: continue
                    bar_len = (c/max_c)*hist_max_w*0.88
                    fig.add_shape(type="rect",
                        x0=bar_right-bar_len, x1=bar_right,
                        y0=bin_edges[bi], y1=bin_edges[bi+1],
                        fillcolor=fill2, line=dict(color=col, width=0.7), layer="below")
                fig.add_shape(type="line",
                    x0=bar_right, x1=bar_right, y0=y_lo, y1=y_hi,
                    line=dict(color=col, width=0.8, dash="dot"))

            bx = box_cx + 0.18 + ki*0.29

            if len(vals) == 0:
                continue   # ← no ghost legend entry added here

            st_ = compute_stats(vals, sd_mult)
            if st_ is None: continue

            # ── Box legend entry — only added when we have real data ──
            if cut not in added_box_legend:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    name=f"{cut} box/hist",
                    marker=dict(color=col, size=9, symbol="square",
                                line=dict(color=col, width=1)),
                    legendgroup=f"box_{cut}", showlegend=True))
                added_box_legend.add(cut)

            # Whiskers
            for y0, y1 in [(st_["q3"], st_["hi"]), (st_["lo"], st_["q1"])]:
                fig.add_shape(type="line", x0=bx, x1=bx, y0=y0, y1=y1,
                    line=dict(color=col, width=1.5))
                fig.add_shape(type="line",
                    x0=bx-box_box_w*0.7, x1=bx+box_box_w*0.7, y0=y1, y1=y1,
                    line=dict(color=col, width=1.8))
            # IQR box
            fig.add_shape(type="rect",
                x0=bx-box_box_w, x1=bx+box_box_w, y0=st_["q1"], y1=st_["q3"],
                fillcolor=fill, line=dict(color=col, width=1.8))
            # Median
            fig.add_shape(type="line",
                x0=bx-box_box_w, x1=bx+box_box_w, y0=st_["med"], y1=st_["med"],
                line=dict(color=col, width=2.8))
            # Mean
            fig.add_trace(go.Scatter(x=[bx], y=[st_["mean"]], mode="markers",
                marker=dict(symbol="circle-open", size=10, color=col,
                            line=dict(color=col, width=2.0)),
                legendgroup=f"box_{cut}", showlegend=False,
                hovertemplate=f"<b>Mean ({cut}–{cond})</b>: %{{y:.4f}}<extra></extra>"))
            # Outliers
            if len(st_["outliers"]) > 0:
                fig.add_trace(go.Scatter(
                    x=[bx]*len(st_["outliers"]), y=st_["outliers"].tolist(),
                    mode="markers",
                    marker=dict(symbol="circle-open", size=8, color=col,
                                line=dict(color=col, width=1.5)),
                    legendgroup=f"box_{cut}", showlegend=False,
                    hovertemplate=f"<b>Outlier ({cut})</b>: %{{y:.4f}}<extra></extra>"))

            # ── X marks — label uses only samples ACTUALLY PRESENT in filtered data ──
            np.random.seed(42 + ki*100 + ci*10)
            rng_x = box_box_w * 0.60

            for ph_val in sorted(csub["pH"].unique()):
                for folder_val in sorted(csub[csub["pH"]==ph_val]["folder"].unique()):
                    for t_val in sorted(csub["test"].unique()):
                        mask = ((csub["pH"]==ph_val) &
                                (csub["folder"]==folder_val) &
                                (csub["test"]==t_val))
                        tph_sub = csub[mask]
                        tvals   = tph_sub["value"].dropna().values
                        if len(tvals) == 0: continue

                        trids    = tph_sub["row_id"].values
                        mark_col = get_mark_colour(cut, ph_val, folder_val)
                        symbol   = TEST_SYMBOL.get(str(t_val), "x")

                        # ── Build legend label from ACTUAL samples in filtered data ──
                        actual_samples = sorted(tph_sub["sample"].unique())
                        samples_short  = ",".join(s.replace("S","") for s in actual_samples)
                        leg_label      = f"{cut} pH{ph_val} F{folder_val} ({samples_short})"
                        leg_key        = f"{cut}_pH{ph_val}_F{folder_val}_T{t_val}"
                        leg_name       = f"{leg_label}  T{t_val}"
                        show_m         = leg_key not in added_mark_legend
                        jitter         = np.random.uniform(-rng_x, rng_x, len(tvals))

                        custom = tph_sub[["row_id","sample","folder","test",
                                          "point","pH","direction"]].values.tolist()

                        fig.add_trace(go.Scatter(
                            x=(bx + jitter).tolist(),
                            y=tvals.tolist(),
                            mode="markers",
                            marker=dict(symbol=symbol, size=7, color=mark_col,
                                        line=dict(color=mark_col, width=1.8),
                                        opacity=0.92),
                            name=leg_name,
                            legendgroup=leg_key,
                            showlegend=show_m,
                            customdata=custom,
                            hovertemplate=(
                                f"<b>{cut} — {cond}</b><br>"
                                "Sample: %{customdata[1]}<br>"
                                "pH: %{customdata[5]}  |  "
                                "Folder: %{customdata[2]}  |  "
                                "Test: %{customdata[3]}  |  "
                                "Point: %{customdata[4]}<br>"
                                "Direction: %{customdata[6]}<br>"
                                "Value: %{y:.5f}<br>"
                                "<i>Click to exclude</i><extra></extra>"
                            ),
                        ))
                        if show_m:
                            added_mark_legend.add(leg_key)

            fig.add_annotation(
                x=bx, y=y_lo - y_pad*0.55,
                text=f"n={st_['n']}",
                showarrow=False, font=dict(size=9, color=col),
                xanchor="center")

    # Divider
    fig.add_shape(type="line",
        x0=n_c*slot + box_gap*0.4, x1=n_c*slot + box_gap*0.4,
        y0=y_lo, y1=y_hi,
        line=dict(color="#8899BB", width=1.2))

    pH_str     = f"pH {pH_filter}" if pH_filter != "Both" else "pH 1 & 4"
    dir_str    = f" | {dir_filter}" if not is_ocp and dir_filter != "Both" else ""
    sample_str = f" | {sample_filter}" if sample_filter else ""
    x_end      = n_c*slot + box_gap + n_c*box_slot + 0.1

    fig.update_layout(
        title=dict(
            text=f"<b>{param}</b>  |  {pH_str}{dir_str}{sample_str}",
            font=dict(size=15, color="#2E4057"), x=0.5),
        xaxis=dict(
            tickvals=tick_hist_vals + tick_box_vals,
            ticktext=tick_hist_text + tick_box_text,
            tickfont=dict(size=11, color="#2E4057"),
            showgrid=False, zeroline=False, range=[-0.08, x_end]),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(size=11, color="#333")),
            range=[y_lo - y_pad*0.7, y_hi + y_pad*0.15],
            gridcolor="#E8EDF5", gridwidth=0.8,
            zeroline=True, zerolinecolor="#CCCCCC", zerolinewidth=0.8),
        legend=dict(
            title=dict(text="Legend", font=dict(size=10)),
            orientation="v", x=1.01, y=0.99,
            bgcolor="rgba(248,250,252,0.93)",
            bordercolor="#C0CAD8", borderwidth=1,
            font=dict(size=9), tracegroupgap=2),
        plot_bgcolor="white", paper_bgcolor="white",
        height=540, margin=dict(l=72, r=240, t=65, b=72),
        clickmode="event",
    )

    if n_c > 0:
        fig.add_annotation(
            x=(n_c*slot)/2 - 0.3, y=y_hi + y_pad*0.08,
            text="◀  Distribution (histogram)",
            showarrow=False, font=dict(size=9.5, color="#667799"),
            xanchor="center")
        fig.add_annotation(
            x=n_c*slot + box_gap + (n_c*box_slot)/2, y=y_hi + y_pad*0.08,
            text="Box & Whisker  ▶",
            showarrow=False, font=dict(size=9.5, color="#667799"),
            xanchor="center")

    return fig


# ─────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <style>
    .block-container{padding-top:1rem}
    h1{color:#2E4057;font-size:1.55rem}
    </style>""", unsafe_allow_html=True)

    st.title("📊 Electrolyzer Whisker — SS316L Corrosion Study")
    st.caption(
        "Left = histogram  ·  Right = box-whisker  ·  "
        "Colour = subsample folder  ·  Shape = test number  ·  "
        "Click marks to exclude"
    )

    if "deleted_ids"  not in st.session_state: st.session_state.deleted_ids  = set()
    if "deleted_meta" not in st.session_state: st.session_state.deleted_meta = {}
    if "df"           not in st.session_state: st.session_state.df           = None

    with st.sidebar:
        st.header("📂 Data source")
        src = st.radio("Load from:", ["Pre-filled Excel","Upload raw files"])

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
                    st.warning("Place ElectrolyzerWhisker_Final.xlsx in the app folder, "
                               "or upload files above.")

        if st.session_state.df is None:
            st.stop()

        df = st.session_state.df

        st.divider()
        st.header("⚙ Filters")
        params_av = sorted(df["parameter"].unique())
        param = st.selectbox("Parameter", params_av,
            index=params_av.index("OCP1") if "OCP1" in params_av else 0)

        pH_opts   = ["Both"] + sorted(df["pH"].dropna().unique().tolist())
        pH_filter = st.selectbox("pH", pH_opts)

        is_ocp     = param.startswith("OCP")
        dir_filter = ("Both" if is_ocp
                      else st.selectbox("Direction",["Both","Anodic","Cathodic"]))

        samp_opts     = [""] + sorted(df["sample"].unique())
        sample_filter = st.selectbox("Sample (blank = all)", samp_opts)

        st.divider()
        st.header("📐 Statistics")
        r2_min    = st.slider("R² minimum",      0.0, 1.0, 0.90, 0.01)
        sd_mult   = st.slider("Outlier × StDev", 1.0, 4.0, 2.0,  0.5)
        log_icorr = st.checkbox("Log₁₀(Icorr)", value=True)

        st.divider()
        st.header("🎨 Colour key")
        groups = [
            ("LC pH 1",[("#E53935","F02 (S60)"),("#B71C1C","F03 (S52,S63)"),
                        ("#F06292","F04 (S53)"),("#FF7043","F05 (S50)"),
                        ("#7B1FA2","F09 (S64)")]),
            ("LC pH 4",[("#FB8C00","F01 (S50,53,63,64)"),("#F9A825","F02 (S51,S61)"),
                        ("#FDD835","F05 (S63,S64)"),("#A1887F","F10 (S52,S60)")]),
            ("WJ pH 1",[("#1565C0","F03 (S70,S73)"),("#283593","F06 (S74)"),
                        ("#4527A0","F07 (S72)")]),
            ("WJ pH 4",[("#00897B","F01 (S73,S74)"),("#00ACC1","F02 (S70,S71)"),
                        ("#43A047","F03 (S74)"),("#7CB342","F05 (S74)"),
                        ("#26A69A","F09 (S73)"),("#0288D1","F10 (S72)")]),
        ]
        for group_title, entries in groups:
            st.markdown(f"*{group_title}*")
            for hex_c, label in entries:
                st.markdown(
                    f'<span style="color:{hex_c};font-size:17px;font-weight:bold">✕</span>'
                    f' <span style="font-size:11px;color:#333">{label}</span>',
                    unsafe_allow_html=True)
        st.markdown("**Shape = test**  ·  ✕ T1  ·  ✚ T2")

        st.divider()
        st.download_button(
            "⬇ Download filtered CSV",
            data=df[~df["row_id"].isin(st.session_state.deleted_ids)].to_csv(index=False).encode(),
            file_name="electrolyzer_filtered.csv",
            mime="text/csv", use_container_width=True)

        if st.session_state.deleted_meta:
            exc_df = pd.DataFrame(list(st.session_state.deleted_meta.values()))
            st.download_button(
                "⬇ Download excluded log (CSV)",
                data=exc_df.to_csv(index=False).encode(),
                file_name="electrolyzer_excluded.csv",
                mime="text/csv", use_container_width=True)

    # Metrics
    df       = st.session_state.df
    param_df = df[df["parameter"] == param]
    active   = param_df[~param_df["row_id"].isin(st.session_state.deleted_ids)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total (this param)", len(param_df))
    c2.metric("Active",             len(active))
    c3.metric("Excluded",           len(st.session_state.deleted_ids))
    c4.metric("Conditions",         active["condition"].nunique())

    # Plot
    fig = make_figure(
        df=df, param=param, pH_filter=pH_filter, dir_filter=dir_filter,
        r2_min=r2_min, sd_mult=sd_mult, log_icorr=log_icorr,
        deleted_ids=st.session_state.deleted_ids,
        sample_filter=sample_filter,
    )

    event = st.plotly_chart(
        fig, use_container_width=True, on_select="rerun",
        key=(f"chart_{param}_{pH_filter}_{dir_filter}_"
             f"{sample_filter}_{len(st.session_state.deleted_ids)}"),
    )

    if event and event.get("selection") and event["selection"].get("points"):
        for pt in event["selection"]["points"]:
            cdata = pt.get("customdata")
            if cdata is None: continue
            if isinstance(cdata, list) and len(cdata) >= 7:
                rid=str(cdata[0]); samp=str(cdata[1]); folder=str(cdata[2])
                test=str(cdata[3]); point=str(cdata[4])
                ph=str(cdata[5]);   direction=str(cdata[6])
                raw_val = pt.get("y", float("nan"))
            else:
                rid=str(cdata[0]) if isinstance(cdata,list) else str(cdata)
                samp=folder=test=point=ph=direction="?"
                raw_val=pt.get("y",float("nan"))

            if rid and rid not in st.session_state.deleted_ids:
                row_info = df[df["row_id"]==rid]
                if not row_info.empty:
                    r = row_info.iloc[0]
                    meta = {
                        "row_id":    rid,
                        "parameter": r.get("parameter", param),
                        "sample":    r.get("sample", samp),
                        "cut":       r.get("cut","?"),
                        "condition": r.get("condition","?"),
                        "pH":        r.get("pH", ph),
                        "direction": r.get("direction", direction),
                        "folder":    r.get("folder", folder),
                        "test":      r.get("test", test),
                        "point":     r.get("point", point),
                        "source":    r.get("source","?"),
                        "value":     float(r.get("value", raw_val)),
                        "r2":        r.get("r2", None),
                    }
                else:
                    meta = {"row_id":rid,"parameter":param,"sample":samp,"cut":"?",
                            "condition":"?","pH":ph,"direction":direction,"folder":folder,
                            "test":test,"point":point,"source":"?","value":raw_val,"r2":None}
                st.session_state.deleted_ids.add(rid)
                st.session_state.deleted_meta[rid] = meta
                st.rerun()

    st.caption(
        "💡 **Colour** = subsample folder  ·  **Shape** = test (✕=T1, ✚=T2)  ·  "
        "Hover for full details  ·  Click to exclude"
    )

    # Excluded points log
    n_del = len(st.session_state.deleted_ids)
    if n_del > 0:
        st.subheader(f"🗑 Excluded Points  ({n_del})")
        if st.button("↺ Restore all excluded points", use_container_width=True):
            st.session_state.deleted_ids  = set()
            st.session_state.deleted_meta = {}
            st.rerun()

        meta_rows = list(st.session_state.deleted_meta.values())
        if meta_rows:
            exc_df = pd.DataFrame(meta_rows)[[
                "parameter","sample","cut","condition","pH",
                "direction","folder","test","point","source","value","r2"
            ]].copy()
            exc_df["value"] = exc_df["value"].round(6)
            exc_df["r2"]    = exc_df["r2"].apply(
                lambda x: round(float(x),4) if pd.notna(x) else "—")
            exc_df.columns  = ["Parameter","Sample","Cut","Condition","pH",
                                "Direction","Folder","Test","Point","Source","Value","R²"]

            def _style_exc(row):
                bg = "#FAD7D3" if row["Cut"]=="LC" else "#D0E8F7"
                return [f"background-color:{bg}" if c=="Cut" else ""
                        for c in exc_df.columns]

            st.dataframe(exc_df.style.apply(_style_exc, axis=1),
                         use_container_width=True,
                         height=min(50+35*len(exc_df), 400))

            st.markdown("**Restore individual points:**")
            for rid, meta in list(st.session_state.deleted_meta.items()):
                label = (f"↺  {meta['parameter']}  ·  {meta['sample']}  ·  "
                         f"{meta['condition']}  ·  pH {meta['pH']}  ·  "
                         f"Folder {meta['folder']}  ·  Test {meta['test']}  ·  "
                         f"Point {meta['point']}  ·  Value = {meta['value']:.5f}")
                if st.button(label, key=f"restore_{rid}", use_container_width=True):
                    st.session_state.deleted_ids.discard(rid)
                    st.session_state.deleted_meta.pop(rid, None)
                    st.rerun()

    # Stats table
    st.subheader("📋 Summary Statistics")

    sub = df[~df["row_id"].isin(st.session_state.deleted_ids)]
    sub = sub[sub["parameter"]==param].copy()
    if pH_filter != "Both":
        sub = sub[sub["pH"]==str(pH_filter)]
    if not is_ocp and dir_filter != "Both":
        sub = sub[sub["direction"].str.upper()==dir_filter.upper()]
    if sample_filter:
        sub = sub[sub["sample"].str.upper()==sample_filter.upper()]
    if not is_ocp:
        sub = sub[(sub["r2"].isna())|(sub["r2"]>=r2_min)]
    if param == "Icorr" and log_icorr:
        sub = sub[sub["value"]>0].copy()
        sub["value"] = np.log10(sub["value"])

    stat_rows = []
    for cond in COND_ORDER:
        for cut in ["LC","WJ"]:
            vals = sub[(sub["condition"]==cond)&(sub["cut"]==cut)]["value"].dropna().values
            if len(vals)==0: continue
            st_ = compute_stats(vals, sd_mult)
            if st_ is None: continue
            stat_rows.append({
                "Condition":  cond, "Cut": cut, "n": st_["n"],
                "n_outlier":  len(st_["outliers"]),
                "Min":        round(float(st_["clean"].min()),5),
                "Q1":         round(float(st_["q1"]),5),
                "Median":     round(float(st_["med"]),5),
                "Q3":         round(float(st_["q3"]),5),
                "Max":        round(float(st_["clean"].max()),5),
                "Mean_clean": round(float(st_["mean"]),5),
                "StDev":      round(float(st_["sd"]),5),
                "Lo_Whisker": round(float(st_["lo"]),5),
                "Hi_Whisker": round(float(st_["hi"]),5),
            })

    if stat_rows:
        sdf = pd.DataFrame(stat_rows)
        def _color_row(row):
            bg = "#FAD7D3" if row["Cut"]=="LC" else "#D0E8F7"
            return [f"background-color:{bg}" if i==1 else "" for i in range(len(row))]
        st.dataframe(sdf.style.apply(_color_row, axis=1),
                     use_container_width=True, height=300)
    else:
        st.info("No data for the selected filters.")


if __name__ == "__main__":
    main()
