"""
Electrolyzer Whisker Plot App  v3.3
SS316L Corrosion Study

Changes v3.3:
  - NEW: "🗑 Exclude by legend group" multiselect below the chart.
    Selecting one or more legend groups (e.g. "S63_04  ●T1") excludes ALL
    row_ids belonging to that group; the whisker auto-scales accordingly.
  - Excluded-by-legend entries appear in the same excluded-points table and
    can be restored individually or via "Restore all".
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
#  COLOURS  (cut × pH × folder)
# ─────────────────────────────────────────────────────────────
FOLDER_COLOURS = {
    ("LC", "1", "01"): "#FF1744",
    ("LC", "1", "02"): "#E53935",
    ("LC", "1", "03"): "#B71C1C",
    ("LC", "1", "04"): "#F06292",
    ("LC", "1", "05"): "#FF7043",
    ("LC", "1", "06"): "#E040FB",
    ("LC", "1", "07"): "#D81B60",
    ("LC", "1", "09"): "#7B1FA2",
    ("LC", "1", "10"): "#C2185B",
    ("LC", "4", "01"): "#FB8C00",
    ("LC", "4", "02"): "#F9A825",
    ("LC", "4", "03"): "#FFD600",
    ("LC", "4", "04"): "#FFAB40",
    ("LC", "4", "05"): "#FDD835",
    ("LC", "4", "06"): "#FF6F00",
    ("LC", "4", "08"): "#FFB300",
    ("LC", "4", "09"): "#A5D6A7",
    ("LC", "4", "10"): "#A1887F",
    ("LC", "4", "27"): "#26C6DA",
    ("LC", "4", "29"): "#00BFA5",
    ("WJ", "1", "01"): "#0D47A1",
    ("WJ", "1", "02"): "#1E88E5",
    ("WJ", "1", "03"): "#1565C0",
    ("WJ", "1", "04"): "#3949AB",
    ("WJ", "1", "06"): "#283593",
    ("WJ", "1", "07"): "#4527A0",
    ("WJ", "1", "08"): "#5C6BC0",
    ("WJ", "1", "09"): "#039BE5",
    ("WJ", "4", "01"): "#00897B",
    ("WJ", "4", "02"): "#00ACC1",
    ("WJ", "4", "03"): "#43A047",
    ("WJ", "4", "04"): "#F57C00",
    ("WJ", "4", "05"): "#7CB342",
    ("WJ", "4", "06"): "#009688",
    ("WJ", "4", "07"): "#8D6E63",
    ("WJ", "4", "09"): "#26A69A",
    ("WJ", "4", "10"): "#0288D1",
}
FALLBACK_COLOUR = "#9E9E9E"
TEST_SYMBOL     = {"1": "circle", "2": "x", "3": "diamond", "4": "square"}

SYMBOL_OPTIONS = [
    "circle", "square", "diamond", "triangle-up", "triangle-down",
    "star", "hexagram", "pentagon", "cross", "x",
    "circle-open", "square-open", "diamond-open", "triangle-up-open",
]
SYMBOL_GLYPH = {
    "circle": "●", "square": "■", "diamond": "◆", "triangle-up": "▲",
    "triangle-down": "▼", "star": "★", "hexagram": "✶", "pentagon": "⬟",
    "cross": "✚", "x": "✕", "circle-open": "○", "square-open": "□",
    "diamond-open": "◇", "triangle-up-open": "△",
}
_HOLLOW_KEYS = ("open", "x", "cross", "asterisk", "line", "y-up", "y-down")
def is_filled_symbol(sym):
    return not any(k in str(sym) for k in _HOLLOW_KEYS)

CUT_COL  = {"LC": "#C0392B",               "WJ": "#1F618D"}
CUT_FILL = {"LC": "rgba(220,100,80,0.15)", "WJ": "rgba(50,130,180,0.15)"}

COND_ORDER  = ["AC", "Brushed", "Pickled", "B&P", "BPP"]
PARAM_UNITS = {
    "OCP1": "V", "OCP2": "V", "OCP3": "V",
    "Ecorr": "V", "Icorr": "A dm⁻²", "Epp": "V",
}

SAMPLE_META = {
    "50": ("LC", "AC"),  "51": ("LC", "Brushed"), "52": ("LC", "Pickled"), "53": ("LC", "B&P"),
    "60": ("LC", "AC"),  "61": ("LC", "Brushed"), "62": ("LC", "Pickled"),
    "63": ("LC", "B&P"), "64": ("LC", "BPP"),
    "70": ("WJ", "AC"),  "71": ("WJ", "Brushed"), "72": ("WJ", "Pickled"),
    "73": ("WJ", "B&P"), "74": ("WJ", "BPP"),
}

PH_MAP = {
    "50": {"01": 4, "02": 4, "04": 1, "05": 1},
    "51": {"01": 4, "02": 4, "03": 1, "05": 4, "06": 4},
    "52": {"01": 1, "03": 4, "04": 4, "10": 4},
    "53": {"01": 4, "02": 4, "03": 1, "04": 1},
    "60": {"01": 1, "02": 1, "09": 4, "10": 4},
    "61": {"01": 4, "02": 4, "03": 4, "04": 4, "06": 1, "07": 1},
    "62": {"01": 4, "02": 1, "04": 1, "05": 4, "06": 1, "09": 4, "10": 4},
    "63": {"01": 4, "02": 4, "03": 4,
           "04": {"1": 1, "2": 4},
           "05": 1, "07": 1, "08": 4,
           "09": {"1": 4, "2": 1}},
    "64": {"01": 1, "03": 4, "05": 4, "06": 1, "09": 4,
           "10": {"1": 4, "2": 1},
           "27": 4, "29": 4},
    "70": {"01": 4, "02": 4, "03": 1, "04": 1},
    "71": {"01": 4, "02": 4, "03": 1},
    "72": {"02": 4, "07": 1, "08": 1, "10": 4},
    "73": {"01": 1, "02": 1, "03": 1, "04": 4, "05": 4, "06": 4, "07": 4,
           "09": {"1": 4, "2": 1},
           "10": 4},
    "74": {"01": 4, "02": 4, "03": 4, "05": 4, "06": 1, "07": 1},
}


def resolve_ph(sample, folder, test):
    entry = PH_MAP.get(str(sample), {}).get(str(folder))
    if entry is None:
        return None
    if isinstance(entry, dict):
        m = re.search(r"\d+", str(test))
        t = m.group(0) if m else None
        if t and t in entry:
            return entry[t]
        return None
    return entry


def get_mark_colour(cut, ph, folder):
    return FOLDER_COLOURS.get((cut, str(ph), str(folder)), FALLBACK_COLOUR)

# ─────────────────────────────────────────────────────────────
#  PARSERS
# ─────────────────────────────────────────────────────────────
def parse_lsv_path(f):
    s   = re.search(r"Sample (\d+)", str(f))
    fol = re.search(r"Sample \d+\\(\d+)\\", str(f))
    t   = re.search(r"Test (\d+)", str(f))
    p   = re.search(r"Point (\d+)", str(f), re.IGNORECASE)
    return (s.group(1) if s else None, fol.group(1) if fol else None,
            t.group(1) if t else None, p.group(1) if p else None)

def parse_ocp_path(fp, fn):
    s     = re.search(r"Sample (\d+)", str(fp))
    fol   = re.search(r"Sample \d+\\(\d+)\\", str(fp))
    t     = re.search(r"Test (\d+)", str(fp))
    p     = re.search(r"Point (\d+)", str(fp), re.IGNORECASE)
    ocp_n = re.search(r"ocp(\d+)", str(fn), re.IGNORECASE)
    return (s.group(1) if s else None, fol.group(1) if fol else None,
            t.group(1) if t else None, p.group(1) if p else None,
            ocp_n.group(1) if ocp_n else "1")

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_uploaded(lsv_bytes, ocp_bytes):
    df = pd.read_excel(io.BytesIO(lsv_bytes), sheet_name="LSV")
    parsed = df["File"].apply(lambda x: pd.Series(parse_lsv_path(x)))
    parsed.columns = ["sample", "folder", "test", "point"]
    df = pd.concat([df, parsed], axis=1)
    df["sample"] = df["sample"].astype(str)
    df["folder"] = df["folder"].astype(str)
    df["pH"]        = df.apply(
        lambda r: resolve_ph(r["sample"], r["folder"], r.get("test")), axis=1)
    df["cut"]       = df["sample"].map(lambda s: SAMPLE_META.get(s, ("", ""))[0])
    df["condition"] = df["sample"].map(lambda s: SAMPLE_META.get(s, ("", ""))[1])

    main_lsv = df[
        df["sample"].isin(SAMPLE_META) &
        df["pH"].notna() &
        df["scan_direction"].notna()
    ].copy()

    rows = []
    param_map = {"Ecorr_fitted_V": "Ecorr", "Icorr_abs": "Icorr", "Epp_V": "Epp"}
    for _, r in main_lsv.iterrows():
        for pcol, pname in param_map.items():
            val = r.get(pcol)
            if pd.isna(val): continue
            r2 = r.get("R2_log")
            rows.append({
                "sample":    f"S{r['sample']}",
                "cut":       r["cut"],
                "condition": r["condition"],
                "pH":        str(int(r["pH"])),
                "direction": str(r["scan_direction"]),
                "parameter": pname,
                "value":     float(val),
                "r2":        float(r2) if pd.notna(r2) else None,
                "test":      str(r["test"])  if pd.notna(r.get("test"))  else "1",
                "folder":    str(r["folder"]),
                "point":     str(r["point"]) if pd.notna(r.get("point")) else "?",
                "source":    "LSV",
                "row_id":    f"lsv_{r.name}_{pcol}",
            })

    ocp = pd.read_excel(io.BytesIO(ocp_bytes), sheet_name="Sheet1")
    ocp_parsed = ocp.apply(
        lambda r: pd.Series(parse_ocp_path(r["file_path"], r["file_name"])), axis=1)
    ocp_parsed.columns = ["sample", "folder", "test", "point", "ocp_num"]
    ocp = pd.concat([ocp, ocp_parsed], axis=1)
    ocp["sample"] = ocp["sample"].astype(str)
    ocp["folder"] = ocp["folder"].astype(str)
    ocp["pH"]        = ocp.apply(
        lambda r: resolve_ph(r["sample"], r["folder"], r.get("test")), axis=1)
    ocp["cut"]       = ocp["sample"].map(lambda s: SAMPLE_META.get(s, ("", ""))[0])
    ocp["condition"] = ocp["sample"].map(lambda s: SAMPLE_META.get(s, ("", ""))[1])

    main_ocp = ocp[
        ocp["sample"].isin(SAMPLE_META) &
        ocp["pH"].notna() &
        ocp["last_voltage_v"].notna()
    ].copy()

    for _, r in main_ocp.iterrows():
        n       = str(r["ocp_num"]).strip()
        ocp_lbl = f"OCP{n}" if n else "OCP1"
        rows.append({
            "sample":    f"S{r['sample']}",
            "cut":       r["cut"],
            "condition": r["condition"],
            "pH":        str(int(r["pH"])),
            "direction": "Both",
            "parameter": ocp_lbl,
            "value":     float(r["last_voltage_v"]),
            "r2":        None,
            "test":      str(r["test"])  if pd.notna(r.get("test"))  else "1",
            "folder":    str(r["folder"]),
            "point":     str(r["point"]) if pd.notna(r.get("point")) else "?",
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
            ["include", "sample", "cut", "condition", "pH", "direction",
             "parameter", "value", "unit", "folder", "test", "r2"]
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
        return raw[["sample", "cut", "condition", "pH", "direction", "parameter",
                    "value", "r2", "test", "folder", "point", "source", "row_id"]].copy()
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────────────────────
def compute_stats(vals: np.ndarray, sd_mult: float):
    if len(vals) == 0: return None
    s   = np.sort(vals); n = len(s)
    q1  = np.percentile(s, 25)
    med = np.percentile(s, 50)
    q3  = np.percentile(s, 75)
    mu  = s.mean()
    sd  = s.std(ddof=1) if n > 1 else 0.0
    lo  = mu - sd_mult * sd
    hi  = mu + sd_mult * sd
    clean    = s[(s >= lo) & (s <= hi)]
    if len(clean) == 0: clean = s
    lo_w     = clean.min()
    hi_w     = clean.max()
    outliers = s[(s < lo_w) | (s > hi_w)]
    return dict(n=n, q1=q1, med=med, q3=q3, lo=lo_w, hi=hi_w,
                mean=clean.mean(), sd=sd, outliers=outliers, clean=clean, raw=s)

# ─────────────────────────────────────────────────────────────
#  LEGEND GROUP → ROW IDS MAPPING
# ─────────────────────────────────────────────────────────────
def build_legendgroup_map(df, param, pH_filter, dir_filter, r2_min,
                          sample_filter, deleted_ids, log_icorr,
                          symbol_overrides=None):
    """
    Returns a dict:  legend_name -> list[row_id]
    Only includes mark-data traces (not box/whisker traces).
    Mirrors the grouping logic in make_figure exactly.
    """
    symbol_overrides = symbol_overrides or {}

    def mark_symbol(test):
        t = str(test)
        return symbol_overrides.get(t, TEST_SYMBOL.get(t, "circle"))

    is_ocp = param.startswith("OCP")
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

    for col_name, default in [("test", "1"), ("folder", "01"), ("pH", "1"), ("point", "?")]:
        if col_name not in sub.columns: sub[col_name] = default
        sub[col_name] = sub[col_name].fillna(default).astype(str)

    if param == "Icorr" and log_icorr:
        sub = sub[sub["value"] > 0].copy()

    group_map = {}  # leg_name -> [row_ids]

    for cond in COND_ORDER:
        if cond not in sub["condition"].unique():
            continue
        for cut in ["LC", "WJ"]:
            csub = sub[(sub["condition"] == cond) & (sub["cut"] == cut)]
            if csub.empty:
                continue
            for ph_val in sorted(csub["pH"].unique()):
                ph_sub = csub[csub["pH"] == ph_val]
                for folder_val in sorted(ph_sub["folder"].unique()):
                    for t_val in sorted(csub["test"].unique()):
                        mask = ((csub["pH"]     == ph_val)     &
                                (csub["folder"] == folder_val) &
                                (csub["test"]   == t_val))
                        tph = csub[mask]
                        if tph.empty:
                            continue
                        symbol = mark_symbol(t_val)
                        glyph  = SYMBOL_GLYPH.get(symbol, "•")
                        t_lbl  = f"{glyph}T{t_val}"
                        actual = sorted(tph["sample"].unique())
                        samp_fol = ", ".join(f"{s}_{folder_val}" for s in actual)
                        leg_name = f"{samp_fol}  {t_lbl}"
                        rids = tph["row_id"].tolist()
                        if leg_name not in group_map:
                            group_map[leg_name] = []
                        group_map[leg_name].extend(rids)

    return group_map


# ─────────────────────────────────────────────────────────────
#  DISCOVER PRESENT (for customise panel)
# ─────────────────────────────────────────────────────────────
def discover_present(df, param, pH_filter, dir_filter, r2_min,
                     sample_filter, deleted_ids):
    is_ocp = param.startswith("OCP")
    sub = df[df["parameter"] == param].copy()
    if not is_ocp:
        if dir_filter != "Both":
            sub = sub[sub["direction"].str.upper() == dir_filter.upper()]
        sub = sub[(sub["r2"].isna()) | (sub["r2"] >= r2_min)]
    if pH_filter != "Both":
        sub = sub[sub["pH"] == str(pH_filter)]
    if sample_filter:
        sub = sub[sub["sample"].str.upper() == sample_filter.upper()]
    sub = sub[~sub["row_id"].isin(deleted_ids)]
    for c, d in [("folder", "01"), ("pH", "1"), ("test", "1")]:
        if c not in sub.columns: sub[c] = d
        sub[c] = sub[c].fillna(d).astype(str)

    combos = {}
    for (cut, ph, fol), g in sub.groupby(["cut", "pH", "folder"]):
        if not cut: continue
        labels = sorted({f"{s}_{fol}" for s in g["sample"].unique()})
        combos[(cut, str(ph), str(fol))] = labels
    tests = sorted(sub["test"].unique())
    return combos, tests


# ─────────────────────────────────────────────────────────────
#  FIGURE
# ─────────────────────────────────────────────────────────────
def make_figure(df, param, pH_filter, dir_filter,
                r2_min, sd_mult, log_icorr, deleted_ids, sample_filter,
                color_overrides=None, symbol_overrides=None):

    color_overrides  = color_overrides  or {}
    symbol_overrides = symbol_overrides or {}

    def mark_colour(cut, ph, folder):
        key = (cut, str(ph), str(folder))
        if key in color_overrides:
            return color_overrides[key]
        return FOLDER_COLOURS.get(key, FALLBACK_COLOUR)

    def mark_symbol(test):
        t = str(test)
        if t in symbol_overrides:
            return symbol_overrides[t]
        return TEST_SYMBOL.get(t, "circle")

    is_ocp = param.startswith("OCP")

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

    for col_name, default in [("test", "1"), ("folder", "01"), ("pH", "1"), ("point", "?")]:
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
                           showarrow=False, font_size=18, font_color="#888")
        fig.update_layout(height=560, plot_bgcolor="white", paper_bgcolor="white")
        return fig

    cuts_with_data = set(sub["cut"].unique())
    both_cuts      = ("LC" in cuts_with_data) and ("WJ" in cuts_with_data)
    CUT_OFFSET     = {
        "LC": -0.22 if both_cuts else 0.0,
        "WJ":  0.22 if both_cuts else 0.0,
    }

    all_v = sub["value"].dropna().values
    y_min = all_v.min(); y_max = all_v.max()
    y_rng = max(y_max - y_min, 1e-9)
    y_pad = y_rng * 0.14
    y_lo  = y_min - y_pad
    y_hi  = y_max + y_pad

    slot   = 1.0
    box_hw = 0.14
    cap_hw = 0.10

    added_box_legend  = set()
    added_mark_legend = set()
    tick_vals, tick_text = [], []

    for ci, cond in enumerate(conds):
        cx = ci * slot + slot / 2
        tick_vals.append(cx)
        tick_text.append(cond)

        if ci > 0:
            fig.add_shape(type="line",
                x0=ci*slot, x1=ci*slot, y0=y_lo, y1=y_hi,
                line=dict(color="#B0BEC5", width=1.5))

        if ci % 2 == 1:
            fig.add_shape(type="rect",
                x0=ci*slot, x1=(ci+1)*slot, y0=y_lo, y1=y_hi,
                fillcolor="rgba(240,244,248,0.55)",
                line=dict(width=0), layer="below")

        for cut in ["LC", "WJ"]:
            if cut not in cuts_with_data:
                continue

            col  = CUT_COL[cut]
            fill = CUT_FILL[cut]
            bx   = cx + CUT_OFFSET[cut]

            csub = sub[(sub["condition"] == cond) & (sub["cut"] == cut)]
            vals = csub["value"].dropna().values
            if len(vals) == 0:
                continue

            st_ = compute_stats(vals, sd_mult)
            if st_ is None:
                continue

            if cut not in added_box_legend:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    name=f"{cut}  box",
                    marker=dict(color=col, size=13, symbol="square",
                                line=dict(color=col, width=2)),
                    legendgroup=f"box_{cut}", showlegend=True))
                added_box_legend.add(cut)

            fig.add_shape(type="line", x0=bx, x1=bx,
                y0=st_["q3"], y1=st_["hi"],
                line=dict(color=col, width=2.2))
            fig.add_shape(type="line",
                x0=bx-cap_hw, x1=bx+cap_hw, y0=st_["hi"], y1=st_["hi"],
                line=dict(color=col, width=2.5))
            fig.add_shape(type="line", x0=bx, x1=bx,
                y0=st_["lo"], y1=st_["q1"],
                line=dict(color=col, width=2.2))
            fig.add_shape(type="line",
                x0=bx-cap_hw, x1=bx+cap_hw, y0=st_["lo"], y1=st_["lo"],
                line=dict(color=col, width=2.5))
            fig.add_shape(type="rect",
                x0=bx-box_hw, x1=bx+box_hw, y0=st_["q1"], y1=st_["q3"],
                fillcolor=fill, line=dict(color=col, width=2.5))
            fig.add_shape(type="line",
                x0=bx-box_hw, x1=bx+box_hw, y0=st_["med"], y1=st_["med"],
                line=dict(color=col, width=3.5))
            fig.add_trace(go.Scatter(
                x=[bx], y=[st_["mean"]], mode="markers",
                marker=dict(symbol="circle-open", size=12, color=col,
                            line=dict(color=col, width=2.5)),
                legendgroup=f"box_{cut}", showlegend=False,
                hovertemplate=f"<b>Mean ({cut}–{cond})</b>: %{{y:.5f}}<extra></extra>"))
            if len(st_["outliers"]) > 0:
                fig.add_trace(go.Scatter(
                    x=[bx]*len(st_["outliers"]), y=st_["outliers"].tolist(),
                    mode="markers",
                    marker=dict(symbol="circle-open", size=10, color=col,
                                line=dict(color=col, width=2.0)),
                    legendgroup=f"box_{cut}", showlegend=False,
                    hovertemplate=f"<b>Outlier ({cut})</b>: %{{y:.5f}}<extra></extra>"))

            np.random.seed(42 + (0 if cut == "LC" else 1)*100 + ci*10)
            jitter_w = box_hw * 0.55

            for ph_val in sorted(csub["pH"].unique()):
                ph_sub = csub[csub["pH"] == ph_val]
                for folder_val in sorted(ph_sub["folder"].unique()):
                    for t_val in sorted(csub["test"].unique()):
                        mask = ((csub["pH"]     == ph_val)     &
                                (csub["folder"] == folder_val) &
                                (csub["test"]   == t_val))
                        tph   = csub[mask]
                        tvals = tph["value"].dropna().values
                        if len(tvals) == 0: continue

                        mark_col = mark_colour(cut, ph_val, folder_val)
                        symbol   = mark_symbol(t_val)
                        filled   = is_filled_symbol(symbol)
                        glyph    = SYMBOL_GLYPH.get(symbol, "•")
                        t_lbl    = f"{glyph}T{t_val}"

                        actual   = sorted(tph["sample"].unique())
                        samp_fol = ", ".join(f"{s}_{folder_val}" for s in actual)
                        leg_key  = f"{cut}_pH{ph_val}_F{folder_val}_T{t_val}"
                        leg_name = f"{samp_fol}  {t_lbl}"
                        show_m   = leg_key not in added_mark_legend

                        jitter = np.random.uniform(-jitter_w, jitter_w, len(tvals))
                        custom = tph[["row_id", "sample", "folder", "test",
                                      "point", "pH", "direction"]].values.tolist()

                        fig.add_trace(go.Scatter(
                            x=(bx + jitter).tolist(),
                            y=tvals.tolist(),
                            mode="markers",
                            marker=dict(
                                symbol=symbol,
                                size=9 if filled else 8,
                                color=mark_col if filled else "rgba(0,0,0,0)",
                                line=dict(color=mark_col, width=2.2),
                                opacity=0.92,
                            ),
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
                x=bx, y=y_lo - y_pad*0.50,
                text=f"n={st_['n']}",
                showarrow=False,
                font=dict(size=10, color=col, family="Arial"),
                xanchor="center")

    if conds:
        first_cx = slot / 2
        for cut_lbl in ["LC", "WJ"]:
            if cut_lbl not in cuts_with_data:
                continue
            fig.add_annotation(
                x=first_cx + CUT_OFFSET[cut_lbl],
                y=y_hi + y_pad * 0.06,
                text=f"<b>{cut_lbl}</b>",
                showarrow=False,
                font=dict(size=12, color=CUT_COL[cut_lbl], family="Arial"),
                xanchor="center")

    pH_str     = f"pH {pH_filter}" if pH_filter != "Both" else "pH 1 & 4"
    dir_str    = f" | {dir_filter}" if not is_ocp and dir_filter != "Both" else ""
    sample_str = f" | {sample_filter}" if sample_filter else ""

    fig.update_layout(
        title=dict(
            text=f"<b>{param}</b>  |  {pH_str}{dir_str}{sample_str}",
            font=dict(size=17, color="#1A237E", family="Arial"),
            x=0.5, pad=dict(b=10)),
        xaxis=dict(
            tickvals=tick_vals, ticktext=tick_text,
            tickfont=dict(size=13, color="#1A237E", family="Arial"),
            title=dict(text="Condition",
                       font=dict(size=13, color="#333", family="Arial")),
            showgrid=False, zeroline=False,
            range=[-0.5, n_c*slot - 0.5],
            showline=True, linewidth=2.5, linecolor="#455A64",
            mirror=False, ticks="outside", tickwidth=2,
            ticklen=7, tickcolor="#455A64"),
        yaxis=dict(
            title=dict(text=ylabel,
                       font=dict(size=13, color="#333", family="Arial")),
            tickfont=dict(size=12, color="#333", family="Arial"),
            gridcolor="#CFD8DC", gridwidth=1.2,
            zeroline=True, zerolinecolor="#90A4AE", zerolinewidth=1.5,
            showline=True, linewidth=2.5, linecolor="#455A64",
            mirror=False, ticks="outside", tickwidth=2,
            ticklen=7, tickcolor="#455A64",
            range=[y_lo - y_pad*0.6, y_hi + y_pad*0.25]),
        legend=dict(
            title=dict(text="<b>Legend</b>",
                       font=dict(size=13, color="#1A237E", family="Arial")),
            orientation="v", x=1.02, y=1.00,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.97)",
            bordercolor="#90A4AE", borderwidth=2,
            font=dict(size=11, family="Arial", color="#222"),
            tracegroupgap=6, itemsizing="constant", itemwidth=40),
        plot_bgcolor="white", paper_bgcolor="white",
        height=580, margin=dict(l=80, r=280, t=75, b=80),
        clickmode="event",
    )

    return fig


# ─────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <style>
    .block-container{padding-top:1rem}
    h1{color:#1A237E;font-size:1.6rem;font-family:Arial}
    </style>""", unsafe_allow_html=True)

    st.title("📊 Electrolyzer Whisker — SS316L Corrosion Study")
    st.caption(
        "● circle = Test 1  ·  ✕ x = Test 2  ·  "
        "Colour = subsample folder  ·  Click marks to exclude"
    )

    if "deleted_ids"      not in st.session_state: st.session_state.deleted_ids      = set()
    if "deleted_meta"     not in st.session_state: st.session_state.deleted_meta     = {}
    if "df"               not in st.session_state: st.session_state.df               = None
    if "color_overrides"  not in st.session_state: st.session_state.color_overrides  = {}
    if "symbol_overrides" not in st.session_state: st.session_state.symbol_overrides = {}

    with st.sidebar:
        st.header("📂 Data source")
        src = st.radio("Load from:", ["Upload raw files", "Pre-filled Excel"])

        if src == "Upload raw files":
            lf = st.file_uploader("batch_fit_summary.xlsx", type=["xlsx"])
            of = st.file_uploader("ocp_summary.xlsx",       type=["xlsx"])
            if lf and of:
                with st.spinner("Parsing files…"):
                    try:
                        st.session_state.df = load_uploaded(lf.read(), of.read())
                        st.success(f"✅ {len(st.session_state.df):,} rows loaded")
                    except Exception as e:
                        st.error(f"Parse error: {e}")
        else:
            if st.session_state.df is None:
                with st.spinner("Loading pre-filled data…"):
                    st.session_state.df = load_prefilled()
                if st.session_state.df is not None:
                    st.success(f"✅ {len(st.session_state.df):,} rows loaded")
                else:
                    st.warning("Place ElectrolyzerWhisker_Final.xlsx in the "
                               "app folder, or switch to Upload above.")

        if st.session_state.df is None:
            st.info("Upload both files to get started.")
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
                      else st.selectbox("Direction", ["Both", "Anodic", "Cathodic"]))

        samp_opts     = [""] + sorted(df["sample"].unique())
        sample_filter = st.selectbox("Sample (blank = all)", samp_opts)

        st.divider()
        st.header("📐 Statistics")
        r2_min    = st.slider("R² minimum",      0.0, 1.0, 0.90, 0.01)
        sd_mult   = st.slider("Outlier × StDev", 1.0, 4.0, 2.0,  0.5)
        log_icorr = st.checkbox("Log₁₀(Icorr)", value=True)

        st.divider()
        st.header("🎨 Mark encoding")
        st.markdown(
            "**Shape = test**<br>"
            '<span style="font-size:15px">●</span>'
            '<span style="font-size:11px"> Filled circle = Test 1</span><br>'
            '<span style="font-size:15px">✕</span>'
            '<span style="font-size:11px"> X mark = Test 2</span>',
            unsafe_allow_html=True)
        st.markdown("**Colour = subsample folder**")
        groups = [
            ("LC pH 1", [
                ("#FF1744", "S52_01, S60_01, S64_01"),
                ("#E53935", "S60_02, S62_02"),
                ("#B71C1C", "S51_03, S53_03"),
                ("#F06292", "S50_04, S53_04, S62_04, S63_04 (T1)"),
                ("#FF7043", "S50_05, S63_05"),
                ("#E040FB", "S61_06, S62_06, S64_06"),
                ("#D81B60", "S61_07, S63_07"),
                ("#7B1FA2", "S63_09 (T2)"),
                ("#C2185B", "S64_10 (T2)"),
            ]),
            ("LC pH 4", [
                ("#FB8C00", "S50_01, S51_01, S53_01, S61_01, S62_01, S63_01"),
                ("#F9A825", "S50_02, S51_02, S53_02, S61_02, S63_02"),
                ("#FFD600", "S52_03, S61_03, S63_03, S64_03"),
                ("#FFAB40", "S52_04, S61_04, S63_04 (T2)"),
                ("#FDD835", "S51_05, S62_05, S64_05"),
                ("#FF6F00", "S51_06"),
                ("#FFB300", "S63_08"),
                ("#A5D6A7", "S60_09, S62_09, S63_09 (T1), S64_09"),
                ("#A1887F", "S52_10, S60_10, S62_10, S64_10 (T1)"),
                ("#26C6DA", "S64_27"),
                ("#00BFA5", "S64_29"),
            ]),
            ("WJ pH 1", [
                ("#0D47A1", "S73_01"),
                ("#1E88E5", "S73_02"),
                ("#1565C0", "S70_03, S71_03, S73_03"),
                ("#3949AB", "S70_04"),
                ("#283593", "S74_06"),
                ("#4527A0", "S72_07, S74_07"),
                ("#5C6BC0", "S72_08"),
                ("#039BE5", "S73_09 (T2)"),
            ]),
            ("WJ pH 4", [
                ("#00897B", "S70_01, S71_01, S74_01"),
                ("#00ACC1", "S70_02, S71_02, S72_02, S74_02"),
                ("#43A047", "S74_03"),
                ("#F57C00", "S73_04"),
                ("#7CB342", "S73_05, S74_05"),
                ("#009688", "S73_06"),
                ("#8D6E63", "S73_07"),
                ("#26A69A", "S73_09 (T1)"),
                ("#0288D1", "S72_10, S73_10"),
            ]),
        ]
        for group_title, entries in groups:
            st.markdown(f"*{group_title}*")
            for hex_c, label in entries:
                st.markdown(
                    f'<span style="color:{hex_c};font-size:16px">●</span>'
                    f' <span style="font-size:11px;color:#333">{label}</span>',
                    unsafe_allow_html=True)

        st.divider()
        st.header("✏️ Customise plot")
        combos, tests = discover_present(
            df, param, pH_filter, dir_filter, r2_min,
            sample_filter, st.session_state.deleted_ids)

        with st.expander("Shapes (per test)", expanded=False):
            if not tests:
                st.caption("No data for the current filters.")
            for t in tests:
                base_sym = TEST_SYMBOL.get(str(t), "circle")
                cur_sym  = st.session_state.symbol_overrides.get(str(t), base_sym)
                idx      = (SYMBOL_OPTIONS.index(cur_sym)
                            if cur_sym in SYMBOL_OPTIONS else 0)
                chosen = st.selectbox(
                    f"Test {t}  {SYMBOL_GLYPH.get(base_sym, '')}",
                    SYMBOL_OPTIONS, index=idx, key=f"sym_{t}")
                if chosen != base_sym:
                    st.session_state.symbol_overrides[str(t)] = chosen
                else:
                    st.session_state.symbol_overrides.pop(str(t), None)

        with st.expander("Colours (per sample / folder)", expanded=False):
            if not combos:
                st.caption("No data for the current filters.")
            for key in sorted(combos):
                cut, ph, fol = key
                base_col = FOLDER_COLOURS.get(key, FALLBACK_COLOUR)
                cur_col  = st.session_state.color_overrides.get(key, base_col)
                label    = (", ".join(combos[key])
                            + f"   ({cut} · pH{ph} · F{fol})")
                chosen = st.color_picker(
                    label, value=cur_col,
                    key=f"col_{cut}_{ph}_{fol}")
                if chosen.upper() != base_col.upper():
                    st.session_state.color_overrides[key] = chosen
                else:
                    st.session_state.color_overrides.pop(key, None)

        if st.button("↺ Reset colours & shapes", use_container_width=True):
            st.session_state.color_overrides  = {}
            st.session_state.symbol_overrides = {}
            st.rerun()

        st.divider()
        st.download_button(
            "⬇ Download filtered CSV",
            data=df[~df["row_id"].isin(st.session_state.deleted_ids)
                   ].to_csv(index=False).encode(),
            file_name="electrolyzer_filtered.csv",
            mime="text/csv", use_container_width=True)

        if st.session_state.deleted_meta:
            exc_df = pd.DataFrame(list(st.session_state.deleted_meta.values()))
            st.download_button(
                "⬇ Download excluded log (CSV)",
                data=exc_df.to_csv(index=False).encode(),
                file_name="electrolyzer_excluded.csv",
                mime="text/csv", use_container_width=True)

    df       = st.session_state.df
    param_df = df[df["parameter"] == param]
    active   = param_df[~param_df["row_id"].isin(st.session_state.deleted_ids)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total (this param)", len(param_df))
    c2.metric("Active",             len(active))
    c3.metric("Excluded",           len(st.session_state.deleted_ids))
    c4.metric("Conditions",         active["condition"].nunique())

    # ── Build legend-group → row_ids map for the exclude widget ──
    lg_map = build_legendgroup_map(
        df, param, pH_filter, dir_filter, r2_min, sample_filter,
        st.session_state.deleted_ids, log_icorr,
        symbol_overrides=st.session_state.symbol_overrides,
    )

    fig = make_figure(
        df=df, param=param, pH_filter=pH_filter, dir_filter=dir_filter,
        r2_min=r2_min, sd_mult=sd_mult, log_icorr=log_icorr,
        deleted_ids=st.session_state.deleted_ids,
        sample_filter=sample_filter,
        color_overrides=st.session_state.color_overrides,
        symbol_overrides=st.session_state.symbol_overrides,
    )

    _ov = abs(hash(
        str(sorted(st.session_state.color_overrides.items())) +
        str(sorted(st.session_state.symbol_overrides.items()))
    ))
    event = st.plotly_chart(
        fig, use_container_width=True, on_select="rerun",
        key=(f"chart_{param}_{pH_filter}_{dir_filter}_"
             f"{sample_filter}_{len(st.session_state.deleted_ids)}_{_ov}"),
    )

    # ── Legend-group bulk exclude widget ──────────────────────
    if lg_map:
        with st.expander("🗑 Exclude by legend group (click-to-remove from whisker)", expanded=False):
            st.caption(
                "Select one or more legend groups below to exclude **all** their "
                "points at once. The whisker and statistics will auto-scale."
            )
            lg_names = sorted(lg_map.keys())
            chosen_groups = st.multiselect(
                "Legend groups to exclude:",
                options=lg_names,
                default=[],
                key=f"lg_exclude_{param}_{pH_filter}_{dir_filter}_{sample_filter}",
                placeholder="Start typing a sample name…",
            )
            if st.button("🗑 Exclude selected groups", use_container_width=True,
                         disabled=len(chosen_groups) == 0,
                         key=f"lg_exclude_btn_{param}"):
                df_ref = st.session_state.df
                for grp_name in chosen_groups:
                    for rid in lg_map.get(grp_name, []):
                        if rid not in st.session_state.deleted_ids:
                            row_info = df_ref[df_ref["row_id"] == rid]
                            if not row_info.empty:
                                r = row_info.iloc[0]
                                meta = {
                                    "row_id":    rid,
                                    "parameter": r.get("parameter", param),
                                    "sample":    r.get("sample", "?"),
                                    "cut":       r.get("cut", "?"),
                                    "condition": r.get("condition", "?"),
                                    "pH":        r.get("pH", "?"),
                                    "direction": r.get("direction", "?"),
                                    "folder":    r.get("folder", "?"),
                                    "test":      r.get("test", "?"),
                                    "point":     r.get("point", "?"),
                                    "source":    r.get("source", "?"),
                                    "value":     float(r.get("value", float("nan"))),
                                    "r2":        r.get("r2", None),
                                }
                            else:
                                meta = {
                                    "row_id": rid, "parameter": param,
                                    "sample": "?", "cut": "?", "condition": "?",
                                    "pH": "?", "direction": "?", "folder": "?",
                                    "test": "?", "point": "?", "source": "?",
                                    "value": float("nan"), "r2": None,
                                }
                            st.session_state.deleted_ids.add(rid)
                            st.session_state.deleted_meta[rid] = meta
                st.rerun()

    # ── Point-click exclude (existing behaviour) ──────────────
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
                rid=str(cdata[0]) if isinstance(cdata, list) else str(cdata)
                samp=folder=test=point=ph=direction="?"
                raw_val=pt.get("y", float("nan"))

            if rid and rid not in st.session_state.deleted_ids:
                row_info = df[df["row_id"]==rid]
                if not row_info.empty:
                    r = row_info.iloc[0]
                    meta = {
                        "row_id":    rid,
                        "parameter": r.get("parameter", param),
                        "sample":    r.get("sample", samp),
                        "cut":       r.get("cut", "?"),
                        "condition": r.get("condition", "?"),
                        "pH":        r.get("pH", ph),
                        "direction": r.get("direction", direction),
                        "folder":    r.get("folder", folder),
                        "test":      r.get("test", test),
                        "point":     r.get("point", point),
                        "source":    r.get("source", "?"),
                        "value":     float(r.get("value", raw_val)),
                        "r2":        r.get("r2", None),
                    }
                else:
                    meta = {
                        "row_id":rid, "parameter":param, "sample":samp, "cut":"?",
                        "condition":"?", "pH":ph, "direction":direction,
                        "folder":folder, "test":test, "point":point,
                        "source":"?", "value":raw_val, "r2":None}
                st.session_state.deleted_ids.add(rid)
                st.session_state.deleted_meta[rid] = meta
                st.rerun()

    st.caption(
        "💡 **●** = Test 1  ·  **✕** = Test 2  ·  "
        "Colour = subsample folder  ·  Hover for details  ·  Click to exclude"
    )

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
                "parameter", "sample", "cut", "condition", "pH",
                "direction", "folder", "test", "point", "source", "value", "r2"
            ]].copy()
            exc_df["value"] = exc_df["value"].round(6)
            exc_df["r2"]    = exc_df["r2"].apply(
                lambda x: round(float(x), 4) if pd.notna(x) else "—")
            exc_df.columns  = ["Parameter", "Sample", "Cut", "Condition", "pH",
                                "Direction", "Folder", "Test", "Point",
                                "Source", "Value", "R²"]

            def _style_exc(row):
                bg = "#FAD7D3" if row["Cut"] == "LC" else "#D0E8F7"
                return [f"background-color:{bg}" if c == "Cut" else ""
                        for c in exc_df.columns]

            st.dataframe(exc_df.style.apply(_style_exc, axis=1),
                         use_container_width=True,
                         height=min(50+35*len(exc_df), 400))

            st.markdown("**Restore individual points:**")
            for rid, meta in list(st.session_state.deleted_meta.items()):
                label = (
                    f"↺  {meta['parameter']}  ·  {meta['sample']}  ·  "
                    f"{meta['condition']}  ·  pH {meta['pH']}  ·  "
                    f"Folder {meta['folder']}  ·  Test {meta['test']}  ·  "
                    f"Point {meta['point']}  ·  Value = {meta['value']:.5f}"
                )
                if st.button(label, key=f"restore_{rid}",
                             use_container_width=True):
                    st.session_state.deleted_ids.discard(rid)
                    st.session_state.deleted_meta.pop(rid, None)
                    st.rerun()

    st.subheader("📋 Summary Statistics")

    sub = df[~df["row_id"].isin(st.session_state.deleted_ids)]
    sub = sub[sub["parameter"] == param].copy()
    if pH_filter != "Both":
        sub = sub[sub["pH"] == str(pH_filter)]
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
            if len(vals) == 0: continue
            st_ = compute_stats(vals, sd_mult)
            if st_ is None: continue
            stat_rows.append({
                "Condition":  cond, "Cut": cut, "n": st_["n"],
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
        st.dataframe(sdf.style.apply(_color_row, axis=1),
                     use_container_width=True, height=300)
    else:
        st.info("No data for the selected filters.")


if __name__ == "__main__":
    main()
