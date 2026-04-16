"""
Interactive MLP predictions: historical backtest + 2026 in-app inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from extra_credit.mlp_core import ACTIVATIONS, KPB_FEATURES, MLP_predict_regression, feature_diff_names
from streamlit_app._paths import extra_credit_data_dir, extra_credit_dir, predictions_dir

st.set_page_config(page_title="MLP explorer", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    .block { padding-bottom: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("MLP Margin Explorer")
st.markdown(
    "Compare **predicted** tournament point margin to **actual** margin where results exist. "
    "Use the filters below to explore historical results and this year's 2026 March Madness matchups."
)

pred_dir = predictions_dir()
hist_path = pred_dir / "mlp.csv"
ec_dir = extra_credit_dir()
ec_data_dir = extra_credit_data_dir()
checkpoint_path = ec_dir / "mlp_checkpoint.npz"
matchups_2026_path = ec_data_dir / "tournament_matchups_2026.csv"
kpb_2026_path = ec_data_dir / "kenpom_barttorvik_2026.csv"

if not hist_path.is_file():
    st.error(f"Missing `{hist_path}`. Train the MLP notebook and export `mlp.csv`.")
    st.stop()


@st.cache_data
def load_historical() -> pd.DataFrame:
    return pd.read_csv(hist_path)


def load_checkpoint():
    z = np.load(checkpoint_path, allow_pickle=True)
    feature_cols = list(z["feature_cols"])
    weights = []
    biases = []
    i = 0
    while f"w{i}" in z.files:
        weights.append(z[f"w{i}"])
        biases.append(z[f"b{i}"])
        i += 1
    mean = z["scaler_mean"]
    scale = z["scaler_scale"]
    hidden_names = list(z["hidden_names"])
    activations = [ACTIVATIONS[str(name)] for name in hidden_names]
    return weights, biases, activations, mean, scale, feature_cols


def pair_games(tm: pd.DataFrame) -> pd.DataFrame:
    tm = tm.copy()
    tm["_byn"] = tm["BY YEAR NO"].astype(int)
    tm = tm.sort_values("_byn", ascending=False)
    rows = []
    for i in range(0, len(tm) - 1, 2):
        a, b = tm.iloc[i], tm.iloc[i + 1]
        if a["CURRENT ROUND"] != b["CURRENT ROUND"]:
            raise ValueError(f"Mismatched CURRENT ROUND for pair starting {a['TEAM']} / {b['TEAM']}")
        s1, s2 = a["SCORE"], b["SCORE"]
        actual = np.nan
        if pd.notna(s1) and str(s1).strip() != "" and pd.notna(s2) and str(s2).strip() != "":
            actual = float(int(s1) - int(s2))
        rows.append(
            {
                "YEAR": int(a["YEAR"]),
                "CURRENT_ROUND": int(a["CURRENT ROUND"]),
                "TEAM1": a["TEAM"],
                "TEAM2": b["TEAM"],
                "TEAM1_SCORE": s1 if str(s1).strip() != "" else np.nan,
                "TEAM2_SCORE": s2 if str(s2).strip() != "" else np.nan,
                "POINT_DIFFERENTIAL": actual,
            }
        )
    return pd.DataFrame(rows)


def normalize_2026_round(value: int) -> int:
    value = int(value)
    round_map = {
        0: 64,
        1: 64,
        2: 32,
        4: 16,
        8: 8,
        16: 4,
        32: 2,
    }
    return round_map.get(value, value)


@st.cache_data
def build_2026_in_app() -> pd.DataFrame:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing `{checkpoint_path}`. Run the checkpoint-save cell at the bottom of `mlp_model.ipynb`."
        )
    if not matchups_2026_path.is_file() or not kpb_2026_path.is_file():
        raise FileNotFoundError(
            f"Missing `{matchups_2026_path}` or `{kpb_2026_path}`. Keep the slim 2026 files in `extra_credit/data/`."
        )

    weights, biases, activations, mean, scale, ck_cols = load_checkpoint()
    tm = pd.read_csv(matchups_2026_path)
    kpb = pd.read_csv(kpb_2026_path)
    games = pair_games(tm)

    kpb_stats = kpb[["YEAR", "TEAM"] + KPB_FEATURES].copy()
    team1_stats = kpb_stats.add_prefix("T1_").rename(columns={"T1_YEAR": "YEAR", "T1_TEAM": "TEAM1"})
    merged = games.merge(team1_stats, on=["YEAR", "TEAM1"], how="left")
    team2_stats = kpb_stats.add_prefix("T2_").rename(columns={"T2_YEAR": "YEAR", "T2_TEAM": "TEAM2"})
    merged = merged.merge(team2_stats, on=["YEAR", "TEAM2"], how="left")

    feature_cols = feature_diff_names()
    for c in KPB_FEATURES:
        merged[f"DIFF_{c}"] = merged[f"T1_{c}"] - merged[f"T2_{c}"]

    if ck_cols != feature_cols:
        raise ValueError(f"Feature mismatch checkpoint {ck_cols} vs current {feature_cols}")

    missing = merged[feature_cols].isna().any(axis=1)
    if missing.any():
        bad = merged.loc[missing, ["TEAM1", "TEAM2"]]
        raise ValueError(f"Missing 2026 feature rows for:\n{bad}")

    X = merged[feature_cols].to_numpy(dtype=float)
    Xs = (X - mean) / scale
    pred = np.asarray(MLP_predict_regression(Xs, weights, biases, activations=activations)).reshape(-1)

    out = merged[["YEAR", "CURRENT_ROUND", "TEAM1", "TEAM2"]].copy()
    out["CURRENT_ROUND"] = out["CURRENT_ROUND"].map(normalize_2026_round)
    out["Round (teams remaining)"] = out["CURRENT_ROUND"]
    out["SPLIT"] = "2026_tournament"
    out["PRED_MARGIN"] = pred
    out["POINT_DIFFERENTIAL"] = merged["POINT_DIFFERENTIAL"]
    out["ABS_ERROR"] = np.abs(out["POINT_DIFFERENTIAL"] - out["PRED_MARGIN"])
    out["SQ_ERROR"] = (out["POINT_DIFFERENTIAL"] - out["PRED_MARGIN"]) ** 2
    out["TEAM1_SCORE"] = merged["TEAM1_SCORE"]
    out["TEAM2_SCORE"] = merged["TEAM2_SCORE"]
    return out.sort_values(["CURRENT_ROUND", "TEAM1", "TEAM2"]).reset_index(drop=True)


mode = st.radio(
    "Dataset",
    ["Historical evaluation (2008–2025)", "2026 tournament (MLP + checkpoint)"],
    horizontal=True,
)

if mode.startswith("Historical"):
    df = load_historical()
    st.caption(f"{len(df):,} games")

    years = sorted(df["YEAR"].dropna().unique())
    default_max = max(years) if years else None
    y_pick = st.multiselect("Season (YEAR)", options=years, default=[default_max] if default_max else years[:1])
    split_pick = st.multiselect(
        "Split",
        options=sorted(df["SPLIT"].dropna().unique()),
        default=list(df["SPLIT"].dropna().unique()),
    )
    rounds = sorted(df["CURRENT_ROUND"].dropna().unique())
    r_pick = st.multiselect("Round (teams remaining)", options=rounds, default=rounds)
    team_q = st.text_input("Filter by team name (optional)", "").strip()

    d = df[df["YEAR"].isin(y_pick) & df["SPLIT"].isin(split_pick) & df["CURRENT_ROUND"].isin(r_pick)]
    if team_q:
        mask = d["TEAM1"].str.contains(team_q, case=False, na=False) | d["TEAM2"].str.contains(team_q, case=False, na=False)
        d = d[mask]

    st.subheader("Predicted vs Actual Margin")
    st.caption("Each point is one tournament game. Diagonal = perfect prediction (Team 1 margin).")

    fig = px.scatter(
        d,
        x="PRED_MARGIN",
        y="POINT_DIFFERENTIAL",
        color="SPLIT",
        hover_data=["YEAR", "CURRENT_ROUND", "TEAM1", "TEAM2"],
        labels={"PRED_MARGIN": "Predicted margin (Team 1)", "POINT_DIFFERENTIAL": "Actual margin (Team 1)"},
        color_discrete_sequence=["#1e3a5f", "#D2691E"],
    )
    lo = float(min(d["PRED_MARGIN"].min(), d["POINT_DIFFERENTIAL"].min()))
    hi = float(max(d["PRED_MARGIN"].max(), d["POINT_DIFFERENTIAL"].max()))
    pad = max(3.0, (hi - lo) * 0.05)
    fig.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad],
            y=[lo - pad, hi + pad],
            mode="lines",
            line=dict(dash="dash", color="rgba(80,80,80,0.5)"),
            name="Perfect",
        )
    )
    fig.update_layout(template="plotly_white", height=520)
    st.plotly_chart(fig, use_container_width=True)

    err = (d["POINT_DIFFERENTIAL"] - d["PRED_MARGIN"]).abs()
    c1, c2, c3 = st.columns(3)
    c1.metric("Games shown", f"{len(d):,}")
    c2.metric("MAE (subset)", f"{err.mean():.2f}")
    c3.metric("RMSE (subset)", f"{((err ** 2).mean()) ** 0.5:.2f}")

    st.subheader("Games")
    st.dataframe(d.sort_values(["YEAR", "CURRENT_ROUND", "TEAM1"]), use_container_width=True, hide_index=True)

else:
    try:
        d26 = build_2026_in_app()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.caption(f"{len(d26):,} paired games computed in-app from the saved MLP checkpoint and current 2026 source data.")

    r_pick = st.multiselect(
        "Round (teams remaining)",
        options=sorted(d26["Round (teams remaining)"].dropna().unique()),
        default=sorted(d26["Round (teams remaining)"].dropna().unique()),
    )
    team_q = st.text_input("Filter by team name (optional)", "", key="t26").strip()
    d = d26[d26["Round (teams remaining)"].isin(r_pick)]
    if team_q:
        mask = d["TEAM1"].str.contains(team_q, case=False, na=False) | d["TEAM2"].str.contains(team_q, case=False, na=False)
        d = d[mask]

    st.subheader("Predicted vs Actual Margin")
    dd = d[d["POINT_DIFFERENTIAL"].notna()]
    if len(dd) == 0:
        st.caption("No rows with both scores yet.")
        fig = px.scatter(title="No completed games in filtered set")
    else:
        fig = px.scatter(
            dd,
            x="PRED_MARGIN",
            y="POINT_DIFFERENTIAL",
            hover_data=["Round (teams remaining)", "TEAM1", "TEAM2", "TEAM1_SCORE", "TEAM2_SCORE"],
            labels={"PRED_MARGIN": "Predicted margin (Team 1)", "POINT_DIFFERENTIAL": "Actual margin (Team 1)"},
            color_discrete_sequence=["#1e3a5f"],
        )
        lo = float(min(dd["PRED_MARGIN"].min(), dd["POINT_DIFFERENTIAL"].min()))
        hi = float(max(dd["PRED_MARGIN"].max(), dd["POINT_DIFFERENTIAL"].max()))
        pad = max(3.0, (hi - lo) * 0.05)
        fig.add_trace(
            go.Scatter(
                x=[lo - pad, hi + pad],
                y=[lo - pad, hi + pad],
                mode="lines",
                line=dict(dash="dash", color="rgba(80,80,80,0.5)"),
                name="Perfect",
            )
        )
    fig.update_layout(template="plotly_white", height=480)
    st.plotly_chart(fig, use_container_width=True)

    err = (dd["POINT_DIFFERENTIAL"] - dd["PRED_MARGIN"]).abs() if len(dd) else pd.Series(dtype=float)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games shown", f"{len(d):,}")
    c2.metric("Games with results", f"{len(dd):,}")
    c3.metric("MAE (subset)", f"{err.mean():.2f}" if len(dd) else "N/A")
    c4.metric("RMSE (subset)", f"{((err ** 2).mean()) ** 0.5:.2f}" if len(dd) else "N/A")

    st.subheader("Games")
    show = d[
        [
            "YEAR",
            "Round (teams remaining)",
            "TEAM1",
            "TEAM2",
            "PRED_MARGIN",
            "POINT_DIFFERENTIAL",
            "ABS_ERROR",
            "SQ_ERROR",
            "TEAM1_SCORE",
            "TEAM2_SCORE",
        ]
    ].sort_values(["Round (teams remaining)", "TEAM1"])
    st.dataframe(show, use_container_width=True, hide_index=True)
