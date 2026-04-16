"""
DS4420 March Madness -- landing page.
Run locally:  streamlit run streamlit_app/Home.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="March Madness Margin Models",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(90deg, #0F1C2E 0%, #1e3a5f 50%, #0F1C2E 100%);
        color: #FAF8F5;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #D2691E;
    }
    .hero h1 { margin: 0; font-size: 1.75rem; }
    .hero p { margin: 0.5rem 0 0 0; opacity: 0.92; font-size: 1rem; }
    </style>
    <div class="hero">
        <h1>March Madness: Predicting Point Margins</h1>
        <p>DS4420 Final Project</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Team")
st.markdown(
    "**Nathalie Abello, Akshitha Bhashetty, Eftyghia Kourtelidis**"
)

st.markdown("---")
st.markdown("### What this project does")
st.markdown(
    """
    March Madness is hard to forecast: upsets and tight games are the norm, but fans still want
    credible estimates of **how much** a team might win or lose by, not only who advances.

    In this project, we predict **tournament point differential from Team 1's perspective**, using:

    - **Baseline** -- season-average point-differential differences between teams
    - **Time series** -- momentum / hot-or-cold team dynamics from past margins
    - **Multi-layer perceptron (MLP)** -- KenPom / BartTorvik-style matchup features and a small neural net
    - **Bayesian** -- uncertainty-aware updating of team strength

    This app highlights the **MLP, our strongest-performing model**. You can explore its
    **historical tournament predictions** across train/test seasons, and also see how the same
    model performs on **2026 March Madness matchups** by comparing its predicted margins with
    the results we observed.
    """
)

st.markdown("---")
st.markdown("### Where the data comes from")
st.markdown(
    """
    - [Kaggle March Madness-style datasets](https://www.kaggle.com/datasets/nishaanamin/march-madness-data/data)
    - [Supplementary ratings (e.g. T-Rank / BartTorvik ecosystem)](https://adamcwisports.blogspot.com/p/data.html)
    - [NCAA-style competition data (e.g. Machine Learning Mania)](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)

    Full citations and narrative live in the course report and repo `README.md`.
    """
)
