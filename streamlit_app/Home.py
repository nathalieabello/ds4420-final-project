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
    .hero h1 { margin: 0; font-size: 1.75rem; color: #FAF8F5; }
    .hero p { margin: 0.5rem 0 0 0; opacity: 0.92; font-size: 1rem; color: #FAF8F5; }
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
    March Madness is one of the hardest sporting events to predict: upsets are common,
    games are close, and the bracket is unforgiving. Most prediction models focus on
    picking winners, but we wanted to go further: **by how much will a team win or lose?**

    Using tournament game data from 2008-2025, we built and compared three machine learning
    approaches alongside a simple baseline, each taking a different angle on the same question:

    - **Baseline** - predicts margins using the difference in each team's regular-season average point differential
    - **Time Series** - captures team momentum by modeling legacy strength and recent form using Auto-ARIMA
    - **Multi-Layer Perceptron (MLP)** - a manually implemented neural network trained on opponent-adjusted efficiency metrics
    - **Bayesian Linear Regression** - a probabilistic model that predicts margins while quantifying uncertainty in each prediction

    The MLP was our best-performing model, achieving a test MAE of 9.10 points and correctly
    identifying the winner in roughly 3 out of every 4 games. This app lets you explore its
    predictions on historical tournament matchups from 2008-2025, as well as see how it held
    up on the 2026 March Madness bracket.
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
