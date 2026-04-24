"""Rhyme — find historical analogs for the current macro/market environment.

Upload a CSV or JSON with a date column followed by one column per time
series (e.g. returns, yield changes, spreads). The app slides a window of
configurable length across history, clusters the windows, and surfaces the
windows most similar to the most-recent one."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    build_windows,
    cluster_windows,
    find_analogs,
    load_panel,
    project_2d,
)

st.set_page_config(page_title="Rhyme", layout="wide")
st.title("Rhyme")
st.caption("History doesn't repeat, but it rhymes — find today's closest analogs.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    st.caption(
        "Format: first column is date, remaining columns are numeric series "
        "(returns, yield changes, vol, etc.). Daily or weekly both work."
    )

    st.header("Parameters")
    window_size = st.slider("Window size (rows)", 5, 120, 20)
    n_clusters = st.slider("Number of regime clusters", 2, 10, 4)
    top_k = st.slider("Top analogs to show", 3, 25, 8)

if uploaded is None:
    st.info("Upload a file in the sidebar to begin. A sample is in `sample_data/`.")
    st.stop()

panel = load_panel(uploaded)
st.subheader("Input panel")
st.write(
    f"{len(panel):,} rows × {panel.shape[1]} series "
    f"from {panel.index.min().date()} to {panel.index.max().date()}"
)
st.dataframe(panel.tail(10), use_container_width=True)

ws = build_windows(panel, window_size)
labels = cluster_windows(ws, n_clusters)
coords = project_2d(ws)

ref_idx = len(ws.end_dates) - 1
ref_date = ws.end_dates[ref_idx]
ref_cluster = int(labels[ref_idx])

col1, col2, col3 = st.columns(3)
col1.metric("Reference window ends", ref_date.strftime("%Y-%m-%d"))
col2.metric("Reference cluster", f"#{ref_cluster}")
col3.metric("Windows analyzed", f"{len(ws.end_dates):,}")

st.subheader("Top historical analogs")
analogs = find_analogs(ws, ref_idx, top_k=top_k, min_gap=window_size)
analogs_display = analogs.copy()
analogs_display["end_date"] = analogs_display["end_date"].dt.strftime("%Y-%m-%d")
analogs_display["start_date"] = (
    analogs["end_date"] - pd.Timedelta(days=window_size)
).dt.strftime("%Y-%m-%d")
date_to_idx = {d: i for i, d in enumerate(ws.end_dates)}
analogs_display["cluster"] = [int(labels[date_to_idx[d]]) for d in analogs["end_date"]]
st.dataframe(
    analogs_display[["start_date", "end_date", "distance", "cluster"]],
    use_container_width=True,
)

st.subheader("Regime map (PCA of windows)")
map_df = pd.DataFrame(
    {
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "cluster": [f"#{c}" for c in labels],
        "end_date": ws.end_dates,
    }
)
fig = px.scatter(
    map_df,
    x="pc1",
    y="pc2",
    color="cluster",
    hover_data={"end_date": True, "pc1": ":.2f", "pc2": ":.2f"},
    opacity=0.6,
)
fig.add_trace(
    go.Scatter(
        x=[coords[ref_idx, 0]],
        y=[coords[ref_idx, 1]],
        mode="markers",
        marker=dict(size=18, color="black", symbol="star"),
        name="reference",
    )
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Reference window vs. top analogs")
series_pick = st.selectbox("Series", list(panel.columns))
ref_slice = panel[series_pick].iloc[ref_idx : ref_idx + window_size].reset_index(drop=True)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=ref_slice.values, name="reference", line=dict(width=3)))
for _, row in analogs.head(5).iterrows():
    end = row["end_date"]
    end_pos = panel.index.get_loc(end)
    start_pos = end_pos - window_size + 1
    s = panel[series_pick].iloc[start_pos : end_pos + 1].reset_index(drop=True)
    fig2.add_trace(go.Scatter(y=s.values, name=str(end.date()), opacity=0.6))
fig2.update_layout(xaxis_title="days into window", yaxis_title=series_pick)
st.plotly_chart(fig2, use_container_width=True)
