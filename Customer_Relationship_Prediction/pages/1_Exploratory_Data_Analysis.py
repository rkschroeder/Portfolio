import plotly.express as px
import streamlit as st

from customer_behavior import app_support, config, palette, preprocessing

st.set_page_config(page_title="EDA · Customer Behavior", page_icon="\U0001F4CA", layout="wide")
st.title("Exploratory Data Analysis")

raw_features, raw_labels = app_support.load_raw_data()

st.markdown(
    f"Raw dataset: **{raw_features.shape[0]:,} rows × {raw_features.shape[1]} columns**, "
    "mixing numerical and categorical (anonymized) features."
)

st.header("Label distribution")
st.caption(
    "All three targets are highly imbalanced — the rare positive class is what makes "
    "these predictions hard."
)

label_cols = st.columns(3)
for col, target in zip(label_cols, config.TARGETS):
    label_df = raw_labels[target]
    label_column = label_df.columns[0]
    counts = label_df[label_column].value_counts().rename({-1: "No", 1: "Yes"})
    counts = counts.reindex(["No", "Yes"])

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map={"No": palette.CATEGORICAL[0], "Yes": palette.CATEGORICAL[1]},
        labels={"x": "", "y": "Count"},
        title=target.capitalize(),
    )
    fig.update_layout(**palette.PLOTLY_LAYOUT_DEFAULTS, showlegend=False, height=350)
    col.plotly_chart(fig, width="stretch")

with st.expander("View label counts as a table"):
    table = {
        target: raw_labels[target][raw_labels[target].columns[0]].value_counts().rename({-1: "No", 1: "Yes"})
        for target in config.TARGETS
    }
    st.dataframe(table)

st.header("Missing data")

nan_percentage = preprocessing.missing_value_percentage(raw_features)
up_to_20 = (nan_percentage <= 20).sum()
between_20_50 = ((nan_percentage > 20) & (nan_percentage <= 50)).sum()
above_50 = (nan_percentage > 50).sum()

col1, col2 = st.columns([1, 2])
with col1:
    summary_fig = px.bar(
        x=["Up to 20%", "20%–50%", "Above 50%"],
        y=[up_to_20, between_20_50, above_50],
        color=["Up to 20%", "20%–50%", "Above 50%"],
        color_discrete_sequence=[
            palette.SEQUENTIAL_BLUE[1],
            palette.SEQUENTIAL_BLUE[2],
            palette.SEQUENTIAL_BLUE[4],
        ],
        labels={"x": "Missing-value range", "y": "Number of columns"},
        title="Columns grouped by missing-value share",
    )
    summary_fig.update_layout(**palette.PLOTLY_LAYOUT_DEFAULTS, showlegend=False, height=420)
    st.plotly_chart(summary_fig, width="stretch")
    st.caption(
        f"The cleaning pipeline keeps columns with ≤20% missing "
        f"({config.COLUMN_NON_NULL_THRESHOLD:.0%} non-null threshold): {up_to_20} of "
        f"{raw_features.shape[1]} columns."
    )

with col2:
    top_missing = nan_percentage.head(30)
    detail_fig = px.bar(
        x=top_missing.index,
        y=top_missing.values,
        labels={"x": "Column", "y": "Missing (%)"},
        title="30 columns with the most missing values",
        color_discrete_sequence=[palette.SEQUENTIAL_BLUE[3]],
    )
    detail_fig.update_layout(**palette.PLOTLY_LAYOUT_DEFAULTS, height=420)
    detail_fig.update_xaxes(tickangle=45)
    st.plotly_chart(detail_fig, width="stretch")

st.header("Categorical cardinality")
_, categorical_cols = preprocessing.get_column_types(raw_features)
cardinality = preprocessing.cardinality_by_column(raw_features, categorical_cols)

cardinality_fig = px.bar(
    x=cardinality.index,
    y=cardinality.values,
    labels={"x": "Column", "y": "Unique values"},
    title="Unique values per categorical column",
    color_discrete_sequence=[palette.CATEGORICAL[2]],
)
cardinality_fig.update_layout(**palette.PLOTLY_LAYOUT_DEFAULTS, height=450)
cardinality_fig.update_xaxes(tickangle=90)
cardinality_fig.add_hline(
    y=config.HIGH_CARDINALITY_THRESHOLD,
    line_dash="dash",
    line_color=palette.STATUS["critical"],
    annotation_text=f"drop threshold ({config.HIGH_CARDINALITY_THRESHOLD})",
)
st.plotly_chart(cardinality_fig, width="stretch")
st.caption(
    "Columns above the dashed line are dropped before modeling — too many unique "
    "values to encode usefully."
)