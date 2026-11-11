import json, os
import pandas as pd
import plotly.express as px
import streamlit as st
from classifier.predictor import triage, load_labels

st.set_page_config(page_title="TicketTriage AI", layout="wide")
ADAPTER = os.environ.get("ADAPTER_DIR", "artifacts/adapter_curated")
VARIANTS = ["base", "raw", "curated"]


def read(path):
    return json.load(open(path)) if os.path.exists(path) else None


@st.cache_resource
def get_generator():
    from training.inference import load_generator
    return load_generator(ADAPTER)


metrics = {v: read(f"reports/metrics_{v}.json") for v in VARIANTS}
page = st.sidebar.radio("TicketTriage AI", ["Overview", "Classifier", "Evaluation", "Dataset"])

if page == "Overview":
    st.title("TicketTriage AI")
    st.write("Fine-tuned Cohere Labs model that labels banking support messages with one of "
             "77 Banking77 intents and routes them to a department.")
    cols = st.columns(3)
    for col, v in zip(cols, VARIANTS):
        m = metrics[v]
        col.metric(f"{v} accuracy", f"{m['accuracy']:.1%}" if m else "not evaluated yet")
    if metrics["base"] and metrics["curated"]:
        st.metric("Improvement over base (accuracy points)",
                  f"{(metrics['curated']['accuracy'] - metrics['base']['accuracy']) * 100:.1f}")

elif page == "Classifier":
    st.header("Ticket classifier")
    examples = ["I transferred money yesterday but it has not arrived",
                "Why was I charged a fee when I used the ATM?",
                "I lost my card, please block it"]
    choice = st.selectbox("Example messages", [""] + examples)
    text = st.text_area("Customer message", value=choice, height=120)
    if st.button("Classify"):
        if not os.path.exists(ADAPTER):
            st.error(f"Adapter not found at {ADAPTER}. Train the model first.")
        else:
            with st.spinner("Running model"):
                result = triage(text, get_generator(), load_labels())
            st.json(result.model_dump())

elif page == "Evaluation":
    st.header("Evaluation on the held-out Banking77 test set")
    done = [m for m in metrics.values() if m]
    if not done:
        st.info("No evaluation reports yet. Run evaluation/run_eval.py.")
    else:
        df = pd.DataFrame([{k: m[k] for k in ("variant", "accuracy", "macro_f1", "weighted_f1",
                                              "invalid_output_rate", "latency_s_per_sample")} for m in done])
        st.dataframe(df.round(4), use_container_width=True)
        st.plotly_chart(px.bar(df, x="variant", y=["accuracy", "macro_f1"], barmode="group"),
                        use_container_width=True)
        pick = st.selectbox("Variant details", [m["variant"] for m in done])
        m = metrics[pick]
        labels = load_labels()
        st.plotly_chart(px.imshow(m["confusion_matrix"], x=labels, y=labels, height=800,
                                  labels=dict(x="predicted", y="true")), use_container_width=True)
        st.subheader("Misclassified examples")
        st.dataframe(pd.DataFrame(m["errors"]), use_container_width=True)

else:
    st.header("Dataset")
    stats = read("reports/curation_stats.json")
    if stats:
        st.json(stats)
    path = "data/processed/train_raw.jsonl"
    if os.path.exists(path):
        df = pd.read_json(path, lines=True)
        counts = df["intent"].value_counts().reset_index()
        st.plotly_chart(px.bar(counts, x="intent", y="count"), use_container_width=True)
    st.caption("Source: PolyAI/banking77, CC-BY-4.0")
