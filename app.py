import streamlit as st
from pipeline import DualGuardPipeline

st.set_page_config(page_title="DualGuard", page_icon="🛡️", layout="wide")

st.title("🛡️ DualGuard")
st.caption("Two-Stage Detect-then-Correct Architecture for RAG Hallucination Verification")

# ---- Sidebar: Knowledge base setup ----
st.sidebar.header("Knowledge Base")
input_mode = st.sidebar.radio("Provide documents via:", ["Sample documents", "Paste your own"])

if input_mode == "Sample documents":
    documents = [
        "The Eiffel Tower was completed in 1889 for the World's Fair.",
        "It is located in Paris, France, on the Champ de Mars.",
        "The tower is 330 meters tall including antennas.",
        "Gustave Eiffel's company designed and built the tower.",
    ]
    st.sidebar.info(f"{len(documents)} sample chunks loaded.")
else:
    raw_text = st.sidebar.text_area(
        "Paste documents (one chunk per line)", height=250
    )
    documents = [line.strip() for line in raw_text.split("\n") if line.strip()]

st.sidebar.divider()
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 10, 5)
max_attempts = st.sidebar.slider("Max correction attempts", 1, 5, 3)

# ---- Main: Query input ----
query = st.text_input("Enter your question:", placeholder="e.g., When was the Eiffel Tower built and how tall is it?")

run_button = st.button("Run DualGuard Pipeline", type="primary")

# ---- Pipeline execution ----
if run_button:
    if not documents:
        st.error("Please provide at least one document chunk.")
    elif not query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Retrieving, generating, and verifying..."):
            pipeline = DualGuardPipeline(documents)
            pipeline.retriever.retrieve.__defaults__ = None  # placeholder, top_k passed explicitly below
            results = pipeline.run(query)

        st.success("Done!")
        st.divider()
        st.subheader("Verified Answer")

        # ---- Render each sentence with color-coded trust ----
        for item in results:
            if item["trust"] == "HIGH_TRUST":
                bg_color = "#d4edda"   # green
                border_color = "#28a745"
                label = "✅ HIGH TRUST"
            else:
                bg_color = "#f8d7da"   # red
                border_color = "#dc3545"
                label = "⚠️ LOW TRUST — FLAGGED"

            st.markdown(
                f"""
                <div style="
                    background-color:{bg_color};
                    border-left: 5px solid {border_color};
                    padding: 12px 16px;
                    border-radius: 6px;
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 15px; color:#111;">{item['sentence']}</div>
                    <div style="font-size: 12px; color:#333; margin-top:6px;">
                        <b>{label}</b> &nbsp;|&nbsp;
                        Faithfulness: {item['faith_score']} &nbsp;|&nbsp;
                        Relevance: {item['rel_score']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if item["citation"]:
                with st.expander("📄 View citation source"):
                    st.write(item["citation"])

        # ---- Summary stats ----
        st.divider()
        st.subheader("Summary")
        total = len(results)
        high_trust = sum(1 for r in results if r["trust"] == "HIGH_TRUST")
        flagged = total - high_trust

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sentences", total)
        col2.metric("High Trust", high_trust)
        col3.metric("Flagged", flagged)