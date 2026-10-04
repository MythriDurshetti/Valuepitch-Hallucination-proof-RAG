import streamlit as st
from rag_pipeline import answer_question

st.set_page_config(
    page_title="FinanceBench Hallucination-Proof QA",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Hallucination-Proof Document Q&A")
st.caption("Answers are generated only from the provided FinanceBench evidence.")

st.info(
    "Ask a question about the provided financial documents. "
    "If the information is not supported by the documents, the system will abstain."
)

question = st.text_area(
    "Enter your question:",
    placeholder="Example: What was the company's revenue in 2018?",
    height=100
)

if st.button("🔍 Ask Question", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the documents..."):
            try:
                result = answer_question(question)

                st.subheader("Answer")

                if result["answer"] == "I don't know based on the provided documents":
                    st.warning(result["answer"])
                else:
                    st.success(result["answer"])

                if result["sources"]:
                    st.subheader("📌 Sources")

                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(
                            f"Source {i} — {source['doc_name']} — Page {source['page']}"
                        ):
                            st.write(
                                f"**Document:** {source['doc_name']}"
                            )
                            st.write(
                                f"**Page:** {source['page']}"
                            )
                            st.write(
                                f"**Similarity:** {source['score']:.3f}"
                            )
                            st.write("**Evidence:**")
                            st.write(source["text"])

            except Exception as e:
                st.error(f"Something went wrong: {e}")