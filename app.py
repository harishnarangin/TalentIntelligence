import json
from pathlib import Path

import streamlit as st
from TalentIntelligenceAgent import TalentIntelligenceAgent


def find_data_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "input",
        script_dir.parent / "R" / "Talent_Intelligence_Project" / "input",
        Path.cwd(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find the input directory containing the workforce CSVs.")


@st.cache_resource
def load_agent() -> TalentIntelligenceAgent:
    data_dir = find_data_dir()
    return TalentIntelligenceAgent(data_dir)


st.set_page_config(page_title="Talent Intelligence Agent", page_icon="🧠", layout="wide")
st.title("Talent Intelligence Agent")
st.write("Ask workforce questions and the app will use the repo-backed intelligence agent from GitHub to reason over your CSV datasets.")

agent = load_agent()

with st.sidebar:
    st.header("Try these questions")
    sample_prompts = [
        "How many employees are in the dataset?",
        "What is the attrition breakdown?",
        "What is the average engagement score?",
        "What is the learning trend?",
        "How are employees connected to managers and skills?",
        "What is the performance and potential distribution?",
    ]
    for prompt in sample_prompts:
        if st.button(prompt):
            st.session_state["query"] = prompt

    st.header("Query")
    user_query = st.text_input("Ask about workforce insights", value=st.session_state.get("query", "How many employees are in the dataset?"))

if user_query:
    try:
        result = agent.explain(user_query)
    except Exception as exc:
        st.error(f"Failed to evaluate the query: {exc}")
        st.exception(exc)
    else:
        answer_text = result.get("answer") or "Sorry, I couldn't generate an answer for that query."
        st.subheader("Answer")
        st.write(answer_text)

        if "answer" not in result:
            st.warning(f"Result missing 'answer' key. Keys available: {list(result.keys())}")

        st.subheader("Reasoning path")
        for step in result.get("reasoning", []):
            st.markdown(f"- {step}")

        st.subheader("Source and code info")
        st.markdown("- Repository: [TalentIntelligence](https://github.com/harishnarangin/TalentIntelligence)")
        st.markdown("- Code source: `TalentIntelligenceAgent.py`")

        st.subheader("Graph overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes", result.get("graph_nodes", 0))
        col2.metric("Edges", result.get("graph_edges", 0))
        col3.metric("Intent", result.get("intent", "unknown"))

        st.subheader("Key metrics")
        st.json(result.get("metrics", {}))

        st.subheader("Data sources used")
        st.write(result.get("sources", []))
