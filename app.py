import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import streamlit as st

DATASET_FILES = {
    "core": "01_core_workforce.csv",
    "termination": "02_termination.csv",
    "listening": "03_employee_listening.csv",
    "learning": "04_learning.csv",
    "csat": "05_customer_satisfaction.csv",
    "talent": "06_talent_management.csv",
    "acquisition": "07_talent_acquisition.csv",
    "contractor": "08_contractor_rejoiner.csv",
    "job_architecture": "job_architecture.csv",
    "manager": "manager_network.csv",
    "performance": "performance_calibration.csv",
    "skills": "skills_taxonomy.csv",
}


def find_data_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "R" / "Talent_Intelligence_Project" / "input",
        script_dir / "input",
        Path.cwd(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find the input directory containing the workforce CSVs.")


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").strip().lower())


def pick_value(row: Dict[str, str], *names: str) -> Optional[str]:
    for name in names:
        if name in row and row.get(name) not in (None, ""):
            return row.get(name)
        key = normalize_key(name)
        for row_key, value in row.items():
            if normalize_key(row_key) == key and value not in (None, ""):
                return value
    return None


def to_float(value: Optional[str]) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def read_csv_rows(filename: str, data_dir: Path) -> List[Dict[str, str]]:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_all_data(data_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    return {name: read_csv_rows(filename, data_dir) for name, filename in DATASET_FILES.items()}


class KnowledgeGraphReasoner:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data = load_all_data(data_dir)
        self.graph = self.build_graph()

    def build_graph(self) -> nx.DiGraph():
        g = nx.DiGraph()

        core_rows = self.data.get("core", [])
        manager_rows = self.data.get("manager", [])
        skills_rows = self.data.get("skills", [])
        learning_rows = self.data.get("learning", [])
        talent_rows = self.data.get("talent", [])
        termination_rows = self.data.get("termination", [])
        job_rows = self.data.get("job_architecture", [])

        for row in core_rows:
            emp_id = pick_value(row, "employee_id", "EmployeeID", "id")
            if not emp_id:
                continue
            g.add_node(emp_id, type="Employee")

            job_title = pick_value(row, "job_title", "JobTitle", "title")
            if job_title:
                g.add_node(job_title, type="Job")
                g.add_edge(emp_id, job_title, relation="works_in")

            manager_id = pick_value(row, "manager_id", "ManagerID")
            if manager_id:
                g.add_node(manager_id, type="Manager")
                g.add_edge(emp_id, manager_id, relation="reports_to")

            dept = pick_value(row, "department", "Department")
            if dept:
                g.add_node(dept, type="Department")
                g.add_edge(emp_id, dept, relation="belongs_to")

        for row in manager_rows:
            manager_id = pick_value(row, "manager_id", "ManagerID", "manager")
            employee_id = pick_value(row, "employee_id", "EmployeeID", "employee")
            if manager_id and employee_id:
                g.add_node(manager_id, type="Manager")
                g.add_node(employee_id, type="Employee")
                g.add_edge(manager_id, employee_id, relation="manages")

        for row in skills_rows:
            skill_id = pick_value(row, "skill_id", "SkillID")
            if skill_id:
                g.add_node(skill_id, type="Skill")
            family = pick_value(row, "skill_family", "SkillFamily")
            if family:
                g.add_node(family, type="SkillFamily")
                if skill_id:
                    g.add_edge(skill_id, family, relation="belongs_to")

        for row in learning_rows:
            course_id = pick_value(row, "course_id", "CourseID")
            if course_id:
                g.add_node(course_id, type="Course")
                emp_id = pick_value(row, "employee_id", "EmployeeID")
                if emp_id:
                    g.add_node(emp_id, type="Employee")
                    g.add_edge(emp_id, course_id, relation="completed")

        for row in talent_rows:
            emp_id = pick_value(row, "employee_id", "EmployeeID")
            if emp_id:
                g.add_node(emp_id, type="Employee")
                perf = pick_value(row, "performance_rating", "PerformanceRating")
                if perf:
                    g.add_node(perf, type="Performance")
                    g.add_edge(emp_id, perf, relation="has_rating")

        for row in termination_rows:
            emp_id = pick_value(row, "employee_id", "EmployeeID")
            if emp_id:
                g.add_node(emp_id, type="Employee")
                term = pick_value(row, "termination_type", "TerminationType")
                if term:
                    g.add_node(term, type="Termination")
                    g.add_edge(emp_id, term, relation="ended_as")

        for row in job_rows:
            job_family = pick_value(row, "job_family", "JobFamily")
            job_level = pick_value(row, "job_level", "JobLevel")
            if job_family or job_level:
                label = job_family or job_level or "Role"
                g.add_node(label, type="Role")

        return g

    def infer_intent(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["attrition", "termination", "turnover", "leave", "quit", "retention"]):
            return "attrition"
        if any(k in q for k in ["engagement", "sentiment", "listening", "survey", "stay"]):
            return "engagement"
        if any(k in q for k in ["skill", "learning", "training", "course", "upskill", "capability"]):
            return "learning"
        if any(k in q for k in ["manager", "leadership", "org", "report", "span", "network"]):
            return "manager"
        if any(k in q for k in ["hiring", "candidate", "recruit", "offer", "fill", "hire"]):
            return "hiring"
        if any(k in q for k in ["performance", "promotion", "potential", "succession", "9 box", "calibration"]):
            return "performance"
        if any(k in q for k in ["contractor", "rejoin", "conversion"]):
            return "contractor"
        if any(k in q for k in ["customer", "csat", "satisfaction"]):
            return "csat"
        if any(k in q for k in ["department", "headcount", "employee", "workforce", "people"]):
            return "headcount"
        if any(k in q for k in ["job", "role", "career", "level"]):
            return "job"
        return "general"

    def answer(self, query: str) -> Dict[str, Any]:
        intent = self.infer_intent(query)

        core_rows = self.data.get("core", [])
        termination_rows = self.data.get("termination", [])
        listening_rows = self.data.get("listening", [])
        learning_rows = self.data.get("learning", [])
        manager_rows = self.data.get("manager", [])
        acquisition_rows = self.data.get("acquisition", [])
        talent_rows = self.data.get("talent", [])
        contractor_rows = self.data.get("contractor", [])
        csat_rows = self.data.get("csat", [])
        skills_rows = self.data.get("skills", [])
        job_rows = self.data.get("job_architecture", [])

        if intent == "attrition":
            counts = Counter(pick_value(row, "termination_type", "TerminationType") or "Unknown" for row in termination_rows)
            total_terms = sum(counts.values())
            answer = f"There are {total_terms} termination records in the dataset. The mix is: {', '.join(f'{k}: {v}' for k, v in counts.most_common(5))}."
            return {"intent": intent, "answer": answer, "metrics": {"total_termination_records": total_terms}, "reasoning": ["Employee ↔ Termination outcome", "Termination outcome ↔ retention insight"]}

        if intent == "engagement":
            engagement_scores = [to_float(pick_value(row, "engagement_score", "EngagementScore")) for row in listening_rows]
            sentiment_scores = [to_float(pick_value(row, "sentiment_score", "SentimentScore")) for row in listening_rows]
            avg_eng = round(sum(engagement_scores) / len(engagement_scores), 2) if engagement_scores else 0
            avg_sent = round(sum(sentiment_scores) / len(sentiment_scores), 2) if sentiment_scores else 0
            answer = f"Average engagement is {avg_eng} and average sentiment is {avg_sent}."
            return {"intent": intent, "answer": answer, "metrics": {"avg_engagement": avg_eng, "avg_sentiment": avg_sent}, "reasoning": ["Employee ↔ Listening signal", "Listening signal ↔ workforce sentiment"]}

        if intent == "learning":
            hours = [to_float(pick_value(row, "hours_spent", "HoursSpent")) for row in learning_rows]
            completed = sum(1 for row in learning_rows if str(pick_value(row, "completion_status", "CompletionStatus") or "").lower() == "completed")
            avg_hours = round(sum(hours) / len(hours), 2) if hours else 0
            answer = f"The learning data includes {len(learning_rows)} records, with {completed} completed and an average of {avg_hours} hours per record."
            return {"intent": intent, "answer": answer, "metrics": {"learning_records": len(learning_rows), "completed_records": completed, "avg_hours": avg_hours}, "reasoning": ["Employee ↔ Learning course", "Course ↔ skill growth"]}

        if intent == "manager":
            rel_types = Counter(pick_value(row, "Reporting_Relationship_Type", "relationship_type") or "Unknown" for row in manager_rows)
            answer = f"Manager network relationship types include: {', '.join(f'{k}: {v}' for k, v in rel_types.most_common(5))}."
            return {"intent": intent, "answer": answer, "metrics": {"relationship_types": dict(rel_types)}, "reasoning": ["Employee ↔ Manager", "Manager ↔ org structure"]}

        if intent == "hiring":
            sources = Counter(pick_value(row, "source", "Source") or "Unknown" for row in acquisition_rows)
            stages = Counter(pick_value(row, "stage", "Stage") or "Unknown" for row in acquisition_rows)
            answer = f"Hiring source mix: {', '.join(f'{k}: {v}' for k, v in sources.most_common(5))}. Funnel stages: {', '.join(f'{k}: {v}' for k, v in stages.most_common(5))}."
            return {"intent": intent, "answer": answer, "metrics": {"sources": dict(sources), "stages": dict(stages)}, "reasoning": ["Candidate ↔ Source", "Candidate ↔ hiring funnel"]}

        if intent == "performance":
            perf_ratings = Counter(pick_value(row, "performance_rating", "PerformanceRating") or "Unknown" for row in talent_rows)
            pot_ratings = Counter(pick_value(row, "potential_rating", "PotentialRating") or "Unknown" for row in talent_rows)
            answer = f"Performance ratings: {', '.join(f'{k}: {v}' for k, v in perf_ratings.most_common())}. Potential ratings: {', '.join(f'{k}: {v}' for k, v in pot_ratings.most_common())}."
            return {"intent": intent, "answer": answer, "metrics": {"performance": dict(perf_ratings), "potential": dict(pot_ratings)}, "reasoning": ["Employee ↔ Performance rating", "Performance ↔ potential and succession"]}

        if intent == "contractor":
            converted = sum(1 for row in contractor_rows if str(pick_value(row, "converted_to_ft", "ConvertedToFT") or "").lower() == "yes")
            answer = f"There are {len(contractor_rows)} contractor records and {converted} converted to full-time."
            return {"intent": intent, "answer": answer, "metrics": {"contractor_records": len(contractor_rows), "converted_to_full_time": converted}, "reasoning": ["Contractor ↔ employment conversion", "Conversion ↔ workforce planning"]}

        if intent == "csat":
            scores = [to_float(pick_value(row, "csat_score", "CSATScore")) for row in csat_rows]
            avg_score = round(sum(scores) / len(scores), 2) if scores else 0
            channels = Counter(pick_value(row, "channel", "Channel") or "Unknown" for row in csat_rows)
            answer = f"Average CSAT score is {avg_score}. Channel mix: {', '.join(f'{k}: {v}' for k, v in channels.most_common())}."
            return {"intent": intent, "answer": answer, "metrics": {"avg_csat": avg_score, "channels": dict(channels)}, "reasoning": ["Customer interaction ↔ CSAT score", "CSAT ↔ service quality"]}

        if intent == "job":
            levels = Counter(pick_value(row, "job_level", "JobLevel") or "Unknown" for row in job_rows)
            answer = f"The job architecture dataset shows job levels: {', '.join(f'{k}: {v}' for k, v in levels.most_common())}."
            return {"intent": intent, "answer": answer, "metrics": {"job_levels": dict(levels)}, "reasoning": ["Role ↔ level", "Role ↔ career path"]}

        if intent == "headcount":
            department_counts = Counter(pick_value(row, "department", "Department") or "Unknown" for row in core_rows)
            total = len(core_rows)
            answer = f"There are {total} employees in the dataset. Top departments include: {', '.join(f'{k}: {v}' for k, v in department_counts.most_common(5))}."
            return {"intent": intent, "answer": answer, "metrics": {"employee_count": total, "departments": dict(department_counts)}, "reasoning": ["Employee ↔ Department", "Department ↔ workforce distribution"]}

        summary = {
            "employee_count": len(core_rows),
            "termination_records": len(termination_rows),
            "learning_records": len(learning_rows),
            "manager_relationships": len(manager_rows),
            "hiring_records": len(acquisition_rows),
            "talent_records": len(talent_rows),
            "contractor_records": len(contractor_rows),
            "csat_records": len(csat_rows),
            "skills_records": len(skills_rows),
            "job_architecture_records": len(job_rows),
        }
        answer = f"The knowledge graph connects workforce entities across the CSVs. Current data summary: {json.dumps(summary, indent=2)}"
        return {"intent": intent, "answer": answer, "metrics": summary, "reasoning": ["Employee ↔ Job", "Employee ↔ Manager", "Employee ↔ Learning", "Employee ↔ Performance", "Employee ↔ Termination"]}

    def explain(self, query: str) -> Dict[str, Any]:
        result = self.answer(query)
        result["graph_nodes"] = self.graph.number_of_nodes()
        result["graph_edges"] = self.graph.number_of_edges()
        result["query"] = query
        return result


@st.cache_resource
def load_reasoner() -> KnowledgeGraphReasoner:
    data_dir = find_data_dir()
    return KnowledgeGraphReasoner(data_dir)


st.set_page_config(page_title="Talent Intelligence Agent", page_icon="🧠", layout="wide")
st.title("Talent Intelligence Agent")
st.write("Ask workforce questions and the app will use a knowledge graph built from your CSV files as the reasoning layer.")

reasoner = load_reasoner()

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
        result = reasoner.explain(user_query)
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

        st.subheader("Graph overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes", result.get("graph_nodes", 0))
        col2.metric("Edges", result.get("graph_edges", 0))
        col3.metric("Intent", result.get("intent", "unknown"))

        st.subheader("Key metrics")
        st.json(result.get("metrics", {}))
