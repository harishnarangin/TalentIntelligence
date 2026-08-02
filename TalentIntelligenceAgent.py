import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

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

REPO_URL = "https://github.com/harishnarangin/TalentIntelligence"
CODE_SOURCE = "TalentIntelligenceAgent.py"


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


def read_csv_rows(filename: str, data_dir: Path) -> List[Dict[str, str]]:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_all_data(data_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    return {name: read_csv_rows(filename, data_dir) for name, filename in DATASET_FILES.items()}


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").strip().lower()).strip()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class TalentIntelligenceAgent:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data = load_all_data(data_dir)
        self.graph = self.build_graph()

    def build_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()

        for row in self.data["core"]:
            employee_id = row.get("employee_id") or row.get("EmployeeID") or row.get("id")
            if not employee_id:
                continue
            graph.add_node(employee_id, type="Employee")

            job_title = row.get("job_title") or row.get("JobTitle") or row.get("title")
            if job_title:
                graph.add_node(job_title, type="Job")
                graph.add_edge(employee_id, job_title, relation="works_in")

            manager_id = row.get("manager_id") or row.get("ManagerID")
            if manager_id:
                graph.add_node(manager_id, type="Manager")
                graph.add_edge(employee_id, manager_id, relation="reports_to")

            department = row.get("department") or row.get("Department")
            if department:
                graph.add_node(department, type="Department")
                graph.add_edge(employee_id, department, relation="belongs_to")

        for row in self.data["manager"]:
            manager_id = row.get("manager_id") or row.get("ManagerID") or row.get("manager")
            employee_id = row.get("employee_id") or row.get("EmployeeID") or row.get("employee")
            if manager_id and employee_id:
                graph.add_node(manager_id, type="Manager")
                graph.add_node(employee_id, type="Employee")
                graph.add_edge(manager_id, employee_id, relation="manages")

        for row in self.data["learning"]:
            course = row.get("course_id") or row.get("CourseID")
            employee_id = row.get("employee_id") or row.get("EmployeeID")
            if course and employee_id:
                graph.add_node(course, type="Course")
                graph.add_edge(employee_id, course, relation="completed")

        for row in self.data["skills"]:
            skill_id = row.get("skill_id") or row.get("SkillID")
            skill_family = row.get("skill_family") or row.get("SkillFamily")
            if skill_id:
                graph.add_node(skill_id, type="Skill")
            if skill_id and skill_family:
                graph.add_node(skill_family, type="SkillFamily")
                graph.add_edge(skill_id, skill_family, relation="belongs_to")

        return graph

    def infer_intent(self, query: str) -> str:
        text = query.lower()
        if any(k in text for k in ["attrition", "termination", "turnover", "leave", "quit"]):
            return "attrition"
        if any(k in text for k in ["engagement", "sentiment", "listening", "survey", "stay"]):
            return "engagement"
        if any(k in text for k in ["learning", "training", "course", "upskill", "skill", "capability"]):
            return "learning"
        if any(k in text for k in ["manager", "leadership", "org", "report", "network", "span"]):
            return "manager"
        if any(k in text for k in ["hiring", "candidate", "recruit", "offer", "fill", "hire"]):
            return "hiring"
        if any(k in text for k in ["performance", "promotion", "potential", "succession", "calibration", "9 box"]):
            return "performance"
        if any(k in text for k in ["contractor", "rejoin", "conversion"]):
            return "contractor"
        if any(k in text for k in ["customer", "csat", "satisfaction", "service"]):
            return "csat"
        if any(k in text for k in ["department", "headcount", "employee", "workforce", "people"]):
            return "headcount"
        if any(k in text for k in ["job", "role", "career", "level"]):
            return "job"
        if any(k in text for k in ["skill", "skills", "taxonomy"]):
            return "skills"
        return "general"

    def relevant_sources(self, intent: str) -> List[str]:
        mapping = {
            "headcount": ["core", "manager"],
            "attrition": ["termination"],
            "engagement": ["listening"],
            "learning": ["learning"],
            "manager": ["manager", "core"],
            "hiring": ["acquisition"],
            "performance": ["talent", "performance"],
            "contractor": ["contractor"],
            "csat": ["csat"],
            "job": ["job_architecture", "core"],
            "skills": ["skills"],
            "general": list(DATASET_FILES.keys()),
        }
        return [DATASET_FILES[name] for name in mapping.get(intent, []) if name in DATASET_FILES]

    def answer(self, query: str) -> Dict[str, Any]:
        intent = self.infer_intent(query)
        method = getattr(self, f"answer_{intent}", self.answer_general)
        result = method(query)
        if isinstance(result, str):
            result = {"answer": result}
        result.update({
            "intent": intent,
            "sources": self.relevant_sources(intent),
            "repository": REPO_URL,
            "code_source": CODE_SOURCE,
        })
        return result

    def answer_headcount(self, query: str) -> Dict[str, Any]:
        rows = self.data["core"]
        department_counts = Counter(row.get("department", "") for row in rows if row.get("department"))
        top = department_counts.most_common(5)
        total = len(rows)
        if top:
            top_text = ", ".join(f"{name}: {count}" for name, count in top)
            answer = f"There are {total} employees in the dataset. Top departments: {top_text}."
        else:
            answer = f"There are {total} employees in the dataset."
        return {"answer": answer, "metrics": {"employee_count": total, "top_departments": dict(top)}}

    def answer_attrition(self, query: str) -> Dict[str, Any]:
        rows = self.data["termination"]
        if not rows:
            return {"answer": "No termination data available.", "metrics": {}}
        term_counts = Counter(row.get("termination_type", "") for row in rows if row.get("termination_type"))
        rehire = Counter(row.get("rehire_eligible", "") for row in rows if row.get("rehire_eligible"))
        answer = f"Termination breakdown: {', '.join(f'{name}: {count}' for name, count in term_counts.most_common())}. Rehire eligibility: {', '.join(f'{k}: {v}' for k, v in rehire.most_common())}."
        return {"answer": answer, "metrics": {"termination_counts": dict(term_counts), "rehire_eligibility": dict(rehire)}}

    def answer_engagement(self, query: str) -> Dict[str, Any]:
        rows = self.data["listening"]
        if not rows:
            return {"answer": "No employee listening data available.", "metrics": {}}
        engagement_scores = [safe_float(r.get("engagement_score")) for r in rows if r.get("engagement_score") is not None]
        sentiment_scores = [safe_float(r.get("sentiment_score")) for r in rows if r.get("sentiment_score") is not None]
        avg_eng = round(sum(engagement_scores) / len(engagement_scores), 2) if engagement_scores else 0.0
        avg_sent = round(sum(sentiment_scores) / len(sentiment_scores), 2) if sentiment_scores else 0.0
        answer = f"Average engagement score: {avg_eng}. Average sentiment score: {avg_sent}."
        return {"answer": answer, "metrics": {"avg_engagement": avg_eng, "avg_sentiment": avg_sent}}

    def answer_learning(self, query: str) -> Dict[str, Any]:
        rows = self.data["learning"]
        if not rows:
            return {"answer": "No learning data available.", "metrics": {}}
        completed = sum(1 for r in rows if normalize(r.get("completion_status", "")) == "completed")
        hours = [safe_float(r.get("hours_spent")) for r in rows if r.get("hours_spent") is not None]
        avg_hours = round(sum(hours) / len(hours), 2) if hours else 0.0
        answer = f"Learning records loaded: {len(rows)}. Completed modules: {completed}. Average hours spent per record: {avg_hours}."
        return {"answer": answer, "metrics": {"learning_records": len(rows), "completed_records": completed, "avg_hours": avg_hours}}

    def answer_manager(self, query: str) -> Dict[str, Any]:
        rows = self.data["manager"]
        if not rows:
            return {"answer": "No manager network data available.", "metrics": {}}
        rel_types = Counter(row.get("Reporting_Relationship_Type", "") for row in rows if row.get("Reporting_Relationship_Type"))
        top_types = dict(rel_types.most_common(5))
        answer = f"Manager network relationship types: {', '.join(f'{name}: {count}' for name, count in top_types.items())}."
        return {"answer": answer, "metrics": {"relationship_types": top_types}}

    def answer_hiring(self, query: str) -> Dict[str, Any]:
        rows = self.data["acquisition"]
        if not rows:
            return {"answer": "No talent acquisition data available.", "metrics": {}}
        sources = Counter(row.get("source", "") for row in rows if row.get("source"))
        stages = Counter(row.get("stage", "") for row in rows if row.get("stage"))
        answer = f"Hiring source mix: {', '.join(f'{name}: {count}' for name, count in sources.most_common(5))}. Funnel stages: {', '.join(f'{name}: {count}' for name, count in stages.most_common(5))}."
        return {"answer": answer, "metrics": {"sources": dict(sources), "stages": dict(stages)}}

    def answer_performance(self, query: str) -> Dict[str, Any]:
        rows = self.data["talent"]
        if not rows:
            return {"answer": "No talent management data available.", "metrics": {}}
        ratings = Counter(row.get("performance_rating", "") for row in rows if row.get("performance_rating"))
        potentials = Counter(row.get("potential_rating", "") for row in rows if row.get("potential_rating"))
        answer = f"Performance ratings: {', '.join(f'{name}: {count}' for name, count in ratings.most_common())}. Potential ratings: {', '.join(f'{name}: {count}' for name, count in potentials.most_common())}."
        return {"answer": answer, "metrics": {"performance_ratings": dict(ratings), "potential_ratings": dict(potentials)}}

    def answer_contractor(self, query: str) -> Dict[str, Any]:
        rows = self.data["contractor"]
        if not rows:
            return {"answer": "No contractor/rejoiner data available.", "metrics": {}}
        converted = sum(1 for r in rows if normalize(r.get("converted_to_ft", "")) == "yes")
        rejoin = Counter(row.get("rejoin_event", "") for row in rows if row.get("rejoin_event"))
        answer = f"Contractor conversion count: {converted}. Rejoin events: {dict(rejoin)}."
        return {"answer": answer, "metrics": {"converted_to_ft": converted, "rejoin_events": dict(rejoin)}}

    def answer_csat(self, query: str) -> Dict[str, Any]:
        rows = self.data["csat"]
        if not rows:
            return {"answer": "No customer satisfaction data available.", "metrics": {}}
        scores = [safe_float(r.get("csat_score")) for r in rows if r.get("csat_score") is not None]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        channels = Counter(row.get("channel", "") for row in rows if row.get("channel"))
        answer = f"Average CSAT score: {avg_score}. Channel mix: {dict(channels)}."
        return {"answer": answer, "metrics": {"avg_csat": avg_score, "channels": dict(channels)}}

    def answer_job(self, query: str) -> Dict[str, Any]:
        rows = self.data["job_architecture"]
        if not rows:
            return {"answer": "No job architecture data available.", "metrics": {}}
        levels = Counter(row.get("job_level", "") for row in rows if row.get("job_level"))
        families = Counter(row.get("job_family", "") for row in rows if row.get("job_family"))
        answer = f"Job architecture summary includes {len(levels)} levels and {len(families)} families. Top levels: {', '.join(f'{name}: {count}' for name, count in levels.most_common(5))}."
        return {"answer": answer, "metrics": {"job_levels": dict(levels), "job_families": dict(families)}}

    def answer_skills(self, query: str) -> Dict[str, Any]:
        rows = self.data["skills"]
        if not rows:
            return {"answer": "No skills taxonomy data available.", "metrics": {}}
        families = Counter(row.get("skill_family", "") for row in rows if row.get("skill_family"))
        answer = f"Skills families represented: {', '.join(f'{name}: {count}' for name, count in families.most_common(10))}."
        return {"answer": answer, "metrics": {"skill_families": dict(families)}}

    def answer_general(self, query: str) -> Dict[str, Any]:
        summary = {
            "employees": len(self.data["core"]),
            "termination_records": len(self.data["termination"]),
            "listening_records": len(self.data["listening"]),
            "learning_records": len(self.data["learning"]),
            "manager_relationships": len(self.data["manager"]),
            "hiring_records": len(self.data["acquisition"]),
            "performance_records": len(self.data["talent"]),
            "contractor_records": len(self.data["contractor"]),
            "csat_records": len(self.data["csat"]),
            "skills_records": len(self.data["skills"]),
            "job_architecture_records": len(self.data["job_architecture"]),
        }
        answer = (
            "I can help with workforce questions such as headcount, attrition, engagement, learning, manager network, hiring, performance, contractor conversion, CSAT, job architecture, and skills. "
            f"Available data summary: {json.dumps(summary, indent=2)}"
        )
        return {"answer": answer, "metrics": summary}

    def explain(self, query: str) -> Dict[str, Any]:
        result = self.answer(query)
        result["query"] = query
        result["graph_nodes"] = self.graph.number_of_nodes()
        result["graph_edges"] = self.graph.number_of_edges()
        result["graph_summary"] = {
            "node_types": Counter(nx.get_node_attributes(self.graph, "type").values()),
            "relations": Counter(data.get("relation", "") for _, _, data in self.graph.edges(data=True)),
        }
        return result


def main():
    data_dir = find_data_dir()
    agent = TalentIntelligenceAgent(data_dir)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Enter a workforce question, for example:")
        print("  python TalentIntelligenceAgent.py \"How many employees are in the dataset\"")
        print("  python TalentIntelligenceAgent.py \"What is the attrition breakdown\"")
        return

    result = agent.explain(query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
