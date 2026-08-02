import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List


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


class TalentIntelligenceAgent:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data = load_all_data(data_dir)

    def answer(self, query: str) -> str:
        q = query.lower()

        if any(k in q for k in ["headcount", "employee count", "how many employees", "department", "location"]):
            return self.answer_headcount(q)
        if any(k in q for k in ["attrition", "termination", "turnover", "leave", "quit"]):
            return self.answer_attrition(q)
        if any(k in q for k in ["engagement", "sentiment", "listening", "survey", "stay"]):
            return self.answer_engagement(q)
        if any(k in q for k in ["learning", "training", "course", "skill gain", "upskill"]):
            return self.answer_learning(q)
        if any(k in q for k in ["manager", "network", "span of control", "report", "org"]):
            return self.answer_manager_network(q)
        if any(k in q for k in ["hiring", "candidate", "recruit", "offer", "time to fill"]):
            return self.answer_hiring(q)
        if any(k in q for k in ["performance", "calibration", "promotion", "potential", "9 box", "succession"]):
            return self.answer_talent_management(q)
        if any(k in q for k in ["contractor", "rejoin", "conversion"]):
            return self.answer_contractor(q)
        if any(k in q for k in ["customer", "csat", "satisfaction", "service"]):
            return self.answer_csat(q)
        if any(k in q for k in ["skill", "skills", "taxonomy"]):
            return self.answer_skills(q)

        return self.answer_generic(q)

    def answer_headcount(self, query: str) -> str:
        rows = self.data["core"]
        department_counts = Counter(row.get("department", "") for row in rows if row.get("department"))
        top = department_counts.most_common(5)
        total = len(rows)
        if top:
            top_text = ", ".join(f"{name}: {count}" for name, count in top)
            return f"There are {total} employees in the dataset. Top departments: {top_text}."
        return f"There are {total} employees in the dataset."

    def answer_attrition(self, query: str) -> str:
        rows = self.data["termination"]
        if not rows:
            return "No termination data available."
        term_counts = Counter(row.get("termination_type", "") for row in rows if row.get("termination_type"))
        rehire = Counter(row.get("rehire_eligible", "") for row in rows if row.get("rehire_eligible"))
        summary = []
        for name, count in term_counts.most_common():
            summary.append(f"{name}: {count}")
        rehire_summary = ", ".join(f"{k}: {v}" for k, v in rehire.most_common())
        return f"Termination breakdown: {', '.join(summary)}. Rehire eligibility: {rehire_summary}."

    def answer_engagement(self, query: str) -> str:
        rows = self.data["listening"]
        if not rows:
            return "No employee listening data available."
        engagement_scores = [float(r.get("engagement_score", 0)) for r in rows if r.get("engagement_score")]
        sentiment_scores = [float(r.get("sentiment_score", 0)) for r in rows if r.get("sentiment_score")]
        if engagement_scores:
            avg_eng = round(sum(engagement_scores) / len(engagement_scores), 2)
        else:
            avg_eng = 0
        if sentiment_scores:
            avg_sent = round(sum(sentiment_scores) / len(sentiment_scores), 2)
        else:
            avg_sent = 0
        return f"Average engagement score: {avg_eng}. Average sentiment score: {avg_sent}."

    def answer_learning(self, query: str) -> str:
        rows = self.data["learning"]
        if not rows:
            return "No learning data available."
        completed = sum(1 for r in rows if normalize(r.get("completion_status", "")) == "completed")
        hours = [float(r.get("hours_spent", 0)) for r in rows if r.get("hours_spent")]
        avg_hours = round(sum(hours) / len(hours), 2) if hours else 0
        return f"Learning records loaded: {len(rows)}. Completed modules: {completed}. Average hours spent per record: {avg_hours}."

    def answer_manager_network(self, query: str) -> str:
        rows = self.data["manager"]
        if not rows:
            return "No manager network data available."
        rel_types = Counter(row.get("Reporting_Relationship_Type", "") for row in rows if row.get("Reporting_Relationship_Type"))
        top_types = ", ".join(f"{name}: {count}" for name, count in rel_types.most_common(5))
        return f"Manager network relationship types: {top_types}."

    def answer_hiring(self, query: str) -> str:
        rows = self.data["acquisition"]
        if not rows:
            return "No talent acquisition data available."
        sources = Counter(row.get("source", "") for row in rows if row.get("source"))
        stages = Counter(row.get("stage", "") for row in rows if row.get("stage"))
        source_text = ", ".join(f"{name}: {count}" for name, count in sources.most_common(5))
        stage_text = ", ".join(f"{name}: {count}" for name, count in stages.most_common(5))
        return f"Hiring source mix: {source_text}. Funnel stages: {stage_text}."

    def answer_talent_management(self, query: str) -> str:
        rows = self.data["talent"]
        if not rows:
            return "No talent management data available."
        ratings = Counter(row.get("performance_rating", "") for row in rows if row.get("performance_rating"))
        potential = Counter(row.get("potential_rating", "") for row in rows if row.get("potential_rating"))
        rating_text = ", ".join(f"{name}: {count}" for name, count in ratings.most_common())
        potential_text = ", ".join(f"{name}: {count}" for name, count in potential.most_common())
        return f"Performance ratings: {rating_text}. Potential ratings: {potential_text}."

    def answer_contractor(self, query: str) -> str:
        rows = self.data["contractor"]
        if not rows:
            return "No contractor/rejoiner data available."
        converted = sum(1 for r in rows if normalize(r.get("converted_to_ft", "")) == "yes")
        rejoin = Counter(row.get("rejoin_event", "") for row in rows if row.get("rejoin_event"))
        return f"Contractor conversion count: {converted}. Rejoin events: {dict(rejoin)}."

    def answer_csat(self, query: str) -> str:
        rows = self.data["csat"]
        if not rows:
            return "No customer satisfaction data available."
        scores = [float(r.get("csat_score", 0)) for r in rows if r.get("csat_score")]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0
        channels = Counter(row.get("channel", "") for row in rows if row.get("channel"))
        return f"Average CSAT score: {avg_score}. Channel mix: {dict(channels)}."

    def answer_skills(self, query: str) -> str:
        rows = self.data["skills"]
        if not rows:
            return "No skills taxonomy data available."
        families = Counter(row.get("skill_family", "") for row in rows if row.get("skill_family"))
        return f"Skills families represented: {dict(families)}."

    def answer_generic(self, query: str) -> str:
        summary = {
            "employees": len(self.data["core"]),
            "termination_records": len(self.data["termination"]),
            "listening_records": len(self.data["listening"]),
            "learning_records": len(self.data["learning"]),
            "manager_relationships": len(self.data["manager"]),
            "hiring_records": len(self.data["acquisition"]),
            "performance_records": len(self.data["talent"]),
            "contractor_records": len(self.data["contractor"]),
        }
        return (
            "I can help with workforce questions such as headcount, attrition, engagement, learning, manager network, hiring, performance, contractor conversion, CSAT, and skills. "
            f"Available data summary: {json.dumps(summary, indent=2)}"
        )


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

    answer = agent.answer(query)
    print(answer)


if __name__ == "__main__":
    main()
