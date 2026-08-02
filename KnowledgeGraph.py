import csv
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

def find_data_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "R" / "Talent_Intelligence_Project" / "input",
        script_dir / "input",
        Path.cwd(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find the input directory.")


def read_csv_rows(filename, data_dir):
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_value(row, *keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def build_graph(data_dir):
    g = nx.DiGraph()

    core_rows = read_csv_rows("01_core_workforce.csv", data_dir)
    manager_rows = read_csv_rows("manager_network.csv", data_dir)
    skills_rows = read_csv_rows("skills_taxonomy.csv", data_dir)

    for row in core_rows:
        person_id = safe_value(row, "employee_id", "EmployeeID", "id")
        if person_id is None:
            continue

        g.add_node(person_id, type="Person", name=safe_value(row, "employee_name", "EmployeeName", "name", "full_name"))

        job_title = safe_value(row, "job_title", "JobTitle", "title")
        if job_title:
            job_id = str(job_title)
            g.add_node(job_id, type="Job", name=job_title)
            g.add_edge(person_id, job_id, relation="works_in")

    for row in manager_rows:
        manager_id = safe_value(row, "manager_id", "ManagerID", "manager")
        employee_id = safe_value(row, "employee_id", "EmployeeID", "employee")
        if manager_id and employee_id:
            g.add_node(manager_id, type="Person")
            g.add_node(employee_id, type="Person")
            g.add_edge(manager_id, employee_id, relation="manages")

    for row in skills_rows:
        skill_id = safe_value(row, "skill_id", "SkillID")
        skill_name = safe_value(row, "skill_name", "SkillName", "name")
        if skill_id:
            g.add_node(skill_id, type="Skill", name=skill_name or skill_id)

    return g


def main():
    data_dir = find_data_dir()
    graph = build_graph(data_dir)
    output_path = Path(__file__).resolve().parent / "knowledge_graph.graphml"
    nx.write_graphml(graph, output_path)
    print(f"Knowledge graph created successfully: {output_path}")

    # Visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, node_size=800, font_size=8)
    plt.show()

    # PyVis Visualization
    nt = Network(notebook=True, height="700px", width="100%", cdn_resources="remote")
    for node in graph.nodes():
        nt.add_node(node, label=str(node))
    for u, v, data in graph.edges(data=True):
        nt.add_edge(u, v, label=data.get("relation", ""))
    nt.show("knowledge_graph.html")


if __name__ == "__main__":
    main()
