# core/ideas_core.py

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd


@dataclass
class IdeaNode:
    id: str                 # es. "root", "n1", "n2"
    label: str              # testo da mostrare
    parent_id: Optional[str] = None  # None = nodo radice
    category: str = "Generale"       # es. "Setup", "Risk", "Macro"
    priority: int = 5                # 1–10


def build_nodes_from_df(df: pd.DataFrame) -> List[IdeaNode]:
    """
    Converte una tabella Streamlit (id, label, parent_id, category, priority)
    in una lista di IdeaNode.
    """
    nodes: List[IdeaNode] = []
    for _, row in df.iterrows():
        if not str(row.get("label", "")).strip():
            continue
        node = IdeaNode(
            id=str(row.get("id")),
            label=str(row.get("label")),
            parent_id=str(row.get("parent_id")) if row.get("parent_id") not in (None, "", "None") else None,
            category=str(row.get("category", "Generale")),
            priority=int(row.get("priority", 5)),
        )
        nodes.append(node)
    return nodes

def ideas_to_dot(nodes: List[IdeaNode]) -> str:
    """
    Genera una stringa DOT per st.graphviz_chart a partire dai nodi.
    """
    lines: list[str] = []
    lines.append("digraph G {")
    # layout più leggibile
    lines.append('  graph [rankdir=LR, nodesep=0.4, ranksep=0.7];')
    lines.append(
        '  node [shape=box, style="rounded,filled", color="#555555", '
        'fillcolor="#f5f5f5", fontsize=10];'
    )

    # Nodo radice evidenziato
    for n in nodes:
        safe_label = n.label.replace('"', '\\"')
        style_extra = ""
        if n.parent_id is None:
            style_extra = ', fillcolor="#d6e9ff", color="#3366cc", penwidth=2'
        lines.append(
            f'  "{n.id}" [label="{safe_label}\\n({n.category}, prio {n.priority})"{style_extra}];'
        )

    # Archi parent->child
    for n in nodes:
        if n.parent_id is not None:
            lines.append(f'  "{n.parent_id}" -> "{n.id}";')

    lines.append("}")
    return "\n".join(lines)
