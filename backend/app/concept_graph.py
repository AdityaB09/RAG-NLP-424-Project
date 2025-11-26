from __future__ import annotations
from typing import List, Dict
import re

from .storage import store
from .models import ConceptNode, ConceptEdge


KEY_PHRASE_PATTERN = re.compile(
    r"\b(semantic parsing|syntactic parsing|dependency parsing|"
    r"semantic roles?|first-order logic|neural networks?|recurrent neural networks?|"
    r"transformers?|self-attention|generative ai|large language models?|embeddings?)\b",
    flags=re.IGNORECASE,
)


def extract_concepts_from_text(text: str) -> List[str]:
    matches = re.findall(KEY_PHRASE_PATTERN, text)
    phrases = set(m.lower() if isinstance(m, str) else m[0].lower() for m in matches)
    return list(phrases)


def build_concept_graph() -> Dict[str, List[Dict]]:
    nodes: Dict[str, ConceptNode] = {}
    edges: Dict[tuple, ConceptEdge] = {}

    # Document nodes
    for doc_id, doc in store.documents.items():
        nodes[doc_id] = ConceptNode(
            id=doc_id, label=doc.title, type="document"
        )

    # Concept nodes & edges
    for chunk in store.chunks.values():
        concepts = extract_concepts_from_text(chunk.text)
        if not concepts:
            continue

        # Document -> concept edges
        for c in concepts:
            concept_id = f"concept:{c}"
            if concept_id not in nodes:
                nodes[concept_id] = ConceptNode(
                    id=concept_id, label=c, type="concept"
                )
            key = (chunk.doc_id, concept_id)
            if key not in edges:
                edges[key] = ConceptEdge(source=chunk.doc_id, target=concept_id, weight=1.0)
            else:
                edges[key].weight += 1.0

        # Concept <-> concept co-occurrence edges
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                c1 = f"concept:{concepts[i]}"
                c2 = f"concept:{concepts[j]}"
                key = tuple(sorted((c1, c2)))
                if key not in edges:
                    edges[key] = ConceptEdge(source=key[0], target=key[1], weight=1.0)
                else:
                    edges[key].weight += 1.0

    node_dicts = [
        {"id": n.id, "label": n.label, "type": n.type}
        for n in nodes.values()
    ]
    edge_dicts = [
        {"source": e.source, "target": e.target, "weight": e.weight}
        for e in edges.values()
    ]

    return {"nodes": node_dicts, "edges": edge_dicts}
