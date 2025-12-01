from __future__ import annotations
from typing import List, Dict
import re

from .storage import store
from .models import ConceptNode, ConceptEdge


# Heuristic phrase matcher tuned for your course slides
KEY_PHRASE_PATTERN = re.compile(
    r"\b("
    r"naive bayes|logistic regression|deep learning|"
    r"semantic parsing|syntactic parsing|dependency parsing|semantic roles?|"
    r"context[- ]free grammar|cfgs?|cky|chart parsing|"
    r"neural networks?|recurrent neural networks?|rnns?|"
    r"transformers?|self[- ]attention|multi[- ]head attention|"
    r"generative ai|large language models?|llms?|"
    r"pretrain (and|&) prompt|pretrain (and|&) finetune"
    r")\b",
    flags=re.IGNORECASE,
)


def _extract_concepts_from_text(text: str) -> List[str]:
    matches = KEY_PHRASE_PATTERN.findall(text)
    # matches can be tuples because of groups – take the first element
    raw_phrases = []
    for m in matches:
        if isinstance(m, tuple):
            raw_phrases.append(m[0])
        else:
            raw_phrases.append(m)
    phrases = {p.lower() for p in raw_phrases}
    normed = {
        re.sub(r"\s+", " ", p.replace("-", " ")).strip() for p in phrases if p.strip()
    }
    return list(normed)


def build_concept_graph() -> Dict[str, List[Dict]]:
    """
    Build a bipartite-ish graph:
      - document nodes (each uploaded PDF)
      - concept nodes (key phrases)
      - doc<->concept edges (frequency)
      - concept<->concept edges (co-occurrence within chunks)
    """
    nodes: Dict[str, ConceptNode] = {}
    edges: Dict[tuple, ConceptEdge] = {}

    # Document nodes
    for doc_id, doc in store.documents.items():
        nodes[doc_id] = ConceptNode(
            id=doc_id,
            label=doc.title,
            type="document",
        )

    # Concept extraction over chunks
    for chunk in store.chunks.values():
        concepts = _extract_concepts_from_text(chunk.text)
        if not concepts:
            continue

        # doc -> concept edges
        for c in concepts:
            concept_id = f"concept:{c}"
            if concept_id not in nodes:
                nodes[concept_id] = ConceptNode(
                    id=concept_id,
                    label=c,
                    type="concept",
                )
            key = (chunk.doc_id, concept_id)
            if key not in edges:
                edges[key] = ConceptEdge(
                    source=chunk.doc_id, target=concept_id, weight=1.0
                )
            else:
                edges[key].weight += 1.0

        # concept <-> concept co-occurrence inside this chunk
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
        {"source": e.source, "target": e.target, "weight": float(e.weight)}
        for e in edges.values()
    ]

    return {"nodes": node_dicts, "edges": edge_dicts}
