# app/model_infer.py
import json
from typing import Any, Dict, List

import cohere
from .rules_fallback import fallback_options

# 🔑 Put your Cohere API key here directly
COHERE_API_KEY = "zzfS0ldXvIB3tcIJJ81qw98Dix6qEiTFWiviZiAZ"

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)


def _build_prompt(question: str, facts, evidence_docs):
    """Builds the structured prompt for Cohere."""
    fact_lines = "\n".join(f"- {f.get('text','')}" for f in (facts or [])) or "- (none)"

    uniq_sources, doc_lines = [], []
    if evidence_docs:
        seen = set()
        for d in evidence_docs:
            src = (d.get("source") or "").strip()
            if src and src not in seen:
                uniq_sources.append(src)
                seen.add(src)
            txt = (d.get("text", "") or "").replace("\n", " ")[:300]
            doc_lines.append(f"- ({src}) {txt}")

    src_lines = "\n".join(f"- {s}" for s in uniq_sources) or "- (none)"
    docs_block = "\n".join(doc_lines) or "- (none)"

    return f"""You are a clinical reasoning assistant. Output STRICT JSON ONLY.

Schema:
{{
  "options": [
    {{
      "title": str,
      "rationale": str,
      "steps": [str],
      "risks": [str],
      "contraindications": [str],
      "monitoring": [str],
      "citations": [str]   // filenames ONLY from the list below
    }},
    {{...}}, {{...}}
  ]
}}

Rules:
- Cite ONLY from these allowed filenames:
{src_lines}
- If none fit, use [].
- Keep lists concise (≤6). No text outside JSON.

<Question>
{question.strip()}

<Patient_Facts>
{fact_lines}

<Evidence_Snippets>
{docs_block}

<JSON>
"""


def generate_options(
    question: str,
    facts: List[Dict[str, Any]],
    evidence_docs: List[Dict[str, Any]],
    max_new_tokens=500,
) -> Dict[str, Any]:
    """
    Calls Cohere API with structured prompt.
    Returns parsed JSON with clinical reasoning options.
    """
    try:
        prompt = _build_prompt(question, facts, evidence_docs)

        response = co.chat(
            model="command-a-03-2025",  # Cohere reasoning model
            message=prompt,
            temperature=0.2,
            max_tokens=max_new_tokens,
        )

        text = response.text.strip()
        jstart, jend = text.find("{"), text.rfind("}")
        if jstart == -1 or jend == -1:
            raise ValueError("model_output_not_json")

        obj = json.loads(text[jstart:jend + 1])
        obj["options"] = list(obj.get("options", []))[:3]  # Limit to 3 options
        obj["engine"] = "cohere-command-r-plus"
        return obj

    except Exception as e:
        print(f"[warn] Cohere inference failed: {e}")
        return fallback_options(question, facts, evidence_docs)
