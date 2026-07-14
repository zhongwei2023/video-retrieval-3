from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TargetSpec:
    target_object: str
    attributes: str
    visual_hint: str
    english_query: str
    raw: Dict[str, Any]


_SYSTEM = (
    "You extract the visual search target from a user question about a video.\n"
    "Return ONLY JSON with keys: target_object, attributes, visual_hint, english_query.\n"
    "- target_object: the main object to locate/crop.\n"
    "- attributes: constraints from question context.\n"
    "- visual_hint: short visual clue for detection.\n"
    "- english_query: short English phrase optimized for object detection.\n"
    "No extra text, no markdown."
)


def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("LLM did not return JSON object")
    return json.loads(m.group(0))


def parse_target_with_llm(
    query: str,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> TargetSpec:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required for LLM target extraction") from exc

    base_url = base_url or "https://api.deepseek.com"
    model = model or "deepseek-v4-flash"
    api_key = api_key or ""

    if not api_key:
        return TargetSpec(
            target_object=query,
            attributes="",
            visual_hint="",
            english_query=query,
            raw={"fallback": True, "reason": "missing_api_key"},
        )

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Question: {query}"},
        ],
        temperature=0.1,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    text = response.choices[0].message.content
    obj = _extract_json(text)

    return TargetSpec(
        target_object=str(obj.get("target_object") or query),
        attributes=str(obj.get("attributes") or ""),
        visual_hint=str(obj.get("visual_hint") or ""),
        english_query=str(obj.get("english_query") or query),
        raw=obj,
    )
