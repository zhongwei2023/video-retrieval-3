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


SYSTEM_PROMPT = """你是一个专业的视频分析助手。你的任务是理解用户的问题，提取其中的目标物体信息。

你需要输出以下JSON格式:
{
    "primary_target": "用户想要查看的目标物体(简短名词)",
    "context": "目标周围的上下文环境描述",
    "search_prompt": "用于在视频帧中搜索该目标的英文描述(描述目标在画面中的样子)",
    "crop_prompt": "用于精确裁切目标区域的英文描述"
}

注意:
- primary_target 只提取最终需要裁切的目标物体
- search_prompt 要包含上下文信息，帮助在视频帧中定位
- crop_prompt 要精确描述需要裁切的区域
- 只输出JSON，不要有任何其他文字"""


def _chat_json(client, model: str, prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
    """Call LLM and parse JSON response. Mirrors VLMClient.chat_json()."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        stream=False,
    )

    text = (response.choices[0].message.content or "").strip()
    print(f"  [LLM raw {len(text)} chars]")
    print(f"  {text[:500]}")

    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        raise ValueError(f"No JSON object found in LLM response. Full text:\n{text}")

    raw_json = m.group(0)
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        raw_json = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", raw_json)
        return json.loads(raw_json)


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
    except ImportError:
        raise ImportError("openai package is required. Run: pip install openai")

    base_url = base_url or "https://api.deepseek.com"
    model = model or "deepseek-v4-flash"

    if not api_key:
        print("  [SKIP] No --llm_api_key provided, using raw query as target")
        return TargetSpec(
            target_object=query,
            attributes="",
            visual_hint=query,
            english_query=query,
            raw={"fallback": True, "reason": "missing_api_key"},
        )

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    prompt = f'用户问题: "{query}"\n\n请分析这个问题，提取需要在视频中查找的目标物体信息。'

    try:
        result = _chat_json(client, model, prompt)

        primary_target = str(result.get("primary_target", "")).strip()
        context = str(result.get("context", "")).strip()
        search_prompt = str(result.get("search_prompt", "")).strip()
        crop_prompt = str(result.get("crop_prompt", "")).strip()

        if not primary_target:
            raise ValueError("primary_target is empty")
        if len(primary_target) > 200:
            raise ValueError(f"primary_target too long ({len(primary_target)} chars)")

        print(f"\n  {'='*50}")
        print(f"  [目标理解结果]")
        print(f"  {'='*50}")
        print(f"    用户问题:   {query}")
        print(f"    目标物体:   {primary_target}")
        print(f"    上下文约束: {context}")
        print(f"    检索描述:   {search_prompt}")
        print(f"    裁切描述:   {crop_prompt}")
        print(f"  {'='*50}")

        return TargetSpec(
            target_object=primary_target,
            attributes=context,
            visual_hint=search_prompt,
            english_query=search_prompt or primary_target,
            raw=result,
        )

    except Exception as e:
        print(f"\n  [ERROR] LLM extraction failed: {e}")
        print(f"  [FALLBACK] Using raw query as target")
        return TargetSpec(
            target_object=query,
            attributes="",
            visual_hint=query,
            english_query=query,
            raw={"fallback": True, "reason": str(e)},
        )
