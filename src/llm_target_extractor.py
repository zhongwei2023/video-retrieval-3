from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class TargetSpec:
    """LLM-extracted queries for downstream models."""
    detect_query: str   # OWLv2: concise noun phrase, e.g. "computer mouse"
    clip_query: str     # CLIP: descriptive sentence, e.g. "a computer mouse on a desk"
    crop_prompt: str    # cropping hint, e.g. "the mouse, excluding the desk surface"
    raw: Dict[str, Any]


SYSTEM_PROMPT = r"""你是一个专业的视频分析助手。你的任务是理解用户的问题，提取目标物体信息，并为两个视觉模型分别生成最优查询。

输出JSON格式:
{
    "detect_query": "给目标检测模型(OWLv2)的精简查询",
    "clip_query": "给图文匹配模型(CLIP)的描述性查询",
    "crop_prompt": "精确裁切目标区域的英文描述"
}

## detect_query (给OWLv2目标检测模型)
目的：精确描述要检测的物体，使模型能把目标和画面中其他同类物体区分开。
规则：
- 描述目标物体，如果画面中可能有多个同类物体，必须包含区分性特征（颜色、位置等）
- 自然英文表达，可以包含介词短语和简单的位置关系，但所有表达不能杜撰
- 核心测试：如果画面有2个对象（比如小孩或物品），这个查询能不能“挑出”正确的那一个？

用户问题 → detect_query:
  “桌上的鼠标是什么颜色” → “computer mouse”
  “红车的logo是什么牌子” → “red car”
  “穿荧光黄衣服的小孩的鞋子” → “shoes of child in fluorescent yellow clothes”
  “远处那只白色的猫” → “white cat”
  “冰箱里的牛奶” → “milk inside refrigerator”
  “戴眼镜的男生手里的书” → “book held by boy with glasses”

## clip_query (给CLIP图文匹配模型)
目的：从大量视频帧中快速筛选出包含目标的画面。
规则：描述目标在画面中的样子，包含上下文。自然语言英文，像图片标题。5-15个词。

用户问题 → clip_query:
  "桌上的鼠标是什么颜色" → "a computer mouse on a desk"
  "红车的logo是什么牌子" → "a red car on a road"
  "穿荧光黄衣服的小孩的鞋子" → "a child wearing fluorescent yellow clothes"
  "远处那只白色的猫" → "a white cat in the distance"
  "冰箱里的牛奶" → "a milk bottle inside a refrigerator"

## crop_prompt
精确描述需要裁切的区域，说明保留什么、排除什么。
示例: "the computer mouse, excluding the desk surface"

只输出JSON，不要有任何其他文字。"""


def _chat_json(client, model: str, prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
    """Call LLM and parse JSON response."""
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


def _make_fallback_spec(query: str, reason: str) -> TargetSpec:
    """Build a TargetSpec when LLM is unavailable."""
    return TargetSpec(
        detect_query=query,
        clip_query=query,
        crop_prompt=query,
        raw={"fallback": True, "reason": reason},
    )


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
        return _make_fallback_spec(query, "missing_api_key")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    prompt = f'用户问题: "{query}"\n\n请分析这个问题，提取需要在视频中查找的目标物体信息。'

    try:
        result = _chat_json(client, model, prompt)

        detect_query = str(result.get("detect_query", "")).strip()
        clip_query = str(result.get("clip_query", "")).strip()
        crop_prompt = str(result.get("crop_prompt", "")).strip()

        # Fallback chain: if detect_query empty, use clip_query; if both empty, use query
        if not detect_query:
            detect_query = clip_query or query
        if not clip_query:
            clip_query = detect_query or query
        if not crop_prompt:
            crop_prompt = detect_query

        print(f"\n  {'='*50}")
        print(f"  [目标理解结果]")
        print(f"  {'='*50}")
        print(f"    用户问题:     {query}")
        print(f"    OWLv2查询:    {detect_query}")
        print(f"    CLIP查询:     {clip_query}")
        print(f"    裁切描述:     {crop_prompt}")
        print(f"  {'='*50}")

        return TargetSpec(
            detect_query=detect_query,
            clip_query=clip_query,
            crop_prompt=crop_prompt,
            raw=result,
        )

    except Exception as e:
        print(f"\n  [ERROR] LLM extraction failed: {e}")
        print(f"  [FALLBACK] Using raw query as target")
        return _make_fallback_spec(query, str(e))
