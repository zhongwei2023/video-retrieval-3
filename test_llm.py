"""Quick test for LLM target extraction. Run with:
    python test_llm.py --api_key YOUR_KEY
"""
import sys
sys.path.insert(0, ".")

from src.llm_target_extractor import parse_target_with_llm

api_key = sys.argv[1] if len(sys.argv) > 1 else input("API key: ").strip()

queries = [
    "What color is the mouse on the desk?",
    "What color are the shoes of the child wearing the fluorescent yellow clothes?",
    "What brand is the red car logo?",
]

for q in queries:
    print(f"\n{'#'*60}")
    result = parse_target_with_llm(
        q,
        api_key=api_key,
    )
    print(f"\nFINAL: detect_query={result.detect_query!r}  clip_query={result.clip_query!r}  crop_prompt={result.crop_prompt!r}")
