# run.py
import argparse
import json
import os
from tot_engine import ToTEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--question", required=True, help="Question")
    parser.add_argument("--workdir", default="./work", help="Working directory")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--per_expand_limit", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--save_tree", default="./tree.json", help="Path to save the reasoning tree JSON")
    args = parser.parse_args()

    engine = ToTEngine(
        llm_model=args.model,
        api_key=args.api_key,
        workdir=args.workdir,
        max_depth=args.max_depth,
        per_expand_limit=args.per_expand_limit,
        temperature=args.temperature,
    )

    result = engine.run(args.video, args.question)
    tree = engine.export_tree(result)

    print("Final answer:", result.final_answer)
    print("Terminated:", result.terminated)
    print(f"Total nodes: {len(result.nodes)}")

    with open(args.save_tree, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)
    print("Tree saved to:", args.save_tree)

if __name__ == "__main__":
    main()