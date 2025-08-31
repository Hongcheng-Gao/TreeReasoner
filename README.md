# README

A minimal Tree-of-Thought (ToT) pipeline for Video QA using a single tool (video clipping) in a pure multi‑turn dialogue style. The LLM plans exploration paths (time ranges), we clip segments, then feed the tool result back into the same conversation. The model decides to answer, expand with new paths, discard, or terminate early, always returning strict JSON.

## Features

- Single tool: clip a video segment [start_s, end_s].
- Pure multi-turn dialogue: tool results are appended as assistant messages; the same model plans and decides next steps.
- Early termination and discard supported.
- JSON-only responses enforced for programmatic parsing.
- Exports a reasoning tree with nodes, decisions, and segment file paths.

## Project Structure

- tool_video.py
  - clip_video_segment: clips a segment and returns its path and metadata.
  - VideoClipResult dataclass.
- prompts.py
  - ROOT_SYSTEM_PROMPT, PER_NODE_SYSTEM_PROMPT.
  - build_root_user_prompt, build_node_user_prompt.
- tot_engine.py
  - ToTEngine: runs the ToT loop as a multi-turn chat.
  - export_tree: serialize the reasoning tree.
  - PathNode and ToTRunResult dataclasses.
- run.py
  - CLI entry to run the pipeline and save the tree.

## Requirements

- Python 3.9+
- ffmpeg installed and available on PATH
- Python packages:
  - moviepy
  - openai

Install:

- pip install moviepy openai

Note: moviepy requires ffmpeg. On macOS: brew install ffmpeg; Ubuntu: sudo apt-get install ffmpeg; Windows: download from ffmpeg.org and add to PATH.

## Environment

- Set your OpenAI API key:
  - Linux/macOS: export OPENAI_API_KEY="sk-..."
  - Windows (CMD): set OPENAI_API_KEY=sk-...
  - Or pass --api_key to the CLI.

## Usage

Basic run:

- python run.py --video input.mp4 --question "What color is the car that appears?"

Common flags:

- --workdir ./work
- --model gpt-4o-mini
- --max_depth 3
- --per_expand_limit 3
- --temperature 0.2
- --save_tree ./tree.json

Example:

- python run.py --video samples/demo.mp4 --question "How many people walk past the camera?" --max_depth 4 --per_expand_limit 3 --temperature 0.3

Outputs:

- Console:
  - Final answer (if any), whether terminated, total nodes.
- Files:
  - work/segment_*.mp4: clipped segments produced during search.
  - tree.json: serialized reasoning tree including nodes, decisions, and clip paths.

## How It Works

1. Root planning
   - The engine sends a system prompt with rules and a user prompt containing video metadata and the question.
   - The LLM returns JSON with a decision and proposed root paths (P1..Pk). If it can already answer, it returns "answer"; it can also "terminate".

2. Iterative expansion (multi-turn)
   - For each selected path:
     - The engine calls the single tool to clip [start_s, end_s].
     - The engine appends a conversational assistant message describing the tool result:
       - [Tool Result] path, time range, segment duration.
     - The engine adds a per-node system prompt (decision rules) and a user message asking for a JSON decision for that node.
     - The model replies with strict JSON: answer | expand | terminate | discard.
     - If expand, propose 2–3 child paths (P1a, P1b…) with time ranges.
   - The loop continues until the model answers or terminates, or max_depth is reached.

3. Export
   - The engine builds a node dictionary with decisions, clip paths, and relations, saved to tree.json.

## JSON Schema (Model Responses)

The model must always output strict JSON (no extra text) in this shape:

- {
  "decision": "answer" | "expand" | "terminate" | "discard",
  "rationale": "brief reason",
  "proposed_paths": [
    {"id": "P1" or "P1a", "strategy": "short strategy", "start_s": number, "end_s": number}
  ],
  "direct_answer": "short final answer if decision=answer",
  "confidence": 0.0-1.0
}

Notes:

- Path IDs follow a tree convention: P1..Pk for roots; P1a, P1b for children.
- Time ranges must be valid and within the video duration.
- If unsure, propose short probing segments (e.g., 0–5s).

## Design Notes

- Pure multi-turn dialogue: no extra tools or vision module. The only actionable tool is clipping. All reasoning happens via the same model in a continuous conversation.
- Robust JSON parsing: the engine attempts to parse the model reply; on failure, it defaults to a safe termination or a simple fallback in offline mode.
- Deterministic control: you can tune per_expand_limit and max_depth to control breadth and depth.

## Tips

- If the model occasionally emits non-JSON text, lower temperature or reinforce “strict JSON only” in prompts.
- For long videos, encourage a grid of short probes at root (e.g., 0–5, 5–10, 10–15…).
- If you hit token limits, you can prune older turns or summarize very old turns, though this repo keeps full turns by default for simplicity.

## Troubleshooting

- ffmpeg not found: ensure ffmpeg is installed and on PATH. moviepy needs it to write video files.
- Slow clipping: adjust CRF, threads, or segment length. Shorter segments improve iteration speed.
- JSON parse failures: check the raw assistant content in the printed logs or extend the parser to extract JSON from code blocks.
- Permission errors on workdir: run with a writable directory or pass --workdir.

## License

MIT License. Use at your own risk.

## Roadmap (Optional Enhancements)

- Priority queue or scoring for path selection.
- Heuristic early-stop if confidence stays low after N nodes.
- Optional summarization of older dialogue turns to save tokens.
- Optional support for a separate vision model (kept out by design here).