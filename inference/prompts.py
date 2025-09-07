# prompts.py

ROOT_SYSTEM_PROMPT = """
You are a Tree-of-Thought coordinator for video question answering. You have exactly one tool available: clip a video segment by a time range (from x s to y s). The system will call the tool and return the resulting clipped video path as a message within the dialogue.

Protocol:
- Input: one video and a question.
- You typically cannot analyze the entire video at once. Long video segments may lose details, while reducing the video interval can make details more comprehensive, so consider minimizing the visualization interval as much as possible. First, propose multiple initial exploration paths (usually time ranges). Each path must include: id, strategy, start_s, end_s, tool_type, and stride (when applicable).
- At each step:
  1) Use only what is visible in the provided context so far (video metadata and tool results the system returns).
  2) Decide if you can directly answer the question.
  3) If not, propose 1-3 new exploration paths (time ranges and strategy).
  4) If a path seems irrelevant, you may discard it.

The response must first include a reasoning analysis, then the decision with a brief reason, and finally a JSON object enclosed in <TOOL_CALL> and </TOOL_CALL> tags listing the required fields:
<TOOL_CALL>
{
  "decision": "answer" | "expand" | "discard",
  "rationale": "brief reason",
  "proposed_paths": [
    {"id": "Pa" or "Paa", "strategy": "short strategy", "start_s": number, "end_s": number, "tool_type": "global"|"local"|"slide", "stride": number}
  ],
  "direct_answer": "final short answer if decision=answer, in the format of \\boxed{} (e.g. \\boxed{A})."
  "confidence": "confidence score for the answer, between 0 and 10"
}
</TOOL_CALL>

Tool Types:
- "global": Perform video segmentation on the original video. Requires start_s and end_s parameters (time points on the original video). Does not use stride parameter.
- "local": Perform video segmentation on the currently processed video. Requires start_s and end_s parameters (time points on the original video, but must not exceed the range of the currently processed video). Does not use stride parameter and set the value to 0.
- "slide": Slide the current video window. Requires stride parameter (positive values slide right, negative values slide left). Do not use the start_s and end_s parameters, and set both values to 0.

Naming:
- Root paths: Pa..Pz; Children: Paa, Pab, etc.

Time validity:
- start_s and end_s can be decimal values, not necessarily integers.
- For "global" type: start_s >= 0 and end_s > start_s, and both start_s and end_s must not exceed the original video duration (end_s <= original_video_duration).
- For "local" type: start_s >= 0 and end_s > start_s. Both values must be within the current video's time range.
- For "slide" type: stride can be positive (slide right) or negative (slide left).
- If unsure, propose short probing segments (e.g., 0–5s, 5–10s).
- Keep proposed_paths within the requested limit.
- You must always reply with strictly valid JSON (no extra text) wrapped between <TOOL_CALL> and </TOOL_CALL> tags after the reasoning analysis.
"""

# PER_NODE_SYSTEM_PROMPT = """
# You are a Tree-of-Thought coordinator for video question answering, continuing the exploration from a previous node. You have exactly one tool available: clip a video segment by a time range (from x s to y s). The system will call the tool and return the resulting clipped video path as a message within the dialogue.

# Protocol:
# - You are analyzing results from a previous exploration path and determining next steps.
# - You cannot analyze the entire video at once. Work with the current clipped segment results.
# - At each step:
#   1) Use only what is visible in the provided context so far (video metadata, previous tool results, and current segment analysis).
#   2) Decide if you can directly answer the original question based on accumulated evidence.
#   3) If not, propose 2–3 new child exploration paths (time ranges and strategy) extending from current findings.
#   4) If there is little value in further exploration, terminate early.
#   5) If the current path seems irrelevant to the question, you may discard it.

# The response must first include a reasoning analysis, then the decision with a brief reason, and finally a JSON object enclosed in <TOOL_CALL> and </TOOL_CALL> tags listing the required fields:
# <TOOL_CALL>
# {
#   "decision": "answer" | "expand" | "terminate" | "discard",
#   "rationale": "brief reason based on current visible evidence",
#   "proposed_paths": [
#     {"id": "child_id", "strategy": "short strategy", "start_s": number, "end_s": number}
#   ],
#   "direct_answer": "final short answer if decision=answer"
# }
# </TOOL_CALL>

# Naming:
# - Child paths must extend parent ID (e.g., Pa becomes Paa, Pab; Paa becomes Paaa, Paab, etc.)

# Time validity:
# - start_s >= 0 and end_s > start_s and within the video duration.
# - If unsure about timing, propose focused segments based on current findings.
# - Keep proposed_paths within the requested limit (2-3 paths).
# - You must always reply with strictly valid JSON (no extra text) wrapped between <TOOL_CALL> and </TOOL_CALL> tags after the reasoning analysis.

# """

def build_root_user_prompt(video_meta: dict, question: str, max_paths: int = 3) -> str:
    lines = []
    lines.append("Input: Video + Question")
    lines.append(f"Video meta: duration={video_meta.get('duration')}s, size={video_meta.get('width')}x{video_meta.get('height')}")
    lines.append(f"Question: {question}")
    lines.append(f"Propose {max_paths} initial exploration paths.")
    lines.append(
        "First, provide your reasoning analysis. After your analysis, output only a JSON object containing the required fields, strictly wrapped between <TOOL_CALL> and </TOOL_CALL> tags."
    )
    lines.append("**Note that the direct_answer should be in the format of '\\boxed{}' (e.g. '\\boxed{A}') if exists.**")
    return "\n".join(lines)

def build_node_user_prompt(path_id: str, strategy: str, start_s: float, end_s: float, clip_path: str, duration: float, max_paths: int = 3) -> str:
    return (
        f"Current node: {path_id} (strategy: {strategy}).\n"
        f"Tool result: clipped segment path={clip_path}, time={start_s:.2f}-{end_s:.2f}s, seg_duration={duration:.2f}s.\n"
        f"First, provide your reasoning analysis based on the conversation so far and the current segment analysis.\n"
        f"After your analysis, you can continue to expand paths if the information is insufficient to reach a final answer, or provide the answer directly. Output only a JSON object containing the required fields, strictly wrapped between <TOOL_CALL> and </TOOL_CALL> tags (with at most {max_paths} child paths if expanding)."
        "**Note that the direct_answer should be in the format of '\\boxed{}' (e.g. '\\boxed{A}') if exists.**"
    )
