# tot_engine.py
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from moviepy.editor import VideoFileClip

from tool_video import clip_video_segment, VideoClipResult, ensure_dir
from prompts import ROOT_SYSTEM_PROMPT, PER_NODE_SYSTEM_PROMPT, build_root_user_prompt, build_node_user_prompt

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

@dataclass
class PathNode:
    path_id: str
    start_s: float
    end_s: float
    strategy: str
    depth: int
    parent_id: Optional[str] = None
    clip_result: Optional[VideoClipResult] = None
    status: str = "pending"  # pending | processed | discarded
    decision: Optional[str] = None
    direct_answer: Optional[str] = None
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    children: List[str] = field(default_factory=list)

@dataclass
class ToTRunResult:
    root_paths: List[str]
    nodes: Dict[str, PathNode]
    final_answer: Optional[str] = None
    terminated: bool = False

class ToTEngine:
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        workdir: str = "./work",
        max_depth: int = 3,
        per_expand_limit: int = 3,
        temperature: float = 0.2,
    ):
        self.workdir = workdir
        ensure_dir(self.workdir)
        self.max_depth = max_depth
        self.per_expand_limit = per_expand_limit
        self.temperature = temperature

        self.client = None
        self.llm_model = llm_model
        if OpenAI is not None:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        # Full dialogue history (pure multi-turn)
        self.messages: List[Dict[str, str]] = []

    def _chat_to_json(self) -> Dict[str, Any]:
        """
        Send self.messages to the LLM and parse a strict JSON response.
        On failure, return a safe terminate decision.
        """
        if self.client is None:
            # Offline fallback
            dummy = {
                "decision": "expand",
                "rationale": "Offline fallback: propose probing paths.",
                "proposed_paths": [
                    {"id": "P1", "strategy": "probe beginning", "start_s": 0, "end_s": 5},
                    {"id": "P2", "strategy": "probe middle", "start_s": 5, "end_s": 10},
                ],
                "direct_answer": None,
                "confidence": 0.3,
            }
            # Also append dummy assistant message to keep dialogue shape
            self.messages.append({"role": "assistant", "content": json.dumps(dummy, ensure_ascii=False)})
            return dummy

        resp = self.client.chat.completions.create(
            model=self.llm_model,
            temperature=self.temperature,
            messages=self.messages,
        )
        text = resp.choices[0].message.content.strip()
        # Append assistant message to history
        self.messages.append({"role": "assistant", "content": text})

        try:
            data = json.loads(text)
        except Exception:
            # Try extracting from code block
            data = None
            if "```" in text:
                parts = text.split("```")
                for i in range(len(parts)):
                    try:
                        data = json.loads(parts[i])
                        break
                    except Exception:
                        continue
            if data is None:
                data = {"decision": "terminate", "rationale": "JSON parse failure", "proposed_paths": [], "direct_answer": None, "confidence": 0.0}

        data.setdefault("proposed_paths", [])
        data.setdefault("direct_answer", None)
        data.setdefault("confidence", 0.0)
        data.setdefault("rationale", "")
        data.setdefault("decision", "terminate")
        return data

    def _video_meta(self, video_path: str) -> Dict[str, Any]:
        with VideoFileClip(video_path) as clip:
            return {
                "duration": float(clip.duration),
                "fps": float(clip.fps) if clip.fps else None,
                "width": int(clip.w),
                "height": int(clip.h),
                "path": video_path,
            }

    def _sanitize_paths(self, proposed: List[Dict[str, Any]], parent_id: Optional[str], duration: float, depth: int) -> List[PathNode]:
        nodes: List[PathNode] = []
        for i, p in enumerate(proposed):
            pid = str(p.get("id") or f"P{depth}_{i+1}")
            strat = str(p.get("strategy") or "explore")
            start_s = float(p.get("start_s", 0))
            end_s = float(p.get("end_s", max(0.1, min(duration, start_s + 5))))
            if start_s < 0:
                start_s = 0.0
            if end_s <= start_s:
                end_s = min(duration, start_s + 2.0)
            end_s = min(end_s, duration)
            node = PathNode(
                path_id=pid,
                start_s=start_s,
                end_s=end_s,
                strategy=strat,
                depth=depth,
                parent_id=parent_id,
            )
            nodes.append(node)
        return nodes

    def run(self, video_path: str, question: str) -> ToTRunResult:
        meta = self._video_meta(video_path)

        # Initialize dialogue with system and root user prompt
        self.messages = [
            {"role": "system", "content": ROOT_SYSTEM_PROMPT},
            {"role": "user", "content": build_root_user_prompt(meta, question, max_paths=self.per_expand_limit)},
        ]

        # Root planning
        root_decision = self._chat_to_json()
        nodes: Dict[str, PathNode] = {}
        root_paths: List[str] = []

        if root_decision.get("decision") == "answer":
            return ToTRunResult(root_paths=[], nodes=nodes, final_answer=root_decision.get("direct_answer"), terminated=True)
        if root_decision.get("decision") == "terminate":
            return ToTRunResult(root_paths=[], nodes=nodes, final_answer=None, terminated=True)

        proposed = root_decision.get("proposed_paths", [])[: self.per_expand_limit]
        initial_nodes = self._sanitize_paths(proposed, parent_id=None, duration=meta["duration"], depth=1)
        for n in initial_nodes:
            nodes[n.path_id] = n
            root_paths.append(n.path_id)

        final_answer = None
        terminated = False

        # Sequential expansion (dialogue-driven)
        queue = list(root_paths)

        while queue and not terminated:
            pid = queue.pop(0)
            node = nodes[pid]
            if node.depth > self.max_depth:
                continue

            # Tool call: clip segment
            clip_res = clip_video_segment(video_path, node.start_s, node.end_s, workdir=self.workdir)
            node.clip_result = clip_res

            # Add the tool result to conversation as assistant content
            tool_msg = (
                f"[Tool Result] Clipped segment: path={clip_res.path}, "
                f"time={node.start_s:.2f}-{node.end_s:.2f}s, duration={clip_res.duration:.2f}s."
            )
            self.messages.append({"role": "assistant", "content": tool_msg})

            # Switch to per-node system context for decision consistency
            self.messages.append({"role": "system", "content": PER_NODE_SYSTEM_PROMPT})

            # Ask the model to decide next action for this node
            self.messages.append({
                "role": "user",
                "content": build_node_user_prompt(
                    path_id=node.path_id,
                    strategy=node.strategy,
                    start_s=node.start_s,
                    end_s=node.end_s,
                    clip_path=clip_res.path,
                    duration=clip_res.duration,
                    max_paths=self.per_expand_limit
                )
            })

            node_decision = self._chat_to_json()

            node.status = "processed"
            node.decision = node_decision.get("decision")
            node.rationale = node_decision.get("rationale")
            node.confidence = node_decision.get("confidence", 0.0)

            if node.decision == "discard":
                node.status = "discarded"
                continue

            if node.decision == "answer":
                final_answer = node_decision.get("direct_answer")
                node.direct_answer = final_answer
                terminated = True
                break

            if node.decision == "terminate":
                terminated = True
                break

            if node.decision == "expand":
                if node.depth >= self.max_depth:
                    terminated = True
                    break
                child_nodes = self._sanitize_paths(
                    node_decision.get("proposed_paths", [])[: self.per_expand_limit],
                    parent_id=node.path_id,
                    duration=meta["duration"],
                    depth=node.depth + 1,
                )
                for child in child_nodes:
                    nodes[child.path_id] = child
                    node.children.append(child.path_id)
                    queue.append(child.path_id)

        return ToTRunResult(root_paths=root_paths, nodes=nodes, final_answer=final_answer, terminated=terminated)

    @staticmethod
    def export_tree(result: ToTRunResult) -> Dict[str, Any]:
        out_nodes = {}
        for pid, n in result.nodes.items():
            out_nodes[pid] = {
                "path_id": n.path_id,
                "parent_id": n.parent_id,
                "strategy": n.strategy,
                "start_s": n.start_s,
                "end_s": n.end_s,
                "status": n.status,
                "decision": n.decision,
                "direct_answer": n.direct_answer,
                "rationale": n.rationale,
                "confidence": n.confidence,
                "clip_path": n.clip_result.path if n.clip_result else None,
                "children": n.children,
            }
        return {
            "final_answer": result.final_answer,
            "terminated": result.terminated,
            "root_paths": result.root_paths,
            "nodes": out_nodes,
        }