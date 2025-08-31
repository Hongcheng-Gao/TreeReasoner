# tot_engine.py
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from tool_video import VideoClipper, ClipHandle
import uuid
import time

# ------- Replace with your actual LLM client --------
class LLM:
    def chat(self, system: str, user: str) -> str:
        # 必须返回严格 JSON。请替换为真实模型调用。
        return '{"paths": [], "notes": "placeholder"}'

# ------- Data structures --------
@dataclass
class TimeWindow:
    start_s: float
    end_s: float
    purpose: str

@dataclass
class PathPlan:
    id: str
    rationale: str
    time_windows: List[TimeWindow]
    stop_condition: Optional[str] = None

@dataclass
class Node:
    node_id: str
    path_id: str
    parent_id: Optional[str]
    depth: int
    plans: List[TimeWindow] = field(default_factory=list)
    # 仅保存工具调用的“句柄元数据”，不含内容
    clip_handles: List[ClipHandle] = field(default_factory=list)
    status: str = "pending"  # pending, answered, discarded, expanded
    answer: Optional[str] = None
    confidence: Optional[float] = None

class ToTEngine:
    def __init__(self, llm: LLM, clipper: VideoClipper, question: str, max_nodes: int = 20, max_depth: int = 4):
        self.llm = llm
        self.clipper = clipper
        self.question = question
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.nodes: Dict[str, Node] = {}
        self.frontier: List[str] = []
        self.answers: List[Node] = []
        self.root_plans: List[PathPlan] = []

    def _parse_root(self, text: str) -> List[PathPlan]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        paths = []
        for i, p in enumerate(data.get("paths", []), 1):
            tws = []
            for tw in p.get("time_windows", []):
                try:
                    tws.append(TimeWindow(
                        start_s=float(tw["start_s"]),
                        end_s=float(tw["end_s"]),
                        purpose=str(tw.get("purpose", "")),
                    ))
                except Exception:
                    continue
            if tws:
                pid = p.get("id") or f"P{i}"
                paths.append(PathPlan(
                    id=pid,
                    rationale=p.get("rationale", ""),
                    time_windows=tws,
                    stop_condition=p.get("stop_condition")
                ))
        return paths

    def _parse_node(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"can_answer": False, "proposed_children": [], "discard_this_path": False, "confidence": 0.0}

    def root_summarize(self, video_meta: str) -> List[PathPlan]:
        from prompts import ROOT_SYSTEM_PROMPT, ROOT_USER_PROMPT_TEMPLATE
        user = ROOT_USER_PROMPT_TEMPLATE.format(video_meta=video_meta, question=self.question)
        resp = self.llm.chat(ROOT_SYSTEM_PROMPT, user)
        plans = self._parse_root(resp)
        self.root_plans = plans
        for plan in plans:
            node_id = str(uuid.uuid4())
            node = Node(node_id=node_id, path_id=plan.id, parent_id=None, depth=0,
                        plans=plan.time_windows)
            self.nodes[node_id] = node
            self.frontier.append(node_id)
        return plans

    def expand_node(self, node_id: str):
        from prompts import NODE_SYSTEM_PROMPT, NODE_USER_PROMPT_TEMPLATE
        node = self.nodes[node_id]

        # 执行该节点的时间窗：仅抽取句柄，不读取内容
        clip_handles = []
        for tw in node.plans:
            handle = self.clipper.extract_clip(tw.start_s, tw.end_s)
            clip_handles.append(handle)
            time.sleep(0.005)
        node.clip_handles.extend(clip_handles)

        # 构造路径上下文（仅包含已探索过的窗口和句柄元数据）
        explored_summary = []
        for h in node.clip_handles:
            explored_summary.append(
                f"[{h.start_s:.1f}-{h.end_s:.1f}] id={h.clip_id[:8]} fps={h.fps} frames~{h.frame_count}"
            )
        path_context_str = f"Path {node.path_id} depth={node.depth}; explored windows: " + "; ".join(explored_summary)

        user = NODE_USER_PROMPT_TEMPLATE.format(
            question=self.question,
            path_context=path_context_str,
            observations="(no content available; only metadata above)"
        )
        resp = self.llm.chat(NODE_SYSTEM_PROMPT, user)
        decision = self._parse_node(resp)

        if decision.get("discard_this_path"):
            node.status = "discarded"
            return

        if decision.get("can_answer"):
            node.status = "answered"
            node.answer = decision.get("answer")
            node.confidence = float(decision.get("confidence", 0.0))
            self.answers.append(node)
            return

        children = decision.get("proposed_children", []) or []
        if node.depth + 1 >= self.max_depth or len(children) == 0:
            node.status = "expanded"
            return

        for i, child in enumerate(children):
            child_id = child.get("id") or f"{node.path_id}{chr(ord('a')+i)}"
            tws = []
            for tw in child.get("time_windows", []):
                try:
                    tws.append(TimeWindow(
                        start_s=float(tw["start_s"]),
                        end_s=float(tw["end_s"]),
                        purpose=str(tw.get("purpose", "")),
                    ))
                except Exception:
                    continue
            if not tws:
                continue
            nid = str(uuid.uuid4())
            new_node = Node(node_id=nid, path_id=child_id, parent_id=node.node_id,
                            depth=node.depth + 1, plans=tws)
            self.nodes[nid] = new_node
            self.frontier.append(nid)

        node.status = "expanded"

    def run(self, video_meta: str = "", strategy: str = "fifo"):
        plans = self.root_summarize(video_meta)
        if not plans:
            return {"answer": None, "tree": self.serialize_tree(), "reason": "no_root_plans"}

        nodes_processed = 0
        while self.frontier and nodes_processed < self.max_nodes:
            node_id = self.frontier.pop(0) if strategy == "fifo" else self.frontier.pop()
            self.expand_node(node_id)
            nodes_processed += 1
            if any((n.status == "answered" and (n.confidence or 0) >= 0.75) for n in self.answers):
                break

        best = None
        if self.answers:
            best = max(self.answers, key=lambda n: n.confidence or 0.0)

        return {
            "answer": (best.answer if best else None),
            "confidence": (best.confidence if best else None),
            "tree": self.serialize_tree(),
        }

    def serialize_tree(self) -> Dict[str, Any]:
        by_parent: Dict[Optional[str], List[Node]] = {}
        for n in self.nodes.values():
            by_parent.setdefault(n.parent_id, []).append(n)

        def to_dict(n: Node) -> Dict[str, Any]:
            children = [to_dict(c) for c in by_parent.get(n.node_id, [])]
            return {
                "node_id": n.node_id,
                "path_id": n.path_id,
                "depth": n.depth,
                "status": n.status,
                "plans": [vars(tw) for tw in n.plans],
                "clips": [vars(h) for h in n.clip_handles],
                "answer": n.answer,
                "confidence": n.confidence,
                "children": children
            }

        roots = [to_dict(n) for n in by_parent.get(None, [])]
        return {"roots": roots}