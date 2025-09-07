# tot_engine.py
import json
import os
import base64
import io
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image

from tool_video import clip_video_segment, VideoClipResult, ensure_dir
from prompts import ROOT_SYSTEM_PROMPT, build_root_user_prompt, build_node_user_prompt

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
    tool_type: str = "global"  # global| local | slide
    stride: float = 0.0  # Only for "slide" type
    father_clip_result: Optional[VideoClipResult] = None  # 新增：父节点的裁剪结果
    clip_result: Optional[VideoClipResult] = None
    status: str = "pending"  # pending | processed | discarded
    decision: Optional[str] = None
    direct_answer: Optional[str] = None
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    children: List[str] = field(default_factory=list)
    # New: local dialogue snapshot for this node (to be inherited by its children)
    messages: Optional[List[Dict[str, str]]] = None

@dataclass
class ToTRunResult:
    root_paths: List[str]
    nodes: Dict[str, PathNode]
    final_answer: Optional[str] = None
    terminated: bool = False
    # 新增：收集所有找到的答案
    all_answers: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None

class ToTEngine:
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_depth: int = 3,
        per_expand_limit: int = 3,
        temperature: float = 0.2,
    ):
        self.max_depth = max_depth
        self.per_expand_limit = per_expand_limit
        self.temperature = temperature

        self.client = None
        self.llm_model = llm_model
        if OpenAI is not None:
            self.client = OpenAI(
                api_key="sk-VWyPFNDXKVnTiItv66qXrJZIhfEb5kdxqPJoQ5ACHwDl0ulH",
                # api_key="Empty",
                base_url="https://openai.app.msh.team/v1" # Add proper base URL
            )

        # Full dialogue history (root planning only uses this)
        self.messages: List[Dict[str, str]] = []
        
        # Store extracted frames for reuse
        self.frame_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _encode_video(self, video_path: str) -> str:
        """
        Encode a video file to base64 string.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Base64 encoded string of the video
        """
        try:
            with open(video_path, "rb") as video_file:
                encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"Warning: Failed to encode video {video_path}: {e}")
            return ""

    def _extract_initial_frames(self, video_path: str, fps: float = 10.0) -> List[Dict[str, Any]]:
        """
        Extract frames from video at specified fps and store them with timestamps.
        
        Args:
            video_path: Path to the video file
            fps: Frames per second to extract (default: 10.0)
            
        Returns:
            List of dictionaries containing base64 encoded frames and timestamps
        """
        
        if video_path in self.frame_cache:
            return self.frame_cache[video_path]
            
        frames_data = []
        # 查看原始视频帧数
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            print(f"Original video duration: {duration}s")
            print(f"Original video fps: {clip.fps}")
            print(f"Original video width: {clip.w}")
            print(f"Original video height: {clip.h}")
        # import pdb; pdb.set_trace()
        try:
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                interval = 1.0 / fps  # Time interval between frames
                
                timestamp = 0.0
                while timestamp <= duration:
                    try:
                        # Extract frame at this timestamp
                        frame = clip.get_frame(timestamp)
                        
                        # Convert numpy array to PIL Image
                        pil_image = Image.fromarray(frame.astype('uint8'))
                        
                        # Convert to base64
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format='JPEG')
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        frames_data.append({
                            "base64": f"data:image/jpeg;base64,{img_base64}",
                            "timestamp": timestamp
                        })
                        
                        timestamp += interval
                        
                    except Exception as e:
                        print(f"Warning: Failed to extract frame at {timestamp}s: {e}")
                        timestamp += interval
                        continue
                        
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            
        # Cache the extracted frames
        self.frame_cache[video_path] = frames_data
        return frames_data

    def _find_nearest_frame(self, target_timestamp: float, available_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find the frame with timestamp closest to the target timestamp.
        
        Args:
            target_timestamp: The desired timestamp
            available_frames: List of available frames with timestamps
            
        Returns:
            Frame dictionary with closest timestamp
        """
        if not available_frames:
            return None
            
        closest_frame = min(available_frames, key=lambda f: abs(f["timestamp"] - target_timestamp))
        return closest_frame

    def _extract_frames_with_timestamps(self, video_path: str, start_s: float = None, end_s: float = None) -> List[Dict[str, Any]]:
        """
        Extract 16 frames evenly spaced within the specified time interval,
        using the nearest available frames from the initial extraction.
        
        Args:
            video_path: Path to the video file (or time interval for tools)
            start_s: Start time in seconds (None for full video)
            end_s: End time in seconds (None for full video)
            
        Returns:
            List of dictionaries containing base64 encoded frames and timestamps
        """
        # Get all available frames
        available_frames = self.frame_cache.get(video_path, [])
        if not available_frames:
            # Fallback: extract frames if not cached
            available_frames = self._extract_initial_frames(video_path)
        
        # If no time interval specified, return all frames for initial processing
        if start_s is None or end_s is None:
            return available_frames
        
        # For tool processing: select 16 frames within the specified interval
        target_frames = []
        num_frames = 16
        
        if end_s <= start_s:
            end_s = start_s + 0.1  # Minimum interval
        
        interval_duration = end_s - start_s
        frame_interval = interval_duration / (num_frames - 1) if num_frames > 1 else 0
        
        for i in range(num_frames):
            target_timestamp = start_s + (i * frame_interval)
            nearest_frame = self._find_nearest_frame(target_timestamp, available_frames)
            
            if nearest_frame:
                # Create a copy with the target timestamp for context
                frame_copy = nearest_frame.copy()
                frame_copy["original_timestamp"] = nearest_frame["timestamp"]
                frame_copy["target_timestamp"] = target_timestamp
                target_frames.append(frame_copy)
        
        return target_frames

    def _chat_to_json(self, messages: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Send messages (or self.messages if None) to the LLM and parse a strict JSON response.
        On failure, return a safe terminate decision.
        Only append assistant content to the given thread.
        """
        use_global = messages is None
        msgs = self.messages if use_global else messages

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
            # Keep dialogue shape
            msgs.append({"role": "assistant", "content": json.dumps(dummy, ensure_ascii=False)})
            return dummy

        resp = self.client.chat.completions.create(
            model="google/gemini-2.5-flash",
            temperature=self.temperature,
            messages=msgs,
        )
        text = resp.choices[0].message.content.strip()

        # Append assistant message to the same thread used for the request
        msgs.append({"role": "assistant", "content": text})
        raw_text = text
        import re
        match = re.search(r'<\s*TOOL_CALL\s*>(.*?)<\s*/\s*TOOL_CALL\s*>', raw_text, re.DOTALL)
        text = match.group(1) if match else ''

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
                data = {
                    "decision": "terminate",
                    "rationale": "JSON parse failure",
                    "proposed_paths": [],
                    "direct_answer": None,
                    "confidence": 0.0,
                }

        data.setdefault("proposed_paths", [])
        data.setdefault("direct_answer", None)
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
            start_s = float(p.get("start_s", 0.0))
            end_s = float(p.get("end_s", max(0.1, min(duration, start_s + 5))))
            tool_type = str(p.get("tool_type", "global")).lower()
            stride = float(p.get("stride", 0.0))
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
                tool_type=tool_type,
                stride=stride,
            )
            nodes.append(node)
        return nodes

    def run(self, video_path: str, question: str) -> ToTRunResult:
        # import pdb; pdb.set_trace()
        # print(video_path)
        
        # Set up video-specific working directory
        video_dir = os.path.dirname(video_path)
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        # Create video-specific workdir
        self.workdir = os.path.join(video_dir, f"{video_name}_work")
        ensure_dir(self.workdir)
        
        meta = self._video_meta(video_path)

        # Extract initial frames at fps=4 with timestamp information
        frames_data = self._extract_initial_frames(video_path, fps=10.0)
        
        # Build content list with alternating frames and timestamps
        frame_content = []

        init_segment_frames = self._extract_frames_with_timestamps(
                video_path, 
                start_s=0, 
                end_s=meta["duration"]
        )
        print(len(init_segment_frames))

        for frame_info in init_segment_frames:
            frame_content.append({"type": "image_url", "image_url": {"url": frame_info["base64"]}})
            frame_content.append({"type": "text", "text": f"timestamp: {frame_info['timestamp']}s"})

        frame_content.append({"type": "text", "text": build_root_user_prompt(meta, question, max_paths=self.per_expand_limit)})
        # Initialize dialogue with system and frame-based user prompt
        self.messages = [
            {"role": "system", "content": ROOT_SYSTEM_PROMPT},
            {"role": "user", "content": frame_content},
        ]

        # Root planning using the global thread
        root_decision = self._chat_to_json()
        nodes: Dict[str, PathNode] = {}
        root_paths: List[str] = []

        if root_decision.get("decision") == "answer":
            all_answers = [{
                "path_id": "root",
                "answer": root_decision.get("direct_answer"),
                "rationale": root_decision.get("rationale", "")
            }]
            return ToTRunResult(root_paths=[], nodes=nodes, final_answer=root_decision.get("direct_answer"), terminated=True, all_answers=all_answers)
        if root_decision.get("decision") == "terminate":
            return ToTRunResult(root_paths=[], nodes=nodes, final_answer=None, terminated=True, all_answers=[])

        proposed = root_decision.get("proposed_paths", [])[: self.per_expand_limit]
        initial_nodes = self._sanitize_paths(proposed, parent_id=None, duration=meta["duration"], depth=1)
        for n in initial_nodes:
            nodes[n.path_id] = n
            root_paths.append(n.path_id)

        # 收集所有找到的答案
        all_answers = []
        final_answer = None
        terminated = False

        # Snapshot of root dialogue history to seed the first layer
        root_history = list(self.messages)

        # Sequential expansion (dialogue-driven) with per-path inherited context
        queue = list(root_paths)

        while queue:
            pid = queue.pop(0)
            node = nodes[pid]
            if node.depth > self.max_depth:
                continue

            # Tool call: clip segment
            # father_clip_result = node.father_clip_result if node.father_clip_result else 0.0            
            if node.tool_type in ["global", "local"]:
                clip_res = clip_video_segment(video_path, node.start_s, node.end_s, workdir=self.workdir, tool_type=node.tool_type)
            elif node.tool_type in ["slide"]:  
                clip_res = clip_video_segment(video_path, node.father_clip_result.start_s, node.father_clip_result.end_s, workdir=self.workdir, tool_type=node.tool_type, stride=node.stride)
            node.clip_result = clip_res
            node.clip_result = clip_res

            # Build local thread for this node:
            # - If root-level node: start from root_history
            # - Else: inherit from parent's messages (entire path context)
            if node.parent_id is None:
                local_messages: List[Dict[str, str]] = list(root_history)
            else:
                parent = nodes[node.parent_id]
                # Parent must have messages prepared; defensive copy
                local_messages = list(parent.messages) if parent.messages is not None else list(root_history)

            # Append this node's tool result (local only)
            # tool_msg = (
            #     f"[Tool Result] Time interval analysis: [{clip_res.start_s:.2f}s - {clip_res.end_s:.2f}s] "
            #     f"(duration: {clip_res.duration:.2f}s). Extracted 16 frames evenly spaced within this interval."
            # )
            # local_messages.append({"role": "assistant", "content": tool_msg})

            # Per-node system context (local only)
            # local_messages.append({"role": "system", "content": PER_NODE_SYSTEM_PROMPT})

            # Extract frames from the time interval (16 frames evenly spaced)
            segment_frames = self._extract_frames_with_timestamps(
                video_path, 
                start_s=clip_res.start_s, 
                end_s=clip_res.end_s
            )

            print(len(segment_frames))
            
            # Build content for the segment
            segment_content = []
            for frame_info in segment_frames:
                segment_content.append({"type": "image_url", "image_url": {"url": frame_info["base64"]}})
                # Show both target and original timestamps for context
                if 'original_timestamp' in frame_info:
                    timestamp_text = f"timestamp: {frame_info['original_timestamp']:.2f}s"
                else:
                    timestamp_text = f"timestamp: {frame_info.get('target_timestamp', frame_info['timestamp']):.2f}s"
                segment_content.append({"type": "text", "text": timestamp_text})
            
            # Add tool message and node prompt as text
            segment_content.append({"type": "text", "text": build_node_user_prompt(
                path_id=node.path_id,
                strategy=node.strategy,
                start_s=node.start_s,
                end_s=node.end_s,
                clip_path=clip_res.path,
                duration=clip_res.duration,
                max_paths=self.per_expand_limit
            )})

            # Node-specific user prompt (local only)
            local_messages.append({
                "role": "user",
                "content": segment_content
            })

            # Get decision for this node using the local thread
            # import pdb; pdb.set_trace()
            node_decision = self._chat_to_json(messages=local_messages)

            # Persist the local thread on the node for its children to inherit
            node.messages = local_messages

            # Update node status
            node.status = "processed"
            node.decision = node_decision.get("decision")
            node.rationale = node_decision.get("rationale")
            node.confidence = node_decision.get("confidence", 0.0)

            # Branch handling
            if node.decision == "discard":
                node.status = "discarded"
                continue

            if node.decision == "answer":
                # 收集答案但不终止，继续探索其他路径
                answer_info = {
                    "path_id": node.path_id,
                    "answer": node_decision.get("direct_answer"),
                    "rationale": node_decision.get("rationale", ""),
                    "start_s": node.start_s,
                    "end_s": node.end_s,
                    "strategy": node.strategy,
                    "depth": node.depth
                }
                all_answers.append(answer_info)
                node.direct_answer = node_decision.get("direct_answer")
                
                # 如果这是第一个答案，设置为final_answer
                if final_answer is None:
                    final_answer = node_decision.get("direct_answer")
                    confidence = node_decision.get("confidence", 0.0)

                
                
                print(f"Found answer from path {node.path_id}: {node_decision.get('direct_answer')}")
                # 不break，继续处理其他路径
                continue

            # if node.decision == "terminate":
            #     # 只有在明确要求终止时才终止，但仍然继续处理queue中的其他路径
            #     print(f"Path {node.path_id} chose to terminate")
            #     continue

            if node.confidence < 3:
                continue

            if node.decision == "expand":
                if node.depth >= self.max_depth:
                    print(f"Path {node.path_id} reached max depth {self.max_depth}")
                    continue
                child_nodes = self._sanitize_paths(
                    node_decision.get("proposed_paths", [])[: self.per_expand_limit],
                    parent_id=node.path_id,
                    duration=meta["duration"],
                    depth=node.depth + 1,
                )
                for child in child_nodes:
                    child.father_clip_result = node.clip_result  # 传递父节点的clip_result
                    nodes[child.path_id] = child
                    node.children.append(child.path_id)
                    queue.append(child.path_id)

        # 如果收集到了答案，设置terminated为True
        if all_answers:
            terminated = True
            print(f"Exploration completed. Found {len(all_answers)} answers in total.")
        else:
            print("Exploration completed. No answers found.")

        return ToTRunResult(root_paths=root_paths, nodes=nodes, final_answer=final_answer, terminated=terminated, all_answers=all_answers, confidence=confidence)

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
                "father_clip_path": n.father_clip_result.path if n.father_clip_result else None,  # 新增导出字段
                "father_clip_result": n.father_clip_result,
                "tool_type": n.tool_type,
                "stride": n.stride,
                "clip_path": n.clip_result.path if n.clip_result else None,
                "children": n.children,
                "messages": n.messages,
                "confidence": n.confidence,
            }
        # import pdb; pdb.set_trace()
        return {
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "terminated": result.terminated,
            "root_paths": result.root_paths,
            "nodes": out_nodes,
            "all_answers": result.all_answers,  # 新增：导出所有找到的答案
        }
