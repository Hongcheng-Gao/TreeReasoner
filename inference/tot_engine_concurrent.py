# tot_engine_concurrent.py
import json
import os
import base64
import io
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image
from copy import deepcopy

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
    father_start_s: float = 0.0
    father_end_s: float = 0.0
    clip_result: Optional[VideoClipResult] = None
    status: str = "pending"  # pending | processed | discarded
    decision: Optional[str] = None
    direct_answer: Optional[str] = None
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    children: List[str] = field(default_factory=list)
    messages: Optional[List[Dict[str, str]]] = None

@dataclass
class ToTRunResult:
    root_paths: List[str]
    nodes: Dict[str, PathNode]
    final_answer: Optional[str] = None
    terminated: bool = False
    all_answers: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None

class ConcurrentToTEngine:
    """线程安全的ToT引擎，每个实例都有独立的状态"""
    
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_depth: int = 3,
        per_expand_limit: int = 3,
        temperature: float = 0.2,
        thread_safe: bool = True,
    ):
        self.max_depth = max_depth
        self.per_expand_limit = per_expand_limit
        self.temperature = temperature
        self.thread_safe = thread_safe
        self.thread_id = threading.get_ident() if thread_safe else None
        
        # 每个实例独立的客户端
        self.client = None
        self.llm_model = llm_model
        if OpenAI is not None:
            self.client = OpenAI(
                api_key="Empty",
                base_url="https://x35thinking-toyama-sft-hyy-base.app.msh.team/v1"
            )

        # 实例独立的状态 - 每次run都会重置
        self.messages: List[Dict[str, str]] = []
        self.frame_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.workdir: Optional[str] = None
        
        # 线程安全锁（如果需要）
        if thread_safe:
            self._lock = threading.RLock()
        else:
            self._lock = None

    def _thread_safe_operation(self, func, *args, **kwargs):
        """线程安全操作包装器"""
        if self._lock:
            with self._lock:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _encode_video(self, video_path: str) -> str:
        """
        Encode a video file to base64 string.
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
        线程安全版本 - 使用实例级缓存
        """
        # 检查实例级缓存
        if video_path in self.frame_cache:
            return self.frame_cache[video_path]
            
        frames_data = []
        
        try:
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                print(f"[Thread {self.thread_id}] Video duration: {duration}s, fps: {clip.fps}")
                
                interval = 1.0 / fps
                timestamp = 0.0
                
                while timestamp <= duration:
                    try:
                        frame = clip.get_frame(timestamp)
                        pil_image = Image.fromarray(frame.astype('uint8'))
                        
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
            
        # 存储到实例级缓存
        self.frame_cache[video_path] = frames_data
        return frames_data

    def _find_nearest_frame(self, target_timestamp: float, available_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the frame with timestamp closest to the target timestamp."""
        if not available_frames:
            return None
            
        closest_frame = min(available_frames, key=lambda f: abs(f["timestamp"] - target_timestamp))
        return closest_frame

    def _extract_frames_with_timestamps(self, video_path: str, start_s: float = None, end_s: float = None) -> List[Dict[str, Any]]:
        """
        Extract 16 frames evenly spaced within the specified time interval.
        """
        # 获取可用帧
        available_frames = self.frame_cache.get(video_path, [])
        if not available_frames:
            available_frames = self._extract_initial_frames(video_path)
        
        if start_s is None or end_s is None:
            return available_frames
        
        # 选择16帧
        target_frames = []
        num_frames = 16
        
        if end_s <= start_s:
            end_s = start_s + 0.1
        
        interval_duration = end_s - start_s
        frame_interval = interval_duration / (num_frames - 1) if num_frames > 1 else 0
        
        for i in range(num_frames):
            target_timestamp = start_s + (i * frame_interval)
            nearest_frame = self._find_nearest_frame(target_timestamp, available_frames)
            
            if nearest_frame:
                frame_copy = nearest_frame.copy()
                frame_copy["original_timestamp"] = nearest_frame["timestamp"]
                frame_copy["target_timestamp"] = target_timestamp
                target_frames.append(frame_copy)
        
        return target_frames

    def _chat_to_json(self, messages: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Send messages to the LLM and parse a strict JSON response.
        线程安全版本 - 使用独立的消息副本
        """
        use_global = messages is None
        if use_global:
            msgs = deepcopy(self.messages)  # 使用深拷贝避免状态污染
        else:
            msgs = deepcopy(messages)

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
                "evidence_confidence": 0.3, 
            }
            
            # 更新相应的消息列表
            if use_global:
                self.messages.append({"role": "assistant", "content": json.dumps(dummy, ensure_ascii=False)})
            else:
                messages.append({"role": "assistant", "content": json.dumps(dummy, ensure_ascii=False)})
            return dummy

        try:
            # Retry logic: if resp is None, retry up to 20 times
            resp = None
            for attempt in range(20):
                try:
                    resp = self.client.chat.completions.create(
                        model="x35thinking-toyama-sft-hyy-base",
                        temperature=self.temperature,
                        messages=msgs,
                    )
                    print("response", resp)
                    if resp is not None:
                        break
                except Exception as e:
                    print(f"[Thread {self.thread_id}] API call attempt {attempt + 1} failed: {e}")
                    if attempt == 19:  # Last attempt
                        raise e
            
            if resp is None:
                raise Exception("API call failed after 20 attempts")
                
            text = resp.choices[0].message.content.strip()

            # 更新相应的消息列表
            if use_global:
                self.messages.append({"role": "assistant", "content": text})
            else:
                messages.append({"role": "assistant", "content": text})
            
            # 解析JSON
            raw_text = text
            import re
            match = re.search(r'<\s*TOOL_CALL\s*>(.*?)<\s*/\s*TOOL_CALL\s*>', raw_text, re.DOTALL)
            text = match.group(1) if match else ''

            try:
                data = json.loads(text)
            except Exception:
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
                        "evidence_confidence": 0.0,
                    }

            data.setdefault("proposed_paths", [])
            data.setdefault("direct_answer", None)
            data.setdefault("rationale", "")
            data.setdefault("decision", "terminate")
            return data
            
        except Exception as e:
            print(f"[Thread {self.thread_id}] Error in LLM call: {e}")
            return {
                "decision": "terminate",
                "rationale": f"LLM call failed: {str(e)}",
                "proposed_paths": [],
                "direct_answer": None,
                "evidence_confidence": 0.0,
            }

    def _video_meta(self, video_path: str) -> Dict[str, Any]:
        """获取视频元数据"""
        with VideoFileClip(video_path) as clip:
            return {
                "duration": float(clip.duration),
                "fps": float(clip.fps) if clip.fps else None,
                "width": int(clip.w),
                "height": int(clip.h),
                "path": video_path,
            }

    def _sanitize_paths(self, proposed: List[Dict[str, Any]], parent_id: Optional[str], duration: float, depth: int) -> List[PathNode]:
        """清理和验证提议的路径"""
        nodes: List[PathNode] = []
        for i, p in enumerate(proposed):
            pid = str(p.get("id") or f"P{depth}_{i+1}")
            strat = str(p.get("strategy") or "expand")
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
        """
        主要的运行方法 - 线程安全版本
        每次运行都重置实例状态
        """
        # 重置实例状态
        self.messages = []
        self.frame_cache = {}
        
        # 设置视频特定的工作目录
        video_dir = os.path.dirname(video_path)
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        # 创建线程特定的工作目录
        thread_suffix = f"_t{self.thread_id}" if self.thread_id else ""
        self.workdir = os.path.join(video_dir, f"{video_name}_work{thread_suffix}")
        ensure_dir(self.workdir)
        
        print(f"[Thread {self.thread_id}] Processing {video_name} in {self.workdir}")
        
        try:
            meta = self._video_meta(video_path)

            # 提取初始帧
            frames_data = self._extract_initial_frames(video_path, fps=10.0)
            
            # 构建帧内容
            frame_content = []
            init_segment_frames = self._extract_frames_with_timestamps(
                video_path, 
                start_s=0, 
                end_s=meta["duration"]
            )

            for frame_info in init_segment_frames:
                frame_content.append({"type": "image_url", "image_url": {"url": frame_info["base64"]}})
                frame_content.append({"type": "text", "text": f"timestamp: {frame_info['timestamp']}s"})

            frame_content.append({"type": "text", "text": build_root_user_prompt(meta, question, max_paths=self.per_expand_limit)})
            
            # 初始化对话
            self.messages = [
                {"role": "system", "content": ROOT_SYSTEM_PROMPT},
                {"role": "user", "content": frame_content},
            ]

            # 根规划
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
            confidence = None

            # 根对话历史快照
            root_history = deepcopy(self.messages)

            # 顺序扩展
            queue = list(root_paths)

            while queue:
                pid = queue.pop(0)
                node = nodes[pid]
                if node.depth > self.max_depth:
                    continue
         
                # 工具调用：剪辑片段
                try:
                    if node.tool_type in ["global", "local"]:
                        clip_res = clip_video_segment(video_path, node.start_s, node.end_s, workdir=self.workdir, tool_type=node.tool_type)
                    elif node.tool_type in ["slide"]:  
                        clip_res = clip_video_segment(video_path, node.father_start_s, node.father_end_s, workdir=self.workdir, tool_type=node.tool_type, stride=node.stride)
                    node.clip_result = clip_res
                except Exception as e:
                    print(f"[Thread {self.thread_id}] Error clipping video segment: {e}")
                    continue

                # 构建本地线程
                if node.parent_id is None:
                    local_messages = deepcopy(root_history)
                else:
                    parent = nodes[node.parent_id]
                    local_messages = deepcopy(parent.messages) if parent.messages is not None else deepcopy(root_history)

                # 提取片段帧
                segment_frames = self._extract_frames_with_timestamps(
                    video_path, 
                    start_s=clip_res.start_s, 
                    end_s=clip_res.end_s
                )
                
                # 构建片段内容
                segment_content = []
                for frame_info in segment_frames:
                    segment_content.append({"type": "image_url", "image_url": {"url": frame_info["base64"]}})
                    if 'original_timestamp' in frame_info:
                        timestamp_text = f"timestamp: {frame_info['original_timestamp']:.2f}s"
                    else:
                        timestamp_text = f"timestamp: {frame_info.get('target_timestamp', frame_info['timestamp']):.2f}s"
                    segment_content.append({"type": "text", "text": timestamp_text})
                
                segment_content.append({"type": "text", "text": build_node_user_prompt(
                    path_id=node.path_id,
                    strategy=node.strategy,
                    start_s=node.start_s,
                    end_s=node.end_s,
                    clip_path=clip_res.path,
                    duration=clip_res.duration,
                    max_paths=self.per_expand_limit
                )})

                local_messages.append({
                    "role": "user",
                    "content": segment_content
                })

                # 获取节点决策
                node_decision = self._chat_to_json(messages=local_messages)

                # 保存本地线程到节点
                node.messages = local_messages

                # 更新节点状态
                node.status = "processed"
                node.decision = node_decision.get("decision")
                node.rationale = node_decision.get("rationale")
                node.confidence = node_decision.get("evidence_confidence", 0.0)

                # 分支处理
                if node.decision == "discard":
                    node.status = "discarded"
                    continue

                if node.decision == "answer":
                    answer_info = {
                        "path_id": node.path_id,
                        "answer": node_decision.get("direct_answer"),
                        "rationale": node_decision.get("rationale", ""),
                        "start_s": node.start_s,
                        "end_s": node.end_s,
                        "strategy": node.strategy,
                        "depth": node.depth,
                        "evidence_confidence": node_decision.get("evidence_confidence", 0.0)
                    }
                    all_answers.append(answer_info)
                    node.direct_answer = node_decision.get("direct_answer")
                    
                    if final_answer is None:
                        final_answer = node_decision.get("direct_answer")
                        confidence = node_decision.get("evidence_confidence", 0.0)

                    print(f"[Thread {self.thread_id}] Found answer from path {node.path_id}: {node_decision.get('direct_answer')}")
                    continue

                if node.confidence and node.confidence < 4 and node.confidence > 0:
                    continue

                if node.decision == "expand":
                    if node.depth >= self.max_depth:
                        print(f"[Thread {self.thread_id}] Path {node.path_id} reached max depth {self.max_depth}")
                        continue
                    
                    child_nodes = self._sanitize_paths(
                        node_decision.get("proposed_paths", [])[: self.per_expand_limit],
                        parent_id=node.path_id,
                        duration=meta["duration"],
                        depth=node.depth + 1,
                    )
                    for child in child_nodes:
                        child.father_start_s = node.start_s
                        child.father_end_s = node.end_s
                        nodes[child.path_id] = child
                        node.children.append(child.path_id)
                        queue.append(child.path_id)

            # 设置结果状态
            if all_answers:
                terminated = True
                print(f"[Thread {self.thread_id}] Exploration completed. Found {len(all_answers)} answers.")
            else:
                print(f"[Thread {self.thread_id}] Exploration completed. No answers found.")

            return ToTRunResult(
                root_paths=root_paths, 
                nodes=nodes, 
                final_answer=final_answer, 
                terminated=terminated, 
                all_answers=all_answers, 
                confidence=confidence
            )
            
        except Exception as e:
            print(f"[Thread {self.thread_id}] Error in run method: {e}")
            import traceback
            traceback.print_exc()
            return ToTRunResult(root_paths=[], nodes={}, final_answer=None, terminated=True, all_answers=[])

    @staticmethod
    def export_tree(result: ToTRunResult) -> Dict[str, Any]:
        """导出推理树"""
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
                "father_start_s": n.father_start_s,
                "father_end_s": n.father_end_s,
                "tool_type": n.tool_type,
                "stride": n.stride,
                "clip_path": n.clip_result.path if n.clip_result else None,
                "children": n.children,
                "messages": n.messages,
                "evidence_confidence": n.confidence,
            }
        
        return {
            "final_answer": result.final_answer,
            "evidence_confidence": result.confidence,
            "terminated": result.terminated,
            "root_paths": result.root_paths,
            "nodes": out_nodes,
            "all_answers": result.all_answers,
        } 
