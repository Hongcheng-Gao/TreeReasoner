# tool_video.py
from dataclasses import dataclass
from typing import Optional
import uuid

@dataclass
class ClipHandle:
    clip_id: str
    start_s: float
    end_s: float
    # 仅元数据，不含任何语义内容
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    note: Optional[str] = None

class VideoClipper:
    def __init__(self, video_path: str, default_fps: float = 2.0):
        self.video_path = video_path
        self.default_fps = default_fps
        # 可在此初始化视频句柄、时长、关键帧索引等（不在此示例中实现）

    def extract_clip(self, start_s: float, end_s: float, fps: Optional=float) -> ClipHandle:
        """
        只抽片段并返回一个句柄与基本元数据；不做任何转写/摘要/识别。
        """
        start_s = max(0.0, float(start_s))
        end_s = max(start_s, float(end_s))
        fps = fps if fps is not None else self.default_fps
        duration = end_s - start_s
        frame_count = int(max(1, duration * fps))
        handle = ClipHandle(
            clip_id=str(uuid.uuid4()),
            start_s=start_s,
            end_s=end_s,
            fps=fps,
            frame_count=frame_count,
            note=f"extracted {duration:.1f}s at ~{fps} fps (no transcription/summary)"
        )
        return handle