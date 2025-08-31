# tool_video.py
import os
import uuid
from typing import Optional
from dataclasses import dataclass
from moviepy.editor import VideoFileClip

@dataclass
class VideoClipResult:
    path: str
    start_s: float
    end_s: float
    duration: float

    def to_dict(self):
        return {
            "path": self.path,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration": self.duration,
        }

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def clip_video_segment(
    input_video_path: str,
    start_s: float,
    end_s: float,
    workdir: str = "./work",
    codec: str = "libx264",
    audio_codec: Optional[str] = "aac",
    crf: int = 23,
) -> VideoClipResult:
    """
    The single tool: clip a video segment [start_s, end_s] from input_video_path
    and return the path to the clipped video file.
    """
    if start_s < 0:
        start_s = 0.0
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")

    ensure_dir(workdir)
    uid = str(uuid.uuid4())[:8]
    out_path = os.path.join(workdir, f"segment_{uid}_{int(start_s)}_{int(end_s)}.mp4")

    with VideoFileClip(input_video_path) as clip:
        duration = clip.duration
        if end_s > duration:
            end_s = duration
        subclip = clip.subclip(start_s, end_s)
        subclip.write_videofile(
            out_path,
            codec=codec,
            audio_codec=audio_codec,
            temp_audiofile=os.path.join(workdir, f"temp-audio-{uid}.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None,
            threads=2,
            ffmpeg_params=["-crf", str(crf)],
        )
        seg_duration = subclip.duration

    return VideoClipResult(out_path, start_s, end_s, seg_duration)