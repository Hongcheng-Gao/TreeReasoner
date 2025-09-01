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
    tool_type: str = "raw",
    current_segment_start_s: float = 0.0,
) -> VideoClipResult:
    """
    The single tool: clip a video segment [start_s, end_s] from input_video_path
    and return the path to the clipped video file.
    
    Args:
        input_video_path: Path to the video file (original video for all modes)
        start_s: Start time in seconds
        end_s: End time in seconds
        workdir: Working directory for output files
        codec: Video codec
        audio_codec: Audio codec
        crf: Constant Rate Factor for quality
        tool_type: Clipping strategy - "raw", "segment", or "relative"
        current_segment_start_s: Start time of current segment in original video
    """
    if start_s < 0:
        start_s = 0.0
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")

    ensure_dir(workdir)
    uid = str(uuid.uuid4())[:8]
    
    # Determine the actual time range based on tool_type
    if tool_type == "raw":
        # Use input video path directly with original time range
        actual_start_s = start_s
        actual_end_s = end_s
        result_start_s = start_s
        result_end_s = end_s
    elif tool_type in ["segment", "relative"]:
        # Calculate absolute time points in the original video
        actual_start_s = current_segment_start_s + start_s
        actual_end_s = current_segment_start_s + end_s
        result_start_s = actual_start_s
        result_end_s = actual_end_s
    else:
        raise ValueError(f"Unsupported tool_type: {tool_type}")

    out_path = os.path.join(workdir, f"segment_{uid}_{int(actual_start_s)}_{int(actual_end_s)}.mp4")

    with VideoFileClip(input_video_path) as clip:
        duration = clip.duration
        
        # Ensure end time doesn't exceed video duration
        if actual_end_s > duration:
            actual_end_s = duration
            result_end_s = actual_end_s
        
        # Ensure start time is valid
        if actual_start_s >= duration:
            raise ValueError(f"Start time {actual_start_s}s exceeds video duration {duration}s")
            
        subclip = clip.subclip(actual_start_s, actual_end_s)
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

    return VideoClipResult(out_path, result_start_s, result_end_s, seg_duration)