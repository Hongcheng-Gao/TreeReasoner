import os
import uuid
from typing import Optional
from dataclasses import dataclass
from moviepy.editor import VideoFileClip

@dataclass
class VideoClipResult:
    path: str  # Now represents the original video path instead of a clip path
    start_s: float
    end_s: float
    duration: float
    is_time_interval: bool = True  # Flag to indicate this is a time interval, not a file

    def to_dict(self):
        return {
            "path": self.path,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration": self.duration,
            "is_time_interval": self.is_time_interval,
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
    slide: float = 0.0,
) -> VideoClipResult:
    """
    The single tool: return a time interval [start_s, end_s] from input_video_path
    instead of creating an actual video clip.
    
    Args:
        input_video_path: Path to the video file (original video for all modes)
        start_s: Start time in seconds
        end_s: End time in seconds
        workdir: Working directory for output files (unused but kept for compatibility)
        codec: Video codec (unused but kept for compatibility)
        audio_codec: Audio codec (unused but kept for compatibility)
        crf: Constant Rate Factor for quality (unused but kept for compatibility)
        tool_type: Clipping strategy - "raw", "segment", or "relative"
        slide: Start time of current segment in original video
    """
    if start_s < 0:
        start_s = 0.0
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")

    # Determine the actual time range based on tool_type
    if tool_type in ["global", "local"]:
        # Use input video path directly with original time range
        actual_start_s = start_s
        actual_end_s = end_s
        result_start_s = start_s
        result_end_s = end_s
    elif tool_type in ["slide"]:
        # Calculate absolute time points in the original video
        actual_start_s = slide + start_s
        actual_end_s = slide + end_s
        result_start_s = actual_start_s
        result_end_s = actual_end_s
    else:
        raise ValueError(f"Unsupported tool_type: {tool_type}")

    # Get video duration for validation
    try:
        with VideoFileClip(input_video_path) as clip:
            duration = clip.duration
            
            # Ensure end time doesn't exceed video duration
            if actual_end_s > duration:
                actual_end_s = duration
                result_end_s = actual_end_s
            
            # Ensure start time is valid
            if actual_start_s >= duration:
                raise ValueError(f"Start time {actual_start_s}s exceeds video duration {duration}s")
    except Exception as e:
        print(f"Warning: Could not validate video duration: {e}")
        # Continue with provided times if validation fails

    seg_duration = result_end_s - result_start_s

    # Return time interval instead of actual clip
    return VideoClipResult(
        path=input_video_path,  # Original video path
        start_s=result_start_s, 
        end_s=result_end_s, 
        duration=seg_duration,
        is_time_interval=True
    )