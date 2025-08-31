# run.py
import json
from tot_engine import ToTEngine, LLM
from tool_video import VideoClipper

def main():
    video_path = "demo_video.mp4"
    question = "视频中在10秒到40秒期间，是否有人把红色杯子放在桌子上？"

    video_meta = json.dumps({
        "duration_s": 120.0,
        "fps": 30,
        "scene_bounds": [[0,10], [10,25], [25,45], [45,120]],
        "notes": "No transcript/ASR available. Tool only returns clip handles."
    }, ensure_ascii=False)

    llm = LLM()
    clipper = VideoClipper(video_path)
    engine = ToTEngine(llm=llm, clipper=clipper, question=question, max_nodes=12, max_depth=3)
    result = engine.run(video_meta=video_meta, strategy="fifo")

    print("Answer:", result.get("answer"))
    print("Confidence:", result.get("confidence"))
    print("Tree JSON:")
    print(json.dumps(result.get("tree"), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()