# run.py
import argparse
import json
import os
from tot_engine import ToTEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=False, default="/mnt/moonfs/kimiv-m2/huangzihao/dataset/Video-R1-data/Video-R1-COT-165k.json", help="Input JSON file containing CLEVRER data")
    parser.add_argument("--video", required=False, default="/mnt/moonfs/kimiv-public-m2/huangzihao/data/video/movie_mp4/longvila_videos_O/OJ4QDtCxKbQ.mp4", help="Input video path (used when not processing JSON)")
    parser.add_argument("--question", required=False, default="What is the primary intention behind the video, based on the evolving scenes and contextual clues?  \nD. To contrast street food preparation with high-end restaurant dining in Vietnam.  \nA. To promote a specific Vietnamese restaurant chain's menu and pricing.  \nB. To showcase a curated culinary guide to Vietnamese food experiences, likely tied to a travel or food content creator.  \nC. To document a personal travel experience focused on beach relaxation with incidental food scenes.  \n", help="Question (used when not processing JSON)")
    parser.add_argument("--workdir", default="./work", help="Working directory")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--per_expand_limit", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--save_tree", default="./tree.json", help="Path to save the reasoning tree JSON")
    parser.add_argument("--process_clevrer", action="store_true", help="Process CLEVRER entries from input JSON")
    parser.add_argument("--output_results", default="./clevrer_results.json", help="Path to save CLEVRER processing results")
    parser.add_argument("--data_dir", default="/mnt/moonfs/kimiv-m2/huangzihao/dataset/Video/clevrer_tmp_v5", help="Base data directory for outputs")
    args = parser.parse_args()

    engine = ToTEngine(
        llm_model=args.model,
        api_key=args.api_key,
        workdir=args.workdir,
        max_depth=args.max_depth,
        per_expand_limit=args.per_expand_limit,
        temperature=args.temperature,
    )

    if args.process_clevrer:
        # Process CLEVRER entries from JSON file
        print(f"Loading JSON file: {args.input_json}")
        with open(args.input_json, "r", encoding="utf-8") as f:
            
            data = json.load(f)
        
        # Filter CLEVRER entries
        clevrer_entries = [entry for entry in data if "CLEVRER" in entry.get("path", "")]
        print(f"Found {len(clevrer_entries)} CLEVRER entries")
        
        results = []
        for i, entry in enumerate(clevrer_entries):
            print(f"Processing entry {i+1}/{len(clevrer_entries)}: {entry['path']}")
            
            video_path = entry["path"]
            # import pdb; pdb.set_trace()
            video_path = video_path.replace("./", "/mnt/moonfs/kimiv-m2/huangzihao/dataset/Video-R1-data/")
            
            # Extract video filename without extension for folder name
            video_filename = os.path.basename(video_path)
            # import pdb; pdb.set_trace()
            video_name = os.path.splitext(video_filename)[0]  # Remove .mp4 extension
            
            # Create output directory for this video
            video_output_dir = os.path.join(args.data_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            
            question = entry["problem"]
            solution = entry["solution"]
            
            try:
                result = engine.run(video_path, question)
                tree = engine.export_tree(result)
                
                # Add solution to the result
                entry_result = {
                    "problem_id": entry.get("problem_id"),
                    "video_path": video_path,
                    "question": question,
                    "solution": solution,
                    "final_answer": result.final_answer,
                    "terminated": result.terminated,
                    "total_nodes": len(result.nodes),
                    "reasoning_tree": tree
                }
                results.append(entry_result)
                
                # Save individual result file in the video's directory
                individual_result_path = os.path.join(video_output_dir, "result.json")
                with open(individual_result_path, "w", encoding="utf-8") as f:
                    json.dump(entry_result, f, ensure_ascii=False, indent=2)
                
                # Save reasoning tree separately
                tree_path = os.path.join(video_output_dir, "tree.json")
                with open(tree_path, "w", encoding="utf-8") as f:
                    json.dump(tree, f, ensure_ascii=False, indent=2)
                
                print(f"Final answer: {result.final_answer}")
                print(f"Solution: {solution}")
                print(f"Total nodes: {len(result.nodes)}")
                print(f"Output saved to: {video_output_dir}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error processing entry {i+1}: {e}")
                error_result = {
                    "problem_id": entry.get("problem_id"),
                    "video_path": video_path,
                    "question": question,
                    "solution": solution,
                    "error": str(e)
                }
                results.append(error_result)
                
                # Save error result in the video's directory
                error_result_path = os.path.join(video_output_dir, "error.json")
                with open(error_result_path, "w", encoding="utf-8") as f:
                    json.dump(error_result, f, ensure_ascii=False, indent=2)
        
        # Save overall results summary
        overall_results_path = os.path.join(args.data_dir, "overall_results.json")
        with open(overall_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Overall results saved to: {overall_results_path}")
        
    else:
        # Original single video processing with updated output structure
        video_filename = os.path.basename(args.video)
        video_name = os.path.splitext(video_filename)[0]  # Remove .mp4 extension
        
        # Create output directory for this video
        video_output_dir = os.path.join(args.data_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        result = engine.run(args.video, args.question)
        tree = engine.export_tree(result)

        print("Final answer:", result.final_answer)
        print("Terminated:", result.terminated)
        print(f"Total nodes: {len(result.nodes)}")

        # Save tree in the video's directory
        tree_path = os.path.join(video_output_dir, "tree.json")
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, ensure_ascii=False, indent=2)
        print("Tree saved to:", tree_path)

if __name__ == "__main__":
    main()