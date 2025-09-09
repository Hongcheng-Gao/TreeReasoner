# run_videomme.py
import argparse
import json
import os
import pandas as pd
from tot_engine import ToTEngine

def run_videomme(engine, args):
    """Process Video-MME dataset from parquet file"""
    
    # Read the parquet file
    parquet_path = "/mnt/moonfs/kimiv-m2/huangzihao/dataset/videomme/videomme/test-00000-of-00001.parquet"
    dataset_path = "/mnt/moonfs/kimiv-m2/huangzihao/dataset/videomme"
    
    print(f"Loading Video-MME parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Found {len(df)} Video-MME entries")
    
    if len(df) == 0:
        print("No Video-MME entries found, skipping...")
        return []
    
    # 应用行数范围过滤（如果指定）
    if args.start_row is not None and args.end_row is not None:
        print(f"Processing rows {args.start_row} to {args.end_row-1}")
        df = df.iloc[args.start_row:args.end_row]
        print(f"Filtered to {len(df)} entries")
    
    # 直接使用指定的输出目录
    base_data_dir = args.data_dir
    os.makedirs(base_data_dir, exist_ok=True)
    print(f"Using output directory: {base_data_dir}")
    
    results = []
    successful_count = 0
    
    for i, (orig_idx, row) in enumerate(df.iterrows()):
        print(f"Processing Video-MME entry {i+1}/{len(df)} (original index: {orig_idx}, successful: {successful_count})")
        
        # Construct video path using videoID column
        video_filename = row["videoID"] + ".mp4"
        video_path = os.path.join(dataset_path, "data", video_filename)
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}, skipping...")
            continue
            
        print(f"Video path: {video_path}")
        
        # Extract video name for folder creation
        video_name = row["videoID"]
        
        # Create output directory for this video
        video_output_dir = os.path.join(base_data_dir, f"videomme_{video_name}")
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Prepare question
        question = row["question"]
        print(f"Question: {question}")
        
        # Add options if they exist
        options = row.get("options")
        if options is not None and len(options) > 0:
            if isinstance(options, list):
                options_str = " ".join(options)
            else:
                options_str = str(options)
            question = question + " " + options_str
        
        # Get the ground truth answer
        answer = row["answer"]
        
        try:
            result = engine.run(video_path, question)
            tree = engine.export_tree(result)
            
            # Create result entry
            entry_result = {
                "video_id": row["video_id"],
                "question_id": row.get("question_id"),
                "videoID": row["videoID"],
                "video_path": video_path,
                "question": question,
                "ground_truth_answer": answer,
                "domain": row.get("domain"),
                "sub_category": row.get("sub_category"),
                "task_type": row.get("task_type"),
                "duration": row.get("duration"),
                "final_answer": result.final_answer,
                "terminated": result.terminated,
                "total_nodes": len(result.nodes),
                "all_answers": result.all_answers,
                "total_answers": len(result.all_answers),
                "reasoning_tree": tree,
                "dataset": "Video-MME",
                "original_index": orig_idx
            }
            
            # Save individual result file in the video's directory
            individual_result_path = os.path.join(video_output_dir, "result.json")
            with open(individual_result_path, "w", encoding="utf-8") as f:
                json.dump(entry_result, f, ensure_ascii=False, indent=2)
            
            # Save reasoning tree separately
            tree_path = os.path.join(video_output_dir, "tree.json")
            with open(tree_path, "w", encoding="utf-8") as f:
                json.dump(tree, f, ensure_ascii=False, indent=2)
            
            # Only add to results and increment successful_count if everything above succeeds
            results.append(entry_result)
            successful_count += 1
            
            print(f"Final answer: {result.final_answer}")
            print(f"Ground truth answer: {answer}")
            print(f"Total nodes: {len(result.nodes)}")
            print(f"Total answers found: {len(result.all_answers)}")
            
            # Print all found answers
            if result.all_answers:
                print("All answers found:")
                for j, answer_info in enumerate(result.all_answers, 1):
                    print(f"  {j}. Path {answer_info['path_id']}: {answer_info['answer']} (confidence: {answer_info.get('confidence', 0.0):.2f})")
            
            print(f"Output saved to: {video_output_dir}")
            print(f"✓ Video-MME successful results: {successful_count}")
            print("-" * 50)
            
        except Exception as e:
            print(f"✗ Error processing Video-MME entry {i+1}: {e}")
            print(f"Error occurred, continuing... Video-MME successful results: {successful_count}")
            print("-" * 30)
    
    print(f"Completed Video-MME processing: {successful_count} successful results")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="./work", help="Base working directory (will be overridden for video-specific dirs)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--per_expand_limit", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--save_tree", default="./tree.json", help="Path to save the reasoning tree JSON")
    
    # Video-MME specific arguments
    parser.add_argument("--data_dir", default="/mnt/moonfs/kimiv-m2/huangzihao/dataset/Video/videomme_results", help="Base data directory for outputs")
    
    # 行数范围参数（由shell脚本计算并传入）
    parser.add_argument("--start_row", type=int, help="Start row index (inclusive)")
    parser.add_argument("--end_row", type=int, help="End row index (exclusive)")
    
    args = parser.parse_args()

    engine = ToTEngine(
        llm_model=args.model,
        api_key=args.api_key,
        max_depth=args.max_depth,
        per_expand_limit=args.per_expand_limit,
        temperature=args.temperature,
    )

    # Process Video-MME dataset
    if args.start_row is not None and args.end_row is not None:
        print(f"Starting Video-MME dataset processing for rows {args.start_row}-{args.end_row-1}...")
    else:
        print("Starting Video-MME dataset processing...")
    videomme_results = run_videomme(engine, args)
    
    print(f"Total successful Video-MME results: {len(videomme_results)}")
    
    # 确定结果文件名
    if args.start_row is not None and args.end_row is not None:
        results_filename = f"videomme_results_rows_{args.start_row}_{args.end_row-1}.json"
    else:
        results_filename = "videomme_overall_results.json"
    
    # Save results summary
    overall_results_path = os.path.join(args.data_dir, results_filename)
    with open(overall_results_path, "w", encoding="utf-8") as f:
        json.dump(videomme_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {overall_results_path}")
    
    # Print summary statistics
    successful_count = len([r for r in videomme_results if "error" not in r])
    print(f"\n=== Video-MME Processing Summary ===")
    print(f"Video-MME: {successful_count} successful results")

if __name__ == "__main__":
    main() 