# run.py
import argparse
import json
import os
from tot_engine import ToTEngine

def process_dataset_entries(data, dataset_filter_func, dataset_name, target_count, engine, args):
    """Process entries for a specific dataset until target_count successful results are achieved"""
    
    # Get all entries for this dataset
    all_entries = [entry for entry in data if dataset_filter_func(entry)]
    print(f"Found {len(all_entries)} {dataset_name} entries")
    
    if len(all_entries) == 0:
        print(f"No {dataset_name} entries found, skipping...")
        return []
    
    results = []
    successful_count = 0
    
    for i, entry in enumerate(all_entries):
        if successful_count >= target_count:
            print(f"Reached target of {target_count} successful {dataset_name} results, stopping...")
            break
            
        print(f"Processing {dataset_name} entry {i+1}/{len(all_entries)} (successful: {successful_count}/{target_count}): {entry['path']}")
        
        video_path = entry["path"]
        video_path = video_path.replace("./", "/mnt/moonfs/kimiv-m2/huangzihao/dataset/Video-R1-data/")
        
        # Extract video filename without extension for folder name
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]  # Remove .mp4 extension
        
        # Create output directory for this video
        video_output_dir = os.path.join(args.data_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        question = entry["problem"]
        print(question)
        option = entry["options"]
        if option is not None:
            options = " ".join(entry["options"])
        else:
            options = ""
        question = question + options

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
                "all_answers": result.all_answers,  # 新增：包含所有找到的答案
                "total_answers": len(result.all_answers),  # 新增：答案总数
                "reasoning_tree": tree,
                "dataset": dataset_name  # Add dataset identifier
            }
            results.append(entry_result)
            successful_count += 1  # Only increment on success
            
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
            print(f"Total answers found: {len(result.all_answers)}")
            
            # 打印所有找到的答案
            if result.all_answers:
                print("All answers found:")
                for j, answer_info in enumerate(result.all_answers, 1):
                    print(f"  {j}. Path {answer_info['path_id']}: {answer_info['answer']} (confidence: {answer_info.get('confidence', 0.0):.2f})")
            
            print(f"Output saved to: {video_output_dir}")
            print(f"✓ {dataset_name} successful results: {successful_count}/{target_count}")
            print("-" * 50)
            
        except Exception as e:
            print(f"✗ Error processing {dataset_name} entry {i+1}: {e}")
            # Note: We don't add error results to the main results list,
            # don't increment successful_count, and don't save error JSON files
            
            print(f"Error occurred, continuing... {dataset_name} successful results: {successful_count}/{target_count}")
            print("-" * 30)
    print(f"Completed {dataset_name} processing: {successful_count} successful results")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=False, default="/mnt/moonfs/kimiv-m2/huangzihao/dataset/Video-R1-data/Video-R1-COT-165k.json", help="Input JSON file containing video data")
    parser.add_argument("--video", required=False, default="/mnt/moonfs/kimiv-public-m2/zihao/test/1.mov", help="Input video path (used when not processing JSON)")
    parser.add_argument("--question", required=False, default="带你到天空去这个歌词出现在第几秒")
    parser.add_argument("--workdir", default="./work", help="Base working directory (will be overridden for video-specific dirs)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--per_expand_limit", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--save_tree", default="./tree.json", help="Path to save the reasoning tree JSON")
    
    # Dataset processing options
    parser.add_argument("--process_clevrer", action="store_true", help="Process CLEVRER entries from input JSON")
    parser.add_argument("--process_other_datasets", action="store_true", help="Process other datasets (LLaVA-Video-178K, NeXT-QA, PerceptionTest, STAR) from input JSON")
    parser.add_argument("--num_llava", type=int, default=None, help="Number of LLaVA-Video-178K entries to process (if not specified, process all)")
    parser.add_argument("--num_nextqa", type=int, default=1000, help="Number of NeXT-QA successful results to process (default: 1000)")
    parser.add_argument("--num_perceptiontest", type=int, default=1000, help="Number of PerceptionTest successful results to process (default: 1000)")
    parser.add_argument("--num_star", type=int, default=1000, help="Number of STAR successful results to process (default: 1000)")
    
    parser.add_argument("--output_results", default="./clevrer_results.json", help="Path to save processing results")
    parser.add_argument("--data_dir", default="/mnt/moonfs/kimiv-m2/huangzihao/dataset/Video/0904_v3", help="Base data directory for outputs")
    args = parser.parse_args()

    engine = ToTEngine(
        llm_model=args.model,
        api_key=args.api_key,
        max_depth=args.max_depth,
        per_expand_limit=args.per_expand_limit,
        temperature=args.temperature,
    )

    # Check if any dataset processing is requested
    if args.process_clevrer or args.process_other_datasets:
        # Process entries from JSON file for specified datasets
        print(f"Loading JSON file: {args.input_json}")
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        all_results = []
        
        if args.process_clevrer:
            # Original CLEVRER processing logic (unchanged)
            clevrer_entries = [entry for entry in data if "CLEVRER" in entry.get("path", "")]
            print(f"Found {len(clevrer_entries)} CLEVRER entries")
            
            # Process CLEVRER entries (keeping original logic)
            for i, entry in enumerate(clevrer_entries):
                print(f"Processing CLEVRER entry {i+1}/{len(clevrer_entries)}: {entry['path']}")
                
                video_path = entry["path"]
                video_path = video_path.replace("./", "/mnt/moonfs/kimiv-m2/huangzihao/dataset/Video-R1-data/")
                
                # Extract video filename without extension for folder name
                video_filename = os.path.basename(video_path)
                video_name = os.path.splitext(video_filename)[0]  # Remove .mp4 extension
                
                # Create output directory for this video
                video_output_dir = os.path.join(args.data_dir, video_name)
                os.makedirs(video_output_dir, exist_ok=True)
                
                question = entry["problem"]
                print(question)
                option = entry["options"]
                if option is not None:
                    options = " ".join(entry["options"])
                else:
                    options = ""
                question = question + options

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
                        "all_answers": result.all_answers,
                        "total_answers": len(result.all_answers),
                        "reasoning_tree": tree,
                        "dataset": "CLEVRER"
                    }
                    all_results.append(entry_result)
                    
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
                    print(f"Total answers found: {len(result.all_answers)}")
                    
                    # 打印所有找到的答案
                    if result.all_answers:
                        print("All answers found:")
                        for j, answer_info in enumerate(result.all_answers, 1):
                            print(f"  {j}. Path {answer_info['path_id']}: {answer_info['answer']} (confidence: {answer_info.get('confidence', 0.0):.2f})")
                    
                    print(f"Output saved to: {video_output_dir}")
                    print("-" * 50)
                    
                except Exception as e:
                    print(f"Error processing CLEVRER entry {i+1}: {e}")
                    # Note: We don't save error JSON files anymore
                    error_result = {
                        "problem_id": entry.get("problem_id"),
                        "video_path": video_path,
                        "question": question,
                        "solution": solution,
                        "error": str(e),
                        "dataset": "CLEVRER"
                    }
                    all_results.append(error_result)
        
        if args.process_other_datasets:
            # Process each dataset separately with success-based counting
            
            # LLaVA-Video-178K (keeping original logic)
            if args.num_llava is not None:
                llava_filter = lambda entry: "LLaVA-Video-178K" in entry.get("path", "") or "llava" in entry.get("path", "").lower()
                llava_results = process_dataset_entries(data, llava_filter, "LLaVA-Video-178K", args.num_llava, engine, args)
                all_results.extend(llava_results)
            
            # NeXT-QA (modified to count successful results)
            nextqa_filter = lambda entry: "NeXT-QA" in entry.get("path", "") or "nextqa" in entry.get("path", "").lower()
            nextqa_results = process_dataset_entries(data, nextqa_filter, "NeXT-QA", args.num_nextqa, engine, args)
            all_results.extend(nextqa_results)
            
            # PerceptionTest (modified to count successful results)
            perception_filter = lambda entry: "PerceptionTest" in entry.get("path", "") or "perception" in entry.get("path", "").lower()
            perception_results = process_dataset_entries(data, perception_filter, "PerceptionTest", args.num_perceptiontest, engine, args)
            all_results.extend(perception_results)
            
            # STAR (modified to count successful results)
            star_filter = lambda entry: "STAR" in entry.get("path", "") or "star" in entry.get("path", "").lower()
            star_results = process_dataset_entries(data, star_filter, "STAR", args.num_star, engine, args)
            all_results.extend(star_results)
        
        print(f"Total successful results across all datasets: {len(all_results)}")
        
        # Save overall results summary
        overall_results_path = os.path.join(args.data_dir, "overall_results.json")
        with open(overall_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Overall results saved to: {overall_results_path}")
        
        # Print summary statistics
        dataset_counts = {}
        for result in all_results:
            dataset = result.get("dataset", "Unknown")
            if "error" not in result:  # Only count successful results
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        print("\n=== Final Summary ===")
        for dataset, count in dataset_counts.items():
            print(f"{dataset}: {count} successful results")
        
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
        print(f"Total answers found: {len(result.all_answers)}")
        
        # 打印所有找到的答案
        if result.all_answers:
            print("All answers found:")
            for i, answer_info in enumerate(result.all_answers, 1):
                print(f"  {i}. Path {answer_info['path_id']}: {answer_info['answer']} (confidence: {answer_info.get('confidence', 0.0):.2f})")

        # Save tree in the video's directory
        tree_path = os.path.join(video_output_dir, "tree.json")
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, ensure_ascii=False, indent=2)
        print("Tree saved to:", tree_path)

if __name__ == "__main__":
    main()