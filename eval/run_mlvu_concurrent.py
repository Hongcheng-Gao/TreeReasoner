# run_mlvu_concurrent.py
import argparse
import json
import os
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tot_engine import ToTEngine

@dataclass
class ProcessingResult:
    """Result of processing a single MLVU entry"""
    success: bool
    original_index: int
    video_name: str
    entry_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0

class ConcurrentMLVUProcessor:
    """High-concurrency MLVU dataset processor with thread safety and rate limiting"""
    
    def __init__(self, 
                 max_workers: int = 256,
                 rate_limit_per_second: float = 2.0,
                 max_retries: int = 2,
                 retry_delay: float = 1.0,
                 timeout_per_task: float = 300.0):
        """
        Initialize the concurrent processor.
        
        Args:
            max_workers: Maximum number of concurrent threads
            rate_limit_per_second: Maximum API calls per second
            max_retries: Number of retry attempts for failed tasks
            retry_delay: Delay between retries in seconds
            timeout_per_task: Timeout for each task in seconds
        """
        self.max_workers = max_workers
        self.rate_limit = Semaphore(int(rate_limit_per_second))
        self.rate_limit_per_second = rate_limit_per_second
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout_per_task = timeout_per_task
        
        # Thread-safe counters and logging
        self.success_count = 0
        self.failure_count = 0
        self.counter_lock = Lock()
        self.last_rate_limit_reset = time.time()
        
        # Progress tracking
        self.completed_tasks = 0
        self.total_tasks = 0
        self.progress_lock = Lock()
        
    def _reset_rate_limit(self):
        """Reset rate limiting semaphore every second"""
        current_time = time.time()
        if current_time - self.last_rate_limit_reset >= 1.0:
            # Release all permits and re-acquire the correct number
            available = self.rate_limit._value
            needed = int(self.rate_limit_per_second) - available
            for _ in range(max(0, needed)):
                try:
                    self.rate_limit.release()
                except ValueError:
                    break
            self.last_rate_limit_reset = current_time
    
    def _create_engine(self, args) -> ToTEngine:
        """Create a new ToTEngine instance for thread-safe processing"""
        return ToTEngine(
            llm_model=args.model,
            api_key=args.api_key,
            max_depth=args.max_depth,
            per_expand_limit=args.per_expand_limit,
            temperature=args.temperature,
        )
    
    def _update_progress(self, completed: int = 1):
        """Thread-safe progress update"""
        with self.progress_lock:
            self.completed_tasks += completed
            if self.total_tasks > 0:
                progress = (self.completed_tasks / self.total_tasks) * 100
                print(f"Progress: {self.completed_tasks}/{self.total_tasks} ({progress:.1f}%)")
    
    def _process_single_entry(self, 
                            entry_data: Tuple[int, int, pd.Series], 
                            args,
                            video_base_path: str,
                            base_data_dir: str) -> ProcessingResult:
        """
        Process a single MLVU entry with rate limiting and error handling.
        
        Args:
            entry_data: Tuple of (entry_index, original_index, row_data)
            args: Command line arguments
            video_base_path: Base path for video files
            base_data_dir: Base directory for output
            
        Returns:
            ProcessingResult object
        """
        entry_idx, orig_idx, row = entry_data
        start_time = time.time()
        
        try:
            # Rate limiting
            self._reset_rate_limit()
            self.rate_limit.acquire()
            
            # Create thread-local engine instance
            engine = self._create_engine(args)
            
            # Construct video path
            prefix = row["prefix"].replace("./MLVU", "")
            video_path = os.path.join(video_base_path, prefix.lstrip("/"), row["video"])
            
            # Check if video file exists
            if not os.path.exists(video_path):
                return ProcessingResult(
                    success=False,
                    original_index=orig_idx,
                    video_name=row["video"],
                    error_message=f"Video file not found: {video_path}"
                )
            
            # Extract video name for folder creation
            video_name = os.path.splitext(os.path.basename(row["video"]))[0]
            
            # Create output directory for this video
            video_output_dir = os.path.join(base_data_dir, f"mlvu_{video_name}")
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Prepare question and options
            question = row["question"]
            options = [eval(row["candidates"])[0], eval(row["candidates"])[1], 
                      eval(row["candidates"])[2], eval(row["candidates"])[3]]
            options_str = " Options: " + " | ".join(options)
            full_question = question + options_str
            
            # Get the ground truth answer
            answer = row["answer"]
            
            # Process with ToT engine
            result = engine.run(video_path, full_question)
            tree = engine.export_tree(result)
            
            # Create result entry
            entry_result = {
                "video_path": video_path,
                "question": full_question,
                "options": options,
                "ground_truth_answer": answer,
                "final_answer": result.final_answer,
                "terminated": result.terminated,
                "total_nodes": len(result.nodes),
                "all_answers": result.all_answers,
                "total_answers": len(result.all_answers),
                "reasoning_tree": tree,
                "dataset": "MLVU_MCQ",
                "original_index": orig_idx,
                "processing_time": time.time() - start_time,
                "thread_id": threading.get_ident()
            }
            
            # Save individual result files
            individual_result_path = os.path.join(video_output_dir, "result.json")
            with open(individual_result_path, "w", encoding="utf-8") as f:
                json.dump(entry_result, f, ensure_ascii=False, indent=2)
            
            # Save reasoning tree separately
            tree_path = os.path.join(video_output_dir, "tree.json")
            with open(tree_path, "w", encoding="utf-8") as f:
                json.dump(tree, f, ensure_ascii=False, indent=2)
            
            # Update success counter
            with self.counter_lock:
                self.success_count += 1
            
            processing_time = time.time() - start_time
            
            print(f"✓ Entry {entry_idx+1} (orig: {orig_idx}) completed in {processing_time:.1f}s")
            print(f"  Video: {video_name}")
            print(f"  Final answer: {result.final_answer}")
            print(f"  Ground truth: {answer}")
            print(f"  Total nodes: {len(result.nodes)}")
            print(f"  Success count: {self.success_count}")
            
            return ProcessingResult(
                success=True,
                original_index=orig_idx,
                video_name=video_name,
                entry_result=entry_result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing entry {entry_idx+1} (orig: {orig_idx}): {str(e)}"
            
            with self.counter_lock:
                self.failure_count += 1
            
            print(f"✗ {error_msg}")
            print(f"  Processing time: {processing_time:.1f}s")
            print(f"  Failure count: {self.failure_count}")
            
            return ProcessingResult(
                success=False,
                original_index=orig_idx,
                video_name=row["video"] if "video" in row else "unknown",
                error_message=error_msg,
                processing_time=processing_time
            )
        finally:
            # Always release the rate limit semaphore
            try:
                self.rate_limit.release()
            except ValueError:
                pass
            
            # Update progress
            self._update_progress()

    def process_with_retries(self, 
                           entry_data: Tuple[int, int, pd.Series], 
                           args,
                           video_base_path: str,
                           base_data_dir: str) -> ProcessingResult:
        """Process entry with retry logic"""
        last_result = None
        
        for attempt in range(self.max_retries + 1):
            result = self._process_single_entry(entry_data, args, video_base_path, base_data_dir)
            
            if result.success:
                return result
            
            last_result = result
            
            if attempt < self.max_retries:
                print(f"Retrying entry {entry_data[0]+1} (attempt {attempt+2}/{self.max_retries+1})")
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        return last_result

def run_mlvu_concurrent(processor: ConcurrentMLVUProcessor, args):
    """Process MLVU_MCQ dataset with high concurrency"""
    
    # Read the TSV file
    tsv_path = "/mnt/moonfs/kimiv-public-m2/LMUData/MLVU_MCQ.tsv"
    video_base_path = "/mnt/moonfs/kimiv-m2/haoning/datasets/video_evals/MLVU/MLVU"
    
    print(f"Loading MLVU_MCQ TSV file: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Found {len(df)} MLVU_MCQ entries")
    
    if len(df) == 0:
        print("No MLVU_MCQ entries found, skipping...")
        return []
    
    # Apply row range filtering (if specified)
    if args.start_row is not None and args.end_row is not None:
        print(f"Processing rows {args.start_row} to {args.end_row-1}")
        df = df.iloc[args.start_row:args.end_row]
        print(f"Filtered to {len(df)} entries")
    
    # Use specified output directory (output_dir takes priority over data_dir)
    base_data_dir = args.output_dir if args.output_dir is not None else args.data_dir
    os.makedirs(base_data_dir, exist_ok=True)
    print(f"Using output directory: {base_data_dir}")
    
    # Setup for concurrent processing
    processor.total_tasks = len(df)
    processor.completed_tasks = 0
    
    # Prepare entry data for processing
    entry_data_list = [(i, orig_idx, row) for i, (orig_idx, row) in enumerate(df.iterrows())]
    
    print(f"Starting concurrent processing with {processor.max_workers} workers")
    print(f"Rate limit: {processor.rate_limit_per_second} requests/second")
    print(f"Max retries: {processor.max_retries}")
    print(f"Timeout per task: {processor.timeout_per_task}s")
    print("-" * 60)
    
    successful_results = []
    failed_results = []
    
    start_time = time.time()
    
    # Execute concurrent processing
    with ThreadPoolExecutor(max_workers=processor.max_workers) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(
                processor.process_with_retries,
                entry_data,
                args,
                video_base_path,
                base_data_dir
            ): entry_data for entry_data in entry_data_list
        }
        
        # Process completed tasks
        for future in as_completed(future_to_entry, timeout=processor.timeout_per_task * len(entry_data_list)):
            entry_data = future_to_entry[future]
            
            try:
                result = future.result(timeout=processor.timeout_per_task)
                
                if result.success:
                    successful_results.append(result.entry_result)
                else:
                    failed_results.append({
                        "original_index": result.original_index,
                        "video_name": result.video_name,
                        "error": result.error_message,
                        "processing_time": result.processing_time
                    })
                    
            except Exception as e:
                entry_idx, orig_idx, row = entry_data
                error_result = {
                    "original_index": orig_idx,
                    "video_name": row["video"] if "video" in row else "unknown",
                    "error": f"Future execution error: {str(e)}",
                    "processing_time": 0.0
                }
                failed_results.append(error_result)
                print(f"✗ Future execution error for entry {entry_idx+1}: {e}")
    
    total_time = time.time() - start_time
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("CONCURRENT MLVU_MCQ PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total entries processed: {len(entry_data_list)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Success rate: {len(successful_results)/len(entry_data_list)*100:.1f}%")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Average time per entry: {total_time/len(entry_data_list):.1f}s")
    print(f"Concurrent workers: {processor.max_workers}")
    print(f"Rate limit: {processor.rate_limit_per_second} req/s")
    
    if failed_results:
        print(f"\nFailed entries:")
        for failure in failed_results:
            print(f"  - Index {failure['original_index']}: {failure['error']}")
    
    return successful_results, failed_results

def main():
    parser = argparse.ArgumentParser(description="Concurrent MLVU_MCQ processor")
    
    # Original arguments
    parser.add_argument("--workdir", default="./work", help="Base working directory")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--per_expand_limit", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--save_tree", default="./tree.json", help="Path to save the reasoning tree JSON")
    
    # MLVU specific arguments
    parser.add_argument("--data_dir", default="/mnt/moonfs/kimiv-m2/huangzihao/dataset/MLVU", 
                       help="Base data directory for outputs")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Custom output directory (if not specified, uses data_dir)")
    parser.add_argument("--start_row", type=int, help="Start row index (inclusive)")
    parser.add_argument("--end_row", type=int, help="End row index (exclusive)")
    
    # Concurrency arguments
    parser.add_argument("--max_workers", type=int, default=256, 
                       help="Maximum number of concurrent workers")
    parser.add_argument("--rate_limit", type=float, default=2.0, 
                       help="Maximum API calls per second")
    parser.add_argument("--max_retries", type=int, default=2, 
                       help="Maximum number of retries for failed tasks")
    parser.add_argument("--retry_delay", type=float, default=1.0, 
                       help="Base delay between retries in seconds")
    parser.add_argument("--timeout", type=float, default=300.0, 
                       help="Timeout per task in seconds")
    
    args = parser.parse_args()
    
    # Create concurrent processor
    processor = ConcurrentMLVUProcessor(
        max_workers=args.max_workers,
        rate_limit_per_second=args.rate_limit,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout_per_task=args.timeout
    )
    
    # Process MLVU_MCQ dataset
    if args.start_row is not None and args.end_row is not None:
        print(f"Starting concurrent MLVU_MCQ dataset processing for rows {args.start_row}-{args.end_row-1}...")
    else:
        print("Starting concurrent MLVU_MCQ dataset processing...")
    
    successful_results, failed_results = run_mlvu_concurrent(processor, args)
    
    # Determine results filename
    if args.start_row is not None and args.end_row is not None:
        results_filename = f"mlvu_concurrent_results_rows_{args.start_row}_{args.end_row-1}.json"
        failures_filename = f"mlvu_concurrent_failures_rows_{args.start_row}_{args.end_row-1}.json"
    else:
        results_filename = "mlvu_concurrent_overall_results.json"
        failures_filename = "mlvu_concurrent_failures.json"
    
    # Use the same output directory for saving results
    output_dir = args.output_dir if args.output_dir is not None else args.data_dir
    
    # Save successful results
    if successful_results:
        overall_results_path = os.path.join(output_dir, results_filename)
        with open(overall_results_path, "w", encoding="utf-8") as f:
            json.dump(successful_results, f, ensure_ascii=False, indent=2)
        print(f"Successful results saved to: {overall_results_path}")
    
    # Save failed results for analysis
    if failed_results:
        failures_path = os.path.join(output_dir, failures_filename)
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(failed_results, f, ensure_ascii=False, indent=2)
        print(f"Failed results saved to: {failures_path}")

if __name__ == "__main__":
    main() 