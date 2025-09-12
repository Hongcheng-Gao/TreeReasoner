# run_longvideobench_concurrent.py
"""
并发处理LongVideoBench数据集的脚本

输出路径配置选项：
1. --data_dir: 默认输出目录（默认: ./longvideobench_results）
2. --output_dir: 自定义输出目录（如果指定，会覆盖data_dir）
3. --video_subdir_prefix: 视频子目录前缀（默认: longvideobench）

使用示例：
# 使用默认输出目录
python run_longvideobench_concurrent.py

# 自定义输出目录
python run_longvideobench_concurrent.py --output_dir /path/to/custom/output

# 自定义子目录前缀
python run_longvideobench_concurrent.py --video_subdir_prefix "my_video"

# 完整自定义
python run_longvideobench_concurrent.py --output_dir /custom/path --video_subdir_prefix "test"
"""
import argparse
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import traceback
from tot_engine_concurrent import ConcurrentToTEngine

class ConcurrentLongVideoBenchProcessor:
    """高并发的LongVideoBench数据集处理器"""
    
    def __init__(self, args, max_workers: int = 1):
        self.args = args
        self.max_workers = max_workers
        self.results_lock = Lock()
        self.progress_lock = Lock()
        self.results = []
        self.successful_count = 0
        self.failed_count = 0
        self.processed_count = 0
        
        # 确定有效的输出目录
        self.output_dir = self._get_effective_output_dir()
        
        # 预先创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_videos': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'output_directory': self.output_dir
        }
    
    def _get_effective_output_dir(self) -> str:
        """获取有效的输出目录路径"""
        if hasattr(self.args, 'output_dir') and self.args.output_dir:
            # 如果指定了custom output_dir，使用它
            return os.path.abspath(self.args.output_dir)
        else:
            # 否则使用data_dir
            return os.path.abspath(self.args.data_dir)
    
    def get_video_output_dir(self, video_id: str) -> str:
        """获取特定视频的输出目录路径"""
        video_subdir_prefix = getattr(self.args, 'video_subdir_prefix', 'longvideobench')
        return os.path.join(self.output_dir, f"{video_subdir_prefix}_{video_id}")
    
    def format_question_with_options(self, question: str, candidates: List[str]) -> str:
        """格式化问题，添加选项A、B、C、D等"""
        formatted_question = question + "\n"
        for i, candidate in enumerate(candidates):
            option_letter = chr(ord("A") + i)
            formatted_question += f"{option_letter}. {candidate}\n"
        return formatted_question.strip()
    
    def create_engine(self) -> ConcurrentToTEngine:
        """为每个线程创建独立的ConcurrentToTEngine实例"""
        return ConcurrentToTEngine(
            llm_model=self.args.model,
            api_key=self.args.api_key,
            max_depth=self.args.max_depth,
            per_expand_limit=self.args.per_expand_limit,
            temperature=self.args.temperature,
            thread_safe=True,
        )
    
    def process_single_video(self, video_data: Tuple[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """处理单个视频的函数，用于并行执行"""
        orig_idx, data_item = video_data
        dataset_base_path = "/mnt/moonfs/kimiv-m2/haoning/datasets/video_evals/LongVideoBench"

        # import pdb; pdb.set_trace()
        
        try:
            # 为每个线程创建独立的engine
            engine = self.create_engine()
            
            # 构建视频路径
            video_filename = data_item["video_path"]
            video_path = os.path.join(dataset_base_path, "videos", video_filename)
            # video_path = os.path.join(dataset_base_path, video_filename)
            
            # 检查视频文件是否存在
            if not os.path.exists(video_path):
                with self.progress_lock:
                    print(f"Video file not found: {video_path}, skipping...")
                return None
            
            # 创建输出目录
            video_id = data_item["video_id"]
            video_output_dir = self.get_video_output_dir(video_id)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # 准备问题 - 格式化为带选项的问题
            question = data_item["question"]
            
            candidates = data_item["candidates"]
            formatted_question = self.format_question_with_options(question, candidates)
            # import pdb; pdb.set_trace()

            # 获取真实答案
            correct_answer = chr(ord("A") + data_item["correct_choice"])
            # correct_answer = candidates[correct_choice] if 0 <= correct_choice < len(candidates) else "Unknown"
            
            # 处理视频
            result = engine.run(video_path, formatted_question)
            tree = engine.export_tree(result)
            
            # 创建结果条目
            entry_result = {
                "video_id": data_item["video_id"],
                "question_id": data_item.get("id"),
                "video_path": video_path,
                "question": question,
                "formatted_question": formatted_question,
                "candidates": candidates,
                "correct_choice": correct_answer,
                "correct_answer": correct_answer,
                "question_wo_referring_query": data_item.get("question_wo_referring_query"),
                "position": data_item.get("position"),
                "topic_category": data_item.get("topic_category"),
                "question_category": data_item.get("question_category"),
                "level": data_item.get("level"),
                "subtitle_path": data_item.get("subtitle_path"),
                "duration_group": data_item.get("duration_group"),
                "starting_timestamp_for_subtitles": data_item.get("starting_timestamp_for_subtitles"),
                "duration": data_item.get("duration"),
                "view_count": data_item.get("view_count"),
                "final_answer": result.final_answer,
                "terminated": result.terminated,
                "total_nodes": len(result.nodes),
                "all_answers": result.all_answers,
                "total_answers": len(result.all_answers),
                "reasoning_tree": tree,
                "dataset": "LongVideoBench",
                "original_index": orig_idx,
                "processing_thread": threading.current_thread().name
            }
            
            # 保存个别结果文件
            individual_result_path = os.path.join(video_output_dir, "result.json")
            with open(individual_result_path, "w", encoding="utf-8") as f:
                json.dump(entry_result, f, ensure_ascii=False, indent=2)
            
            # 保存推理树
            tree_path = os.path.join(video_output_dir, "tree.json")
            with open(tree_path, "w", encoding="utf-8") as f:
                json.dump(tree, f, ensure_ascii=False, indent=2)
            
            # 更新进度信息
            with self.progress_lock:
                self.successful_count += 1
                thread_name = threading.current_thread().name
                print(f"[{thread_name}] ✓ Processed {video_id} | Success: {self.successful_count}")
                print(f"[{thread_name}] Final answer: {result.final_answer}")
                print(f"[{thread_name}] Correct answer: {correct_answer}")
                print(f"[{thread_name}] Nodes: {len(result.nodes)}, Answers: {len(result.all_answers)}")
            
            return entry_result
            
        except Exception as e:
            error_info = {
                'video_id': data_item.get("video_id", "unknown"),
                'original_index': orig_idx,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'thread': threading.current_thread().name
            }
            
            with self.progress_lock:
                self.failed_count += 1
                self.stats['errors'].append(error_info)
                thread_name = threading.current_thread().name
                print(f"[{thread_name}] ✗ Error processing {data_item.get('video_id', 'unknown')}: {e}")
                print(f"[{thread_name}] Failed: {self.failed_count}")
            
            return None
    
    def process_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理视频数据"""
        self.stats['start_time'] = time.time()
        self.stats['total_videos'] = len(data_list)
        
        print(f"Starting concurrent processing with {self.max_workers} workers...")
        print(f"Total videos to process: {len(data_list)}")
        
        # 准备视频数据
        video_data_list = [(orig_idx, data_item) for orig_idx, data_item in enumerate(data_list)]
        
        # 使用ThreadPoolExecutor进行并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="LVBWorker") as executor:
            # 提交所有任务
            future_to_video = {
                executor.submit(self.process_single_video, video_data): video_data[0] 
                for video_data in video_data_list
            }
            
            # 使用tqdm显示进度条
            with tqdm(total=len(video_data_list), desc="Processing videos", unit="video") as pbar:
                for future in as_completed(future_to_video):
                    orig_idx = future_to_video[future]
                    try:
                        result = future.result()
                        if result is not None:
                            with self.results_lock:
                                self.results.append(result)
                        
                        with self.progress_lock:
                            self.processed_count += 1
                            pbar.set_postfix({
                                'Success': self.successful_count,
                                'Failed': self.failed_count,
                                'Rate': f"{(self.successful_count/self.processed_count)*100:.1f}%" if self.processed_count > 0 else "0%"
                            })
                        
                    except Exception as e:
                        with self.progress_lock:
                            self.failed_count += 1
                            print(f"Future execution error for video index {orig_idx}: {e}")
                    
                    pbar.update(1)
        
        self.stats['end_time'] = time.time()
        self.stats['successful'] = self.successful_count
        self.stats['failed'] = self.failed_count
        
        return self.results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """保存最终结果和统计信息"""
        # 确定结果文件名
        if self.args.start_idx is not None and self.args.end_idx is not None:
            results_filename = f"longvideobench_concurrent_results_idx_{self.args.start_idx}_{self.args.end_idx-1}.json"
            stats_filename = f"longvideobench_concurrent_stats_idx_{self.args.start_idx}_{self.args.end_idx-1}.json"
        else:
            results_filename = "longvideobench_concurrent_overall_results.json"
            stats_filename = "longvideobench_concurrent_overall_stats.json"
        
        # 保存结果
        overall_results_path = os.path.join(self.output_dir, results_filename)
        with open(overall_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 计算处理时间和速度
        total_time = self.stats['end_time'] - self.stats['start_time']
        videos_per_second = self.stats['successful'] / total_time if total_time > 0 else 0
        
        # 更新统计信息
        self.stats.update({
            'total_time_seconds': total_time,
            'videos_per_second': videos_per_second,
            'success_rate': (self.stats['successful'] / self.stats['total_videos']) * 100 if self.stats['total_videos'] > 0 else 0,
            'max_workers': self.max_workers
        })
        
        # 保存统计信息
        stats_path = os.path.join(self.output_dir, stats_filename)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== Concurrent Processing Summary ===")
        print(f"Total videos: {self.stats['total_videos']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {self.stats['success_rate']:.1f}%")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Processing speed: {videos_per_second:.2f} videos/second")
        print(f"Max workers: {self.max_workers}")
        print(f"Results saved to: {overall_results_path}")
        print(f"Statistics saved to: {stats_path}")
        
        if self.stats['errors']:
            print(f"Errors encountered: {len(self.stats['errors'])}")
            print("Check stats file for detailed error information")


def run_longvideobench_concurrent(args):
    """并发处理LongVideoBench数据集的主函数"""
    
    # 读取JSON文件
    json_path = "/mnt/moonfs/kimiv-m2/haoning/datasets/video_evals/LongVideoBench/lvb_val.json"
    
    print(f"Loading LongVideoBench JSON file: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print(f"Found {len(data_list)} LongVideoBench entries")
    
    if len(data_list) == 0:
        print("No LongVideoBench entries found, skipping...")
        return []
    
    # 应用索引范围过滤（如果指定）
    if args.start_idx is not None and args.end_idx is not None:
        print(f"Processing entries {args.start_idx} to {args.end_idx-1}")
        data_list = data_list[args.start_idx:args.end_idx]
        print(f"Filtered to {len(data_list)} entries")
    
    # 确定最佳worker数量
    max_workers = getattr(args, 'max_workers', 1)
    if max_workers > len(data_list):
        max_workers = len(data_list)
    
    # 创建并发处理器
    processor = ConcurrentLongVideoBenchProcessor(args, max_workers=max_workers)
    
    # 处理视频
    results = processor.process_batch(data_list)
    
    # 保存结果
    processor.save_results(results)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="./work", help="Base working directory")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--per_expand_limit", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--save_tree", default="./tree.json", help="Path to save the reasoning tree JSON")
    
    # LongVideoBench specific arguments
    parser.add_argument("--data_dir", default="./longvideobench_results", help="Base output directory for all results")
    parser.add_argument("--output_dir", help="Custom output directory (overrides data_dir if provided)")
    parser.add_argument("--video_subdir_prefix", default="longvideobench", help="Prefix for individual video subdirectories")
    
    # 索引范围参数
    parser.add_argument("--start_idx", type=int, help="Start index (inclusive)")
    parser.add_argument("--end_idx", type=int, help="End index (exclusive)")
    
    # 并发参数
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of concurrent workers")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for result saving")
    
    args = parser.parse_args()
    
    # 确定有效的输出目录
    effective_output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else args.data_dir
    effective_output_dir = os.path.abspath(effective_output_dir)
    
    print(f"Starting concurrent LongVideoBench processing with {args.max_workers} workers...")
    print(f"Output directory: {effective_output_dir}")
    if args.start_idx is not None and args.end_idx is not None:
        print(f"Processing entries {args.start_idx}-{args.end_idx-1}")
    
    # 处理LongVideoBench数据集
    longvideobench_results = run_longvideobench_concurrent(args)
    
    print(f"\n=== Final Results ===")
    print(f"Total successful LongVideoBench results: {len(longvideobench_results)}")


if __name__ == "__main__":
    main() 