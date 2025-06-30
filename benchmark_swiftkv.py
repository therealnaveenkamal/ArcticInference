#!/usr/bin/env python3
"""
SwiftKV Backend Comparison Benchmark Tool

This tool benchmarks SwiftKV performance with different attention backends
(FlashAttention vs FlashInfer) to compare their performance characteristics.
"""

import argparse
import asyncio
import json
import time
from typing import Dict, List, Optional
import subprocess
import os
import sys
from pathlib import Path

class SwiftKVBenchmark:
    def __init__(self, 
                 model: str = "Snowflake/Llama-3.1-SwiftKV-8B-Instruct-FP8",
                 batch_sizes: List[int] = None,
                 requests: int = 100,
                 concurrency: int = 8,
                 prompt_length: int = 2000,
                 output_length: int = 100,
                 result_dir: str = "results"):
        
        self.model = model
        self.batch_sizes = batch_sizes or [1, 4, 8, 16]
        self.requests = requests
        self.concurrency = concurrency
        self.prompt_length = prompt_length
        self.output_length = output_length
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        
        # Backend configurations to test
        self.backends = {
            "FlashAttention": {
                "attention_backend": None,  # Default
                "env_vars": {}
            },
            "FlashInfer": {
                "attention_backend": "FLASHINFER",
                "env_vars": {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}
            }
        }
        
        self.results = {}
    
    def run_vllm_benchmark(self, backend_name: str, backend_config: Dict) -> Optional[Dict]:
        """Run vLLM benchmark for a specific backend."""
        
        print(f"\n{'='*60}")
        print(f"Testing {backend_name}")
        print(f"{'='*60}")
        
        # Set environment variables
        env = os.environ.copy()
        env.update(backend_config["env_vars"])
        
        # Create benchmark command
        cmd = [
            "python", "-m", "vllm.benchmarks.serve",
            "--model", self.model,
            "--dataset-name", "random",
            "--random-input-len", str(self.prompt_length),
            "--random-output-len", str(self.output_length),
            "--num-prompts", str(self.requests),
            "--max-concurrency", str(self.concurrency),
            "--save-result",
            "--result-filename", f"{backend_name.lower()}_result.json"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        if backend_config["env_vars"]:
            print(f"Environment: {backend_config['env_vars']}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, 
                env=env, 
                capture_output=True, 
                text=True, 
                check=True,
                cwd=self.result_dir
            )
            duration = time.time() - start_time
            
            # Parse results
            result_file = self.result_dir / f"{backend_name.lower()}_result.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                metrics = {
                    'throughput': data.get('total_token_throughput', 0),
                    'ttft_ms': data.get('mean_ttft_ms', 0),
                    'tpot_ms': data.get('mean_tpot_ms', 0),
                    'e2e_ms': data.get('mean_e2el_ms', 0),
                    'duration': duration,
                    'success_rate': data.get('request_success', 100)
                }
                
                print(f"✅ Completed in {duration:.1f}s")
                print(f"   Throughput: {metrics['throughput']:.2f} tokens/s")
                print(f"   TTFT: {metrics['ttft_ms']:.2f} ms")
                print(f"   TPOT: {metrics['tpot_ms']:.2f} ms")
                print(f"   E2E Latency: {metrics['e2e_ms']:.2f} ms")
                
                return metrics
            else:
                print("❌ No result file found")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Benchmark failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return None
    
    def run_batch_size_benchmarks(self, backend_name: str, backend_config: Dict) -> Dict:
        """Run benchmarks with different batch sizes for a backend."""
        
        print(f"\n{'='*60}")
        print(f"Batch Size Benchmarks for {backend_name}")
        print(f"{'='*60}")
        
        batch_results = {}
        
        for batch_size in self.batch_sizes:
            print(f"\n--- Testing batch size: {batch_size} ---")
            
            # Adjust concurrency for batch size
            adjusted_concurrency = min(self.concurrency, batch_size)
            
            # Create temporary config for this batch size
            temp_benchmark = SwiftKVBenchmark(
                model=self.model,
                batch_sizes=[batch_size],
                requests=self.requests,
                concurrency=adjusted_concurrency,
                prompt_length=self.prompt_length,
                output_length=self.output_length,
                result_dir=self.result_dir
            )
            
            result = temp_benchmark.run_vllm_benchmark(backend_name, backend_config)
            if result:
                batch_results[batch_size] = result
        
        return batch_results
    
    def run_comparison(self) -> Dict:
        """Run benchmarks for all backends and compare results."""
        
        print(f"{'='*80}")
        print("SWIFTKV BACKEND COMPARISON BENCHMARK")
        print(f"{'='*80}")
        print(f"Model: {self.model}")
        print(f"Batch sizes: {self.batch_sizes}")
        print(f"Requests per batch: {self.requests}")
        print(f"Max concurrency: {self.concurrency}")
        print(f"Prompt length: {self.prompt_length}")
        print(f"Output length: {self.output_length}")
        print(f"Results directory: {self.result_dir}")
        print(f"{'='*80}")
        
        all_results = {}
        
        # Run benchmarks for each backend
        for backend_name, backend_config in self.backends.items():
            print(f"\n🔄 Testing {backend_name}...")
            
            # Run single benchmark
            result = self.run_vllm_benchmark(backend_name, backend_config)
            if result:
                all_results[backend_name] = result
            
            # Run batch size benchmarks
            batch_results = self.run_batch_size_benchmarks(backend_name, backend_config)
            if batch_results:
                all_results[f"{backend_name}_batch"] = batch_results
        
        # Save comprehensive results
        results_file = self.result_dir / "swiftkv_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def print_comparison_summary(self, results: Dict):
        """Print a comprehensive comparison summary."""
        
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Compare main results
        if "FlashAttention" in results and "FlashInfer" in results:
            fa = results["FlashAttention"]
            fi = results["FlashInfer"]
            
            print(f"{'Metric':<20} {'FlashAttention':<15} {'FlashInfer':<15} {'Ratio':<10} {'Improvement':<12}")
            print("-" * 80)
            
            for metric in ['throughput', 'ttft_ms', 'tpot_ms']:
                fa_val = fa[metric]
                fi_val = fi[metric]
                ratio = fi_val / fa_val if fa_val != 0 else 0
                improvement = (ratio - 1) * 100
                
                print(f"{metric:<20} {fa_val:<15.2f} {fi_val:<15.2f} {ratio:<10.3f} {improvement:+.1f}%")
            
            print(f"\n{'='*40}")
            print("PERFORMANCE SUMMARY")
            print(f"{'='*40}")
            
            if fi['throughput'] > fa['throughput']:
                improvement = (fi['throughput'] / fa['throughput'] - 1) * 100
                print(f"✅ FlashInfer is {improvement:.1f}% faster than FlashAttention")
            else:
                degradation = (1 - fi['throughput'] / fa['throughput']) * 100
                print(f"⚠️  FlashInfer is {degradation:.1f}% slower than FlashAttention")
            
            # Latency comparison
            if fi['ttft_ms'] < fa['ttft_ms']:
                improvement = (1 - fi['ttft_ms'] / fa['ttft_ms']) * 100
                print(f"✅ FlashInfer TTFT is {improvement:.1f}% faster")
            else:
                degradation = (fi['ttft_ms'] / fa['ttft_ms'] - 1) * 100
                print(f"⚠️  FlashInfer TTFT is {degradation:.1f}% slower")
        
        # Print batch size results if available
        if "FlashAttention_batch" in results and "FlashInfer_batch" in results:
            print(f"\n{'='*60}")
            print("BATCH SIZE COMPARISON")
            print(f"{'='*60}")
            
            fa_batch = results["FlashAttention_batch"]
            fi_batch = results["FlashInfer_batch"]
            
            print(f"{'Batch Size':<12} {'FA Throughput':<15} {'FI Throughput':<15} {'Ratio':<10}")
            print("-" * 60)
            
            for batch_size in sorted(set(fa_batch.keys()) & set(fi_batch.keys())):
                fa_throughput = fa_batch[batch_size]['throughput']
                fi_throughput = fi_batch[batch_size]['throughput']
                ratio = fi_throughput / fa_throughput if fa_throughput != 0 else 0
                
                print(f"{batch_size:<12} {fa_throughput:<15.2f} {fi_throughput:<15.2f} {ratio:<10.3f}")
        
        print(f"\nDetailed results saved to: {self.result_dir}/swiftkv_comparison_results.json")


def main():
    parser = argparse.ArgumentParser(
        description="SwiftKV Backend Comparison Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_swiftkv.py                                    # Default benchmark
  python benchmark_swiftkv.py --batch-sizes 1,4,8,16            # Custom batch sizes
  python benchmark_swiftkv.py --requests 200 --concurrency 16   # More requests
  python benchmark_swiftkv.py --prompt-length 4000              # Long prompts
        """
    )
    
    parser.add_argument(
        "--model",
        default="Snowflake/Llama-3.1-SwiftKV-8B-Instruct-FP8",
        help="Model to benchmark (default: Snowflake/Llama-3.1-SwiftKV-8B-Instruct-FP8)"
    )
    
    parser.add_argument(
        "--batch-sizes",
        default="1,4,8,16",
        help="Comma-separated list of batch sizes to test (default: 1,4,8,16)"
    )
    
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests per batch size (default: 100)"
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent requests (default: 8)"
    )
    
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=2000,
        help="Length of input prompts (default: 2000)"
    )
    
    parser.add_argument(
        "--output-length",
        type=int,
        default=100,
        help="Length of output tokens (default: 100)"
    )
    
    parser.add_argument(
        "--result-dir",
        default="results",
        help="Directory to save results (default: results)"
    )
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    # Create and run benchmark
    benchmark = SwiftKVBenchmark(
        model=args.model,
        batch_sizes=batch_sizes,
        requests=args.requests,
        concurrency=args.concurrency,
        prompt_length=args.prompt_length,
        output_length=args.output_length,
        result_dir=args.result_dir
    )
    
    results = benchmark.run_comparison()
    benchmark.print_comparison_summary(results)


if __name__ == "__main__":
    main() 