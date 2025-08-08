#!/usr/bin/env python3

import torch
import ttnn
import numpy as np
import time
import json
import os
import argparse
from all_operations import ALL_OPERATIONS
from precision_counter import PrecisionCounter, PRECISION_PASS_THRESHOLD


class TimingStats:
    def __init__(self, times=None):
        self.individual_times_ms = times if times else []
        self.mean_ms = 0.0
        self.variance_ms = 0.0
        self.std_dev_ms = 0.0
        self.confidence_interval_ms = 0.0
        self.calculate_stats()
    
    def calculate_stats(self):
        if not self.individual_times_ms:
            return
        
        self.mean_ms = sum(self.individual_times_ms) / len(self.individual_times_ms)
        
        variance_sum = sum((t - self.mean_ms) ** 2 for t in self.individual_times_ms)
        self.variance_ms = variance_sum / len(self.individual_times_ms)
        
        self.std_dev_ms = np.sqrt(self.variance_ms)
        self.confidence_interval_ms = 2.0 * self.std_dev_ms  # 95% confidence


class BenchmarkResult:
    def __init__(self, test_name, operation_name, tensor_elements, 
                 warmup_stats, benchmark_stats, correctness_passed=True,
                 has_reference=False, warmup_iterations=0, benchmark_iterations=0,
                 correctness_percentage=100.0, max_absolute_error=0.0,
                 mean_absolute_error=0.0, inf_matches=0, total_elements=0,
                 precision_metrics=None):
        self.test_name = test_name
        self.operation_name = operation_name
        self.tensor_elements = tensor_elements
        self.warmup_stats = warmup_stats
        self.benchmark_stats = benchmark_stats
        self.correctness_passed = correctness_passed
        self.has_reference = has_reference
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.correctness_percentage = correctness_percentage
        self.max_absolute_error = max_absolute_error
        self.mean_absolute_error = mean_absolute_error
        self.inf_matches = inf_matches
        self.total_elements = total_elements
        self.precision_metrics = precision_metrics or {}


class OperationBenchmark:
    def __init__(self, device):
        self.device = device
        self.device.enable_program_cache()
        self.precision_counter = PrecisionCounter()
        
        # FIXED STRUCTURE - Separate dicts for each test type, keyed by EXACT element count
        self.current_operation = None
        self.ttnn_results = {}      # {element_count: BenchmarkResult}
        self.generic_results = {}   # {element_count: BenchmarkResult}
        
        # Pre-allocate tensors for different sizes
        self.base_tensors = {}
        self._preallocate_tensors()
    
    def _preallocate_tensors(self):
        """Pre-allocate base tensors for different sizes in [0,1] range"""
        print("🔄 Pre-allocating base tensors...")
        
        # Define tensor sizes (tiles)
        sizes = {
            "Small": (64, 64),    # 64x64 tiles
            "Medium": (128, 128), # 128x128 tiles  
            "Large": (256, 256),  # 256x256 tiles
            "Huge": (512, 512)    # 512x512 tiles
        }
        
        for size_name, (num_tile_rows, num_tile_cols) in sizes.items():
            print(f"  Creating {size_name} tensor: {num_tile_rows}x{num_tile_cols} tiles")
            
            # Create tensor with values in [0,1] range
            base_data = torch.rand(1, num_tile_rows * num_tile_cols, 32, 32).to(torch.bfloat16)
            base_tensor = ttnn.from_torch(
                base_data, 
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, 
                device=self.device, 
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            
            self.base_tensors[size_name] = base_tensor
            print(f"  ✅ {size_name} tensor created: {base_tensor.logical_volume()} elements")
        
        print("✅ All base tensors pre-allocated")
    
    def rescale_tensor_to_range(self, base_tensor, min_val, max_val):
        """Rescale a tensor from [0,1] range to [min_val, max_val] range"""
        # Calculate scale and shift
        scale = max_val - min_val
        shift = min_val
        
        # Scale tensor: first multiply by range size
        scaled_tensor = ttnn.multiply(base_tensor, scale)
        
        # Shift tensor: add minimum value
        result_tensor = ttnn.add(scaled_tensor, shift)
        
        # Clean up intermediate tensor
        scaled_tensor.deallocate()
        
        return result_tensor
    
    def benchmark_operation_test(self, test_name, input_tensor, op_test, num_iterations=-1, run_precision_test=False):
        """Benchmark both TTNN and Generic implementations from OperationTest"""
        
        # Benchmark TTNN implementation
        ttnn_result = self.benchmark_ttnn_implementation(
            test_name, input_tensor, op_test.operation_name, 
            op_test.ttnn_func, num_iterations
        )
        
        # Benchmark Generic implementation
        generic_result = self.benchmark_generic_implementation(
            test_name, input_tensor, op_test.operation_name,
            lambda tensor: op_test.candidate_func(tensor), 
            num_iterations, run_precision_test, reference_func=op_test.ttnn_func
        )
        
        return ttnn_result, generic_result
    

    
    def benchmark_ttnn_implementation(self, test_name, input_tensor, operation_name, operation_func, num_iterations=-1):
        """Benchmark TTNN implementation and store in ttnn_results"""
        result = self._run_benchmark(test_name, f"TTNN {operation_name}", input_tensor, operation_func, num_iterations, False, None)
        element_count = result.tensor_elements
        self.ttnn_results[element_count] = result
        print(f"📦 TTNN result stored: {element_count} elements")
        return result
    
    def benchmark_generic_implementation(self, test_name, input_tensor, operation_name, operation_func, num_iterations=-1, run_precision_test=False, reference_func=None):
        """Benchmark Generic implementation and store in generic_results"""
        result = self._run_benchmark(test_name, f"Generic {operation_name}", input_tensor, operation_func, num_iterations, run_precision_test, reference_func)
        element_count = result.tensor_elements
        self.generic_results[element_count] = result
        print(f"📦 Generic result stored: {element_count} elements")
        return result
    
    def _run_benchmark(self, test_name, operation_name, input_tensor, operation_func, num_iterations, run_precision_test, reference_func):
        """Internal method that actually runs the benchmark - copied from benchmark_single_implementation"""
        print(f"\n=== {test_name} - {operation_name} ===")
        
        # Automatically adjust iterations based on tensor size
        if num_iterations == -1:
            total_elements = input_tensor.logical_volume()
            if total_elements < 1000000:          # < 1M elements
                num_iterations = 100
            elif total_elements < 10000000:       # < 10M elements
                num_iterations = 50
            elif total_elements < 50000000:       # < 50M elements
                num_iterations = 20
            else:                                 # >= 50M elements
                num_iterations = 10
        
        print(f"Tensor size: {input_tensor.logical_volume()} elements, iterations: {num_iterations}")
        
        # Warmup with individual timing
        warmup_iterations = 2
        warmup_times = []
        print(f"🔥 Starting warmup phase with {warmup_iterations} iterations...")
        
        for i in range(warmup_iterations):
            print(f"  Warmup iteration {i + 1}/{warmup_iterations}")
            
            start_time = time.time()
            result = operation_func(input_tensor)
            ttnn.synchronize_device(self.device)
            end_time = time.time()
            
            # Release the result tensors
            result.deallocate()
            
            iter_ms = (end_time - start_time) * 1000.0
            warmup_times.append(iter_ms)
            print(f"    Warmup {i + 1} time: {iter_ms:.2f} ms")
        
        warmup_stats = TimingStats(warmup_times)
        print(f"📊 Warmup stats - Mean: {warmup_stats.mean_ms:.2f} ms, Std: {warmup_stats.std_dev_ms:.2f} ms, 95% CI: ±{warmup_stats.confidence_interval_ms:.2f} ms")
        
        # Benchmark with individual timing
        benchmark_times = []
        print(f"📈 Starting {operation_name} benchmark ({num_iterations} iterations)...")
        start_time = time.time()
        
        for i in range(num_iterations):
            if i % 5 == 0 or i == num_iterations - 1:
                current_time = time.time()
                elapsed = current_time - start_time
                print(f"  Iteration {i + 1}/{num_iterations} ({elapsed:.0f}s elapsed)")
            
            iter_start = time.time()
            result = operation_func(input_tensor)
            ttnn.synchronize_device(self.device)
            iter_end = time.time()
            
            # Release the result tensors
            result.deallocate()
            
            iter_ms = (iter_end - iter_start) * 1000.0
            benchmark_times.append(iter_ms)
        
        print(f"✅ {operation_name} benchmark completed")
        
        benchmark_stats = TimingStats(benchmark_times)
        
        print("\n📊 === PERFORMANCE RESULTS ===")
        print(f"Warmup - Mean: {warmup_stats.mean_ms:.2f} ms, Std: {warmup_stats.std_dev_ms:.2f} ms, 95% CI: ±{warmup_stats.confidence_interval_ms:.2f} ms")
        print(f"Benchmark - Mean: {benchmark_stats.mean_ms:.2f} ms, Std: {benchmark_stats.std_dev_ms:.2f} ms, 95% CI: ±{benchmark_stats.confidence_interval_ms:.2f} ms")
        
        # Verify correctness if reference function is provided
        correctness_passed = True
        has_reference = reference_func is not None
        precision_metrics = {}
        
        # Default values for when no reference is provided
        total_elements = input_tensor.logical_volume()
        
        if reference_func and test_name.startswith('Small'):
            print(f"🔍 Analyzing TTNN vs Generic precision (Small tensor only)...")
            
            test_result = operation_func(input_tensor)
            reference_result = reference_func(input_tensor)
            
            # Convert to torch tensors for precision analysis
            input_torch = ttnn.to_torch(input_tensor).float()
            test_torch = ttnn.to_torch(test_result).float()
            reference_torch = ttnn.to_torch(reference_result).float()
            
            # Use centralized normalization method from PrecisionCounter
            precision_metrics = self.precision_counter.calculate_normalized_error_stats(
                input_torch, reference_torch, test_torch, torch.bfloat16
            )
            
            # Pass/fail logic: pass if normalized error < PRECISION_PASS_THRESHOLD (including 0.0)
            correctness_passed = precision_metrics['normalized_error_mean'] < PRECISION_PASS_THRESHOLD
            
            print("📊 === TTNN vs GENERIC PRECISION RESULTS ===")
            print(f"Raw mean absolute error: {precision_metrics['raw_mean_abs_error']:.6e}")
            print(f"Raw max absolute error: {precision_metrics['raw_max_abs_error']:.6e}")
            print(f"Normalized error mean: {precision_metrics['normalized_error_mean']:.4f}")
            print(f"Normalized error max: {precision_metrics['normalized_error_max']:.4f}")
            print(f"Pass threshold: < {PRECISION_PASS_THRESHOLD}")
            print(f"Status: {'PASSED' if correctness_passed else 'FAILED'} (normalized_error_mean < {PRECISION_PASS_THRESHOLD}: {precision_metrics['normalized_error_mean'] < PRECISION_PASS_THRESHOLD})")
            
            # Additional precision details for small tensors
            if run_precision_test:
                print("🔬 === DETAILED PRECISION ANALYSIS ===")
                try:
                    print(f"  Input quantization error (mean): {precision_metrics.get('input_quant_error_mean', 0):.6e}")
                    print(f"  Output quantization error (mean): {precision_metrics.get('output_quant_error_mean', 0):.6e}")
                    print(f"  Machine epsilon (bfloat16): {precision_metrics.get('machine_epsilon', 0):.2e}")
                    print(f"  Mantissa bits: {precision_metrics.get('mantissa_bits', 0)}")
                except Exception as e:
                    print(f"  Warning: Detailed precision analysis failed: {e}")
            
            # Cleanup
            test_result.deallocate()
            reference_result.deallocate()
        elif reference_func:
            print("🔍 Skipping precision analysis (only done for Small tensors)")
        else:
            print("🔍 No reference operation provided - skipping precision analysis")
        
        print("✅ Test completed successfully!\n")
        
        # Store results
        result = BenchmarkResult(
            test_name=test_name,
            operation_name=operation_name,
            tensor_elements=input_tensor.logical_volume(),
            warmup_stats=warmup_stats,
            benchmark_stats=benchmark_stats,
            correctness_passed=correctness_passed,
            has_reference=has_reference,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=num_iterations,
            correctness_percentage=0.0,  # Not used anymore
            max_absolute_error=0.0,      # Not used anymore  
            mean_absolute_error=0.0,     # Not used anymore
            inf_matches=0,               # Not used anymore
            total_elements=total_elements,
            precision_metrics=precision_metrics
        )
        
        return result
    
    def start_operation(self, operation_key):
        """Start benchmarking a new operation - clear previous results"""
        self.current_operation = operation_key
        self.ttnn_results.clear()
        self.generic_results.clear()
        print(f"🚀 Started operation: {operation_key}")
    
    def save_current_operation(self, base_output_dir="test/benchmark"):
        """Save current operation results to JSON"""
        if not self.current_operation:
            print("⚠️ No current operation to save")
            return None
            
        # Create json directory
        json_dir = os.path.join(base_output_dir, "json")
        os.makedirs(json_dir, exist_ok=True)
        
        # Build results list - all TTNN first, then all Generic
        all_results = []
        
        # Add TTNN results (sorted by element count)
        for element_count in sorted(self.ttnn_results.keys()):
            result = self.ttnn_results[element_count]
            all_results.append({
                "test_name": result.test_name,
                "operation_name": result.operation_name,
                "tensor_elements": result.tensor_elements,
                "warmup_iterations": result.warmup_iterations,
                "benchmark_iterations": result.benchmark_iterations,
                "warmup_stats": {
                    "mean_ms": result.warmup_stats.mean_ms,
                    "std_dev_ms": result.warmup_stats.std_dev_ms,
                    "confidence_interval_ms": result.warmup_stats.confidence_interval_ms,
                    "individual_times_ms": result.warmup_stats.individual_times_ms
                },
                "benchmark_stats": {
                    "mean_ms": result.benchmark_stats.mean_ms,
                    "std_dev_ms": result.benchmark_stats.std_dev_ms,
                    "confidence_interval_ms": result.benchmark_stats.confidence_interval_ms,
                    "individual_times_ms": result.benchmark_stats.individual_times_ms
                },
                "has_reference": result.has_reference,
                "precision_metrics": result.precision_metrics
            })
        
        # Add Generic results (sorted by element count)
        for element_count in sorted(self.generic_results.keys()):
            result = self.generic_results[element_count]
            all_results.append({
                "test_name": result.test_name,
                "operation_name": result.operation_name,
                "tensor_elements": result.tensor_elements,
                "warmup_iterations": result.warmup_iterations,
                "benchmark_iterations": result.benchmark_iterations,
                "warmup_stats": {
                    "mean_ms": result.warmup_stats.mean_ms,
                    "std_dev_ms": result.warmup_stats.std_dev_ms,
                    "confidence_interval_ms": result.warmup_stats.confidence_interval_ms,
                    "individual_times_ms": result.warmup_stats.individual_times_ms
                },
                "benchmark_stats": {
                    "mean_ms": result.benchmark_stats.mean_ms,
                    "std_dev_ms": result.benchmark_stats.std_dev_ms,
                    "confidence_interval_ms": result.benchmark_stats.confidence_interval_ms,
                    "individual_times_ms": result.benchmark_stats.individual_times_ms
                },
                "has_reference": result.has_reference,
                "precision_metrics": result.precision_metrics
            })
        
        # Calculate speedup analysis for each tensor size
        speedup_analysis = []
        
        # Compare TTNN vs Generic for each matching tensor size
        for element_count in sorted(self.ttnn_results.keys()):
            if element_count in self.generic_results:
                ttnn_result = self.ttnn_results[element_count]
                generic_result = self.generic_results[element_count]
                
                ttnn_time = ttnn_result.benchmark_stats.mean_ms
                generic_time = generic_result.benchmark_stats.mean_ms
                
                # Calculate speedup: TTNN is baseline, Generic is evaluated against it
                # speedup_factor = ttnn_time / generic_time (how many times faster/slower Generic is)
                if generic_time > 0:
                    speedup_factor = ttnn_time / generic_time
                    
                    if speedup_factor >= 1.0:
                        # Generic is faster - speedup_factor > 1.0 (like 6.0x)
                        speedup_str = f"{speedup_factor:.1f}x"
                        faster_impl = "Generic"
                        is_generic_faster = True
                    else:
                        # Generic is slower - speedup_factor < 1.0 (like 0.52x)
                        speedup_str = f"{speedup_factor:.2f}x"
                        faster_impl = "TTNN"
                        is_generic_faster = False
                    
                    speedup_analysis.append({
                        "tensor_size": element_count,
                        "test_name": ttnn_result.test_name.replace(" " + self.current_operation, ""),  # Extract size name
                        "ttnn_time_ms": ttnn_time,
                        "generic_time_ms": generic_time,
                        "speedup_factor": speedup_factor,
                        "speedup_display": speedup_str,
                        "faster_implementation": faster_impl,
                        "is_generic_faster": is_generic_faster
                    })
        
        # Find precision analysis from smallest Generic result
        precision_analysis = {"description": "No precision analysis available", "status": "NOT_ANALYZED"}
        if self.generic_results:
            smallest_count = min(self.generic_results.keys())
            smallest_generic = self.generic_results[smallest_count]
            if smallest_generic.precision_metrics:
                precision_analysis = {
                    "description": "TTNN vs Generic precision analysis",
                    "tensor_size": smallest_generic.tensor_elements,
                    "raw_abs_mean": smallest_generic.precision_metrics.get('raw_mean_abs_error', 0.0),
                    "raw_abs_max": smallest_generic.precision_metrics.get('raw_max_abs_error', 0.0),
                    "norm_abs_mean": smallest_generic.precision_metrics.get('normalized_error_mean', 0.0),
                    "norm_abs_max": smallest_generic.precision_metrics.get('normalized_error_max', 0.0),
                    "pass_threshold": PRECISION_PASS_THRESHOLD,
                    "status": "PASSED" if smallest_generic.correctness_passed else "FAILED"
                }
        
        # Create JSON structure
        results_json = {
            "operation_name": self.current_operation,
            "speedup_analysis": speedup_analysis,
            "precision_analysis": precision_analysis,
            "benchmark_results": all_results,
            "summary": {
                "total_tests": len(all_results),
                "ttnn_tests": len(self.ttnn_results),
                "generic_tests": len(self.generic_results)
            }
        }
        
        # Save JSON file
        json_filename = os.path.join(json_dir, f"{self.current_operation.lower()}.json")
        with open(json_filename, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"📁 {self.current_operation} results saved to {json_filename}")
        return json_filename
    
    def print_benchmark_summary(self):
        """Print a summary table of current operation results"""
        print("\n")
        print("=" * 140)
        print("                                             BENCHMARK SUMMARY RESULTS")
        print("=" * 140)
        print("\n")
        print(f"{'Test Name':<25} | {'Operation':<26} | {'Elements':<12} | {'Warmup Mean(±CI)':<16} | {'Benchmark Mean(±CI)':<19} | {'Correct%':<9} | {'MaxErr':<7}")
        print("-" * 24 + "-|-" + "-" * 25 + "-|-" + "-" * 11 + "-|-" + "-" * 15 + "-|-" + "-" * 18 + "-|-" + "-" * 8 + "-|-" + "-" * 6)
        
        # Collect all results from both dictionaries
        all_results = []
        
        # Add TTNN results (sorted by element count)
        for element_count in sorted(self.ttnn_results.keys()):
            all_results.append(self.ttnn_results[element_count])
        
        # Add Generic results (sorted by element count)
        for element_count in sorted(self.generic_results.keys()):
            all_results.append(self.generic_results[element_count])
        
        for result in all_results:
            elements_str = str(result.tensor_elements)
            
            if result.has_reference:
                correctness = "PASSED" if result.correctness_passed else "FAILED"
                max_err_str = "N/A"  # We don't use max_absolute_error anymore
            else:
                correctness = "N/A"
                max_err_str = "N/A"
            
            warmup_str = f"{result.warmup_stats.mean_ms:.2f}(±{result.warmup_stats.confidence_interval_ms:.2f})"
            benchmark_str = f"{result.benchmark_stats.mean_ms:.2f}(±{result.benchmark_stats.confidence_interval_ms:.2f})"
            
            print(f"{result.test_name:<25} | {result.operation_name:<26} | {elements_str:<12} | {warmup_str:<16} | {benchmark_str:<19} | {correctness:<9} | {max_err_str:<7}")
        
        # Print summary stats
        print("\n")
        print("=" * 140)
        print("                                                SUMMARY STATS")
        print("=" * 140)
        
        if all_results:
            total_tests_with_ref = 0
            passed_tests = 0
            
            for result in all_results:
                if result.has_reference:
                    total_tests_with_ref += 1
                    if result.correctness_passed:
                        passed_tests += 1
            
            print(f"Total Tests:               {len(all_results)}")
            print(f"TTNN Tests:                {len(self.ttnn_results)}")
            print(f"Generic Tests:             {len(self.generic_results)}")
            print(f"Tests with Reference:      {total_tests_with_ref}")
            print(f"Tests Passed:              {passed_tests}")
        
        print("=" * 140)
        print("\n")



    
    def generate_visualization_for_operation(self, op_name, base_output_dir="test/benchmark"):
        """Generate visualization plots for a specific operation"""
        try:
            import subprocess
            import sys
            
            json_dir = os.path.join(base_output_dir, "json")
            png_dir = os.path.join(base_output_dir, "png")
            json_file = os.path.join(json_dir, f"{op_name.lower()}.json")
            
            print(f"📊 Generating plots for {op_name}...")
            
            # Call visualize_benchmark_results.py with the single JSON file
            result = subprocess.run([
                sys.executable, 'visualize_benchmark_results.py',
                json_file,
                '--output-dir', png_dir
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                png_file = os.path.join(png_dir, f"{op_name.lower()}.png")
                print(f"📊 {op_name} visualization saved to {png_file}")
                return True
            else:
                print(f"    ⚠️ Warning: {op_name} visualization plots failed")
                if result.stderr:
                    print(f"    Error: {result.stderr}")
                return False
                        
        except Exception as e:
            print(f"⚠️ Warning: Could not generate visualization plots for {op_name}: {e}")
            return False
    
    def cleanup(self):
        """Clean up pre-allocated tensors"""
        print("🧹 Cleaning up pre-allocated tensors...")
        for size_name, tensor in self.base_tensors.items():
            tensor.deallocate()
            print(f"  ✅ {size_name} tensor deallocated")
        self.base_tensors.clear()








def main():
    print("🚀 Starting Operation Benchmark Suite...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark operations from all_operations.py')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Run benchmark for all operations')
    group.add_argument('--op', type=str, help='Run benchmark for specific operation (e.g., EXP)')
    parser.add_argument('--output-dir', default='test/benchmark', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Open device
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    
    try:
        # Benchmark mode
        benchmark = OperationBenchmark(device)
        
        # Test different tensor sizes
        test_shapes = ["Small", "Medium", "Large", "Huge"]
        
        operations_to_test = []
        if args.all:
            operations_to_test = list(ALL_OPERATIONS.keys())
        elif args.op:
            if args.op.upper() in ALL_OPERATIONS:
                operations_to_test = [args.op.upper()]
            else:
                print(f"Error: Operation '{args.op.upper()}' not found")
                print(f"Available operations: {list(ALL_OPERATIONS.keys())}")
                return
        
        for op_name in operations_to_test:
            op_test = ALL_OPERATIONS[op_name]
            print(f"\n🔥 === Testing {op_name} operations ===")
            
            # Start new operation - this clears previous results
            benchmark.start_operation(op_name)
            
            for shape_name in test_shapes:
                print(f"\n🔥 === {shape_name} tensor size ===")
                
                # Get base tensor
                base_tensor = benchmark.base_tensors[shape_name]
                
                # Rescale tensor to appropriate range
                scaled_tensor = benchmark.rescale_tensor_to_range(
                    base_tensor,
                    op_test.input_range[0],
                    op_test.input_range[1]
                )
                
                # Benchmark both TTNN and Generic implementations
                # Always run precision test for Small tensors only
                run_precision_test = shape_name == "Small"
                benchmark.benchmark_operation_test(
                    f"{shape_name} {op_name}", 
                    scaled_tensor, 
                    op_test,
                    run_precision_test=run_precision_test
                )
                
                # Clean up scaled tensor
                scaled_tensor.deallocate()
            
            # After completing all tests for this operation, save results and generate plots
            print(f"\n📁 === Saving results for {op_name} ===")
            
            # Save current operation results
            json_file = benchmark.save_current_operation(args.output_dir)
            
            if json_file:
                # Generate visualization plots for this operation
                print(f"\n📊 === Generating plots for {op_name} ===")
                benchmark.generate_visualization_for_operation(op_name, args.output_dir)
                print(f"✅ {op_name} operation completed - JSON saved and plots generated!")
            else:
                print(f"⚠️ {op_name} operation completed but no results to save!")
        
        # Print final summary of all operations
        print("\n📊 === FINAL SUMMARY ===")
        benchmark.print_benchmark_summary()
        
        print("\n🎉 ALL BENCHMARKS COMPLETED! 🎉")
        
        # Cleanup
        benchmark.cleanup()
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close device
        ttnn.close_device(device)


if __name__ == "__main__":
    main()