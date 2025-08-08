import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
import torch
import ttnn
from all_operations import ALL_OPERATIONS


def create_comparison_plot(operation_name, input_values, results, save_path):
    """Create comparison plots for an operation"""
    plt.figure(figsize=(15, 35))  # Increased height for 7 subplots (4 regular + 3 normalized)
    
    # Extract unique input values and corresponding outputs for plotting
    # Sort by input values for plotting
    indices = np.argsort(input_values)
    sorted_inputs = input_values[indices]
    
    # Plot 1: Function comparison with markers
    plt.subplot(7, 1, 1)  # Changed to 7 rows
    markers = ['^', 's', 'D']  # triangle, square, diamond
    colors = ['b', 'g', 'r']
    
    for i, (name, values) in enumerate(results.items()):
        sorted_values = values[indices]
        plt.plot(sorted_inputs, sorted_values, 
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                markevery=max(1, int(len(sorted_inputs)/20)),  # Only show markers every nth point, minimum 1
                linestyle='-',
                linewidth=2,
                label=name)
    
    plt.title(f'Comparison of {operation_name} Implementations')
    plt.xlabel('Input')
    plt.ylabel(f'{operation_name}(x)')
    plt.legend()
    plt.grid(True)
    
    # Separate difference plots for each pair
    if len(results) >= 2:
        names = list(results.keys())
        colors = ['b', 'g', 'r']
        
        # Find maximum absolute difference across all pairs for consistent y-axis
        max_diff = 0
        all_comparisons = []
        
        # Define all possible comparisons
        if len(names) >= 2:
            all_comparisons.append((0, 1, f"{names[0]} vs {names[1]}"))
        if len(names) >= 3:
            all_comparisons.append((0, 2, f"{names[0]} vs {names[2]}"))
            all_comparisons.append((1, 2, f"{names[1]} vs {names[2]}"))
        
        # Calculate max difference for consistent scaling
        for idx1, idx2, _ in all_comparisons:
            diff = results[names[idx1]][indices] - results[names[idx2]][indices]
            # Handle NaN and Inf values
            valid_diff = diff[np.isfinite(diff)]
            if len(valid_diff) > 0:
                max_diff = max(max_diff, np.max(np.abs(valid_diff)))
            else:
                # If all values are NaN/Inf, use a default range
                max_diff = max(max_diff, 1.0)
        
        # Plot 2: PyTorch vs TTNN (or first vs second)
        if len(all_comparisons) >= 1:
            plt.subplot(7, 1, 2)
            idx1, idx2, label = all_comparisons[0]
            diff = results[names[idx1]][indices] - results[names[idx2]][indices]
            plt.scatter(sorted_inputs, diff, 
                      color=colors[0],
                      s=4,
                      alpha=0.6)
            plt.title(f'Difference: {label}')
            plt.xlabel('Input')
            plt.ylabel('Difference')
            plt.grid(True)
            # Ensure max_diff is finite before setting ylim
            if np.isfinite(max_diff) and max_diff > 0:
                plt.ylim(-max_diff*1.05, max_diff*1.05)
        
        # Plot 3: PyTorch vs Generic (or first vs third)
        if len(all_comparisons) >= 2:
            plt.subplot(7, 1, 3)
            idx1, idx2, label = all_comparisons[1]
            diff = results[names[idx1]][indices] - results[names[idx2]][indices]
            plt.scatter(sorted_inputs, diff, 
                      color=colors[1],
                      s=4,
                      alpha=0.6)
            plt.title(f'Difference: {label}')
            plt.xlabel('Input')
            plt.ylabel('Difference')
            plt.grid(True)
            # Ensure max_diff is finite before setting ylim
            if np.isfinite(max_diff) and max_diff > 0:
                plt.ylim(-max_diff*1.05, max_diff*1.05)
        
        # Plot 4: TTNN vs Generic (or second vs third)
        if len(all_comparisons) >= 3:
            plt.subplot(7, 1, 4)
            idx1, idx2, label = all_comparisons[2]
            diff = results[names[idx1]][indices] - results[names[idx2]][indices]
            plt.scatter(sorted_inputs, diff, 
                      color=colors[2],
                      s=4,
                      alpha=0.6)
            plt.title(f'Difference: {label}')
            plt.xlabel('Input')
            plt.ylabel('Difference')
            plt.grid(True)
            # Ensure max_diff is finite before setting ylim
            if np.isfinite(max_diff) and max_diff > 0:
                plt.ylim(-max_diff*1.05, max_diff*1.05)
        
        # Add normalized error plots (plots 5, 6, 7) - same structure as difference plots but normalized
        try:
            from precision_counter import PrecisionCounter, PRECISION_PASS_THRESHOLD
            counter = PrecisionCounter()
            input_tensor = torch.tensor(input_values, dtype=torch.float32)
            
            # Calculate all normalized differences and find max for symmetric scaling
            all_normalized_diffs = []
            normalized_comparisons = []
            normalized_stats = {}
            
            for idx1, idx2, label in all_comparisons:
                # Use centralized normalization method from PrecisionCounter
                reference_tensor = torch.tensor(results[names[idx1]][indices], dtype=torch.float32)
                test_tensor = torch.tensor(results[names[idx2]][indices], dtype=torch.float32)
                input_subset = input_tensor[indices] if len(indices) < len(input_tensor) else input_tensor
                
                # Get normalized error statistics using centralized method
                stats = counter.calculate_normalized_error_stats(
                    input_subset, reference_tensor, test_tensor, torch.bfloat16
                )
                
                # Calculate additional statistics for compatibility
                diff = results[names[idx1]][indices] - results[names[idx2]][indices]
                
                # Calculate quantization errors for plotting (still needed for visualization)
                input_quant_error = counter._get_quantization_error(input_tensor, torch.bfloat16).numpy()
                output_true = torch.tensor(results[names[idx1]], dtype=torch.float32)
                output_quant_error = counter._get_quantization_error(output_true, torch.bfloat16).numpy()
                max_quant_error = np.maximum(input_quant_error, output_quant_error)
                
                # Calculate normalized error for plotting (signed!)
                normalized_diff = diff / np.clip(max_quant_error[indices], a_min=1e-10, a_max=None)
                
                # Store for max calculation and plotting
                finite_mask = np.isfinite(normalized_diff)
                if np.sum(finite_mask) > 0:
                    valid_norm_diff = normalized_diff[finite_mask]
                    all_normalized_diffs.extend(valid_norm_diff)
                    normalized_comparisons.append((normalized_diff, idx1, idx2, label))
                    
                    # Use centralized statistics with additional median and std
                    normalized_stats[label] = {
                        "mean_abs_normalized_error": stats['normalized_error_mean'],
                        "max_abs_normalized_error": stats['normalized_error_max'],
                        "median_abs_normalized_error": float(np.median(np.abs(valid_norm_diff))),
                        "mean_signed_normalized_error": float(np.mean(valid_norm_diff)),
                        "std_normalized_error": float(np.std(valid_norm_diff))
                    }
            
            # Print normalized error statistics
            if normalized_stats:
                print(f"\nNormalized Error Statistics for {operation_name}:")
                print("-" * 60)
                for label, stats in normalized_stats.items():
                    mean_error = stats['mean_abs_normalized_error']
                    passed = mean_error < PRECISION_PASS_THRESHOLD
                    pass_emoji = "✅" if passed else "❌"
                    pass_text = "Passed" if passed else "Failed"
                    
                    print(f"{label}:")
                    print(f"  Mean Abs Normalized Error: {mean_error:.4f}")
                    print(f"  Max Abs Normalized Error:  {stats['max_abs_normalized_error']:.4f}")
                    print(f"  Median Abs Normalized Error: {stats['median_abs_normalized_error']:.4f}")
                    print(f"  Std Normalized Error: {stats['std_normalized_error']:.4f}")
                    print(f"  Test Result: {pass_emoji} {pass_text} (Mean < {PRECISION_PASS_THRESHOLD})")
                    print()
            
            # Plot 5: Normalized Error for first comparison
            if len(normalized_comparisons) >= 1:
                plt.subplot(7, 1, 5)
                normalized_diff, idx1, idx2, label = normalized_comparisons[0]
                finite_mask = np.isfinite(normalized_diff)
                if np.sum(finite_mask) > 0:
                    plt.scatter(sorted_inputs[finite_mask], normalized_diff[finite_mask], 
                              color=colors[0], s=4, alpha=0.6)
                    # Calculate individual scale for this plot, centered at 0
                    valid_diff = normalized_diff[finite_mask]
                    max_abs_diff = np.max(np.abs(valid_diff)) if len(valid_diff) > 0 else 1.0
                    if np.isfinite(max_abs_diff) and max_abs_diff > 0:
                        plt.ylim(-max_abs_diff*1.05, max_abs_diff*1.05)
                plt.title(f'Normalized Error: {label}')
                plt.xlabel('Input')
                plt.ylabel('BFloat16 Quantization Normalized Error')
                plt.grid(True)
                plt.axhline(y=0.0, color='black', linestyle='-', alpha=0.8)
                # Add +/- threshold borders
                plt.axhline(y=PRECISION_PASS_THRESHOLD, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label=f'+{PRECISION_PASS_THRESHOLD} threshold')
                plt.axhline(y=-PRECISION_PASS_THRESHOLD, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label=f'-{PRECISION_PASS_THRESHOLD} threshold')
            
            # Plot 6: Normalized Error for second comparison
            if len(normalized_comparisons) >= 2:
                plt.subplot(7, 1, 6)
                normalized_diff, idx1, idx2, label = normalized_comparisons[1]
                finite_mask = np.isfinite(normalized_diff)
                if np.sum(finite_mask) > 0:
                    plt.scatter(sorted_inputs[finite_mask], normalized_diff[finite_mask], 
                              color=colors[1], s=4, alpha=0.6)
                    # Calculate individual scale for this plot, centered at 0
                    valid_diff = normalized_diff[finite_mask]
                    max_abs_diff = np.max(np.abs(valid_diff)) if len(valid_diff) > 0 else 1.0
                    if np.isfinite(max_abs_diff) and max_abs_diff > 0:
                        plt.ylim(-max_abs_diff*1.05, max_abs_diff*1.05)
                plt.title(f'Normalized Error: {label}')
                plt.xlabel('Input')
                plt.ylabel('BFloat16 Quantization Normalized Error')
                plt.grid(True)
                plt.axhline(y=0.0, color='black', linestyle='-', alpha=0.8)
                # Add +/- threshold borders
                plt.axhline(y=PRECISION_PASS_THRESHOLD, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label=f'+{PRECISION_PASS_THRESHOLD} threshold')
                plt.axhline(y=-PRECISION_PASS_THRESHOLD, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label=f'-{PRECISION_PASS_THRESHOLD} threshold')
            
            # Plot 7: Normalized Error for third comparison
            if len(normalized_comparisons) >= 3:
                plt.subplot(7, 1, 7)
                normalized_diff, idx1, idx2, label = normalized_comparisons[2]
                finite_mask = np.isfinite(normalized_diff)
                if np.sum(finite_mask) > 0:
                    plt.scatter(sorted_inputs[finite_mask], normalized_diff[finite_mask], 
                              color=colors[2], s=4, alpha=0.6)
                    # Calculate individual scale for this plot, centered at 0
                    valid_diff = normalized_diff[finite_mask]
                    max_abs_diff = np.max(np.abs(valid_diff)) if len(valid_diff) > 0 else 1.0
                    if np.isfinite(max_abs_diff) and max_abs_diff > 0:
                        plt.ylim(-max_abs_diff*1.05, max_abs_diff*1.05)
                plt.title(f'Normalized Error: {label}')
                plt.xlabel('Input')
                plt.ylabel('BFloat16 Quantization Normalized Error')
                plt.grid(True)
                plt.axhline(y=0.0, color='black', linestyle='-', alpha=0.8)
                # Add +/- threshold borders
                plt.axhline(y=PRECISION_PASS_THRESHOLD, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label=f'+{PRECISION_PASS_THRESHOLD} threshold')
                plt.axhline(y=-PRECISION_PASS_THRESHOLD, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label=f'-{PRECISION_PASS_THRESHOLD} threshold')
                    
        except ImportError:
            print("Warning: precision_counter module not available, skipping normalized error plots")
            normalized_stats = {}
        except Exception as e:
            print(f"Warning: Error calculating normalized errors: {e}")
            normalized_stats = {}
    else:
        normalized_stats = {}
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as '{save_path}'")
    
    return normalized_stats  # Return stats for JSON saving


def create_top_down_sampling(input_range, max_points=1024):
    """
    Create top-down/hierarchical sampling points within the input range.
    
    Args:
        input_range: Tuple of (min_value, max_value)
        max_points: Maximum number of points to generate (default 1024)
    
    Returns:
        List of input values ordered from coarse to fine detail
    """
    min_val, max_val = input_range
    
    # Start with endpoints
    top_down_points = [min_val, max_val]
    current_segments = [(min_val, max_val)]
    
    # Build levels from coarse to fine
    while len(top_down_points) < max_points and current_segments:
        next_segments = []
        level_points = []
        
        for start, end in current_segments:
            # Add midpoint
            mid = (start + end) / 2.0
            level_points.append(mid)
            
            # Create new segments for next level
            next_segments.append((start, mid))
            next_segments.append((mid, end))
        
        # Add this level's points
        top_down_points.extend(level_points)
        current_segments = next_segments
        
        # Stop if we would exceed max_points in next iteration
        if len(top_down_points) + len(current_segments) > max_points:
            break
    
    return top_down_points


def save_results_json(operation_name, input_range, top_down_results, save_path, normalized_stats=None):
    """Save comparison results as JSON for analysis in top-down/hierarchical format"""
    json_data = []
    
    # Add metadata and statistics at the top
    metadata = {
        "operation": operation_name,
        "total_samples": len(top_down_results["input_values"]),
        "input_range": [float(input_range[0]), float(input_range[1])],
        "data_structure": "top_down_tree",
        "description": "Data organized from coarse to fine detail for hierarchical analysis"
    }
    
    # Add normalized error statistics if available
    if normalized_stats:
        metadata["normalized_error_statistics"] = normalized_stats
        metadata["normalized_error_description"] = "Normalized error statistics (error / max(input_quant_error, output_quant_error))"
    
    json_data.append(metadata)
    
    # Add traditional statistics summary
    results = top_down_results["results"]
    if len(results) > 1:
        reference_name = list(results.keys())[0]
        reference_values = results[reference_name]
        stats = {"operation": operation_name, "statistics": {}}
        
        for name, values in list(results.items())[1:]:
            abs_diff = np.abs(values - reference_values)
            stats["statistics"][name] = {
                "mean_absolute_error": float(np.mean(abs_diff)),
                "max_absolute_error": float(np.max(abs_diff)),
                "rmse": float(np.sqrt(np.mean(abs_diff**2)))
            }
        json_data.append(stats)
    
    # Add data points in top-down order (already ordered from coarse to fine)
    data_points = []
    input_values = top_down_results["input_values"]
    for i, input_val in enumerate(input_values):
        entry = {"input": float(input_val)}
        for name, values in results.items():
            # Clean up the key name for readability
            key_name = name.lower().replace(' ', '_').replace('pytorch', 'torch')
            entry[key_name] = float(values[i])
        data_points.append(entry)
    
    # Add the top-down-ordered data
    json_data.append({
        "top_down_data": data_points,
        "ordering_description": "Data points ordered from coarse to fine detail: endpoints first, then midpoints, then quarter-points, etc."
    })
    
    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Top-down-ordered JSON data saved as '{save_path}'")



def run_operation_comparison(operation_name, device):
    """Run comparison for a single operation"""
    if operation_name not in ALL_OPERATIONS:
        print(f"Error: Operation '{operation_name}' not found in ALL_OPERATIONS")
        print(f"Available operations: {list(ALL_OPERATIONS.keys())}")
        return
    
    print(f"Running comparison for {operation_name}...")
    
    # Get operation test configuration
    op_test = ALL_OPERATIONS[operation_name]
    input_range = op_test.input_range
    
    # Generate dense input values for plotting (high resolution)
    num_samples = 10000
    dense_input_values = np.linspace(input_range[0], input_range[1], num_samples)
    dense_input_tensor = torch.tensor(dense_input_values, dtype=torch.float32)
    
    # Run all three implementations on dense data (for plotting)
    dense_results = {}
    
    # PyTorch reference (dense)
    torch_results = op_test.torch_func(dense_input_tensor).float().numpy()
    dense_results["PyTorch"] = torch_results
    print(f"✓ PyTorch implementation completed (dense)")
    
    # TTNN implementation (dense)
    ttnn_tensor = ttnn.from_torch(
        dense_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    
    ttnn_result = op_test.ttnn_func(ttnn_tensor)
    ttnn_results = ttnn.to_torch(ttnn_result).float().numpy()
    dense_results["TTNN"] = ttnn_results
    print(f"✓ TTNN implementation completed (dense)")
    
    # Generic implementation (dense)
    ttnn_tensor = ttnn.from_torch(
        dense_input_tensor, 
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, 
        device=device, 
        memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    
    generic_result = op_test.candidate_func(ttnn_tensor, inplace=True)
    generic_results = ttnn.to_torch(generic_result).float().numpy()
    dense_results["Generic"] = generic_results
    print(f"✓ Generic implementation completed (dense)")
    
    # Generate top-down input values for JSON (sparse, hierarchical)
    top_down_input_values = create_top_down_sampling(input_range, max_points=1024)
    top_down_input_tensor = torch.tensor(top_down_input_values, dtype=torch.float32)
    
    # Run all three implementations on top-down data (for JSON)
    top_down_results = {"input_values": top_down_input_values, "results": {}}
    
    # PyTorch reference (top-down)
    torch_top_down_results = op_test.torch_func(top_down_input_tensor).float().numpy()
    top_down_results["results"]["PyTorch"] = torch_top_down_results
    print(f"✓ PyTorch implementation completed (top-down)")
    
    # TTNN implementation (top-down)
    ttnn_top_down_tensor = ttnn.from_torch(
        top_down_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    
    ttnn_top_down_result = op_test.ttnn_func(ttnn_top_down_tensor)
    ttnn_top_down_results = ttnn.to_torch(ttnn_top_down_result).float().numpy()
    top_down_results["results"]["TTNN"] = ttnn_top_down_results
    print(f"✓ TTNN implementation completed (top-down)")
    
    # Generic implementation (top-down)
    ttnn_top_down_tensor = ttnn.from_torch(
        top_down_input_tensor, 
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, 
        device=device, 
        memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    
    generic_top_down_result = op_test.candidate_func(ttnn_top_down_tensor, inplace=True)
    generic_top_down_results = ttnn.to_torch(generic_top_down_result).float().numpy()
    top_down_results["results"]["Generic"] = generic_top_down_results
    print(f"✓ Generic implementation completed (top-down)")
    
    # Create plots and save results
    if dense_results:
        # Create test/plot directories if they don't exist
        plot_base_dir = "test/plot"
        json_dir = os.path.join(plot_base_dir, "json")
        png_dir = os.path.join(plot_base_dir, "png")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)
        
        # Generate file paths
        png_path = os.path.join(png_dir, f"{operation_name.lower()}.png")
        json_path = os.path.join(json_dir, f"{operation_name.lower()}.json")
        
        # Create plot with dense sampling (for visualization)
        normalized_stats = create_comparison_plot(operation_name, dense_input_values, dense_results, png_path)
        
        # Save JSON with top-down sampling (for LLM-friendly analysis)
        save_results_json(operation_name, input_range, top_down_results, json_path, normalized_stats)
        
        print(f"Comparison for {operation_name} completed successfully!")
    else:
        print(f"No valid results to plot for {operation_name}")


def main():
    parser = argparse.ArgumentParser(description='Plot operation comparisons')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Run comparison for all operations')
    group.add_argument('--op', type=str, help='Run comparison for specific operation (e.g., SELU)')
    
    args = parser.parse_args()
    
    # Initialize device once
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    print("✓ Device initialized")
    
    try:
        if args.all:
            print("Running comparisons for all operations...")
            for op_name in ALL_OPERATIONS.keys():
                run_operation_comparison(op_name, device)
                print("-" * 50)
        elif args.op:
            run_operation_comparison(args.op.upper(), device)
    finally:
        # Close device at the end
        ttnn.close_device(device)
        print("✓ Device closed")


if __name__ == "__main__":
    main()