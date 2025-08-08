#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import re
import os
import argparse
from scipy import stats


def parse_json_data(json_data):
    """Parse JSON benchmark data"""
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Handle different JSON structures
    if isinstance(data, list):
        benchmark_results = data
    elif isinstance(data, dict):
        if 'benchmark_results' in data:
            benchmark_results = data['benchmark_results']
        else:
            raise ValueError("No benchmark_results found in JSON")
    else:
        raise ValueError("Unsupported JSON structure")
    
    # Flatten the nested structure
    flattened_data = []
    for result in benchmark_results:
        base_data = {
            'test_name': result.get('test_name', ''),
            'operation_name': result.get('operation_name', ''),
            'tensor_elements': result.get('tensor_elements', 0),
        }
        
        # Add benchmark data - focus ONLY on benchmark_stats
        if 'benchmark_stats' in result:
            benchmark_stats = result['benchmark_stats']
            
            # Extract individual times for recalculation
            individual_times = benchmark_stats.get('individual_times_ms', [])
            
            # Calculate mean directly from individual times
            mean_ms = np.mean(individual_times)
            
            # Calculate confidence interval (95% CI)
            if len(individual_times) > 1:
                std_dev = np.std(individual_times, ddof=1)  # Sample standard deviation
                n = len(individual_times)
                confidence_interval = 1.96 * (std_dev / np.sqrt(n))  # 95% CI
            else:
                std_dev = 0
                confidence_interval = 0
            
            benchmark_data = base_data.copy()
            benchmark_data.update({
                'timing_type': 'benchmark',
                'mean_ms': mean_ms,
                'std_dev_ms': std_dev,
                'confidence_interval_ms': confidence_interval,
                'individual_times_ms': individual_times,
                'has_reference': result.get('has_reference', False)
            })
            
            # Add precision metrics if available
            if 'precision_metrics' in result:
                precision_metrics = result['precision_metrics']
                benchmark_data.update({
                    'normalized_error_mean': precision_metrics.get('normalized_error_mean', 0.0),
                    'normalized_error_max': precision_metrics.get('normalized_error_max', 0.0),
                    'raw_mean_abs_error': precision_metrics.get('raw_mean_abs_error', 0.0)
                })
            
            flattened_data.append(benchmark_data)

    return pd.DataFrame(flattened_data)


def extract_size_parameter(df):
    """Extract size parameter from test names"""
    size_values = []
    
    for test_name in df['test_name']:
        if pd.isna(test_name):
            size_values.append(None)
            continue
            
        test_str = str(test_name)
        
        # Map size names to numeric values based on tiles
        if 'Small' in test_str:
            size = 64 * 64 * 32 * 32  # 64x64 tiles * 32x32 elements per tile
        elif 'Medium' in test_str:
            size = 128 * 128 * 32 * 32  # 128x128 tiles
        elif 'Large' in test_str:
            size = 256 * 256 * 32 * 32  # 256x256 tiles
        elif 'Huge' in test_str:
            size = 512 * 512 * 32 * 32  # 512x512 tiles
        else:
            size = None
        
        size_values.append(size)
    
    return size_values


def extract_implementation_type(operation_name):
    """Extract implementation type from operation name"""
    if 'TTNN' in operation_name:
        return 'TTNN'
    elif 'Generic' in operation_name:
        return 'Generic'
    else:
        return 'Unknown'


def extract_base_operation(operation_name):
    """Extract base operation name"""
    # Remove implementation prefixes
    op_base = operation_name.replace('TTNN ', '').replace('Generic ', '')
    return op_base.strip()


def create_performance_plot(df, title, save_path, speedup_data=None):
    """Create performance comparison plot with speedup arrows"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extract implementation types and base operations
    df['implementation'] = df['operation_name'].apply(extract_implementation_type)
    df['base_operation'] = df['operation_name'].apply(extract_base_operation)
    df['size'] = extract_size_parameter(df)
    
    # Define styles
    styles = {
        'TTNN': {'marker': 'o', 'linestyle': '-', 'color': 'blue'},
        'Generic': {'marker': 's', 'linestyle': '--', 'color': 'orange'}
    }
    
    # Get unique base operations
    base_operations = sorted(df['base_operation'].unique())
    
    for base_op in base_operations:
        op_data = df[df['base_operation'] == base_op]
        
        for impl in ['TTNN', 'Generic']:
            impl_data = op_data[op_data['implementation'] == impl]
            
            if len(impl_data) == 0:
                continue
            
            # Sort by size
            impl_data = impl_data.sort_values(by='size')
            sizes = impl_data['size'].values
            means = impl_data['mean_ms'].values
            
            # Get style
            style = styles.get(impl, {'marker': 'o', 'linestyle': '-', 'color': 'black'})
            
            # Plot
            label = f"{base_op} ({impl})"
            ax.loglog(sizes, means,
                     marker=style['marker'],
                     linestyle=style['linestyle'],
                     markersize=8,
                     color=style['color'],
                     label=label)
            
            # Add value labels near each point
            for i, (size, mean_time) in enumerate(zip(sizes, means)):
                # Format the time value
                if mean_time >= 10:
                    time_str = f"{mean_time:.1f}ms"
                elif mean_time >= 1:
                    time_str = f"{mean_time:.2f}ms"
                else:
                    time_str = f"{mean_time:.3f}ms"
                
                # Position the label slightly offset from the point
                offset_x = size * 1.15  # Slightly to the right
                offset_y = mean_time * 1.1  # Slightly above
                
                ax.annotate(time_str, 
                           xy=(size, mean_time),
                           xytext=(offset_x, offset_y),
                           fontsize=9,
                           color=style['color'],
                           ha='left',
                           va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   edgecolor=style['color'], 
                                   alpha=0.8))
            
            # Add confidence intervals
            if 'confidence_interval_ms' in impl_data.columns:
                lower_bounds = means - impl_data['confidence_interval_ms'].values/2
                upper_bounds = means + impl_data['confidence_interval_ms'].values/2
                ax.fill_between(sizes, lower_bounds, upper_bounds,
                               alpha=0.1, color=style['color'])
    
    # Add speedup arrows and annotations
    if speedup_data:
        ttnn_data = df[df['implementation'] == 'TTNN'].sort_values(by='size')
        generic_data = df[df['implementation'] == 'Generic'].sort_values(by='size')
        
        # Create mapping of sizes to data points
        ttnn_points = {row['size']: (row['size'], row['mean_ms']) for _, row in ttnn_data.iterrows()}
        generic_points = {row['size']: (row['size'], row['mean_ms']) for _, row in generic_data.iterrows()}
        
        for speedup_info in speedup_data:
            tensor_size = speedup_info['tensor_size']
            
            # Find corresponding points in the plot data
            if tensor_size in ttnn_points and tensor_size in generic_points:
                ttnn_point = ttnn_points[tensor_size]
                generic_point = generic_points[tensor_size]
                
                # Arrow ALWAYS points from TTNN to Generic
                # Color indicates whether Generic is faster (green) or slower (red)
                speedup_factor = speedup_info['speedup_factor']
                speedup_display = speedup_info['speedup_display']
                is_generic_faster = speedup_info.get('is_generic_faster', False)
                
                # Arrow always goes from TTNN to Generic
                start_point = ttnn_point
                end_point = generic_point
                
                if is_generic_faster:
                    # Generic is faster - GREEN = good
                    arrow_color = 'green'
                    text_color = 'darkgreen'
                else:
                    # Generic is slower - RED = bad
                    arrow_color = 'red'
                    text_color = 'darkred'
                
                # Calculate arrow position (middle point between the two implementations)
                mid_x = start_point[0]
                mid_y_log = np.sqrt(start_point[1] * end_point[1])  # Geometric mean for log scale
                
                # Calculate arrow properties
                dx = 0  # No horizontal movement
                dy = end_point[1] - start_point[1]
                
                # Draw arrow
                ax.annotate('', xy=end_point, xytext=start_point,
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2, alpha=0.7))
                
                # Add speedup text annotation
                # Position text slightly to the right of the arrow
                text_x = mid_x * 1.3
                text_y = mid_y_log
                
                # Create speedup text with background
                if is_generic_faster:
                    speedup_text = f"Generic {speedup_display} faster"
                else:
                    speedup_text = f"Generic {speedup_display} slower"
                
                ax.annotate(speedup_text,
                           xy=(mid_x, mid_y_log),
                           xytext=(text_x, text_y),
                           fontsize=10,
                           fontweight='bold',
                           color=text_color,
                           ha='left',
                           va='center',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   edgecolor=arrow_color, 
                                   alpha=0.9))
    
    # Set more detailed axis labels for log-log plot
    from matplotlib.ticker import LogFormatter, LogLocator
    
    # X-axis (tensor sizes)
    ax.set_xlabel('Tensor Size (elements)', fontsize=12)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    
    # Y-axis (time in ms)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, which='major')
    ax.grid(True, alpha=0.1, which='minor')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10, loc='upper left',
             bbox_to_anchor=(0, -0.15), ncol=2)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved to {save_path}")
    plt.close()








def analyze_benchmark_results(json_path, output_dir='test/benchmark'):
    """Analyze and visualize benchmark results"""
    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"Successfully loaded benchmark data from {json_path}")
        
        # Parse JSON data
        df = parse_json_data(data)
        print(f"Data shape: {df.shape}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract speedup data from JSON
        speedup_data = data.get('speedup_analysis', [])
        
        # Create performance plot - use operation name from JSON
        operation_name = data.get('operation_name', 'unknown')
        performance_plot_path = os.path.join(output_dir, f"{operation_name.lower()}.png")
        plot_title = f"{operation_name} Performance Comparison"
        create_performance_plot(df, plot_title, performance_plot_path, speedup_data)
        
        # Print summary statistics
        print("\n" + "="*80)
        print("BENCHMARK ANALYSIS SUMMARY")
        print("="*80)
        
        # Performance summary
        ttnn_data = df[df['operation_name'].str.contains('TTNN', na=False)]
        generic_data = df[df['operation_name'].str.contains('Generic', na=False)]
        
        if len(ttnn_data) > 0 and len(generic_data) > 0:
            print(f"TTNN Operations: {len(ttnn_data)}")
            print(f"  Mean execution time: {ttnn_data['mean_ms'].mean():.2f} ms")
            print(f"  Min execution time: {ttnn_data['mean_ms'].min():.2f} ms")
            print(f"  Max execution time: {ttnn_data['mean_ms'].max():.2f} ms")
            
            print(f"\nGeneric Operations: {len(generic_data)}")
            print(f"  Mean execution time: {generic_data['mean_ms'].mean():.2f} ms")
            print(f"  Min execution time: {generic_data['mean_ms'].min():.2f} ms")
            print(f"  Max execution time: {generic_data['mean_ms'].max():.2f} ms")
        
        # Print speedup analysis
        if speedup_data:
            print(f"\nSpeedup Analysis (TTNN as baseline):")
            for speedup_info in speedup_data:
                size_name = speedup_info['test_name']
                speedup_display = speedup_info['speedup_display']
                is_generic_faster = speedup_info.get('is_generic_faster', False)
                ttnn_time = speedup_info['ttnn_time_ms']
                generic_time = speedup_info['generic_time_ms']
                
                if is_generic_faster:
                    comparison = f"Generic is {speedup_display} faster than TTNN"
                else:
                    comparison = f"Generic is {speedup_display} slower than TTNN"
                
                print(f"  {size_name}: {comparison}")
                print(f"    TTNN: {ttnn_time:.2f}ms, Generic: {generic_time:.2f}ms")
        
        print("="*80)
        print(f"Analysis complete. Plots saved to {output_dir}")
        
    except Exception as e:
        print(f"Error analyzing benchmark results: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('json_path', help='Path to benchmark JSON file')
    parser.add_argument('--output-dir', default='test/benchmark', 
                       help='Output directory for plots (default: test/benchmark)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file {args.json_path} not found")
        return
    
    analyze_benchmark_results(args.json_path, args.output_dir)


if __name__ == "__main__":
    main()