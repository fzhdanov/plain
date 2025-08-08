# Plain Kernels Repository

A simplified version of kernels designed to provide LLMs with focused access for generating specialized compute operations. This repository contains manually designed seed kernels and an ecosystem of tools for LLM-assisted kernel development.

## Overview

This repository is a "plain" (simplified) version of kernels that enables Large Language Models (LLMs) to generate new mathematical operations with narrow, focused access to the codebase. The approach combines manually crafted seed kernels with automated generation tools to create a comprehensive library of mathematical functions.

## Architecture

### Seed Kernels (Extracted and Simplified)

Three foundational kernels were extracted from the original TTNN codebase and manually simplified to serve as examples and building blocks. The extraction process involved:

- **Unfolding templated arguments**: Removed template parameters and specialized to single execution paths
- **Inlining external calls**: Moved external function dependencies directly into kernel scope
- **Eliminating abstractions**: Flattened complex hierarchical structures into direct implementations

This created self-contained, readable kernels that LLMs can easily understand and modify:

1. **EXP_PRECISE** (`elementwise_sfpu/compute/exp.cpp`)
   - High-precision exponential function implementation
   - Uses Horner form polynomial approximation with exponent handling
   - Handles negative inputs via reciprocal calculation
   - Serves as the foundation for many other mathematical operations

2. **CBRT** (`elementwise_sfpu/compute/cbrt.cpp`) 
   - Cube root function using exp/log decomposition: `cbrt(x) = exp(log(|x|)/3) * sign(x)`
   - Implements inline reciprocal, exp and logarithm functions
   - Demonstrates how to compose multiple mathematical operations

3. **TANH** (`elementwise_sfpu/compute/tanh.cpp`)
   - Hyperbolic tangent using lookup table (LUT) approach
   - Utilizes compile-time coefficient encoding for 8-bit float representation
   - Shows efficient approximation techniques for transcendental functions

### LLM-Generated Kernels

All other operations in the repository were generated using an LLM agentic tool with the following standardized prompt:

```
Given examples of exp and log @cbrt_eltwise_sfpu.cpp, implement [FUNCTION_NAME] function. Add it to @all_operations.py
Test correctness with: python plot.py --op [OPNAME]
Optionally test performance with: python benchmark.py --op [OPNAME]
```

For challenging cases requiring accuracy analysis:
```
Check accuracy at @test/plot/json/
Check performanc at @test/benchmark/json/
```

This approach has successfully generated 10+ mathematical operations including:
- Activation functions: `softplus`, `swish`, `selu`, `softsign`
- Hyperbolic functions: `sinh`, `cosh`, `atanh` (multiple variants)
- Hard functions: `hardsigmoid`, `hardswish`, `hardtan`
- Exponential variants: `exp_approx`, `exp_scaled`, `exp_approx_fast`

## Design Philosophy: LLM-First Development

This repository embraces a paradigm shift optimized for LLM capabilities rather than traditional human development practices:

### Human vs LLM Optimization

**Humans are smart and lazy** - they naturally minimize effort by building abstractions, templates, and reusable components. This leads to:
- Complex inheritance hierarchies
- Templated generic solutions
- Shared dependencies and tight coupling
- DRY (Don't Repeat Yourself) principles

**LLMs are persistent and literal** - they excel at pattern matching and repetitive tasks but struggle with complex abstractions. This suggests:
- Explicit, self-contained implementations
- Minimal external dependencies
- Duplicated code over shared complexity
- WET (Write Everything Twice) acceptance

### Practical Implications

For kernel development, this philosophy proves advantageous:

1. **Shape/Datatype Specialization**: Instead of one templated kernel handling all cases, LLMs can generate specialized kernels for each combination of shape, datatype, and operation parameters.

2. **Dependency Elimination**: Rather than importing complex libraries, each kernel contains all necessary mathematical functions inline.

3. **Explicit Over Generic**: Direct implementations are preferred over parameterized abstractions, even if this means more total code.

4. **Pattern Replication**: LLMs can efficiently generate hundreds of similar kernels with small variations, something humans would abstract away but which provides optimal performance.

5. **Precise Modification Opportunities**: Keeping all dependencies in one place and untied from other components enables targeted optimizations. For example, in `cosh`/`sinh` implementations, the LLM created a specialized `exp(|x|)` kernel and calculates both `e^|x|` and `e^(-|x|)` at the same cost. This structural optimization would be difficult in tightly coupled, shared code.

This approach trades traditional software engineering principles for LLM compatibility and specialization benefits - a worthwhile exchange in performance-critical domains where custom optimization matters more than code maintainability.

## Tools Ecosystem

### Plot Tool (`plot.py`)

The plotting tool serves dual purposes: **human visualization** and **LLM analysis**.

#### Key Features:
- **Top-Down/Hierarchical Sampling**: Creates data points from coarse to fine detail (endpoints → midpoints → quarter-points, etc.)
- **Dense Visualization**: Uses 10,000 points for smooth human-readable plots
- **Sparse LLM Data**: Generates 1,024 hierarchically ordered points for efficient LLM analysis

#### Usage:
```bash
python plot.py --op SOFTPLUS    # Test specific operation
python plot.py --all           # Test all operations
```

#### Output:
- **PNG files** (`test/plot/png/`): 7-panel comparison plots showing:
  - Function comparison (PyTorch vs TTNN vs Generic)
  - Raw difference plots (3 comparisons)
  - Normalized error plots (3 comparisons with precision thresholds)
- **JSON files** (`test/plot/json/`): LLM-friendly hierarchical data format

#### Why This Format is LLM-Friendly:

1. **Hierarchical Structure**: Data organized from coarse to fine detail allows LLMs to quickly identify major issues before examining fine-grained errors
2. **Value Proximity**: Each data point contains all implementation results nearby, making it easy for LLMs to spot discrepancies
3. **Metadata First**: Statistics and error metrics at the top provide immediate context
4. **Top-Down Ordering**: Top-down approach progressing from broad patterns to specific details

Example JSON structure:
```json
[
  {
    "operation": "SOFTPLUS",
    "total_samples": 1024,
    "normalized_error_statistics": {...}
  },
  {
    "top_down_data": [
      {"input": -5.0, "torch": 0.006738, "ttnn": 0.006738, "generic": 0.006738},
      {"input": 5.0, "torch": 5.006738, "ttnn": 5.007812, "generic": 5.007812},
      {"input": 0.0, "torch": 0.693147, "ttnn": 0.693359, "generic": 0.693359},
      ...
    ]
  }
]
```

### Benchmark Tool (`benchmark.py`)

Performance testing tool that measures execution speed and precision across different tensor sizes.

#### Usage:
```bash
python benchmark.py --op EXP      # Benchmark specific operation  
python benchmark.py --all         # Benchmark all operations
```

#### Features:
- **Multi-size Testing**: Small (64×64), Medium (128×128), Large (256×256), Huge (512×512) tensors
- **Statistical Analysis**: Warmup iterations, confidence intervals, standard deviation
- **Precision Analysis**: Compares TTNN vs Generic implementations using normalized error metrics
- **Speedup Analysis**: Calculates relative performance between implementations

#### Output:
- **JSON files** (`test/benchmark/json/`): Detailed timing and precision data
- **PNG files** (`test/benchmark/png/`): Performance visualization plots
- **Console**: Real-time progress and summary statistics

### Precision Counter (`precision_counter.py`)

A sophisticated accuracy analysis tool that accounts for the **quantized nature** of low-precision floating-point formats.

#### Why Precision Counter is Critical:

Modern AI accelerators use quantized formats like `bfloat16` that have inherent precision limitations. Raw error metrics can be misleading because they don't account for:

1. **Quantization Errors**: The unavoidable error introduced by representing real numbers in limited precision
2. **Scale-Dependent Errors**: Errors that should be evaluated relative to the representable precision at each magnitude
3. **Format-Specific Limitations**: Different floating-point formats (float16, bfloat16, float32) have different precision characteristics

#### Key Features:

- **Normalized Error Calculation**: `error / max(input_quant_error, output_quant_error)`
- **Format-Aware Analysis**: Understands mantissa bits, exponent ranges, and machine epsilon for each data type
- **Quantization Modeling**: Simulates the actual quantization process to estimate unavoidable errors
- **Pass/Fail Thresholds**: Uses `PRECISION_PASS_THRESHOLD = 2.0` for normalized errors

#### Example: HARDTAN Analysis

The `hardtan.png` visualization demonstrates this perfectly:
- **Raw errors** might show concerning absolute differences
- **Normalized errors** reveal these differences are within expected quantization bounds
- **Threshold lines** at ±2.0 show the acceptable error range for bfloat16 precision

This prevents false alarms where implementations appear "inaccurate" but are actually limited by the fundamental precision of the number format being used.

## Repository Structure

```
plain/
├── elementwise_sfpu/
│   ├── compute/           # Individual operation kernels (.cpp)
│   └── general/           # Shared reader/writer/compute kernels
├── test/
│   ├── plot/
│   │   ├── json/          # LLM-friendly accuracy data
│   │   └── png/           # Human-friendly accuracy plots
│   └── benchmark/
│       ├── json/          # LLM-friendly performance data
│       └── png/           # Human-friendly performance plots
├── all_operations.py      # Operation registry and configuration
├── plot.py               # Visualization and analysis tool
├── benchmark.py          # Performance testing tool
├── precision_counter.py  # Accuracy analysis tool
└── generic_unary.py      # Kernel execution framework
```

## Usage Examples

### Testing a New Operation

1. **Implement the kernel** (manually or with LLM assistance)
2. **Add to registry**:
   ```python
   # In all_operations.py
   "NEW_OP": OperationTestUnary(
       operation_name="NEW_OP",
       input_range=(-5.0, 5.0),
       torch_func=torch.new_op,
       ttnn_func=ttnn.new_op,
       candidate_func=GenericUnary("elementwise_sfpu/compute/new_op.cpp")
   )
   ```
3. **Test correctness**:
   ```bash
   python plot.py --op NEW_OP
   ```
4. **Analyze accuracy** (check `test/plot/json/new_op.json` for detailed metrics)
5. **Test performance** (optional):
   ```bash
   python benchmark.py --op NEW_OP
   ```

### LLM Generation Workflow

The standardized prompts ensure consistent, high-quality kernel generation:

**Basic Generation**:
```
Given examples of exp and log @cbrt_eltwise_sfpu.cpp, implement softplus function. 
Add it to @all_operations.py
Test correctness with: python plot.py --op SOFTPLUS
```

**With Performance Testing**:
```
Given examples of exp and log @cbrt_eltwise_sfpu.cpp, implement swish function.
Add it to @all_operations.py  
Test correctness with: python plot.py --op SWISH
Test performance with: python benchmark.py --op SWISH
```

**For Accuracy-Critical Cases**:
```
Given examples of exp and log @cbrt_eltwise_sfpu.cpp, implement atanh function.
Add it to @all_operations.py
Test correctness with: python plot.py --op ATANH
Check accuracy at test/plot/json/atanh.json 
```

**For Device-Critical Operations** (operations that may cause device hangs):
```
Given examples of exp and log @cbrt_eltwise_sfpu.cpp, implement atanh function.
Add it to @all_operations.py

CRITICAL TESTING PROTOCOL:
1. Check for correctness with: timeout 10 python plot.py --op ATANH
2. ALWAYS use timeout 10 - if the operation times out, the device is frozen!
3. DO NOT INCREASE THE TIMEOUT. 10 seconds is enough to complete the operation. If 10 seconds is not enough - the device is frozen!
4. If the device is frozen:
- Reset the device: tt-smi --reset 0
- Check the device is recovered: python plot.py --op EXP
- Continue only after confirming that the device is working

DEBUGGING (if needed):
Enable kernel debug logs with evn var: TT_METAL_DPRINT_CORES=0,0 python plot.py --op ATAN("TT_METAL_DPRINT_CORES=all" for all cores)

Never run untested operations without timeout - frozen device require hardware reset
```
