import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union

# Normalized error threshold for pass/fail determination
PRECISION_PASS_THRESHOLD = 2.0


class PrecisionCounter:
    """
    A class that performs nature-aware precision counting by tracking quantization errors
    for different data types and calculating normalized errors.
    
    This class computes quantization errors that arise from different floating-point
    representations (float16, bfloat16, float32) and calculates error metrics that
    account for the inherent precision limitations of each data type.
    """
    
    def __init__(self):
        # Machine epsilon values for different data types
        self.eps_dict = {
            torch.float16: 9.77e-04,    # half precision
            torch.bfloat16: 3.91e-03,   # brain floating point
            torch.float32: 1.19e-07,    # single precision
            torch.float64: 2.22e-16,    # double precision
        }
        
        # Mantissa bits for different data types
        self.mantissa_bits = {
            torch.float16: 10,
            torch.bfloat16: 7,
            torch.float32: 23,
            torch.float64: 52,
        }
        
        # Exponent bits for different data types
        self.exponent_bits = {
            torch.float16: 5,
            torch.bfloat16: 8,
            torch.float32: 8,
            torch.float64: 11,
        }
    
    def _get_quantization_error(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Calculate the quantization error for a given tensor and data type.
        
        This implements a more accurate quantization error model that properly scales
        near zero to create growing "boxes" in normalized error plots. The model uses:
        1. Relative error for larger values (proportional to magnitude)
        2. A smooth transition to absolute error near zero
        3. Proper scaling that grows as values approach zero
        
        Args:
            tensor: Input tensor (assumed to be in float32)
            dtype: Target data type for quantization
            
        Returns:
            Quantization error tensor
        """
        # Convert to float32 if not already
        tensor_f32 = tensor.float()
        abs_tensor = torch.abs(tensor_f32)
        
        # Define quantization parameters based on actual floating-point characteristics
        quantization_params = {
            torch.float16: {
                'mantissa_bits': 10,
                'min_normal': 2**(-14),  # Minimum normal number
                'relative_eps': 2**(-10),  # 2^(-mantissa_bits)
                'subnormal_scale': 2**(-24),  # 2^(-(mantissa_bits + exponent_bias))
            },
            torch.bfloat16: {
                'mantissa_bits': 7,
                'min_normal': 2**(-126),  # Minimum normal number  
                'relative_eps': 2**(-7),   # 2^(-mantissa_bits)
                'subnormal_scale': 2**(-133),  # 2^(-(mantissa_bits + exponent_bias))
            },
            torch.float32: {
                'mantissa_bits': 23,
                'min_normal': 2**(-126),
                'relative_eps': 2**(-23),
                'subnormal_scale': 2**(-149),
            },
            torch.float64: {
                'mantissa_bits': 52,
                'min_normal': 2**(-1022),
                'relative_eps': 2**(-52),
                'subnormal_scale': 2**(-1074),
            },
        }
        
        # Get parameters for this data type
        params = quantization_params.get(dtype, quantization_params[torch.float32])
        relative_eps = params['relative_eps']
        min_normal = params['min_normal']
        subnormal_scale = params['subnormal_scale']
        
        # Calculate quantization error with proper near-zero behavior
        # For normal numbers: error = |value| * relative_eps
        # For subnormal region: error scales up as we approach zero
        
        # Relative error component (dominates for larger values)
        relative_error = abs_tensor * relative_eps
        
        # Near-zero scaling component that grows as values approach zero
        # This creates the "growing boxes" effect you want to see
        # Use a smooth transition function that scales up near zero
        zero_scale_factor = torch.where(
            abs_tensor < min_normal,
            # In subnormal region: error grows as 1/|value| but clamped
            torch.clamp(min_normal / torch.clamp(abs_tensor, min=subnormal_scale), 
                       min=1.0, max=100.0),
            # In normal region: no additional scaling
            torch.ones_like(abs_tensor)
        )
        
        # Combine relative error with near-zero scaling
        # The scaling factor makes errors grow near zero, creating the "boxes"
        scaled_relative_error = relative_error * zero_scale_factor
        
        # Ensure minimum error floor to prevent division by zero
        min_error = subnormal_scale * 10  # Small but non-zero floor
        
        quant_error = torch.maximum(scaled_relative_error, 
                                   torch.full_like(abs_tensor, min_error))
        
        return quant_error
    
    def _simulate_quantization(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Simulate quantization by converting to the target dtype and back to float32.
        
        Args:
            tensor: Input tensor in float32
            dtype: Target data type for quantization
            
        Returns:
            Quantized tensor converted back to float32
        """
        if dtype == torch.float32:
            return tensor.float()
        
        # Convert to target dtype and back to float32
        try:
            quantized = tensor.to(dtype).float()
        except Exception:
            # If conversion fails, return original tensor
            quantized = tensor.float()
            
        return quantized
    

    
    def calculate_precision_metrics(
        self,
        input_tensor: torch.Tensor,
        output_true: torch.Tensor,
        output_eval: torch.Tensor,
        dtype: torch.dtype
    ) -> Dict[str, float]:
        """
        Calculate precision-aware error metrics.
        
        Args:
            input_tensor: Input tensor
            output_true: True/reference output tensor
            output_eval: Evaluated/computed output tensor
            dtype: Data type used in computation
            
        Returns:
            Dictionary containing various error metrics
        """
        # Convert all tensors to float32 for calculation
        input_f32 = input_tensor.float()
        output_true_f32 = output_true.float()
        output_eval_f32 = output_eval.float()
        
        # Calculate quantization errors
        input_quant_error = self._get_quantization_error(input_f32, dtype)
        output_true_quant_error = self._get_quantization_error(output_true_f32, dtype)
        output_eval_quant_error = self._get_quantization_error(output_eval_f32, dtype)
        
        # Calculate raw difference
        raw_diff = torch.abs(output_eval_f32 - output_true_f32)
        
        # Calculate normalized error (scaled by max quantization error)
        max_quant_error = torch.maximum(input_quant_error, output_true_quant_error)
        normalized_error = raw_diff / torch.clamp(max_quant_error, min=1e-10)
        
        # Calculate statistics (use only finite values)
        def safe_mean(tensor):
            finite_mask = torch.isfinite(tensor)
            if finite_mask.sum() > 0:
                return tensor[finite_mask].mean().item()
            return float('nan')
        
        def safe_max(tensor):
            finite_mask = torch.isfinite(tensor)
            if finite_mask.sum() > 0:
                return tensor[finite_mask].max().item()
            return float('nan')
        
        def safe_std(tensor):
            finite_mask = torch.isfinite(tensor)
            if finite_mask.sum() > 0:
                return tensor[finite_mask].std().item()
            return float('nan')
        
        metrics = {
            # Raw error metrics
            'raw_mean_abs_error': safe_mean(raw_diff),
            'raw_max_abs_error': safe_max(raw_diff),
            'raw_std_error': safe_std(raw_diff),
            
            # Normalized error metrics (scaled by max quantization error)
            'normalized_error_mean': safe_mean(normalized_error),
            'normalized_error_max': safe_max(normalized_error),
            'normalized_error_std': safe_std(normalized_error),
            
            # Quantization error statistics
            'input_quant_error_mean': safe_mean(input_quant_error),
            'output_quant_error_mean': safe_mean(output_true_quant_error),
            
            # Data type information
            'dtype_str': str(dtype),
            'mantissa_bits': self.mantissa_bits.get(dtype, -1),
            'exponent_bits': self.exponent_bits.get(dtype, -1),
            'machine_epsilon': self.eps_dict.get(dtype, float('nan')),
        }
        
        return metrics
    
    def compare_dtypes(
        self,
        input_tensor: torch.Tensor,
        output_true: torch.Tensor,
        output_eval: torch.Tensor,
        dtypes: list = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare precision metrics across multiple data types.
        
        Args:
            input_tensor: Input tensor
            output_true: True/reference output tensor
            output_eval: Evaluated/computed output tensor
            dtypes: List of data types to compare (default: [float16, bfloat16, float32])
            
        Returns:
            Dictionary with dtype names as keys and metrics as values
        """
        if dtypes is None:
            dtypes = [torch.float16, torch.bfloat16, torch.float32]
        
        results = {}
        
        for dtype in dtypes:
            dtype_name = str(dtype).split('.')[-1]  # Extract 'float16' from 'torch.float16'
            results[dtype_name] = self.calculate_precision_metrics(
                input_tensor, output_true, output_eval, dtype
            )
        
        return results
    
    def generate_report(
        self,
        input_tensor: torch.Tensor,
        output_true: torch.Tensor,
        output_eval: torch.Tensor,
        dtype: torch.dtype,
        operation_name: str = "Unknown"
    ) -> str:
        """
        Generate a human-readable report of precision metrics.
        
        Args:
            input_tensor: Input tensor
            output_true: True/reference output tensor
            output_eval: Evaluated/computed output tensor
            dtype: Data type used in computation
            operation_name: Name of the operation being analyzed
            
        Returns:
            Formatted report string
        """
        metrics = self.calculate_precision_metrics(input_tensor, output_true, output_eval, dtype)
        
        report = f"""
=== Precision Analysis Report ===
Operation: {operation_name}
Data Type: {metrics['dtype_str']}
Mantissa Bits: {metrics['mantissa_bits']}
Exponent Bits: {metrics['exponent_bits']}
Machine Epsilon: {metrics['machine_epsilon']:.2e}

Raw Error Metrics:
  Mean Absolute Error: {metrics['raw_mean_abs_error']:.6e}
  Max Absolute Error:  {metrics['raw_max_abs_error']:.6e}
  Std Deviation:       {metrics['raw_std_error']:.6e}

Normalized Error Metrics (Max-based):
  Mean: {metrics['normalized_error_mean']:.4f}
  Max:  {metrics['normalized_error_max']:.4f}
  Std:  {metrics['normalized_error_std']:.4f}

Quantization Error Estimates:
  Input Error (mean):     {metrics['input_quant_error_mean']:.6e}
  Output Error (mean):    {metrics['output_quant_error_mean']:.6e}
=================================
        """
        
        return report.strip()

    def calculate_normalized_error_stats(
        self, 
        input_tensor: torch.Tensor, 
        reference_tensor: torch.Tensor, 
        test_tensor: torch.Tensor, 
        dtype: torch.dtype = torch.bfloat16
    ) -> Dict[str, float]:
        """
        Calculate normalized error statistics using the same approach as plot_comparison.py.
        
        This method:
        1. Calculates raw differences between test and reference tensors
        2. Computes quantization errors for input and reference tensors
        3. Normalizes differences by max(input_quant_error, output_quant_error)
        4. Returns statistics on the normalized errors
        
        Args:
            input_tensor: Original input tensor (float32)
            reference_tensor: Reference/ground truth output (float32) 
            test_tensor: Test output to compare (float32)
            dtype: Data type to use for quantization error calculation
            
        Returns:
            Dictionary with raw and normalized error statistics
        """
        # Calculate raw differences
        diff = test_tensor - reference_tensor
        raw_abs_diff = torch.abs(diff)
        raw_mean_abs_error = torch.mean(raw_abs_diff).item()
        raw_max_abs_error = torch.max(raw_abs_diff).item()
        
        # Calculate quantization errors (same as plot_comparison.py)
        input_quant_error = self._get_quantization_error(input_tensor, dtype)
        output_quant_error = self._get_quantization_error(reference_tensor, dtype)
        max_quant_error = torch.maximum(input_quant_error, output_quant_error)
        
        # Calculate normalized error (same as plot_comparison.py)
        normalized_diff = diff / torch.clamp(max_quant_error, min=1e-10)
        finite_mask = torch.isfinite(normalized_diff)
        
        if torch.sum(finite_mask) > 0:
            valid_norm_diff = normalized_diff[finite_mask]
            abs_norm_diff = torch.abs(valid_norm_diff)
            normalized_error_mean = torch.mean(abs_norm_diff).item()
            normalized_error_max = torch.max(abs_norm_diff).item()
        else:
            normalized_error_mean = 0.0
            normalized_error_max = 0.0
        
        return {
            'raw_mean_abs_error': raw_mean_abs_error,
            'raw_max_abs_error': raw_max_abs_error,
            'normalized_error_mean': normalized_error_mean,
            'normalized_error_max': normalized_error_max
        }


# Example usage and test functions
def test_precision_counter():
    """Test the PrecisionCounter class with a simple example."""
    
    # Create test data
    input_vals = torch.linspace(-8.0, 8.0, 10000)
    output_true = torch.exp(input_vals)  # True exponential
    
    output_eval = torch.exp(input_vals.to(torch.bfloat16)).float()
    
    # Create precision counter
    counter = PrecisionCounter()
    
    # Test with bfloat16
    print("Testing with bfloat16:")
    metrics = counter.calculate_precision_metrics(
        input_vals, output_true, output_eval, torch.bfloat16
    )
    
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # Generate report
    report = counter.generate_report(
        input_vals, output_true, output_eval, torch.bfloat16, "EXP"
    )
    print(report)
    
    print("\n" + "="*50 + "\n")
    
    # Compare across data types
    comparison = counter.compare_dtypes(
        input_vals, output_true, output_eval
    )
    
    print("Comparison across data types:")
    for dtype_name, metrics in comparison.items():
        print(f"\n{dtype_name}:")
        print(f"  Raw MAE: {metrics['raw_mean_abs_error']:.6e}")
        print(f"  Machine Epsilon: {metrics['machine_epsilon']:.2e}")


if __name__ == "__main__":
    test_precision_counter()