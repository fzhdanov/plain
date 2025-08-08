import torch
import ttnn
from generic_unary import GenericUnary
from typing import Tuple, Callable


class OperationTestUnary:
    def __init__(self, operation_name: str, input_range: Tuple[float, float], 
                 torch_func: Callable, ttnn_func: Callable, candidate_func: Callable):
        self.operation_name = operation_name
        self.input_range = input_range
        self.torch_func = torch_func
        self.ttnn_func = ttnn_func
        self.candidate_func = candidate_func

ALL_OPERATIONS = {
    "EXP": OperationTestUnary(
        operation_name="EXP ELTWISE SFPU",
        input_range=(-6.0, 6.0),
        torch_func=torch.exp,
        ttnn_func=ttnn.exp,
        candidate_func=GenericUnary(
            "elementwise_sfpu/general/compute_kernel.cpp",
            [
                ("SFPU_OP_EXP_INCLUDE", "1"),
                ("SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);")
            ]
        )
    ),
    "ATANH": OperationTestUnary(
        operation_name="ATANH", 
        input_range=(-0.99, 0.99),
        torch_func=torch.atanh,
        ttnn_func=ttnn.atanh,
        candidate_func=GenericUnary("elementwise_sfpu/compute/atanh.cpp")
    ),
    "ATANH2": OperationTestUnary(
        operation_name="ATANH2",
        input_range=(-0.99, 0.99),
        torch_func=torch.atanh,
        ttnn_func=ttnn.atanh,
        candidate_func=GenericUnary("elementwise_sfpu/compute/atanh2.cpp")
    ),
    "ATANH3": OperationTestUnary(
        operation_name="ATANH3",
        input_range=(-0.99, 0.99),
        torch_func=torch.atanh,
        ttnn_func=ttnn.atanh,
        candidate_func=GenericUnary("elementwise_sfpu/compute/atanh3.cpp")
    ),
    "COSH": OperationTestUnary(
        operation_name="COSH",
        input_range=(-4.0, 4.0),
        torch_func=torch.cosh,
        ttnn_func=ttnn.cosh,
        candidate_func=GenericUnary("elementwise_sfpu/compute/cosh.cpp")
    ),
    "SINH": OperationTestUnary(
        operation_name="SINH",
        input_range=(-4.0, 4.0),
        torch_func=torch.sinh,
        ttnn_func=ttnn.sinh,
        candidate_func=GenericUnary("elementwise_sfpu/compute/sinh.cpp")
    ),
    "SOFTSIGN": OperationTestUnary(
        operation_name="SOFTSIGN",
        input_range=(-8.0, 8.0),
        torch_func=torch.nn.functional.softsign,
        ttnn_func=ttnn.softsign,
        candidate_func=GenericUnary("elementwise_sfpu/compute/softsign.cpp")
    ),
    "SWISH": OperationTestUnary(
        operation_name="SWISH",
        input_range=(-5.0, 5.0),
        torch_func=lambda x: x * torch.sigmoid(x),
        ttnn_func=ttnn.swish,
        candidate_func=GenericUnary("elementwise_sfpu/compute/swish.cpp")
    ),
    "CBRT": OperationTestUnary(
        operation_name="CBRT",
        input_range=(-4.0, 4.0),
        torch_func=lambda x: torch.sign(x) * torch.abs(x) ** (1/3),
        ttnn_func=ttnn.cbrt,
        candidate_func=GenericUnary("elementwise_sfpu/compute/cbrt.cpp")
    ),
    "SOFTPLUS": OperationTestUnary(
        operation_name="SOFTPLUS",
        input_range=(-5.0, 5.0),
        torch_func=torch.nn.functional.softplus,
        ttnn_func=ttnn.softplus,
        candidate_func=GenericUnary("elementwise_sfpu/compute/softplus.cpp")
    ),
    "EXP_PRECISE": OperationTestUnary(
        operation_name="EXP_PRECISE",
        input_range=(-4.0, 4.0),
        torch_func=torch.exp,
        ttnn_func=ttnn.exp,
        candidate_func=GenericUnary("elementwise_sfpu/compute/exp.cpp")
    ),
    "HARDSIGMOID": OperationTestUnary(
        operation_name="HARDSIGMOID",
        input_range=(-5.0, 5.0),
        torch_func=lambda x: torch.nn.functional.hardsigmoid(x),
        ttnn_func=ttnn.hardsigmoid,
        candidate_func=GenericUnary("elementwise_sfpu/compute/hardsigmoid.cpp")
    ),
    "HARDSIGMOID_LUT": OperationTestUnary(
        operation_name="HARDSIGMOID_LUT",
        input_range=(-20.0, 20.0),
        torch_func=lambda x: torch.nn.functional.hardsigmoid(x),
        ttnn_func=ttnn.hardsigmoid,
        candidate_func=GenericUnary("elementwise_sfpu/compute/hardsigmoid_lut.cpp")
    ),
    "SELU": OperationTestUnary(
        operation_name="SELU",
        input_range=(-5.0, 5.0),
        torch_func=lambda x: torch.nn.functional.selu(x),  # PyTorch SELU uses fixed alpha=1.6732, scale=1.0507
        ttnn_func=ttnn.selu, 
        candidate_func=GenericUnary("elementwise_sfpu/compute/selu.cpp")
    ),
    "TANH": OperationTestUnary(
        operation_name="TANH",
        input_range=(-4.0, 4.0),
        torch_func=torch.tanh,
        ttnn_func=ttnn.tanh,
        candidate_func=GenericUnary("elementwise_sfpu/compute/tanh.cpp")
    ),
    "SWISH2": OperationTestUnary(
        operation_name="SWISH2",
        input_range=(-5.0, 5.0),
        torch_func=lambda x: x * torch.sigmoid(x),
        ttnn_func=ttnn.swish,
        candidate_func=GenericUnary("elementwise_sfpu/compute/swish2.cpp")
    ),
    "HARDSIGMOID_LUT2": OperationTestUnary(
        operation_name="HARDSIGMOID_LUT2",
        input_range=(-20.0, 20.0),
        torch_func=lambda x: torch.nn.functional.hardsigmoid(x),
        ttnn_func=ttnn.hardsigmoid,
        candidate_func=GenericUnary("elementwise_sfpu/compute/hardsigmoid_lut2.cpp")
    ),
    "EXP_APPROX": OperationTestUnary(
        operation_name="EXP_APPROX",
        input_range=(-4.0, 4.0),
        torch_func=torch.exp,
        ttnn_func=ttnn.exp,
        candidate_func=GenericUnary("elementwise_sfpu/compute/exp_approx.cpp")
    ),
    "EXP_APPROX_FAST": OperationTestUnary(
        operation_name="EXP_APPROX_FAST",
        input_range=(-4.0, 4.0),
        torch_func=torch.exp,
        ttnn_func=ttnn.exp,
        candidate_func=GenericUnary("elementwise_sfpu/compute/exp_approx_fast.cpp")
    ),
    "EXP_APPROX_SKIP_CHECK": OperationTestUnary(
        operation_name="EXP_APPROX_SKIP_CHECK",
        input_range=(-4.0, 4.0),
        torch_func=torch.exp,
        ttnn_func=ttnn.exp,
        candidate_func=GenericUnary("elementwise_sfpu/compute/exp_approx_skip_check.cpp")
    ),
    "EXP_SCALED": OperationTestUnary(
        operation_name="EXP_SCALED",
        input_range=(-4.0, 4.0),
        torch_func=torch.exp,
        ttnn_func=ttnn.exp,
        candidate_func=GenericUnary("elementwise_sfpu/compute/exp_scaled.cpp")
    ),
    "HARDSWISH": OperationTestUnary(
        operation_name="HARDSWISH",
        input_range=(-6.0, 6.0),
        torch_func=torch.nn.functional.hardswish,
        ttnn_func=ttnn.hardswish,
        candidate_func=GenericUnary("elementwise_sfpu/compute/hardswish.cpp")
    ),
    "HARDTAN": OperationTestUnary(
        operation_name="HARDTAN",
        input_range=(-5.0, 5.0),
        torch_func=lambda x: torch.clamp(x, -4.5, 4.5),  # Standard hardtan: clamp between -1 and 1
        ttnn_func=lambda x: ttnn.clamp(x, -4.5, 4.5),  # Use ttnn.clamp as reference
        candidate_func=GenericUnary(
            "elementwise_sfpu/compute/hardtan.cpp",
            extra_compile_args={"min_val": -4.5, "max_val": 4.5}
        )
    ),
    "HARDTAN_DEFINES": OperationTestUnary(
        operation_name="HARDTAN_DEFINES",
        input_range=(-4.0, 4.0),
        torch_func=lambda x: torch.clamp(x, -1.5, 2.0),  # Different clamp values for testing
        ttnn_func=lambda x: ttnn.clamp(x, -1.5, 2.0),  # Use ttnn.clamp as reference
        candidate_func=GenericUnary(
            "elementwise_sfpu/compute/hardtan_defines.cpp",
            defines=[
                ("HARDTAN_MIN_VAL", "-1.5f"),
                ("HARDTAN_MAX_VAL", "2.0f")
            ]
        )
    ),
}