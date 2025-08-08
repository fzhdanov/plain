import ttnn
import torch
from generic_unary import GenericUnary

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--shape', type=str, required=True)
args = parser.parse_args()

print(f'Testing {args.shape}')
input_tensor_torch = torch.randn(*[int(x) for x in args.shape.split(',')])

input_tensor = ttnn.from_torch(
                input_tensor_torch, 
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, 
                device=device, 
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

# Create GenericUnary instance for exp operation
exp_op = GenericUnary(
    compute_kernel_path="elementwise_sfpu/general/compute_kernel.cpp",
    defines=[
        ("SFPU_OP_EXP_INCLUDE", "1"),
        ("SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);")
    ]
)

# Execute the operation
output_tensor = exp_op(input_tensor)
ttnn.synchronize_device(device) # to actually execute op

res_torch = ttnn.to_torch(output_tensor)
assert (res_torch - torch.exp(input_tensor_torch)).abs().mean() < 0.02
assert (res_torch - ttnn.to_torch(ttnn.exp(input_tensor))).abs().max().item() == 0.0
print('Test passed')
ttnn.close_device(device)