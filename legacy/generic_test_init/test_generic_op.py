
import ttnn
import torch
from legacy.generic_test_init.generic_op import get_generic_op_descriptor

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

program_descriptor, output_tensor = get_generic_op_descriptor(input_tensor, device)

ttnn.generic_op([input_tensor, output_tensor], program_descriptor) # dispatch op
ttnn.synchronize_device(device) # to actually execute op
res_torch = ttnn.to_torch(output_tensor)
assert (res_torch - torch.exp(input_tensor_torch)).abs().mean() < 0.02
assert (res_torch - ttnn.to_torch(ttnn.exp(input_tensor))).abs().max().item() == 0.0
print('Test passed')
ttnn.close_device(device)