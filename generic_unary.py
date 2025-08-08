import ttnn
from typing import List, Tuple, Optional, Dict, Any

def read_kernel(file_name):
    with open(file_name, 'r') as file:
        return file.read()

class GenericUnary:
    def __init__(self, compute_kernel_path: str, 
                 defines: List[Tuple[str, str]] = [],
                 extra_compile_args: Dict[str, Any] = None
                 ):
        self.compute_kernel_path = compute_kernel_path
        self.defines = defines
        self.extra_compile_args = extra_compile_args or {}
        self.program_descriptors_cache = {} # map shape to program descriptor

    def __call__(self, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor = None, inplace: bool = False):
        device = input_tensor.device()

        if inplace:
            assert output_tensor is None, "Output tensor must be None when inplace is True"
            output_tensor = input_tensor
            print(f"Inplace mode: {inplace}")
        else:
            if output_tensor is None:
                output_tensor = ttnn.allocate_tensor_on_device(
                    input_tensor.shape,
                    input_tensor.dtype,
                    input_tensor.layout,
                    device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
        shape = tuple(input_tensor.shape)
        # Create cache key that includes both shape and extra compile args
        cache_key = (shape, tuple(sorted(self.extra_compile_args.items())))
        if cache_key not in self.program_descriptors_cache:
            self.program_descriptors_cache[cache_key] = self.get_generic_op_descriptor(input_tensor, output_tensor, device)
       
        # generic_op will override runtime args (like tensor addresses) for the new tesors with the same shape
        program_descriptor = self.program_descriptors_cache[cache_key]
        return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)

    def get_generic_op_descriptor(self, input_tensor, output_tensor, device):
        TILE_SIZE = 32 * 32
        SIZE_OF_DATATYPE = 2 # for bfloat16
        num_tiles = input_tensor.volume() // TILE_SIZE

        compute_grid = device.compute_with_storage_grid_size()
        grid_x = compute_grid.x
        grid_y = compute_grid.y
        num_cores = grid_x * grid_y

        start_core = ttnn.CoreCoord(0, 0)
        end_core = ttnn.CoreCoord(grid_x - 1, grid_y - 1)
        full_grid_range = ttnn.CoreRange(start_core, end_core)
        core_grid = ttnn.CoreRangeSet([full_grid_range])

        base_tiles_per_core = num_tiles // num_cores
        extra_tiles = num_tiles % num_cores
        print(f"Num tiles: {num_tiles}")
        print(f"Num cores: {num_cores}")
        print(f"base tiles: {base_tiles_per_core}")
        print(f"extra tiles: {extra_tiles}")

        is_dram_input = 1  # Assume DRAM memory

        # CB setup
        cb_in_id = 0  # c_0 # the circular buffer expected by kernels (reader/writer/compute)
        cb_out_id = 16  # c_16 # the buffer circular expected by kernels  (reader/writer/compute)

        input_cb_data_format = input_tensor.dtype
        cb_page_size = SIZE_OF_DATATYPE * TILE_SIZE  # 2 bytes per bfloat16 mutliplyed by tyle size
        cb_total_size = 2 * cb_page_size # 2 tiles per buffer for double buffering

        in_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=cb_in_id,
            data_format=input_cb_data_format,
            page_size=cb_page_size
        )

        out_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=cb_out_id,
            data_format=input_cb_data_format,
            page_size=cb_page_size
        )

        in_cb_descriptor = ttnn.CBDescriptor(
            total_size=cb_total_size,
            core_ranges=core_grid,
            format_descriptors=[in_cb_format]
        )

        out_cb_descriptor = ttnn.CBDescriptor(
            total_size=cb_total_size,
            core_ranges=core_grid,
            format_descriptors=[out_cb_format]
        )

        # Create a 2D array of runtime args for all cores
        # Properly distribute work across cores - calculate tile distribution ONCE for all kernels
        reader_rt_args = []
        writer_rt_args = []
        compute_rt_args = []
        
        # Store tile counts per core to ensure consistency across all kernels
        tiles_per_core = []

        tile_start_id = 0
        core_id = 0
        extra_tiles_remaining = extra_tiles
        
        for y in range(grid_y):
            reader_row = []
            writer_row = []
            compute_row = []
            tiles_row = []

            for x in range(grid_x):
                # Calculate tiles for this specific core - SINGLE SOURCE OF TRUTH
                tiles_for_this_core = base_tiles_per_core
                if extra_tiles_remaining > 0:
                    tiles_for_this_core += 1
                    extra_tiles_remaining -= 1

                # Store tile count for this core
                tiles_row.append(tiles_for_this_core)

                # Add arguments for this core with its specific work range
                reader_row.append([input_tensor.buffer_address(), tiles_for_this_core, tile_start_id])
                writer_row.append([output_tensor.buffer_address(), tiles_for_this_core, tile_start_id])
                compute_row.append([tiles_for_this_core])  # Pass tile count directly to compute

                # Update the starting tile ID for the next core
                tile_start_id += tiles_for_this_core
                core_id += 1

            reader_rt_args.append(reader_row)
            writer_rt_args.append(writer_row)
            compute_rt_args.append(compute_row)
            tiles_per_core.append(tiles_row)

        reader_kernel = read_kernel("elementwise_sfpu/general/reader_kernel.cpp")
        writer_kernel = read_kernel("elementwise_sfpu/general/writer_kernel.cpp")
        compute_kernel = read_kernel(self.compute_kernel_path)

        # Define kernel descriptors
        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source=reader_kernel,
            source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
            core_ranges=core_grid,
            compile_time_args=[is_dram_input],
            runtime_args=reader_rt_args,
            config=ttnn.ReaderConfigDescriptor()
        )

        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source=writer_kernel,
            source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
            core_ranges=core_grid,
            compile_time_args=[cb_out_id, is_dram_input],
            runtime_args=writer_rt_args,
            config=ttnn.WriterConfigDescriptor()
        )

        # Calculate the maximum tiles any core will process
        max_tiles_per_core = base_tiles_per_core + (1 if extra_tiles > 0 else 0)

        per_core_block_cnt = max_tiles_per_core  # Max tiles for compile time
        per_core_block_dim = 1
        compile_time_args = [per_core_block_cnt, per_core_block_dim]
        
        # Add extra compile-time arguments if provided
        for key, value in self.extra_compile_args.items():
            # Convert float values to their bit representation as integers
            if isinstance(value, float):
                import struct
                compile_time_args.append(struct.unpack('I', struct.pack('f', value))[0])
            else:
                compile_time_args.append(value)

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source=compute_kernel,
            source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
            core_ranges=core_grid,
            compile_time_args=compile_time_args,
            defines=self.defines,
            runtime_args=compute_rt_args,  # Use the same args calculated above
            config=ttnn.ComputeConfigDescriptor()
        )
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
            semaphores=[],
            cbs=[in_cb_descriptor, out_cb_descriptor]
        )
        return program_descriptor
