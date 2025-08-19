// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Swish2: Alternative Swish Implementation Using LLK Sigmoid
// ========================================================
//
// This implementation provides an alternative swish activation function that uses
// the official LLK (Low-Level Kernel) sigmoid implementation instead of a custom
// exponential-based sigmoid.
//
// Swish function: f(x) = x * sigmoid(x)
//
// Key differences from original swish_eltwise_sfpu.cpp:
// 1. Uses LLK's piecewise linear sigmoid approximation (6-piece lookup table)
// 2. Leverages hardware-optimized constants and approximations
// 3. Consistent with other TTNN sigmoid operations
// 4. Potentially better numerical stability and accuracy
// 5. No custom exponential or reciprocal implementations needed
//
// The LLK sigmoid uses:
// - Piecewise linear approximation for different ranges
// - Anti-symmetric property: sigmoid(-x) = 1 - sigmoid(x)
// - Optimized constants loaded via _sfpu_load_imm32_
//

#include <limits>
#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_sigmoid.h"

namespace NAMESPACE {

// Initialize LLK sigmoid constants
inline void swish2_tile_init() {
    // Initialize LLK sigmoid - this loads the necessary constants for sigmoid
    ckernel::sfpu::sigmoid_init<false>();
}

// Swish2 calculation using LLK sigmoid
inline void _calculate_swish2_llk(const int iterations) {
    // Use the LLK sigmoid piecewise linear implementation directly
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat original_input = sfpi::dst_reg[0];
        sfpi::vFloat val = original_input;
        sfpi::vFloat result = 0.0f;

        // Apply the LLK sigmoid calculation logic directly
        // This is based on the sigmoid implementation from ckernel_sfpu_sigmoid.h
        // The LLK uses anti-symmetric property: sigmoid[-x] = 1 - sigmoid[x]
        
        // Get absolute value for piecewise calculation
        v_if(val < 0.0f) { val = -val; }
        v_endif;

        // Use the LLK sigmoid piecewise linear approximation for positive values
        result = ckernel::sfpu::sigmoid_piecewise_linear_positive(val);

        // Apply anti-symmetric property for negative inputs
        val = original_input;
        v_if(val < 0.0f) { result = 1.0f - result; }
        v_endif;

        // Now compute swish: x * sigmoid(x)
        sfpi::vFloat swish_result = original_input * result;
        
        sfpi::dst_reg[0] = swish_result;
        sfpi::dst_reg++;
    }
}

// Swish2 tile function
inline void swish2_tile(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_<false>(
        _calculate_swish2_llk,
        idst,
        (int)VectorMode::RC,
        8);
#endif
}

void MAIN {
    // Get the actual number of tiles for this core from runtime args
    uint32_t actual_tiles_for_this_core = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    DPRINT_MATH(DPRINT << "Compute core (" << (uint32_t)get_absolute_logical_x() << "," 
                << (uint32_t)get_absolute_logical_y() << ") processing " 
                << actual_tiles_for_this_core << " tiles" << ENDL());

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    swish2_tile_init();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply swish2 operation using LLK sigmoid
        swish2_tile(0);
        
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);

        cb_pop_front(tt::CBIndex::c_0, 1);
        tile_regs_release();
        cb_push_back(tt::CBIndex::c_16, per_core_block_dim);
    }
    
    DPRINT_MATH(DPRINT << "Compute core (" << (uint32_t)get_absolute_logical_x() << "," 
                << (uint32_t)get_absolute_logical_y() << ") finished processing " 
                << actual_tiles_for_this_core << " tiles" << ENDL());
}
}  // namespace NAMESPACE