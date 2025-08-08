// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace NAMESPACE {

// Hardswish activation function implementation
// Hardswish(x) = x * hardsigmoid(x)
// where hardsigmoid(x) = max(0, min(1, (x + 3) / 6))

// Unfolded version of hardswish_tile_init()
inline void hardswish_tile_init_unfolded() {
    // No constants needed - using hardcoded values in functions
}

// Hardsigmoid implementation: max(0, min(1, (x + 3) / 6))
sfpi_inline sfpi::vFloat _sfpu_hardsigmoid_(sfpi::vFloat x)
{
    // Calculate (x + 3) / 6 using multiplication by 1/6
    sfpi::vFloat three = sfpi::vFloat(3.0f);
    sfpi::vFloat one_sixth = sfpi::vFloat(1.0f / 6.0f);  // 1/6
    sfpi::vFloat one = sfpi::vFloat(1.0f);
    sfpi::vFloat zero = sfpi::vFloat(0.0f);
    
    sfpi::vFloat x_plus_3 = x + three;
    sfpi::vFloat result = x_plus_3 * one_sixth;
    
    // Apply max(0, min(1, result))
    // First apply min(1, result)
    v_if (result > one) {
        result = one;
    }
    v_endif;
    
    // Then apply max(0, result)
    v_if (result < zero) {
        result = zero;
    }
    v_endif;
    
    return result;
}

// Hardswish implementation: x * hardsigmoid(x)
sfpi_inline sfpi::vFloat _sfpu_hardswish_(sfpi::vFloat x)
{
    sfpi::vFloat hardsigmoid_x = _sfpu_hardsigmoid_(x);
    sfpi::vFloat result = x * hardsigmoid_x;
    return result;
}

// Unfolded version of calculate_hardswish
inline void _calculate_hardswish_unfolded(const int iterations) {
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        
        // Calculate hardswish using the fused implementation
        sfpi::vFloat result = _sfpu_hardswish_(val);
        
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Define these functions for compatibility
inline void hardswish_tile_init() {
    hardswish_tile_init_unfolded();
}

inline void hardswish_tile(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    llk_math_eltwise_unary_sfpu_params<false>(
        _calculate_hardswish_unfolded,
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
    hardswish_tile_init_unfolded();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply hardswish operation
        hardswish_tile(0);
        
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