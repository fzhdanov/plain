// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace NAMESPACE {

// Unfolded version of hardtan_tile_init
inline void hardtan_tile_init_unfolded() {
    // No initialization needed for hardtan
}

// Unfolded version of calculate_hardtan
inline void _calculate_hardtan_unfolded(const int iterations) {
    // Get compile-time arguments for min_val and max_val (passed as integer bit representations)
    constexpr uint32_t min_val_bits = get_compile_time_arg_val(2);  // Third compile-time arg
    constexpr uint32_t max_val_bits = get_compile_time_arg_val(3);  // Fourth compile-time arg
    
    // Convert from integer bit representation back to float using union
    union { uint32_t i; float f; } min_converter = { min_val_bits };
    union { uint32_t i; float f; } max_converter = { max_val_bits };
    const float min_val = min_converter.f;
    const float max_val = max_converter.f;
    
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        
        // Implement hardtan: max(min_val, min(max_val, x))
        // This is equivalent to: clamp(x, min_val, max_val)
        sfpi::vFloat result = val;
        
        // Apply minimum clamp: max(min_val, x)
        v_if (result < min_val) {
            result = min_val;
        }
        v_endif;
        
        // Apply maximum clamp: min(max_val, x)
        v_if (result > max_val) {
            result = max_val;
        }
        v_endif;
        
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Define these functions for compatibility
inline void hardtan_tile_init() {
    hardtan_tile_init_unfolded();
}

inline void hardtan_tile(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_<false>(
        _calculate_hardtan_unfolded,
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
    hardtan_tile_init_unfolded();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply hardtan operation
        hardtan_tile(0);
        
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