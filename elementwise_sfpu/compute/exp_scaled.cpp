// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"


namespace NAMESPACE {

// Unfolded version of exp_tile_init<false, true>() for scaled version
inline void exp_tile_init_scaled() {
    // Set constants for exponential calculation with scaling
    // These match the constants set in _init_exponential_<false, true>()
    sfpi::vConstFloatPrgm0 = 1.442695f;  // ln2_recip
    sfpi::vConstFloatPrgm1 = 2.0f;
    sfpi::vConstFloatPrgm2 = 0.863281f;
}

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    sfpi::vInt exp = exexp(val);
    v_if (exp >= 0)
    {
        val = setexp(val, 126);
    }
    v_endif;

    // Run series in Horner form
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);
    val              = val * tmp + sfpi::vConst1;

    v_if (exp >= 0)
    {
        val = val * val;
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;
            // Narrow predication on each loop
            v_and(exp >= 0);
            val = val * val;
        }
    }
    v_endif;

    return val;
}

// Unfolded version of _calculate_exponential_<false, true, 8, true, false>
inline void _calculate_exponential_scaled(const int iterations, uint16_t exp_base_scale_factor = 0x3F80) {
    // This is the unfolded version with the following parameters:
    // APPROXIMATION_MODE = false
    // SCALE_EN = true
    // ITERATIONS = 8
    // FAST_APPROX = true
    // SKIP_POSITIVE_CHECK = false
    
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        
        // Apply scaling since SCALE_EN is true
        val = val * sfpi::s2vFloat16b(exp_base_scale_factor);
        
        // APPROXIMATION_MODE is false, so we execute this branch
        sfpi::vFloat result = _sfpu_exp_(sfpi::setsgn(val, 0));

        v_if (val < 0) {
            result = ckernel::sfpu::_sfpu_reciprocal_(result);
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Unfolded version of exp_tile<false, true, true>(0)
inline void exp_tile_scaled(uint32_t idst, uint16_t scale_factor = 0x3F80) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    llk_math_eltwise_unary_sfpu_params<false>(
        [scale_factor](const int iterations) { _calculate_exponential_scaled(iterations, scale_factor); },
        idst,
        (int)VectorMode::RC,
        8);
#endif
}

void MAIN {
    // Get the actual number of tiles for this core from runtime args
    uint32_t actual_tiles_for_this_core = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    // Use a default scale factor instead of trying to get it from compile time args
    uint16_t scale_factor = 0x3F80; // Default to 1.0 in BF16

    DPRINT_MATH(DPRINT << "Compute core (" << (uint32_t)get_absolute_logical_x() << "," 
                << (uint32_t)get_absolute_logical_y() << ") processing " 
                << actual_tiles_for_this_core << " tiles" << ENDL());

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    exp_tile_init_scaled();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply exponent operation with scaling
        exp_tile_scaled(0, scale_factor);
        
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