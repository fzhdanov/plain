// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"


namespace NAMESPACE {

// Unfolded version of exp_tile_init<false, true>()
inline void exp_tile_init_unfolded() {
    // Set constants for exponential calculation
    // These match the constants set in _init_exponential_<false, true>()
    // These constants are expected by _sfpu_reciprocal_ when handling negative inputs
    sfpi::vConstFloatPrgm0 = 1.442695f;  // ln2_recip
    sfpi::vConstFloatPrgm1 = 2.0f;
    sfpi::vConstFloatPrgm2 = 0.863281f;
}

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat x)
{
    // Handle only abs(x) and calculate inverse for negative values later
    sfpi::vFloat val = sfpi::setsgn(x, 0);
    
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

    // For negative inputs, calculate the reciprocal since e^(-x) = 1/e^x
    v_if (x < 0) {
        val = ckernel::sfpu::_sfpu_reciprocal_(val);;
    }
    v_endif;

    return val;
}

// Unfolded version of _calculate_exponential_<false, false, 8, true, false>
inline void _calculate_exponential_unfolded(const int iterations) {
    // This is the unfolded version with the following parameters:
    // APPROXIMATION_MODE = false
    // SCALE_EN = false
    // ITERATIONS = 8
    // FAST_APPROX = true
    // SKIP_POSITIVE_CHECK = false
    
    // Since APPROXIMATION_MODE is false and FAST_APPROX && APPROXIMATION_MODE is false,
    // we only need the 'else' branch from the original function

    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        // SCALE_EN is false, so no scaling code needed

        // APPROXIMATION_MODE is false, so we execute this branch
        sfpi::vFloat result = _sfpu_exp_(val);

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Unfolded version of exp_tile<false, true>(0)
inline void exp_tile_unfolded(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    llk_math_eltwise_unary_sfpu_params<false>(
        _calculate_exponential_unfolded,
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
    exp_tile_init_unfolded();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply exponent operation
        exp_tile_unfolded(0);
        
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
