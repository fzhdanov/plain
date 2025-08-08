// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace NAMESPACE {

// Unfolded version of cosh_tile_init()
inline void cosh_tile_init_unfolded() {
    // Set constants for hyperbolic cosine calculation
    // cosh(x) = (e^x + e^(-x)) / 2
    // We'll use constants for the exponential approximation
    sfpi::vConstFloatPrgm0 = 1.442695f;  // ln2_recip for exp calculation
    sfpi::vConstFloatPrgm1 = 2.0f;       // for division by 2
    sfpi::vConstFloatPrgm2 = 0.863281f;  // constant for exp series
}

// Positive-only SFPU exponential function (simplified version for cosh)
sfpi_inline sfpi::vFloat _sfpu_exp_positive_(sfpi::vFloat val)
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

// Improved fused cosh implementation
sfpi_inline sfpi::vFloat _sfpu_cosh_(sfpi::vFloat val)
{
    // Since cosh(x) = cosh(-x), we can work with absolute value for better numerical stability
    // cosh(x) = (e^|x| + e^(-|x|))/2
    
    // Get absolute value of input
    sfpi::vFloat abs_val = sfpi::setsgn(val, 0);
    
    // Calculate e^|x|
    sfpi::vFloat exp_pos = _sfpu_exp_positive_(abs_val);
    
    // Calculate e^(-|x|) = 1/e^|x|
    sfpi::vFloat exp_neg = ckernel::sfpu::_sfpu_reciprocal_(exp_pos);
    
    // Calculate (e^|x| + e^(-|x|)) / 2
    sfpi::vFloat result = (exp_pos + exp_neg) * sfpi::s2vFloat16b(0.5f);
    
    return result;
}

// Unfolded version of calculate_cosh
inline void _calculate_cosh_unfolded(const int iterations) {
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        
        // Calculate cosh using the improved fused implementation
        sfpi::vFloat result = _sfpu_cosh_(val);
        
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Define these functions for compatibility
inline void cosh_tile_init() {
    cosh_tile_init_unfolded();
}

inline void cosh_tile(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    llk_math_eltwise_unary_sfpu_params<false>(
        _calculate_cosh_unfolded,
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
    cosh_tile_init_unfolded();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply cosh operation
        cosh_tile(0);
        
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