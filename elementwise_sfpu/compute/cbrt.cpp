// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace NAMESPACE {



// Inline reciprocal implementation based on official SFPU reciprocal
sfpi_inline sfpi::vFloat _sfpu_reciprocal_inline_(sfpi::vFloat in)
{
    // Force sign to 1 (make number negative)
    sfpi::vFloat val = sfpi::setsgn(in, 1);

    val = sfpi::setexp(val, 126); // Set exponent to 126 to make the number in 0.5-1
    
    // Use hardcoded constants for reciprocal calculation
    sfpi::vFloat vConstLn2Recip = sfpi::vFloat(1.442695f); // ln2_recip
    sfpi::vFloat two = sfpi::vFloat(2.0f);
    sfpi::vFloat result = vConstLn2Recip * (val * vConstLn2Recip + two);

    // Newton-Raphson iterations (max_iter = 3)
    for (int s_iter = 0; s_iter < 2; s_iter++)
    {
        result = result * (val * result + two);
    }

    sfpi::vInt orig_exp = sfpi::exexp(in);
    sfpi::vInt new_exp = sfpi::exexp(result);

    // "Subtract" exponents, and re-bias.
    // Execute: -1 - exp, then exp += 127
    new_exp -= orig_exp;
    new_exp += 126;

    v_if (new_exp < 0)
    {
        // If rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        result = 0.0f;
        new_exp = 0;
    }
    v_endif;

    // Set newly denormalized exponent to result exponent field
    return sfpi::setexp(result, new_exp);
}

// Unfolded version of cbrt_tile_init()
inline void cbrt_tile_init_unfolded() {
    // No constants needed - using hardcoded values in functions
}


sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat x)
{
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

    v_if (x < 0) {
        val = _sfpu_reciprocal_inline_(val);
    }
    v_endif;

    return val;
}

// Full copy of logarithm implementation from atanh_eltwise_sfpu.cpp
sfpi_inline sfpi::vFloat _sfpu_log_(sfpi::vFloat in)
{
    // Normalize to calculation range
    sfpi::vFloat x = sfpi::setexp(in, 127);  // set exp to exp bias (put in range of 1-2)
    
    // Chebyshev Approximation using Horner Form Multiplication: 3rd Order
    // Coefficients from the official log implementation - using hardcoded values
    sfpi::vFloat a = sfpi::vFloat(0.1058f);  // coefficient A for log series
    sfpi::vFloat b = sfpi::vFloat(-0.7166f); // coefficient B for log series
    sfpi::vFloat series_result = x * (x * (x * a + b) + 2.0871f) + -1.4753f;
    
    // Convert exponent to float
    sfpi::vInt exp = sfpi::exexp(in);
    v_if(exp < 0) { 
        exp = sfpi::setsgn(~exp + 1, 1); 
    }
    v_endif;
    
    sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);
    sfpi::vFloat vConstLn2 = sfpi::vFloat(0.692871f); // ln2 for log calculation
    sfpi::vFloat result = expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)
    
    // Handle special case when input is 0: ln(0) = -inf
    v_if(in == 0.0f) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;
    
    return result;
}

// Fused cubic root implementation using exp/log formula
sfpi_inline sfpi::vFloat _sfpu_cbrt_(sfpi::vFloat x)
{
    sfpi::vFloat sign = sfpi::vFloat(1.0f);
    v_if(x < 0.0f) {
        sign = sfpi::vFloat(-1.0f);
    }
    v_endif;

    sfpi::vFloat abs_x = sfpi::setsgn(x, 0);
    sfpi::vFloat log_abs_x = _sfpu_log_(abs_x);
    sfpi::vFloat one_third = sfpi::vFloat(0.333333f);
    sfpi::vFloat log_div_3 = log_abs_x * one_third;
    sfpi::vFloat result = _sfpu_exp_(log_div_3) * sign;
    return result;

}

// Unfolded version of calculate_cbrt
inline void _calculate_cbrt_unfolded(const int iterations) {
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        
        // Calculate cubic root using the fused implementation
        sfpi::vFloat result = _sfpu_cbrt_(val);
        
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Define these functions for compatibility
inline void cbrt_tile_init() {
    cbrt_tile_init_unfolded();
}

inline void cbrt_tile(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_<false>(
        _calculate_cbrt_unfolded,
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
    cbrt_tile_init_unfolded();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply cubic root operation
        cbrt_tile(0);
        
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