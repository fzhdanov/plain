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

// Unfolded version of swish_tile_init()
inline void swish_tile_init_unfolded() {
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

// Optimized exp function that computes exp for non-positive values only
// This avoids the reciprocal branch since input x <= 0
sfpi_inline sfpi::vFloat _sfpu_exp_nonpos_(sfpi::vFloat x)
{
    sfpi::vFloat val = sfpi::setsgn(x, 0);  // Get absolute value

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

    // Since x <= 0, we always need reciprocal (exp(x) = 1/exp(|x|))
    val = _sfpu_reciprocal_inline_(val);

    return val;
}

// Optimized sigmoid function utilizing symmetry: sigmoid(-x) = 1 - sigmoid(x)
// This avoids the reciprocal branch in exp by always computing exp(-|x|)
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x)
{
    sfpi::vFloat abs_x = sfpi::setsgn(x, 0);  // Get |x|
    sfpi::vFloat neg_abs_x = -abs_x;          // Get -|x| (always <= 0)
    
    // Calculate exp(-|x|) using optimized function (no reciprocal branch)
    sfpi::vFloat exp_neg_abs_x = _sfpu_exp_nonpos_(neg_abs_x);
    
    sfpi::vFloat one = sfpi::vFloat(1.0f);
    sfpi::vFloat denom = one + exp_neg_abs_x;              // 1 + exp(-|x|)
    sfpi::vFloat sigmoid_abs_x = _sfpu_reciprocal_inline_(denom);  // sigmoid(|x|)
    
    // Use sigmoid symmetry: sigmoid(-x) = 1 - sigmoid(x)
    sfpi::vFloat result;
    v_if (x >= 0) {
        result = sigmoid_abs_x;                            // sigmoid(x) = sigmoid(|x|)
    }
    v_else {
        result = one - sigmoid_abs_x;                      // sigmoid(x) = 1 - sigmoid(|x|)
    }
    v_endif;
    
    return result;
}

// Optimized swish function: x * sigmoid(x)
// Uses sigmoid symmetry to avoid reciprocal branch in exp computation
sfpi_inline sfpi::vFloat _sfpu_swish_(sfpi::vFloat x)
{
    sfpi::vFloat sigmoid_x = _sfpu_sigmoid_(x);
    sfpi::vFloat result = x * sigmoid_x;
    return result;
}

// Unfolded version of calculate_swish
inline void _calculate_swish_unfolded(const int iterations) {
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        
        // Calculate swish using the implementation
        sfpi::vFloat result = _sfpu_swish_(val);
        
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Define these functions for compatibility
inline void swish_tile_init() {
    swish_tile_init_unfolded();
}

inline void swish_tile(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_<false>(
        _calculate_swish_unfolded,
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
    swish_tile_init_unfolded();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply swish operation
        swish_tile(0);
        
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
