// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace NAMESPACE {

// SELU constants
sfpi_inline sfpi::vFloat get_selu_lambda() {
    return sfpi::vFloat(1.0507f); // λ (lambda) value for SELU
}

sfpi_inline sfpi::vFloat get_selu_alpha() {
    return sfpi::vFloat(1.6733f); // α (alpha) value for SELU
}

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

// Inline exp implementation copied from cbrt_eltwise_sfpu.cpp
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
        // Need to use reciprocal instead of division for SFPU
        val = _sfpu_reciprocal_inline_(val);
    }
    v_endif;

    return val;
}

// SELU implementation
sfpi_inline sfpi::vFloat _sfpu_selu_(sfpi::vFloat x)
{
    sfpi::vFloat lambda = get_selu_lambda();
    sfpi::vFloat alpha = get_selu_alpha();
    sfpi::vFloat result;
    
    v_if (x >= 0.0f) {
        // For x >= 0: SELU(x) = λ * x
        result = lambda * x;
    }
    v_else {
        // For x < 0: SELU(x) = λ * α * (exp(x) - 1)
        sfpi::vFloat exp_x = _sfpu_exp_(x);
        result = lambda * alpha * (exp_x - sfpi::vFloat(1.0f));
    }
    v_endif;
    
    return result;
}

// Unfolded version of selu_tile_init()
inline void selu_tile_init_unfolded() {
    // No constants needed - using hardcoded values in functions
}

// Unfolded version of calculate_selu
inline void _calculate_selu_unfolded(const int iterations) {
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        
        // Calculate SELU
        sfpi::vFloat result = _sfpu_selu_(val);
        
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Define these functions for compatibility
inline void selu_tile_init() {
    selu_tile_init_unfolded();
}

inline void selu_tile(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    llk_math_eltwise_unary_sfpu_params<false>(
        _calculate_selu_unfolded,
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
    selu_tile_init_unfolded();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply SELU operation
        selu_tile(0);
        
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