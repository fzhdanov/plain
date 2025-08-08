// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace NAMESPACE {

// Unfolded version of atanh_tile_init()
inline void atanh_tile_init_unfolded() {
    // Set constants for atanh calculation using inline logarithm and reciprocal
    sfpi::vConstFloatPrgm0 = 0.692871f;  // ln2 for log calculation
    sfpi::vConstFloatPrgm1 = 0.1058f;    // coefficient A for log series
    sfpi::vConstFloatPrgm2 = -0.7166f;   // coefficient B for log series
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

// Fused atanh implementation with inline logarithm and reciprocal
sfpi_inline sfpi::vFloat _sfpu_atanh_(sfpi::vFloat x)
{
    // atanh(x) = (1/2) * ln((1+x)/(1-x))
    // Valid for -1 < x < 1
    
    // Calculate 1+x
    sfpi::vFloat one = sfpi::vFloat(1.0f);
    sfpi::vFloat one_plus_x = one + x;
    
    // Calculate 1-x
    sfpi::vFloat one_minus_x = one - x;
    
    // Calculate (1+x)/(1-x) using inline reciprocal
    sfpi::vFloat ratio = one_plus_x * _sfpu_reciprocal_inline_(one_minus_x);
    
    // Inline logarithm calculation based on official SFPU log implementation
    // Normalize to calculation range
    sfpi::vFloat x_log = sfpi::setexp(ratio, 127);  // set exp to exp bias (put in range of 1-2)
    
    // Chebyshev Approximation using Horner Form Multiplication: 3rd Order
    // Coefficients from the official log implementation
    sfpi::vFloat a = sfpi::vConstFloatPrgm1;  // 0.1058f
    sfpi::vFloat b = sfpi::vConstFloatPrgm2;  // -0.7166f
    sfpi::vFloat series_result = x_log * (x_log * (x_log * a + b) + 2.0871f) + -1.4753f;
    
    // Convert exponent to float
    sfpi::vInt exp = sfpi::exexp(ratio);
    v_if(exp < 0) { 
        exp = sfpi::setsgn(~exp + 1, 1); 
    }
    v_endif;
    
    sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat log_result = expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)
    
    // Handle special case when ratio is 0: ln(0) = -inf
    v_if(ratio == 0.0f) {
        log_result = -std::numeric_limits<float>::infinity();
    }
    v_endif;
    
    // Multiply by 0.5 to get atanh result
    sfpi::vFloat result = log_result * sfpi::vFloat(0.5f);
    
    return result;
}

// Unfolded version of calculate_atanh
inline void _calculate_atanh_unfolded(const int iterations) {
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        
        // Check if input is in valid range (-1, 1)
        sfpi::vFloat abs_val = sfpi::setsgn(val, 0);  // Get absolute value
        
        sfpi::vFloat result;
        v_if (abs_val < sfpi::vFloat(1.0f)) {
            // Input is in valid range, calculate atanh
            result = _sfpu_atanh_(val);
        }
        v_else {
            // Input is outside valid range, return NaN
            result = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN());
        }
        v_endif;
        
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Define these functions for compatibility
inline void atanh_tile_init() {
    atanh_tile_init_unfolded();
}

inline void atanh_tile(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    llk_math_eltwise_unary_sfpu_params<false>(
        _calculate_atanh_unfolded,
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
    
    atanh_tile_init_unfolded();
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);

        
        // Apply atanh operation

        atanh_tile(0);
        
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