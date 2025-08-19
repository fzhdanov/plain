// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace NAMESPACE {



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


// Improved Taylor series for small values: atanh(x) = x + x³/3 + x⁵/5 + x⁷/7
sfpi_inline sfpi::vFloat _sfpu_atanh_taylor_better_(sfpi::vFloat abs_x)
{
    sfpi::vFloat x2 = abs_x * abs_x;        // x²
    sfpi::vFloat x3 = x2 * abs_x;           // x³
    sfpi::vFloat x5 = x3 * x2;              // x⁵
    sfpi::vFloat x7 = x5 * x2;              // x⁷

    // 4-term odd polynomial for better accuracy when |x| <= 0.5
    return abs_x + x3 * sfpi::vFloat(1/3.0f) + x5 * sfpi::vFloat(1/5.0f) + x7 * sfpi::vFloat(1/7.0f);
}

// Expansion for large values: atanh(1-u) ≈ -½ln(2u) - u/4 - u²/24 - u³/48 + ...
// where u = 1 - ax for ax > 0.5
sfpi_inline sfpi::vFloat _sfpu_atanh_large_(sfpi::vFloat u, sfpi::vFloat log_u)
{
    sfpi::vFloat u2 = u * u;                   // u²
    sfpi::vFloat u3 = u2 * u;                  // u³
    //sfpi::vFloat u4 = u3 * u;                  // u⁴
    
    // Calculate ½ln(2/u) = ½ln(2) - ½ln(u)
    sfpi::vFloat half_log_2_over_u = sfpi::vFloat(0.34657359f) - sfpi::vFloat(0.5f) * log_u;  // ½ln(2) ≈ 0.34657359
    
    // Calculate series: -u/4 - u²/24 - u³/48 - u⁴/80
    return half_log_2_over_u + u * sfpi::vFloat(-1/4.0f) +          // -u/4
                         u2 * sfpi::vFloat(-1/24.0f) +        // -u²/24
                         u3 * sfpi::vFloat(-1/48.0f);// +        // -u³/48
                         //u4 * sfpi::vFloat(-1/80.0f);         // -u⁴/80
}



// Main atanh function using the exp.cpp approach: calculate positive part, handle sign
sfpi_inline sfpi::vFloat _sfpu_atanh3_(sfpi::vFloat x)
{
    sfpi::vFloat ax = sfpi::setsgn(x, 0);
    sfpi::vFloat out;
    sfpi::vFloat u = sfpi::vFloat(1.0f) - ax;
    sfpi::vFloat log_u = _sfpu_log_(u);
    sfpi::vFloat out1 = _sfpu_atanh_taylor_better_(ax);
    sfpi::vFloat out2 = _sfpu_atanh_large_(u, log_u);
    v_if (ax < 0.75f) {
        out = out1;
    }
    v_else {
        out = out2;
    }
    v_endif;

    v_if (x < 0.0f) 
    { 
        out = -out; 
    } 
    v_endif;

    v_if (ax >= 1.0f) {
        out = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN());
    }
    v_endif;


    return out;
}

// Unfolded version of calculate_atanh
inline void _calculate_atanh3_unfolded(const int iterations) {
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = _sfpu_atanh3_(val);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}


inline void atanh3_tile(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
_llk_math_eltwise_unary_sfpu_params_<false>(
        _calculate_atanh3_unfolded,
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
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);

        
        // Apply atanh3 operation
        atanh3_tile(0);
        
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
