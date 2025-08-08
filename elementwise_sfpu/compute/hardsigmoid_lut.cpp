// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace compile_time_float {

    // Compile-time absolute value
    constexpr double abs_ct(double x) {
        return x < 0 ? -x : x;
    }
    
    // Compile-time power of 2
    constexpr double pow2_ct(int exp) {
        return exp == 0 ? 1.0 : 
               exp > 0 ? 2.0 * pow2_ct(exp - 1) : 
               0.5 * pow2_ct(exp + 1);
    }
    
    // Compile-time round
    constexpr int round_ct(double x) {
        return static_cast<int>(x + 0.5);
    }
    
    // Encode a single float to 8-bit format at compile time
    constexpr uint8_t encode_8bit_float_ct(double value) {
        if (value == 0.0) {
            return 0xFF;
        }
        
        bool sign = value < 0;
        double abs_value = abs_ct(value);
        
        double best_error = 1e10;
        uint8_t best_encoding = 0xFF;
        
        // Try all possible exponent extenders (0-7)
        for (int exp_extender = 0; exp_extender < 8; ++exp_extender) {
            int exponent = 127 - exp_extender;
            
            if (exponent - 127 > 100) continue; // Avoid overflow
            
            double power_of_2 = pow2_ct(exponent - 127);
            if (power_of_2 == 0) continue;
            
            double mantissa_float = (abs_value / power_of_2) - 1.0;
            
            // Check if mantissa is in valid range [0, 15/16]
            if (mantissa_float < 0 || mantissa_float >= 1.0) continue;
            
            // Quantize mantissa to 4 bits
            int mantissa_quantized = round_ct(mantissa_float * 16.0);
            if (mantissa_quantized > 15) mantissa_quantized = 15;
            
            // Reconstruct the value to check error
            double reconstructed_mantissa = mantissa_quantized / 16.0;
            double reconstructed_value = (1.0 + reconstructed_mantissa) * power_of_2;
            if (sign) reconstructed_value = -reconstructed_value;
            
            double error = abs_ct(reconstructed_value - value);
            
            if (error < best_error) {
                best_error = error;
                best_encoding = (sign ? 0x80 : 0x00) | (exp_extender << 4) | mantissa_quantized;
            }
        }
        
        return best_encoding;
    }
    
    // Encode coefficient pair to 16-bit lookup register at compile time
    constexpr uint16_t encode_coefficients_ct(double A, double B) {
        uint8_t A_encoded = encode_8bit_float_ct(A);
        uint8_t B_encoded = encode_8bit_float_ct(B);
        return (static_cast<uint16_t>(A_encoded) << 8) | B_encoded;
    }
}

namespace NAMESPACE {

// Unfolded version of hardsigmoid_lut_tile_init
inline void hardsigmoid_lut_tile_init_unfolded() {
    // No initialization needed for the LUT-based hardsigmoid
}

// Unfolded version of calculate_hardsigmoid_lut
inline void _calculate_hardsigmoid_lut_unfolded(const int iterations) {
    // For hardsigmoid: max(0, min(1, (x + 3) / 6))
    
    // Linear coefs (must be pre-scaled because hardsigmoid in scaled space is just identity)
    //[0, 1], symmetrical, controlling [-1, 1]
    sfpi::vUInt l0 = compile_time_float::encode_coefficients_ct(0.5, 0.0);
    // [1, 2] range, controlling [-2, -1] U [1, 2]
    sfpi::vUInt l1 = compile_time_float::encode_coefficients_ct(0.0, 0.5);
    // [2, inf] range, controlling [-inf, -2] U [2, inf]
    sfpi::vUInt l2 = compile_time_float::encode_coefficients_ct(0.0, 0.5);

    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        
        
        // Now apply LUT to the scaled input
        sfpi::vFloat result = sfpi::lut(val * (1.0f/3.0f), l0, l1, l2) + 0.5f;
        
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Define these functions for compatibility
inline void hardsigmoid_lut_tile_init() {
    hardsigmoid_lut_tile_init_unfolded();
}

inline void hardsigmoid_lut_tile(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    llk_math_eltwise_unary_sfpu_params<false>(
        _calculate_hardsigmoid_lut_unfolded,
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
    hardsigmoid_lut_tile_init_unfolded();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply hardsigmoid_lut operation
        hardsigmoid_lut_tile(0);
        
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