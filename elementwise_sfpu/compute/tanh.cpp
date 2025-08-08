// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//#include <cstdint>
//#include "compute_kernel_api/common.h"
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

// Unfolded version of tanh_tile_init<false>()
inline void tanh_tile_init_unfolded() {
    // Set constants for tanh calculation
    // These match the constants set in tanh_init<false>()
    //uint16_t imm0 = compile_time_float::encode_coefficients_ct(0.90625, 0.0); //0x1DFF;  // 0.90625*x
    //uint16_t imm1 = compile_time_float::encode_coefficients_ct(0.09375, 0.8125); //0x481A;  // 0.09375*x + 0.8125
    //uint16_t imm2 = compile_time_float::encode_coefficients_ct(0.0, 1.0); //0xFF00;  // 1
    
    // Use the correct namespace for _sfpu_load_imm16_
    //ckernel::sfpu::_sfpu_load_imm16_(0, imm0);
    //ckernel::sfpu::_sfpu_load_imm16_(1, imm1);
    //ckernel::sfpu::_sfpu_load_imm16_(2, imm2);
}

// Unfolded version of calculate_tanh<false, 8>
inline void _calculate_tanh_unfolded(const int iterations) {
    // This is the unfolded version with the following parameters:
    // APPROXIMATION_MODE = false
    // ITERATIONS = 8
    
    // SFPU microcode
    sfpi::vUInt l0 = compile_time_float::encode_coefficients_ct(0.90625, 0.0);//sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = compile_time_float::encode_coefficients_ct(0.09374, 0.8125);//sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = compile_time_float::encode_coefficients_ct(0.0, 1.0);//sfpi::l_reg[sfpi::LRegs::LReg2];

    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        val = sfpi::lut(val, l0, l1, l2);
        sfpi::dst_reg[0] = val;

        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
}

// Define these functions for compatibility with the SFPU_OP_CHAIN_0 define
inline void tanh_tile_init() {
    tanh_tile_init_unfolded();
}

inline void tanh_tile(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    llk_math_eltwise_unary_sfpu_params<false>(
        _calculate_tanh_unfolded,
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
    tanh_tile_init_unfolded();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply tanh operation
        tanh_tile(0);
        
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