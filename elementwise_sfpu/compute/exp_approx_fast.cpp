// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_exp.h"


namespace NAMESPACE {

// Unfolded version of exp_tile_init<true, true>()
inline void exp_tile_init_approx_fast() {
    // Set constants for exponential calculation with approximation and fast mode
    // These match the constants set in _init_exponential_<true, true>()
    
    constexpr float LN2_RECIP = 1.4426950408889634f;
    constexpr float A = 256.0f * LN2_RECIP;
    constexpr float B_minus_C = 32500.818359375f;
    constexpr float THRESHOLD = -88.5f;
    
    TTI_SFPLOADI(0, 0xA, ckernel::sfpu::lo16(THRESHOLD));
    TTI_SFPLOADI(0, 0x8, ckernel::sfpu::hi16(THRESHOLD));
    TTI_SFPCONFIG(0, 14, 0); // SFPCONFIG Dest 14 = LREG[14] = -88.5
    
    TTI_SFPLOADI(0, 0xA, ckernel::sfpu::lo16(A));
    TTI_SFPLOADI(0, 0x8, ckernel::sfpu::hi16(A));
    TTI_SFPCONFIG(0, 12, 0); // SFPCONFIG Dest 12 = LREG[12] = A
    
    TTI_SFPLOADI(0, 0xA, ckernel::sfpu::lo16(B_minus_C));
    TTI_SFPLOADI(0, 0x8, ckernel::sfpu::hi16(B_minus_C));
    TTI_SFPCONFIG(0, 13, 0); // SFPCONFIG Dest 13 = LREG[13] = (B-C)
    
    // Set up macro instructions
    TTI_SFPLOADI(0, 0xA, 0x00E1);
    TTI_SFPLOADI(0, 0x8, 0x9200);
    TTI_SFPCONFIG(0, 0, 0);
    TTI_SFPNOP;
    
    TTI_SFPMAD(12, 0, 13, 13, 0);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 14);
    TTI_SFPSHFT(15, 0, 15, 1);
    
    // Set up sequences
    TTI_SFPLOADI(0, 0xA, 0x0004);
    TTI_SFPLOADI(0, 0x8, 0x1300);
    TTI_SFPCONFIG(0, 5, 0);
    
    TTI_SFPLOADI(0, 0xA, 0x85DF);
    TTI_SFPLOADI(0, 0x8, 0x6316);
    TTI_SFPCONFIG(0, 4, 0);
    
    // Reset LoadMacroConfig
    TTI_SFPCONFIG(0, 8, 1);
}

// Unfolded version of _calculate_exponential_<true, false, 8, true, false>
inline void _calculate_exponential_approx_fast(const int iterations) {
    // This is the unfolded version with the following parameters:
    // APPROXIMATION_MODE = true
    // SCALE_EN = false
    // ITERATIONS = 8
    // FAST_APPROX = true
    // SKIP_POSITIVE_CHECK = false
    
    // Since FAST_APPROX && APPROXIMATION_MODE is true, we use the fast approximation code
    
    // Sanitize the input values and calculate the approximate exponential value
    for (int d = 0; d < iterations; d++) {
        TTI_SFPLOADMACRO(4, 0, 3, 0);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(5, 0, 3, 2);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, 3, 4);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, 3, 6);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(4, 0, 3, 8);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(5, 0, 3, 10);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, 3, 12);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, 3, 14);
        
        TTI_SFPLOADMACRO(0, 0, 3, 0);
        TTI_SFPLOADMACRO(1, 0, 3, 2);
        TTI_SFPLOADMACRO(2, 0, 3, 4);
        TTI_SFPLOADMACRO(3, 0, 3, 6);
        TTI_SFPLOADMACRO(0, 0, 3, 8);
        TTI_SFPLOADMACRO(1, 0, 3, 10);
        TTI_SFPLOADMACRO(2, 0, 3, 12);
        TTI_SFPLOADMACRO(3, 0, 3, 14);
        TTI_SFPNOP;
    }
}

// Unfolded version of exp_tile<true, true>(0)
inline void exp_tile_approx_fast(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_<true>(
        _calculate_exponential_approx_fast,
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
    exp_tile_init_approx_fast();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply exponent operation with approximation and fast mode
        exp_tile_approx_fast(0);
        
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