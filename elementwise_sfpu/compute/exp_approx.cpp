// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"


namespace NAMESPACE {

// Unfolded version of exp_tile_init<true, false>()
inline void exp_tile_init_approx() {
    // Set constants for exponential calculation with approximation mode
    // These match the constants set in _init_exponential_<true, false>()
    sfpi::vConstFloatPrgm0 = 1.442695f; // ln2_recip
    sfpi::vConstFloatPrgm1 = sfpi::s2vFloat16b(p_exp::C23_73);
    sfpi::vConstFloatPrgm2 = sfpi::s2vFloat16b(p_exp::ADJ_EXP);
}

// Unfolded version of _calculate_exponential_<true, false, 8, false, false>
inline void _calculate_exponential_approx(const int iterations) {
    // This is the unfolded version with the following parameters:
    // APPROXIMATION_MODE = true
    // SCALE_EN = false
    // ITERATIONS = 8
    // FAST_APPROX = false
    // SKIP_POSITIVE_CHECK = false
    
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        
        // APPROXIMATION_MODE is true, so we execute this branch
        v_if (val >= 89) {
            // Algorithm is incorrect for inputs >= 89, so saturate output to infinity.
            sfpi::vFloat val_inf = std::numeric_limits<float>::infinity();
            sfpi::dst_reg[0] = val_inf;
        }
        v_elseif (val < -42) {
            // Algorithm is incorrect for inputs < -42, so saturate output to 0.
            sfpi::dst_reg[0] = 0.0f;
        }
        v_else {
            // * by 1/ln2 and add convert to 7.3 FxP format
            sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;
            sfpi::vFloat c23_73 = sfpi::vConstFloatPrgm1;
            sfpi::vInt adj_exp = sfpi::vConstIntPrgm2;
            val = val * vConstLn2Recip + c23_73;

            // Remove Exponent of 7 and bias the Mantissa to 127.
            sfpi::vInt val_short = adj_exp + sfpi::reinterpret<sfpi::vInt>(val);

            // SHL to move integer bits to exponent
            val_short <<= 10 - p_exp::FRAC_BITS;
            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(val_short);
        }
        v_endif;
        
        sfpi::dst_reg++;
    }
}

// Unfolded version of exp_tile<true, false>(0)
inline void exp_tile_approx(uint32_t idst) {
// THIS IFDEF IS EXTREMELY IMPORTANT FOR THE COMPILER!!!!
// USE THIS SYNTAX TO CALL THE FUNCTION!!!!
// DO NOT CALL IT DIRECTLY!!!!
#ifdef TRISC_MATH
    llk_math_eltwise_unary_sfpu_params<true>(
        _calculate_exponential_approx,
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
    exp_tile_init_approx();
    
    // Process only the tiles assigned to this core
    for (uint32_t tile_index = 0; tile_index < actual_tiles_for_this_core; ++tile_index) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
    
        // Apply exponent operation with approximation mode
        exp_tile_approx(0);
        
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