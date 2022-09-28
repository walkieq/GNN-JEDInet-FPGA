//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "jedi.h"
#include "../nnet_utils/nnet_jedi.h"

#include "weights50p/w1.h"
#include "weights50p/w2.h"
#include "weights50p/w3.h"
#include "weights50p/b1.h"
#include "weights50p/b2.h"
#include "weights50p/b3.h"

#include "weights50p/w4.h"
#include "weights50p/w5.h"
#include "weights50p/w6.h"
#include "weights50p/b4.h"
#include "weights50p/b5.h"
#include "weights50p/b6.h"

#include "weights50p/w7.h"
#include "weights50p/w8.h"
#include "weights50p/w9.h"
#include "weights50p/b7.h"
#include "weights50p/b8.h"
#include "weights50p/b9.h"


void jedi(
	    input_t I[N_o][P], // change to number of particles
        result_t result[N_OUTPUT_3]
    ){
        #pragma HLS DATAFLOW
        
        input_t in_dup1[N_o][P];
        input_t in_dup2[N_o][P];
        input_t in_dup3[N_o][P];
        const int factor_in=P;
        const int factor_in_d1=DPP;
        #pragma HLS ARRAY_PARTITION variable=I       complete dim=2
        #pragma HLS ARRAY_PARTITION variable=in_dup1 complete dim=2
        #pragma HLS ARRAY_PARTITION variable=in_dup2 complete dim=2
        #pragma HLS ARRAY_PARTITION variable=in_dup3 complete dim=2

        #pragma HLS ARRAY_PARTITION variable=I       complete dim=1
        #pragma HLS ARRAY_PARTITION variable=in_dup1 complete dim=1
        #pragma HLS ARRAY_PARTITION variable=in_dup2 complete dim=1
        #pragma HLS ARRAY_PARTITION variable=in_dup3 complete dim=1
       
        nnet::jedi_duplicate<input_t, jedi1_config>(I, in_dup1, in_dup2, in_dup3); 

        input_t B[N_e][2*P];
        const int factor_B=2*P;
        #pragma HLS ARRAY_PARTITION variable=B cyclic factor = factor_B
        // mmm1, mmm2, and concat
        nnet::jedi1_mmm<input_t, input_t, jedi1_config>(in_dup1, in_dup2, B);
		
        input_t E[N_e][D_e];
        const int factor_E=D_e;
        #pragma HLS ARRAY_PARTITION variable=E cyclic factor = factor_E
        nnet::jedi_dnn1<input_t, input_t, dense1_config>(B, E, w1, w2, w3, b1, b2, b3);
		
        input_t C[N_o][P + D_e];
        const int factor_C=P+D_e;
        #pragma HLS ARRAY_PARTITION variable=C cyclic factor = factor_C
        nnet::jedi2_mmm<input_t, input_t, jedi2_config>(in_dup3, E, C);

        input_t O[N_o][D_o];
        const int factor_O=D_o;
        #pragma HLS ARRAY_PARTITION variable=O cyclic factor = factor_O
        nnet::jedi_dnn2_t<input_t, input_t, dense2_config>(C, O, w4, w5, w6, b4, b5, b6);
		
        nnet::jedi_dnn3_t<input_t, input_t, dense3_config>(O, result, w7, w8, w9, b7, b8, b9);
        
}
