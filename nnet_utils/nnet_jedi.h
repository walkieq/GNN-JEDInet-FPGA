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

#ifndef NNET_JEDI_H_
#define NNET_JEDI_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_activation.h"
#include "nnet_dense.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

    template<class data_T, class res_T, typename CONFIG_T>
    void jedi1_mmm_rr(
            data_T    data1[CONFIG_T::n_col1][CONFIG_T::n_row1],
            res_T     res[CONFIG_T::n_col2][CONFIG_T::n_row1]// n_col1 = n_row2
    ) {
        const int unroll_factor=CONFIG_T::DPP_p;
        for (int i = 0; i < CONFIG_T::n_col1; i++) {
		    #pragma HLS PIPELINE 
            #pragma HLS dependence variable=res intra false
            for (int k = 0; k < (CONFIG_T::n_col1 - 1); k++) {
                //#pragma HLS dependence variable=res inter false
                for (int j = 0; j < CONFIG_T::n_row1; j++) {
                    res[k+i*(CONFIG_T::n_col1-1)][j] = data1[i][j];
                    //res[j][k+i*(CONFIG_T::n_col1-1)] = data1[i][j];
                }
            }
        }

    }
    template<class data_T, class res_T, typename CONFIG_T>
    void jedi1_mmm_rs(
            data_T    data1[CONFIG_T::n_col1][CONFIG_T::n_row1],// n_col1 = n_row2
            res_T     res  [CONFIG_T::n_col2][CONFIG_T::n_row1]
    ) {

        const int unroll_factor=CONFIG_T::DPP_p;
        int index;
        for (int i = 0; i < CONFIG_T::n_col1; i++) {
		    #pragma HLS PIPELINE 
            #pragma HLS dependence variable=res intra false
            for (int k = 0; k < (CONFIG_T::n_col1 - 1); k++) {
                //#pragma HLS dependence variable=res inter false
                for (int j = 0; j < CONFIG_T::n_row1; j++) {
                    index = (k<i)? k : (k+1);
                    res[k+i*(CONFIG_T::n_col1-1)][j] = data1[index][j];
                }
            }
        }

    }


    template<class data_T, class res_T, typename CONFIG_T>
    void jedi2_mmm_rrt_t(
            //data_T    data1[CONFIG_T::n_row1][CONFIG_T::n_col1],
            data_T    data1[CONFIG_T::n_col1][CONFIG_T::n_row1],// n_col1 = n_row2
            res_T     res  [CONFIG_T::n_col2][CONFIG_T::n_row1]
    ){
        // Do the matrix-multiply
        //
        data_T acc[CONFIG_T::n_row1];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        for (int i = 0; i < CONFIG_T::n_col2; i++) {
	        #pragma HLS PIPELINE 
            for(int k = i*(CONFIG_T::n_col2-1); k < (i+1)*(CONFIG_T::n_col2-1);k++){
                for (int j = 0; j < CONFIG_T::n_row1; j++) {
                    data_T tmp = (k==i*(CONFIG_T::n_col2-1))? ((data_T) 0):acc[j];  
                    acc[j] = tmp + data1[k][j];
                }
            }
            for (int j = 0; j < CONFIG_T::n_row1; j++) {
                res[i][j] = (res_T) acc[j]; 
            }
        }
        
    }

    template<class data_T, class res_T, typename CONFIG_T>
    void jedi2_mmm_rrt(
            //data_T    data1[CONFIG_T::n_row1][CONFIG_T::n_col1],
            data_T    data1[CONFIG_T::n_col1][CONFIG_T::n_row1],// n_col1 = n_row2
            res_T     res  [CONFIG_T::n_col2][CONFIG_T::n_row1]
    ){
        // Do the matrix-multiply
        //
        data_T acc[CONFIG_T::n_row1];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        for (int i = 0; i < CONFIG_T::n_col2; i++) {
	        #pragma HLS PIPELINE 
            //for(int k = i*(CONFIG_T::n_col2-1); k < (i+1)*(CONFIG_T::n_col2-1);k++){
            for(int k = 0; k < (CONFIG_T::n_col2-1);k++){
                for (int j = 0; j < CONFIG_T::n_row1; j++) {
                    int index = i*(CONFIG_T::n_col2-1) + k;
                    data_T tmp = (k==0)? ((data_T) 0):acc[j];  
                    acc[j] = tmp + data1[index][j];
                }
            }
            for (int j = 0; j < CONFIG_T::n_row1; j++) {
                res[i][j] = (res_T) acc[j]; 
            }
        }
        
    }

    template<class data_T, class res_T, typename CONFIG_T>
    void jedi_multiply_b(
            data_T    data1[CONFIG_T::n_row1][CONFIG_T::n_col1],
            bool    data2[CONFIG_T::n_row2][CONFIG_T::n_col2],
            res_T     res[CONFIG_T::n_row1][CONFIG_T::n_col2]) // n_col1 = n_row2
    {

        // Do the matrix-multiply
        Product1: for (int i = 0; i < CONFIG_T::n_row1; i++) {
        
            for (int j = 0; j < CONFIG_T::n_col2; j++) {
				#pragma HLS PIPELINE 
				//#pragma HLS PIPELINE rewind
               res[i][j] = 0;

                for (int k = 0; k < CONFIG_T::n_col1; k++){
                    //#pragma HLS PIPELINE
                    //res[i][j] = res[i][j] + CONFIG_T::template product<data_T, data_T, res_T>::product(data1[i][k], data2[k][j]);
                    if(data2[k][j]) {
                        res[i][j] = res[i][j] + data1[i][k];
                    }else {
                        res[i][j] = res[i][j];
                    }
                }
                std::cout << res[i][j] << ", ";
            }
            std::cout  << "\n ";
        }

    }

    template<class data_T, class res_T, typename CONFIG_T>
    void jedi_multiply(
            data_T    data1[CONFIG_T::n_row1][CONFIG_T::n_col1],
            data_T    data2[CONFIG_T::n_row2][CONFIG_T::n_col2],
            res_T     res[CONFIG_T::n_row1][CONFIG_T::n_col2]) // n_col1 = n_row2
    {

        // Do the matrix-multiply
        Product1: for (int i = 0; i < CONFIG_T::n_row1; i++) {
        
            for (int j = 0; j < CONFIG_T::n_col2; j++) {
				#pragma HLS PIPELINE 
				//#pragma HLS PIPELINE rewind
               res[i][j] = 0;

                for (int k = 0; k < CONFIG_T::n_col1; k++)
                    //#pragma HLS PIPELINE
                    //res[i][j] = res[i][j] + CONFIG_T::template product<data_T, data_T, res_T>::product(data1[i][k], data2[k][j]);
                     res[i][j] = res[i][j] + data1[i][k]*data2[k][j];
            }
        }

    }

    template<class data_T, class res_T, typename CONFIG_T>
    void jedi_concat_t_ux(
            data_T    data1[CONFIG_T::n_col1][CONFIG_T::n_row1],
            data_T    data2[CONFIG_T::n_col2][CONFIG_T::n_row2],
            res_T     res[CONFIG_T::n_col2][CONFIG_T::n_row1+CONFIG_T::n_row2]) // n_col1 must be equal to n_col2
    {
        const int unroll_factor=CONFIG_T::DPP_p;
        for (int i = 0; i < CONFIG_T::n_col1; i++) {
		    #pragma HLS UNROLL factor=unroll_factor 
			#pragma HLS PIPELINE 
			//#pragma HLS PIPELINE rewind
            // To store elements
            // of data1
            for (int j = 0; j < CONFIG_T::n_row1; j++) {
				#pragma HLS UNROLL
                res[i][j] = data1[i][j];
            }
            // To store elements
            // of matrix B
            for (int k = 0; k < CONFIG_T::n_row2; k++) {
				#pragma HLS UNROLL
                res[i][k + CONFIG_T::n_row1] = data2[i][k];
            }
        }
    }

    template<class data_T, class res_T, typename CONFIG_T>
    void jedi_concat_t(
            data_T    data1[CONFIG_T::n_col1][CONFIG_T::n_row1],
            data_T    data2[CONFIG_T::n_col2][CONFIG_T::n_row2],
            res_T     res[CONFIG_T::n_col2][CONFIG_T::n_row1+CONFIG_T::n_row2]) // n_col1 must be equal to n_col2
    {
        const int factor_output=CONFIG_T::n_row1+CONFIG_T::n_row2;
        #pragma HLS ARRAY_PARTITION variable=res cyclic factor = factor_output
        // the two matrices have the same number of column
        for (int i = 0; i < CONFIG_T::n_col1; i++) {
			#pragma HLS PIPELINE 
			//#pragma HLS PIPELINE rewind
            // To store elements
            // of data1
            for (int j = 0; j < CONFIG_T::n_row1; j++) {
				#pragma HLS UNROLL
                res[i][j] = data1[i][j];
            }
            // To store elements
            // of matrix B
            for (int k = 0; k < CONFIG_T::n_row2; k++) {
				#pragma HLS UNROLL
                res[i][k + CONFIG_T::n_row1] = data2[i][k];
            }
        }
    }

    template<class data_T, class res_T, typename CONFIG_T>
    void transposeRr(data_T Rr[][CONFIG_T::N_e_p], data_T RrT[][CONFIG_T::N_o_p]) {

        for(int i = 0; i < CONFIG_T::N_o_p; i++) {
			#pragma HLS PIPELINE
            for(int j= 0; j < CONFIG_T::N_e_p; j++) {
				//#pragma HLS PIPELINE
                RrT[j][i] = Rr[i][j];
            }
        }

    }


    template <class data_t, class res_t, typename CONFIG_T>
    void dnn1(data_t input[], res_t res[], data_t w1[], data_t w2[], data_t w3[], data_t b1[],
              data_t b2[], data_t b3[]) {

        /*
        for(int i = 0; i < 2*P; i++)
            for(int j = 0; j < 30; j++)
                std::cout<< w1_1[i+j];*/
        //  ============= Dense MLP layers 1: for transforming B into E ==================
        /*std::cout << "DNN1: PRINTING MATRIX B \n";
        for(int i = 0; i < 32; i++) 
			std::cout << input[i] << " ";
		std::cout << std::endl;
		std::cout << "DNN1: PRINTING W1 \n";
        for(int i = 0; i < 32; i++) 
			std::cout << w1[i] << " ";
		std::cout << std::endl;*/
        
        data_t layer2_out[CONFIG_T::fc1_out];
        #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::fc1_config>(input, layer2_out, w1, b1); // fc1
		//std::cout << "DNN1: PRINTING OUTPUT OF FC1 FIRST NODE \n";
		//std::cout << layer2_out[0] << std::endl;
		
        data_t layer3_out[CONFIG_T::fc1_out];
        #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
        nnet::relu<data_t, data_t, typename CONFIG_T::relu1_config>(layer2_out, layer3_out); // fc1_relu

        data_t layer4_out[CONFIG_T::fc2_out];
        #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::fc2_config>(layer3_out, layer4_out, w2, b2); // fc2

        data_t layer5_out[CONFIG_T::fc2_out];
        #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
        nnet::relu<data_t, data_t, typename CONFIG_T::relu2_config>(layer4_out, layer5_out); // fc2_relu

        data_t layer6_out[CONFIG_T::fc3_out];
        #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::output1_config>(layer5_out, layer6_out, w3, b3); // output

        nnet::relu<data_t, data_t, typename CONFIG_T::softmax1_config>(layer6_out, res);
        //nnet::softmax<data_t, data_t, typename CONFIG_T::softmax1_config>(layer6_out, res); // output_softmax
    }

    // dnn1 with only 1 layer
    template <class data_t, class res_t, typename CONFIG_T>
    void dnn1_1l( data_t input[], 
                        res_t res[], 
                        data_t w1[], 
                        data_t b1[] ) {

        
        //data_t layer2_out[CONFIG_T::fc1_out];
        //#pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
        //nnet::dense<data_t, data_t, typename CONFIG_T::fc1_config>(input, layer2_out, w1, b1); // fc1
		////std::cout << "DNN1: PRINTING OUTPUT OF FC1 FIRST NODE \n";
		////std::cout << layer2_out[0] << std::endl;
		//
        //data_t layer3_out[CONFIG_T::fc1_out];
        //#pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
        //nnet::relu<data_t, data_t, typename CONFIG_T::relu1_config>(layer2_out, layer3_out); // fc1_relu

        //data_t layer4_out[CONFIG_T::fc2_out];
        //#pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
        //nnet::dense<data_t, data_t, typename CONFIG_T::fc2_config>(layer3_out, layer4_out, w2, b2); // fc2

        //data_t layer5_out[CONFIG_T::fc2_out];
        //#pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
        //nnet::relu<data_t, data_t, typename CONFIG_T::relu2_config>(layer4_out, layer5_out); // fc2_relu

        data_t layer6_out[CONFIG_T::fc3_out];
        #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::output1_config>(input, layer6_out, w1, b1); // output

        nnet::relu<data_t, data_t, typename CONFIG_T::softmax1_config>(layer6_out, res);
        //nnet::softmax<data_t, data_t, typename CONFIG_T::softmax1_config>(layer6_out, res); // output_softmax
    }

    // dnn1 with 2 layers
    template <class data_t, class res_t, typename CONFIG_T>
    void dnn1_2l(  data_t input[], res_t res[], // dnn1 with 2 FC layers 
                data_t w1[], data_t w2[],
                data_t b1[], data_t b2[]) {

        
        data_t layer2_out[CONFIG_T::fc1_out];
        #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::fc1_config>(input, layer2_out, w1, b1); // fc1
		//std::cout << "DNN1: PRINTING OUTPUT OF FC1 FIRST NODE \n";
		//std::cout << layer2_out[0] << std::endl;
		
        data_t layer3_out[CONFIG_T::fc1_out];
        #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
        nnet::relu<data_t, data_t, typename CONFIG_T::relu1_config>(layer2_out, layer3_out); // fc1_relu

        //data_t layer4_out[CONFIG_T::fc2_out];
        //#pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
        //nnet::dense<data_t, data_t, typename CONFIG_T::fc2_config>(layer3_out, layer4_out, w2, b2); // fc2

        //data_t layer5_out[CONFIG_T::fc2_out];
        //#pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
        //nnet::relu<data_t, data_t, typename CONFIG_T::relu2_config>(layer4_out, layer5_out); // fc2_relu

        data_t layer6_out[CONFIG_T::fc3_out];
        #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::output1_config>(layer3_out, layer6_out, w2, b2); // output

        nnet::relu<data_t, data_t, typename CONFIG_T::softmax1_config>(layer6_out, res);
        //nnet::softmax<data_t, data_t, typename CONFIG_T::softmax1_config>(layer6_out, res); // output_softmax
    }
    template <class data_t, class res_T, typename CONFIG_T>
    void dnn2(data_t input[], res_T res[], data_t w1[], data_t w2[], data_t w3[], data_t b1[],
              data_t b2[], data_t b3[]) {
        //  ============= Dense MLP layers 1: for transforming B into E ==================
        data_t layer2_out[CONFIG_T::fc1_out];
        #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::fc4_config>(input, layer2_out, w1, b1); // fc1

        data_t layer3_out[CONFIG_T::fc1_out];
        #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
        nnet::relu<data_t, data_t, typename CONFIG_T::relu3_config>(layer2_out, layer3_out); // fc1_relu

        data_t layer4_out[CONFIG_T::fc2_out];
        #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::fc5_config>(layer3_out, layer4_out, w2, b2); // fc2

        data_t layer5_out[CONFIG_T::fc2_out];
        #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
        nnet::relu<data_t, data_t, typename CONFIG_T::relu4_config>(layer4_out, layer5_out); // fc2_relu

        data_t layer6_out[CONFIG_T::fc3_out];
        #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::output2_config>(layer5_out, layer6_out, w3, b3); // output

        nnet::relu<data_t, data_t, typename CONFIG_T::softmax2_config>(layer6_out, res); // output_softmax
        //nnet::softmax<data_t, data_t, typename CONFIG_T::softmax2_config>(layer6_out, res); // output_softmax
    }

    template <class data_t, class res_T, typename CONFIG_T>
    void dnn3(data_t input[], res_T res[], data_t w1[], data_t w2[], data_t w3[], data_t b1[],
              data_t b2[], data_t b3[]) {

        //  ============= Dense MLP layers 1: for transforming B into E ==================

        //#pragma HLS PIPELINE

        data_t layer2_out[CONFIG_T::fc1_out];
        #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::fc7_config>(input, layer2_out, w1, b1); // fc1

        data_t layer3_out[CONFIG_T::fc1_out];
        #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
        nnet::selu<data_t, data_t, typename CONFIG_T::relu5_config>(layer2_out, layer3_out); // fc1_relu

        data_t layer4_out[CONFIG_T::fc2_out];
        #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::fc8_config>(layer3_out, layer4_out, w2, b2); // fc2

        data_t layer5_out[CONFIG_T::fc2_out];
        #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
        nnet::selu<data_t, data_t, typename CONFIG_T::relu6_config>(layer4_out, layer5_out); // fc2_relu

        data_t layer6_out[CONFIG_T::fc3_out];
        #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
        nnet::dense<data_t, data_t, typename CONFIG_T::output3_config>(layer5_out, layer6_out, w3, b3); // output

        nnet::selu<data_t, data_t, typename CONFIG_T::softmax3_config>(layer6_out, res);
        //nnet::softmax<data_t, data_t, typename CONFIG_T::softmax3_config>(layer6_out, res); // output_softmax
    }

    template<class data_T, typename CONFIG_T>
    void jedi_duplicate(
            data_T in     [CONFIG_T::N_o_p][CONFIG_T::P_p],
            data_T in_dup1[CONFIG_T::N_o_p][CONFIG_T::P_p],
            data_T in_dup2[CONFIG_T::N_o_p][CONFIG_T::P_p],
            data_T in_dup3[CONFIG_T::N_o_p][CONFIG_T::P_p]
            //data_T in_dup3[CONFIG_T::P_p][CONFIG_T::N_o_p] 
    ) {
        
        #pragma HLS PIPELINE
        data_T tmp;
        //const int unroll_factor=CONFIG_T::DPP_p;
        for(int i = 0; i < CONFIG_T::N_o_p; i++) {
            //#pragma HLS UNROLL factor=unroll_factor
            for(int j = 0; j < CONFIG_T::P_p; j++) {
                #pragma HLS UNROLL
                data_T tmp = in[i][j];
                in_dup1[i][j] = tmp;
                in_dup2[i][j] = tmp;
                in_dup3[i][j] = tmp;
            }
        }
    }


    template<class data_T, class res_T, typename CONFIG_T>
    void jedi1_mmm(
            data_T in1[CONFIG_T::N_o_p][CONFIG_T::P_p],
            data_T in2[CONFIG_T::N_o_p][CONFIG_T::P_p],
            res_T  B[CONFIG_T::N_e_p][2*CONFIG_T::P_p]
            //res_T  B[2*CONFIG_T::P_p][CONFIG_T::N_e_p]
    ) {
        #pragma HLS INLINE
        data_T B_top[CONFIG_T::N_e_p][CONFIG_T::P_p];
        data_T B_bot[CONFIG_T::N_e_p][CONFIG_T::P_p];

        const int factor_B_top=CONFIG_T::P_p;
        const int factor_B_bot=CONFIG_T::P_p;
        const int factor_B_d1 =CONFIG_T::DPP_p;
        #pragma HLS ARRAY_PARTITION variable=B_top cyclic factor=factor_B_top dim=2
        #pragma HLS ARRAY_PARTITION variable=B_bot cyclic factor=factor_B_bot dim=2
        #pragma HLS ARRAY_PARTITION variable=B_top cyclic factor=factor_B_d1 dim=1
        #pragma HLS ARRAY_PARTITION variable=B_bot cyclic factor=factor_B_d1 dim=1
        
        // mmm1, input * Rr
        jedi1_mmm_rr<data_T, res_T, typename CONFIG_T::mult_1>(in1, B_top);
        // mmm2, input * Rs
        jedi1_mmm_rs<data_T, res_T, typename CONFIG_T::mult_2>(in2, B_bot);
        // concat
        jedi_concat_t_ux<data_T, res_T, typename CONFIG_T::concat_1>(B_top, B_bot, B);
    }

    template<class data_T, class res_T, typename CONFIG_T>
    void jedi_dnn1_t_ux_opt_latn_debug(
            data_T B[CONFIG_T::N_e_p][2*CONFIG_T::P_p],
            res_T E[CONFIG_T::N_e_p][CONFIG_T::D_e_p],
            data_T w1[],
            //data_T w2[],
            //data_T w3[],
            data_T b1[] 
            //data_T b2[],
            //data_T b3[]
    ){
        data_T cache1[2 * CONFIG_T::P_p];
        data_T E_col[CONFIG_T::D_e_p];
        
        #pragma HLS ARRAY_PARTITION variable=cache1 complete
		#pragma HLS ARRAY_PARTITION variable=E_col complete
		
        const int unroll_factor=CONFIG_T::DPP_p;

        for (int cols = 0; cols < CONFIG_T::N_e_p; cols++) {
			//for (int cols = 0; cols < 1; cols++) {
		    #pragma HLS UNROLL factor=unroll_factor 
			#pragma HLS PIPELINE
            for (int rows = 0; rows < 2 * CONFIG_T::P_p; rows++) {
				//#pragma HLS PIPELINE
                //cache1[rows] = B[rows][cols]; // add to an array of size 2P
                cache1[rows] = B[cols][rows]; // add to an array of size 2P
            }
            nnet::dnn1_1l<data_T, data_T, CONFIG_T>(cache1, E_col, w1, b1);
            /*
            if(cols == 0) {
				
			std::cout << "ECOL is: ";
            for (int rows = 0; rows < CONFIG_T::D_e_p; rows++) {
                std::cout << E_col[rows] << " ";
            }
			}*/

            // copy E_col into cols of E
            for(int rows = 0; rows < CONFIG_T::D_e_p; rows++) {
				#pragma HLS UNROLL
                E[cols][rows] = E_col[rows];
			}
        }
    }

    template<class data_T, class res_T, typename CONFIG_T>
    void jedi_dnn1(
            data_T B[CONFIG_T::N_e_p][2*CONFIG_T::P_p],
            res_T E[CONFIG_T::N_e_p][CONFIG_T::D_e_p],
            data_T w1[],
            data_T w2[],
            data_T w3[],
            data_T b1[],
            data_T b2[],
            data_T b3[]
    ){
        data_T cache1[2 * CONFIG_T::P_p];
        data_T E_col[CONFIG_T::D_e_p];
        
        #pragma HLS ARRAY_PARTITION variable=cache1 complete
		#pragma HLS ARRAY_PARTITION variable=E_col complete
		
        const int unroll_factor=CONFIG_T::DPP_p;

        for (int cols = 0; cols < CONFIG_T::N_e_p; cols++) {
			//for (int cols = 0; cols < 1; cols++) {
		    #pragma HLS UNROLL factor=unroll_factor 
			#pragma HLS PIPELINE
            for (int rows = 0; rows < 2 * CONFIG_T::P_p; rows++) {
                cache1[rows] = B[cols][rows]; // add to an array of size 2P
            }
            /*
            if(cols == 0) {
				
			std::cout << "CACHE1 is: ";
            for (int rows = 0; rows < 2 * CONFIG_T::P_p; rows++) {
                std::cout << cache1[rows] << " ";
            }
            std::cout << std::endl;
			}*/

            nnet::dnn1<data_T, data_T, CONFIG_T>(cache1, E_col, w1, w2, w3, b1, b2, b3);
            /*
            if(cols == 0) {
				
			std::cout << "ECOL is: ";
            for (int rows = 0; rows < CONFIG_T::D_e_p; rows++) {
                std::cout << E_col[rows] << " ";
            }
			}*/

            // copy E_col into cols of E
            for(int rows = 0; rows < CONFIG_T::D_e_p; rows++) {
				#pragma HLS UNROLL
                E[cols][rows] = E_col[rows];
			}
        }
    }

    template<class data_T, class res_T, typename CONFIG_T>
    void jedi2_mmm(
            //data_T I[CONFIG_T::P_p][CONFIG_T::N_o_p],
            data_T I[CONFIG_T::N_o_p][CONFIG_T::P_p],
            data_T E[CONFIG_T::N_e_p][CONFIG_T::D_e_p],
            //res_T C[CONFIG_T::P_p + CONFIG_T::D_e_p][CONFIG_T::N_o_p]
            res_T C[CONFIG_T::N_o_p][CONFIG_T::P_p + CONFIG_T::D_e_p]
    ){
        
        #pragma HLS INLINE
        // declare E_bar array
        data_T E_bar[CONFIG_T::N_o_p][CONFIG_T::D_e_p];
        const int factor_E_bar=CONFIG_T::D_e_p;
        #pragma HLS ARRAY_PARTITION variable=E_bar cyclic factor = factor_E_bar
        // multiply by R_r_T
        jedi2_mmm_rrt<data_T, res_T, typename CONFIG_T::mult_3>(E, E_bar);

        // concatenate I with E_bar
        jedi_concat_t<data_T, res_T, typename CONFIG_T::concat_2>(I, E_bar, C);

    }

    template<class data_T, class res_T, typename CONFIG_T>
    void jedi_dnn2_t(
            //data_T C[CONFIG_T::P_p + CONFIG_T::D_e_p][CONFIG_T::N_o_p],
            data_T C[CONFIG_T::N_o_p][CONFIG_T::P_p + CONFIG_T::D_e_p],
            //res_T O[CONFIG_T::D_o_p][CONFIG_T::N_o_p],
            res_T O[CONFIG_T::N_o_p][CONFIG_T::D_o_p],
            data_T w1[],
            data_T w2[],
            data_T w3[],
            data_T b1[],
            data_T b2[],
            data_T b3[]) {

        // run fO neural network on C -> O
        data_T cache2[CONFIG_T::P_p + CONFIG_T::D_e_p];
        data_T O_col[CONFIG_T::D_o_p];
        
        #pragma HLS ARRAY_PARTITION variable=cache2 complete
		#pragma HLS ARRAY_PARTITION variable=O_col complete

        for (int cols = 0; cols < CONFIG_T::N_o_p; cols++) {
			#pragma HLS PIPELINE
            for (int rows = 0; rows < CONFIG_T::P_p + CONFIG_T::D_e_p; rows++){
                cache2[rows] = C[cols][rows]; // add to an array of size P+D_e
			}

            // this dense layer needs a specific config that has n_in = P+D_e, n_out = D_o
            // pass in the weights somehow, probably in jedi() as parameter
            nnet::dnn2<data_T, data_T, CONFIG_T>(cache2, O_col, w1, w2, w3, b1, b2, b3);

            // copy O_col into cols of O
            for(int rows = 0; rows < CONFIG_T::D_o_p; rows++) {
				#pragma HLS UNROLL
                O[cols][rows] = O_col[rows];
			}

        }

    }

    template<class data_T, class res_T, typename CONFIG_T>
    void jedi_dnn3_full(
            //data_T O[CONFIG_T::D_o_p][CONFIG_T::N_o_p],
            data_T O[CONFIG_T::N_o_p][CONFIG_T::D_o_p],
            res_T  res[CONFIG_T::n_out],
            data_T w1[],
            data_T w2[],
            data_T w3[],
            data_T b1[],
            data_T b2[],
            data_T b3[]
            ) {

        // initialise O_sum with 0s
        data_T O_sum[CONFIG_T::D_o_p];
        
        #pragma HLS ARRAY_PARTITION variable=O_sum complete
        #pragma HLS PIPELINE
        
        //for(int i = 0; i < CONFIG_T::D_o_p; i++) {
		//	#pragma HLS UNROLL
        //    O_sum[i] = 0;
		//}

        // sum every element in each col of O
        // CHANGE THIS, SUM HORIZONTALLY NOT VERTICALLY.
        for(int i = 0; i < CONFIG_T::N_o_p; i++) {
            for(int j = 0; j < CONFIG_T::D_o_p; j++){
                data_T tmp = (i==0)? (data_T) 0 : O_sum[j];
			    //#pragma HLS PIPELINE
			    O_sum[j] = tmp + O[i][j];
                //O_sum[rows] += O[rows][cols];
			}
        }

        // run sigma_c final neural network on O -> output
        // shape D_o -> N, sum all rows of each column to achieve dimension D_o.
        nnet::dnn3<data_T, data_T, CONFIG_T>(O_sum, res, w1, w2, w3, b1, b2, b3);
	}
    template<class data_T, class res_T, typename CONFIG_T>
    void jedi_dnn3_t(
            //data_T O[CONFIG_T::D_o_p][CONFIG_T::N_o_p],
            data_T O[CONFIG_T::N_o_p][CONFIG_T::D_o_p],
            res_T  res[CONFIG_T::n_out],
            data_T w1[],
            data_T w2[],
            data_T w3[],
            data_T b1[],
            data_T b2[],
            data_T b3[]
            ) {

        // initialise O_sum with 0s
        data_T O_sum[CONFIG_T::D_o_p];
        
        #pragma HLS ARRAY_PARTITION variable=O_sum complete
        
        //for(int i = 0; i < CONFIG_T::D_o_p; i++) {
		//	#pragma HLS UNROLL
        //    O_sum[i] = 0;
		//}

        // sum every element in each col of O
        // CHANGE THIS, SUM HORIZONTALLY NOT VERTICALLY.
        for(int i = 0; i < CONFIG_T::N_o_p; i++) {
            #pragma HLS PIPELINE
            for(int j = 0; j < CONFIG_T::D_o_p; j++){
                data_T tmp = (i==0)? (data_T) 0 : O_sum[j];
			    //#pragma HLS PIPELINE
			    O_sum[j] = tmp + O[i][j];
                //O_sum[rows] += O[rows][cols];
			}
        }

        // run sigma_c final neural network on O -> output
        // shape D_o -> N, sum all rows of each column to achieve dimension D_o.
        nnet::dnn3<data_T, data_T, CONFIG_T>(O_sum, res, w1, w2, w3, b1, b2, b3);
	}

    template<class data_T, class res_T, typename CONFIG_T, typename CONFIG_T_DNN1, typename CONFIG_T_DNN2>
    void jedi_fusion_dnn1_1layer_nofsm(
            data_T in1[CONFIG_T::N_o_p][CONFIG_T::P_p],
            res_T O[CONFIG_T_DNN2::N_o_p][CONFIG_T_DNN2::D_o_p],
            data_T w1[],
            data_T b1[],

            data_T w4[],
            data_T w5[],
            data_T w6[],
            data_T b4[],
            data_T b5[],
            data_T b6[] 
    ) {

        #pragma HLS INLINE
        #pragma HLS ARRAY_PARTITION variable=O complete dim=2
        #pragma HLS ARRAY_PARTITION variable=O complete dim=1

        data_T    E[CONFIG_T::N_o_p][CONFIG_T_DNN1::D_e_p];
        //data_T    E[CONFIG_T::N_e_p][CONFIG_T_DNN1::D_e_p];
        #pragma HLS ARRAY_PARTITION variable=E complete dim=2
        #pragma HLS ARRAY_PARTITION variable=E complete dim=1
        //const int factor_E    =CONFIG_T_DNN1::D_e_p;
        //const int factor_E_d1 =CONFIG_T_DNN1::DPP_p;
        //#pragma HLS ARRAY_PARTITION variable=E cyclic factor = factor_E    dim=2
        //#pragma HLS ARRAY_PARTITION variable=E cyclic factor = factor_E_d1 dim=1


        data_T cache_column[2 * CONFIG_T::P_p];
        #pragma HLS ARRAY_PARTITION variable=cache_column complete
        data_T E_col[CONFIG_T_DNN1::D_e_p];
		#pragma HLS ARRAY_PARTITION variable=E_col complete


        data_T acc[CONFIG_T::P_p];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        data_T cache2[CONFIG_T::P_p + CONFIG_T::D_e_p];
        data_T O_col[CONFIG_T_DNN2::D_o_p];
        #pragma HLS ARRAY_PARTITION variable=cache2 complete
		#pragma HLS ARRAY_PARTITION variable=O_col complete


        int index;

        JEDI_FUSION:
        for (int i = 0; i < CONFIG_T::N_o_p; i++) {
		    #pragma HLS PIPELINE REWIND 

            for (int k = 0; k < (CONFIG_T::N_o_p - 1); k++) {

                index = (k<i)? k : (k+1);
                for (int j = 0; j < CONFIG_T::P_p; j++) {
			    	#pragma HLS UNROLL
                    cache_column[j] = in1[i][j];
                }
                for (int j = 0; j < CONFIG_T::P_p; j++) {
			    	#pragma HLS UNROLL
                    cache_column[j+CONFIG_T::P_p] = in1[index][j];
                }
                nnet::dnn1_1l<data_T, data_T, CONFIG_T_DNN1>(cache_column, E_col, w1, b1);
                for(int rows = 0; rows < CONFIG_T_DNN1::D_e_p; rows++) {
			    	#pragma HLS UNROLL
                    E[k][rows] = E_col[rows];
			    }
            } // k

            for(int k = 0; k < (CONFIG_T::N_o_p-1);k++){
			    #pragma HLS UNROLL
                for (int j = 0; j < CONFIG_T::D_e_p; j++) {
                    //int index = i*(CONFIG_T::N_o_p-1) + k;
                    data_T tmp = (k==0)? ((data_T) 0):acc[j];  
                    acc[j] = tmp + E[k][j];
                }
            }

            for (int j = 0; j < CONFIG_T::P_p; j++) {
				#pragma HLS UNROLL
                cache2[j] = (res_T) in1[i][j]; 
            }
            for (int j = 0; j < CONFIG_T::D_e_p; j++) {
				#pragma HLS UNROLL
                cache2[j+CONFIG_T::P_p] = (res_T) acc[j]; 
            }
            nnet::dnn2<data_T, data_T, CONFIG_T_DNN2>(cache2, O_col, w4, w5, w6, b4, b5, b6);
            for(int rows = 0; rows < CONFIG_T_DNN2::D_o_p; rows++) {
				#pragma HLS UNROLL
                O[i][rows] = O_col[rows];
			}
        } // i
    } // function end

    template<class data_T, class res_T, typename CONFIG_T, typename CONFIG_T_DNN1, typename CONFIG_T_DNN2>
    void jedi_fusion_dnn1_1layer(
            data_T in1[CONFIG_T::N_o_p][CONFIG_T::P_p],
            res_T O[CONFIG_T_DNN2::N_o_p][CONFIG_T_DNN2::D_o_p],
            data_T w1[],
            data_T b1[],

            data_T w4[],
            data_T w5[],
            data_T w6[],
            data_T b4[],
            data_T b5[],
            data_T b6[]
    ) {

        #pragma HLS INLINE
        #pragma HLS ARRAY_PARTITION variable=O complete dim=2
        #pragma HLS ARRAY_PARTITION variable=O complete dim=1

        data_T    E[CONFIG_T::N_o_p][CONFIG_T_DNN1::D_e_p];
        #pragma HLS ARRAY_PARTITION variable=E complete dim=2
        #pragma HLS ARRAY_PARTITION variable=E complete dim=1

        data_T cache_column[2 * CONFIG_T::P_p];
        #pragma HLS ARRAY_PARTITION variable=cache_column complete
        data_T E_col[CONFIG_T_DNN1::D_e_p];
		#pragma HLS ARRAY_PARTITION variable=E_col complete


        data_T acc[CONFIG_T::P_p];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        data_T cache2[CONFIG_T::P_p + CONFIG_T::D_e_p];
        data_T O_col[CONFIG_T_DNN2::D_o_p];
        #pragma HLS ARRAY_PARTITION variable=cache2 complete
		#pragma HLS ARRAY_PARTITION variable=O_col complete

        int index;

        int curr_state = 0;
        JEDI_FUSION:
        for (int i = 0; i < CONFIG_T::N_o_p; i++) {
            for (int p = 0; p < 2; p++) { // target II is 2
		        #pragma HLS PIPELINE REWIND 
                switch(curr_state) {
                    case 0:{
                        for (int k = 0; k < 25; k++) {
                            index = (k<i)? k : (k+1);
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j] = in1[i][j];
                            }
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j+CONFIG_T::P_p] = in1[index][j];
                            }
                            nnet::dnn1_1l<data_T, data_T, CONFIG_T_DNN1>(cache_column, E_col, w1, b1);
                            for(int rows = 0; rows < CONFIG_T_DNN1::D_e_p; rows++) {
		    	            	#pragma HLS UNROLL
                                E[k][rows] = E_col[rows];
		    	            }
                        } // k
                        curr_state ++;
                        break;    
                    }// case0
                    case 1:{
                        for (int k = 25; k < 49; k++) {
                            index = (k<i)? k : (k+1);
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j] = in1[i][j];
                            }
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j+CONFIG_T::P_p] = in1[index][j];
                            }
                            nnet::dnn1_1l<data_T, data_T, CONFIG_T_DNN1>(cache_column, E_col, w1, b1);
                            for(int rows = 0; rows < CONFIG_T_DNN1::D_e_p; rows++) {
		    	            	#pragma HLS UNROLL
                                E[k][rows] = E_col[rows];
		    	            }
                        } // k
                        for(int k = 0; k < (CONFIG_T::N_o_p-1);k++){
		    	            #pragma HLS UNROLL
                            for (int j = 0; j < CONFIG_T::D_e_p; j++) {
                                //int index = i*(CONFIG_T::N_o_p-1) + k;
                                data_T tmp = (k==0)? ((data_T) 0):acc[j];  
                                acc[j] = tmp + E[k][j];
                            }
                        }

                        for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	        	#pragma HLS UNROLL
                            cache2[j] = (res_T) in1[i][j]; 
                        }
                        for (int j = 0; j < CONFIG_T::D_e_p; j++) {
		    	        	#pragma HLS UNROLL
                            cache2[j+CONFIG_T::P_p] = (res_T) acc[j]; 
                        }
                        nnet::dnn2<data_T, data_T, CONFIG_T_DNN2>(cache2, O_col, w4, w5, w6, b4, b5, b6);
                        for(int rows = 0; rows < CONFIG_T_DNN2::D_o_p; rows++) {
		    	        	#pragma HLS UNROLL
                            O[i][rows] = O_col[rows];
		    	        }
                        curr_state = 0;
                        break;    
                    } // case2
                    default: break;
                } // switch
                //curr_state ++;
            } // p
        }// i
    } // function end

    template<class data_T, class res_T, typename CONFIG_T, typename CONFIG_T_DNN1, typename CONFIG_T_DNN2>
    void jedi_fusion_dnn1_2layer_debug(
            data_T in1[CONFIG_T::N_o_p][CONFIG_T::P_p],
            res_T O[CONFIG_T_DNN2::N_o_p][CONFIG_T_DNN2::D_o_p],
            data_T w1[],
            data_T w2[],
            data_T b1[],
            data_T b2[],

            data_T w4[],
            data_T w5[],
            data_T w6[],
            data_T b4[],
            data_T b5[],
            data_T b6[] 
    ) {

        #pragma HLS INLINE
        #pragma HLS ARRAY_PARTITION variable=O complete dim=2
        #pragma HLS ARRAY_PARTITION variable=O complete dim=1

        data_T    E[CONFIG_T::N_o_p][CONFIG_T_DNN1::D_e_p];
        //data_T    E[CONFIG_T::N_e_p][CONFIG_T_DNN1::D_e_p];
        #pragma HLS ARRAY_PARTITION variable=E complete dim=2
        #pragma HLS ARRAY_PARTITION variable=E complete dim=1
        //const int factor_E    =CONFIG_T_DNN1::D_e_p;
        //const int factor_E_d1 =CONFIG_T_DNN1::DPP_p;
        //#pragma HLS ARRAY_PARTITION variable=E cyclic factor = factor_E    dim=2
        //#pragma HLS ARRAY_PARTITION variable=E cyclic factor = factor_E_d1 dim=1


        data_T cache_column[2 * CONFIG_T::P_p];
        #pragma HLS ARRAY_PARTITION variable=cache_column complete
        data_T E_col[CONFIG_T_DNN1::D_e_p];
		#pragma HLS ARRAY_PARTITION variable=E_col complete


        data_T acc[CONFIG_T::P_p];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        data_T cache2[CONFIG_T::P_p + CONFIG_T::D_e_p];
        data_T O_col[CONFIG_T_DNN2::D_o_p];
        #pragma HLS ARRAY_PARTITION variable=cache2 complete
		#pragma HLS ARRAY_PARTITION variable=O_col complete


        int index;

        JEDI_FUSION:
        for (int i = 0; i < CONFIG_T::N_o_p; i++) {
		    #pragma HLS PIPELINE REWIND 

            for (int k = 0; k < (CONFIG_T::N_o_p - 1); k++) {

                index = (k<i)? k : (k+1);
                for (int j = 0; j < CONFIG_T::P_p; j++) {
			    	#pragma HLS UNROLL
                    cache_column[j] = in1[i][j];
                }
                for (int j = 0; j < CONFIG_T::P_p; j++) {
			    	#pragma HLS UNROLL
                    cache_column[j+CONFIG_T::P_p] = in1[index][j];
                }
                //dnn1_opt_acc<data_T, data_T, CONFIG_T_DNN1>(cache_column, E_col, w1, w2, b1, b2);
                nnet::dnn1_2l<data_T, data_T, CONFIG_T_DNN1>(cache_column, E_col, w1, w2, b1, b2);
                for(int rows = 0; rows < CONFIG_T_DNN1::D_e_p; rows++) {
			    	#pragma HLS UNROLL
                    E[k][rows] = E_col[rows];
			    }
            } // k

            for(int k = 0; k < (CONFIG_T::N_o_p-1);k++){
			    #pragma HLS UNROLL
                for (int j = 0; j < CONFIG_T::D_e_p; j++) {
                    //int index = i*(CONFIG_T::N_o_p-1) + k;
                    data_T tmp = (k==0)? ((data_T) 0):acc[j];  
                    acc[j] = tmp + E[k][j];
                }
            }

            for (int j = 0; j < CONFIG_T::P_p; j++) {
				#pragma HLS UNROLL
                cache2[j] = (res_T) in1[i][j]; 
            }
            for (int j = 0; j < CONFIG_T::D_e_p; j++) {
				#pragma HLS UNROLL
                cache2[j+CONFIG_T::P_p] = (res_T) acc[j]; 
            }
            nnet::dnn2<data_T, data_T, CONFIG_T_DNN2>(cache2, O_col, w4, w5, w6, b4, b5, b6);
            for(int rows = 0; rows < CONFIG_T_DNN2::D_o_p; rows++) {
				#pragma HLS UNROLL
                O[i][rows] = O_col[rows];
			}
        } // i
    } // function end


    template<class data_T, class res_T, typename CONFIG_T, typename CONFIG_T_DNN1, typename CONFIG_T_DNN2>
    void jedi_fusion_dnn1_2layer(
            data_T in1[CONFIG_T::N_o_p][CONFIG_T::P_p],
            res_T O[CONFIG_T_DNN2::N_o_p][CONFIG_T_DNN2::D_o_p],
            data_T w1[],
            data_T w2[],
            data_T b1[],
            data_T b2[],

            data_T w4[],
            data_T w5[],
            data_T w6[],
            data_T b4[],
            data_T b5[],
            data_T b6[] 
    ) {

        #pragma HLS INLINE
        #pragma HLS ARRAY_PARTITION variable=O complete dim=2
        #pragma HLS ARRAY_PARTITION variable=O complete dim=1

        data_T    E[CONFIG_T::N_o_p][CONFIG_T_DNN1::D_e_p];
        #pragma HLS ARRAY_PARTITION variable=E complete dim=2
        #pragma HLS ARRAY_PARTITION variable=E complete dim=1


        data_T cache_column[2 * CONFIG_T::P_p];
        #pragma HLS ARRAY_PARTITION variable=cache_column complete
        data_T E_col[CONFIG_T_DNN1::D_e_p];
		#pragma HLS ARRAY_PARTITION variable=E_col complete


        data_T acc[CONFIG_T::P_p];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        data_T cache2[CONFIG_T::P_p + CONFIG_T::D_e_p];
        data_T O_col[CONFIG_T_DNN2::D_o_p];
        #pragma HLS ARRAY_PARTITION variable=cache2 complete
		#pragma HLS ARRAY_PARTITION variable=O_col complete

        //const int factor_dpp =CONFIG_T::DPP_p;

        int index;

        int curr_state = 0;
        JEDI_FUSION:
        for (int i = 0; i < CONFIG_T::N_o_p; i++) {
            for (int p = 0; p < 3; p++) { // target II is 3
		        #pragma HLS PIPELINE REWIND 
                switch(curr_state) {
                    case 0:{
                        for (int k = 0; k < 17; k++) {
                            index = (k<i)? k : (k+1);
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j] = in1[i][j];
                            }
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j+CONFIG_T::P_p] = in1[index][j];
                            }
                            //dnn1_opt_acc<data_T, data_T, CONFIG_T_DNN1>(cache_column, E_col, w1, w2, b1, b2);
                            nnet::dnn1_2l<data_T, data_T, CONFIG_T_DNN1>(cache_column, E_col, w1, w2, b1, b2);
                            for(int rows = 0; rows < CONFIG_T_DNN1::D_e_p; rows++) {
		    	            	#pragma HLS UNROLL
                                E[k][rows] = E_col[rows];
		    	            }
                        } // k
                        curr_state ++;
                        break;    
                    }// case0
                    case 1:{
                        for (int k = 17; k < 34; k++) {
                            index = (k<i)? k : (k+1);
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j] = in1[i][j];
                            }
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j+CONFIG_T::P_p] = in1[index][j];
                            }
                            nnet::dnn1_2l<data_T, data_T, CONFIG_T_DNN1>(cache_column, E_col, w1, w2, b1, b2);
                            for(int rows = 0; rows < CONFIG_T_DNN1::D_e_p; rows++) {
		    	            	#pragma HLS UNROLL
                                E[k][rows] = E_col[rows];
		    	            }
                        } // k
                        curr_state ++;
                        break;    
                    }// case0
                    case 2:{
                        for (int k = 34; k < 49; k++) {
                            index = (k<i)? k : (k+1);
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j] = in1[i][j];
                            }
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j+CONFIG_T::P_p] = in1[index][j];
                            }
                            nnet::dnn1_2l<data_T, data_T, CONFIG_T_DNN1>(cache_column, E_col, w1, w2, b1, b2);
                            for(int rows = 0; rows < CONFIG_T_DNN1::D_e_p; rows++) {
		    	            	#pragma HLS UNROLL
                                E[k][rows] = E_col[rows];
		    	            }
                        } // k
                        for(int k = 0; k < (CONFIG_T::N_o_p-1);k++){
		    	            #pragma HLS UNROLL
                            for (int j = 0; j < CONFIG_T::D_e_p; j++) {
                                //int index = i*(CONFIG_T::N_o_p-1) + k;
                                data_T tmp = (k==0)? ((data_T) 0):acc[j];  
                                acc[j] = tmp + E[k][j];
                            }
                        }

                        for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	        	#pragma HLS UNROLL
                            cache2[j] = (res_T) in1[i][j]; 
                        }
                        for (int j = 0; j < CONFIG_T::D_e_p; j++) {
		    	        	#pragma HLS UNROLL
                            cache2[j+CONFIG_T::P_p] = (res_T) acc[j]; 
                        }
                        nnet::dnn2<data_T, data_T, CONFIG_T_DNN2>(cache2, O_col, w4, w5, w6, b4, b5, b6);
                        for(int rows = 0; rows < CONFIG_T_DNN2::D_o_p; rows++) {
		    	        	#pragma HLS UNROLL
                            O[i][rows] = O_col[rows];
		    	        }
                        curr_state = 0;
                        break;    
                    } // case2
                    default: break;
                } // switch
                //curr_state ++;
            } // p
        }// i
    } // function end


    template<class data_T, class res_T, typename CONFIG_T, typename CONFIG_T_DNN1, typename CONFIG_T_DNN2>
    void jedi_fusion_opt_latn2(
            data_T in1[CONFIG_T::N_o_p][CONFIG_T::P_p],
            res_T O[CONFIG_T_DNN2::N_o_p][CONFIG_T_DNN2::D_o_p],
            data_T w1[],
            data_T w2[],
            data_T b1[],
            data_T b2[],

            data_T w4[],
            data_T w5[],
            data_T w6[],
            data_T b4[],
            data_T b5[],
            data_T b6[]
    ) {

        #pragma HLS INLINE
        #pragma HLS ARRAY_PARTITION variable=O complete dim=2
        #pragma HLS ARRAY_PARTITION variable=O complete dim=1

        data_T    E[CONFIG_T::N_o_p][CONFIG_T_DNN1::D_e_p];
        //data_T    E[CONFIG_T::N_e_p][CONFIG_T_DNN1::D_e_p];
        #pragma HLS ARRAY_PARTITION variable=E complete dim=2
        #pragma HLS ARRAY_PARTITION variable=E complete dim=1
        //const int factor_E    =CONFIG_T_DNN1::D_e_p;
        //const int factor_E_d1 =CONFIG_T_DNN1::DPP_p;
        //#pragma HLS ARRAY_PARTITION variable=E cyclic factor = factor_E    dim=2
        //#pragma HLS ARRAY_PARTITION variable=E cyclic factor = factor_E_d1 dim=1


        data_T cache_column[2 * CONFIG_T::P_p];
        #pragma HLS ARRAY_PARTITION variable=cache_column complete
        data_T E_col[CONFIG_T_DNN1::D_e_p];
		#pragma HLS ARRAY_PARTITION variable=E_col complete


        data_T acc[CONFIG_T::P_p];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        data_T cache2[CONFIG_T::P_p + CONFIG_T::D_e_p];
        data_T O_col[CONFIG_T_DNN2::D_o_p];
        #pragma HLS ARRAY_PARTITION variable=cache2 complete
		#pragma HLS ARRAY_PARTITION variable=O_col complete

        //const int factor_dpp =CONFIG_T::DPP_p;

        int index;

        //#pragma HLS ALLOCATION function instances=dnn1 limit=13
        //#pragma HLS ALLOCATION instances=dnn1<CONFIG_T_DNN1> limit=13 function
        int curr_state = 0;
        JEDI_FUSION:
        for (int i = 0; i < CONFIG_T::N_o_p; i++) {
            //for (int curr_state = 0; curr_state < 3; curr_state++) { // target II is 3
            for (int p = 0; p < 2; p++) { // target II is 2
		        #pragma HLS PIPELINE REWIND 
                switch(curr_state) {
                    case 0:{
                        for (int k = 0; k < 25; k++) {
                            index = (k<i)? k : (k+1);
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j] = in1[i][j];
                            }
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j+CONFIG_T::P_p] = in1[index][j];
                            }
                            nnet::dnn1_2l<data_T, data_T, CONFIG_T_DNN1>(cache_column, E_col, w1, w2, b1, b2);
                            for(int rows = 0; rows < CONFIG_T_DNN1::D_e_p; rows++) {
		    	            	#pragma HLS UNROLL
                                E[k][rows] = E_col[rows];
		    	            }
                        } // k
                        curr_state ++;
                        break;    
                    }// case0
                    case 1:{
                        for (int k = 25; k < 49; k++) {
                            index = (k<i)? k : (k+1);
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j] = in1[i][j];
                            }
                            for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	            	#pragma HLS UNROLL
                                cache_column[j+CONFIG_T::P_p] = in1[index][j];
                            }
                            //dnn1<CONFIG_T_DNN1>(cache_column, E_col, w1, w2, w3, b1, b2, b3);
                            nnet::dnn1_2l<data_T, data_T, CONFIG_T_DNN1>(cache_column, E_col, w1, w2, b1, b2);
                            for(int rows = 0; rows < CONFIG_T_DNN1::D_e_p; rows++) {
		    	            	#pragma HLS UNROLL
                                E[k][rows] = E_col[rows];
		    	            }
                        } // k
                        for(int k = 0; k < (CONFIG_T::N_o_p-1);k++){
		    	            #pragma HLS UNROLL
                            for (int j = 0; j < CONFIG_T::D_e_p; j++) {
                                //int index = i*(CONFIG_T::N_o_p-1) + k;
                                data_T tmp = (k==0)? ((data_T) 0):acc[j];  
                                acc[j] = tmp + E[k][j];
                            }
                        }

                        for (int j = 0; j < CONFIG_T::P_p; j++) {
		    	        	#pragma HLS UNROLL
                            cache2[j] = (res_T) in1[i][j]; 
                        }
                        for (int j = 0; j < CONFIG_T::D_e_p; j++) {
		    	        	#pragma HLS UNROLL
                            cache2[j+CONFIG_T::P_p] = (res_T) acc[j]; 
                        }
                        nnet::dnn2<data_T, data_T, CONFIG_T_DNN2>(cache2, O_col, w4, w5, w6, b4, b5, b6);
                        for(int rows = 0; rows < CONFIG_T_DNN2::D_o_p; rows++) {
		    	        	#pragma HLS UNROLL
                            O[i][rows] = O_col[rows];
		    	        }
                        curr_state = 0;
                        break;    
                    } // case2
                    default: break;
                } // switch
                //curr_state ++;
            } // p
        }// i
    } // function end





}

#endif
