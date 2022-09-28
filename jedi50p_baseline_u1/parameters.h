#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "../nnet_utils/nnet_helpers.h"
#include "../nnet_utils/nnet_activation.h"
#include "../nnet_utils/nnet_activation_stream.h"
#include "../nnet_utils/nnet_dense.h"
#include "../nnet_utils/nnet_dense_compressed.h"
#include "../nnet_utils/nnet_dense_stream.h"


#include "../nnet_utils/nnet_activation.h"
#include "../nnet_utils/nnet_mult.h"
#include "../nnet_utils/nnet_dense.h"

//hls-fpga-machine-learning insert layer-config
struct mult_1_struct {
    static const unsigned n_row1 = P;
    static const unsigned n_col1 = N_o;
    static const unsigned n_row2 = N_o;
    static const unsigned n_col2 = N_e;
    static const unsigned DPP_p = DPP;
    typedef jedi_r_id_t r_id_t;
    //static const unsigned io_type = nnet::io_parallel;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct mult_2_struct{
    static const unsigned n_row1 = P;
    static const unsigned n_col1 = N_o;
    static const unsigned n_row2 = N_o;
    static const unsigned n_col2 = N_e;
    static const unsigned DPP_p = DPP;
    typedef jedi_r_id_t r_id_t;
    //static const unsigned io_type = nnet::io_parallel;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct mult_3_struct {
    static const unsigned n_row1 = D_e;
    static const unsigned n_col1 = N_e;
    static const unsigned n_row2 = N_e;
    static const unsigned n_col2 = N_o;
    //static const unsigned io_type = nnet::io_parallel;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct concat_1_struct {
    static const unsigned n_row1 = P;
    static const unsigned n_col1 = N_e;
    static const unsigned n_row2 = P;
    static const unsigned n_col2 = N_e;
    static const unsigned DPP_p = DPP;
};

struct concat_2_struct {
    static const unsigned n_row1 = P;
    static const unsigned n_col1 = N_o;
    static const unsigned n_row2 = D_e;
    static const unsigned n_col2 = N_o;
};

struct jedi1_config {
    static const unsigned P_p = P;
    static const unsigned N_o_p = N_o;
    static const unsigned N_e_p = N_e;
    static const unsigned DPP_p = DPP;
    typedef jedi_r_id_t r_id_t;
    typedef mult_1_struct mult_1;
    typedef mult_2_struct mult_2;
    typedef concat_1_struct concat_1;

};

struct jedi2_config {
    static const unsigned P_p = P;
    static const unsigned N_o_p = N_o;
    static const unsigned N_e_p = N_e;
    static const unsigned D_e_p = D_e;
    typedef mult_3_struct mult_3;
    typedef concat_2_struct concat_2;

};

// fc1 dnn1
struct fc1_config_struct : nnet::dense_config {
    static const unsigned n_in = 2*P;
    static const unsigned n_out = N_LAYER_2;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = R1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 2*P*N_LAYER_2;
    static const bool store_weights_in_bram = false;
    //typedef ap_fixed<16,6> accum_t;
    //typedef float accum_t;
    typedef jedi_accum_t accum_t;
    typedef fc1_bias_t bias_t;
    typedef fc1_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// fc1_relu
struct relu1_config_struct : nnet::activ_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    //typedef ap_fixed<18,8> table_t;
    //typedef float table_t;
    typedef model_default_t table_t;
};

// fc2 dnn1
struct fc2_config_struct : nnet::dense_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned n_out = N_LAYER_4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = R1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = N_LAYER_2 * N_LAYER_4;
    static const bool store_weights_in_bram = false;
    //typedef ap_fixed<16,6> accum_t;
    //typedef float accum_t;
    typedef jedi_accum_t accum_t;
    typedef fc2_bias_t bias_t;
    typedef fc2_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct relu2_config_struct : nnet::activ_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    //typedef ap_fixed<18,8> table_t;
    //typedef float table_t;
    typedef model_default_t table_t;
};

// fc3 dnn1
struct output1_config_struct : nnet::dense_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned n_out = N_OUTPUT_1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = R1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = N_LAYER_4 * N_OUTPUT_1;
    static const bool store_weights_in_bram = false;
	//typedef ap_fixed<16,6> accum_t;
    //typedef float accum_t;    
    typedef jedi_accum_t accum_t;
    typedef fc3_bias_t bias_t;
    typedef fc3_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct selu1_config_struct : nnet::activ_config {
    static const unsigned n_in = N_OUTPUT_1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    //typedef ap_fixed<18,8> table_t;
    //typedef float table_t;
    typedef model_default_t table_t;
};

/*
struct softmax1_config_struct : nnet::activ_config {
    static const unsigned n_in = N_OUTPUT_1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::latency;
    typedef ap_fixed<18,8> exp_table_t;
    typedef ap_fixed<18,8> inv_table_t;
};*/

// fc1 dnn2
struct fc4_config_struct : nnet::dense_config {
    static const unsigned n_in = P+D_e;
    static const unsigned n_out = N_LAYER_6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = R2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = (P+D_e) * N_LAYER_6;
    static const bool store_weights_in_bram = false;
    //typedef ap_fixed<16,6> accum_t;
    //typedef float accum_t;
    typedef jedi_accum_t accum_t;
    typedef fc4_bias_t bias_t;
    typedef fc4_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct relu3_config_struct : nnet::activ_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    //typedef ap_fixed<18,8> table_t;
    //typedef float table_t;
    typedef model_default_t table_t;
};

// fc2 dnn2
struct fc5_config_struct : nnet::dense_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned n_out = N_LAYER_8;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = R2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = N_LAYER_6 * N_LAYER_8;
    static const bool store_weights_in_bram = false;
    //typedef ap_fixed<16,6> accum_t;
    //typedef float accum_t;
    typedef jedi_accum_t accum_t;
    typedef fc5_bias_t bias_t;
    typedef fc5_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct relu4_config_struct : nnet::activ_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    //typedef ap_fixed<18,8> table_t;
    //typedef float table_t;
    typedef model_default_t table_t;
};

// fc3 dnn2
struct output2_config_struct : nnet::dense_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned n_out = N_OUTPUT_2;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = R2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = N_LAYER_8 * N_OUTPUT_2;
    static const bool store_weights_in_bram = false;
    //typedef ap_fixed<16,6> accum_t;
    //typedef float accum_t;
    typedef jedi_accum_t accum_t;
    typedef fc6_bias_t bias_t;
    typedef fc6_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct selu2_config_struct : nnet::activ_config {
    static const unsigned n_in = N_OUTPUT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    //typedef ap_fixed<18,8> table_t;
    //typedef float table_t;
    typedef model_default_t table_t;
};

/*
struct softmax2_config_struct : nnet::activ_config {
    static const unsigned n_in = N_OUTPUT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::latency;
    typedef ap_fixed<18,8> exp_table_t;
    typedef ap_fixed<18,8> inv_table_t;
};*/

// fc1 dnn3
struct fc7_config_struct : nnet::dense_config {
    static const unsigned n_in = D_o;
    static const unsigned n_out = N_LAYER_10;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = R3;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = N_LAYER_10 * D_o;
    static const bool store_weights_in_bram = false;
    //typedef ap_fixed<16,6> accum_t;
    //typedef float accum_t;
    typedef jedi_accum_t accum_t;
    typedef fc7_bias_t bias_t;
    typedef fc7_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct relu5_config_struct : nnet::activ_config {
    static const unsigned n_in = N_LAYER_10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    //typedef ap_fixed<18,8> table_t;
    //typedef float table_t;
    typedef model_default_t table_t;
};

// fc2 dnn3
struct fc8_config_struct : nnet::dense_config {
    static const unsigned n_in = N_LAYER_10;
    static const unsigned n_out = N_LAYER_12;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = R3;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = N_LAYER_10 * N_LAYER_12;
    static const bool store_weights_in_bram = false;
    //typedef ap_fixed<16,6> accum_t;
    //typedef float accum_t;
    typedef jedi_accum_t accum_t;
    typedef fc8_bias_t bias_t;
    typedef fc8_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct relu6_config_struct : nnet::activ_config {
    static const unsigned n_in = N_LAYER_12;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    //typedef ap_fixed<18,8> table_t;
    //typedef float table_t;
    typedef model_default_t table_t;
};

// fc3 dnn3
struct output3_config_struct : nnet::dense_config {
    static const unsigned n_in = N_LAYER_12;
    static const unsigned n_out = N_OUTPUT_3;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = R3;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = N_LAYER_12 * N_OUTPUT_3;
    static const bool store_weights_in_bram = false;
    //typedef ap_fixed<16,6> accum_t;
    //typedef float accum_t;
    typedef jedi_accum_t accum_t;
    typedef fc9_bias_t bias_t;
    typedef fc9_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct selu3_config_struct : nnet::activ_config {
    static const unsigned n_in = N_OUTPUT_3;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    //typedef ap_fixed<18,8> table_t;
    //typedef float table_t;
    typedef model_default_t table_t;
};

/*
struct softmax3_config_struct : nnet::activ_config {
    static const unsigned n_in = N_OUTPUT_3;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::latency;
    typedef ap_fixed<18,8> exp_table_t;
    typedef ap_fixed<18,8> inv_table_t;
};*/


struct dense1_config {
    // add in all the layers of dnn1 here
    static const unsigned P_p = P;
    static const unsigned D_e_p = D_e;
    static const unsigned N_o_p = N_o;
    static const unsigned N_e_p = N_e;
    static const unsigned fc1_out = N_LAYER_2;
    static const unsigned fc2_out = N_LAYER_4;
    static const unsigned fc3_out = N_OUTPUT_1;
    static const unsigned DPP_p = DPP;
    typedef fc1_config_struct fc1_config;
    typedef relu1_config_struct relu1_config;
    typedef fc2_config_struct fc2_config;
    typedef relu2_config_struct relu2_config;
    typedef output1_config_struct output1_config;
    typedef selu1_config_struct softmax1_config;

};

struct dense2_config {
    // add in all the layers of dnn2 here
    static const unsigned P_p = P;
    static const unsigned D_e_p = D_e;
    static const unsigned D_o_p = D_o;
    static const unsigned N_o_p = N_o;
    static const unsigned fc1_out = N_LAYER_6;
    static const unsigned fc2_out = N_LAYER_8;
    static const unsigned fc3_out = N_OUTPUT_2;
    typedef fc4_config_struct fc4_config;
    typedef relu3_config_struct relu3_config;
    typedef fc5_config_struct fc5_config;
    typedef relu4_config_struct relu4_config;
    typedef output2_config_struct output2_config;
    typedef selu2_config_struct softmax2_config;
};

struct dense3_config {
    // add in all the layers of dnn3 here
    static const unsigned D_o_p = D_o;
    static const unsigned N_o_p = N_o;
    static const unsigned n_out = N_OUTPUT_3;
    static const unsigned fc1_out = N_LAYER_10;
    static const unsigned fc2_out = N_LAYER_12;
    static const unsigned fc3_out = N_OUTPUT_3;
    typedef fc7_config_struct fc7_config;
    typedef relu5_config_struct relu5_config;
    typedef fc8_config_struct fc8_config;
    typedef relu6_config_struct relu6_config;
    typedef output3_config_struct output3_config;
    typedef selu3_config_struct softmax3_config;
};

#endif



















