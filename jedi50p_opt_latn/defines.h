#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "../nnet_utils/nnet_types.h"

// add macros e.g #define
#define P 16 // change to 16, set to 2 for testing
#define N_o  50
#define N_e (N_o * (N_o - 1))
#define D_e 12
#define D_o  14
#define N_INPUT_1_1 P
#define N_INPUT_1_2 N_o
//#define N_LAYER_2 50
//#define N_LAYER_4 25
#define N_LAYER_6 32
#define N_LAYER_8 16
#define N_LAYER_10 32
#define N_LAYER_12 16
#define N_OUTPUT_1 D_e
#define N_OUTPUT_2 D_o
#define N_OUTPUT_3 5 // number of jet classes


#define RO  1;
#define DPP 25;// only used to show the real number of fR, not used in the custom FSM  


#define R1 1;
#define R2 RO;
#define R3 RO;


#define bits_total 24
#define bits_integer 12
//#define bits_total 26
//#define bits_integer 14


typedef ap_fixed<bits_total, bits_integer> model_default_t; 
typedef model_default_t model_params_t; 

typedef ap_uint<6> jedi_r_id_t;
typedef ap_fixed<32,16> jedi_accum_t;


typedef model_default_t result_t;
typedef model_default_t itmdia_t;

typedef model_params_t input_t; // default <16,6>, changed to <16,10>
typedef model_params_t fc1_weight_t;
typedef model_params_t fc1_bias_t;
typedef model_params_t fc2_weight_t;
typedef model_params_t fc2_bias_t;
typedef model_params_t fc3_weight_t;
typedef model_params_t fc3_bias_t;

typedef model_params_t fc4_weight_t;
typedef model_params_t fc4_bias_t;
typedef model_params_t fc5_weight_t;
typedef model_params_t fc5_bias_t;
typedef model_params_t fc6_weight_t;
typedef model_params_t fc6_bias_t;

typedef model_params_t fc7_weight_t;
typedef model_params_t fc7_bias_t;
typedef model_params_t fc8_weight_t;
typedef model_params_t fc8_bias_t;
typedef model_params_t fc9_weight_t;
typedef model_params_t fc9_bias_t;


#endif
