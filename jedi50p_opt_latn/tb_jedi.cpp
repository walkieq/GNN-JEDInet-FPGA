#include <iomanip>
#include <fstream>
#include <iostream>
#include "math.h"
#include "jedi.h"
#include "weights50p/input.h" // this file should contain input

int main() {

    result_t result[N_OUTPUT_3];


    //jedi(input[0], Rr, Rr_T, Rs, result);
    int BATCH =3; 
    
    input_t jedi_in[P][N_o];
    input_t jedi_in_t[N_o][P];
    for (int k=0;k<BATCH;k++){

        //for(int i=0;i<P;i++){
        //    for(int j=0;j<N_o;j++){
        //        jedi_in[i][j]=input[k][i][j];
        //    }
        //}
        for(int i=0;i<P;i++){
            for(int j=0;j<N_o;j++){
                jedi_in_t[j][i]=input[k][i][j];
            }
        }
        for(int i = 0; i < N_OUTPUT_3; i++){
            result[i] = 0;
        }
         
        jedi(jedi_in_t, result);

        std::cout << "the final selu output for n = " << N_OUTPUT_3 << " is: ";
        for(int i = 0; i < N_OUTPUT_3; i++) {
            std::cout << result[i] << ", ";
        }
        std::cout << "\n";
    }

    std::cout << "testbench ended \n";
}
