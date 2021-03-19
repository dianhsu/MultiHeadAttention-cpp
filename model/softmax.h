#ifndef __CPP_MODEL_SOFTMAX_H__
#define __CPP_MODEL_SOFTMAX_H__

template<typename T, int DIM, int SEQ>
void softmaxForward(T (&input)[SEQ][DIM], T (&output)[SEQ][DIM]){
    T tmp[DIM];
    for(int i = 0; i < DIM; ++i){
        tmp[i] = 0;
    }
    for(int i = 0; i < SEQ; ++i){
        for(int j = 0; j < DIM; ++j){
            tmp[j] += input[i][j];
        }
    }
    for(int i = 0; i < SEQ; ++i){
        for(int j = 0; j < DIM; ++j){
            output[i][j] = input[i][j] / tmp[j];
        }
    }
}

#endif