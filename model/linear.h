#ifndef __CPP_MODEL_LINEAR_H__
#define __CPP_MODEL_LINEAR_H__

template<typename T, int DIM_IN, int DIM_OUT>
struct LinearParameter{
    T weights[DIM_IN][DIM_OUT];
    T bias[DIM_OUT];
};
template<typename T, int DIM_IN, int DIM_OUT>
void singleLinearForward(T (&input)[DIM_IN], T (&output)[DIM_OUT], LinearParameter<T, DIM_IN, DIM_OUT> &param){
    for(int i = 0; i < DIM_OUT; ++i){
        output[i] = param.bias[i];
    }
    for(int i = 0; i < DIM_IN; ++i){
        for(int j = 0; j < DIM_OUT; ++j){
            output[j] += input[i] * param.weights[i][j];
        }
    }
}
template<typename T, int DIM_IN, int DIM_OUT, int SEQ>
void linearForward(T (&input)[SEQ][DIM_IN], T (&output)[SEQ][DIM_OUT], LinearParameter<T, DIM_IN, DIM_OUT> &param){
    for(int i = 0; i < SEQ; ++i){
        singleLinearForward<T, DIM_IN, DIM_OUT>(input[i], output[i], param);
    }
}

#endif