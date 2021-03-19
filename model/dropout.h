#ifndef __CPP_MODEL_DROPOUT_H__
#define __CPP_MODEL_DROPOUT_H__

template<typename T, int DIM>
void dropoutForward(T (&input)[DIM], T (&output)[DIM], T dropout_rate){
    for(int i = 0; i < DIM; ++i){
        if(input[i] < dropout_rate){
            output[i] = 0;
        }else{
            output[i] = input[i];
        }
    }
}

#endif