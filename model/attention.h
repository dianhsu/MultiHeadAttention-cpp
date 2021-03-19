#ifndef __CPP_MODEL_ATTENTION_H__
#define __CPP_MODEL_ATTENTION_H__

#include <cmath>

#include "linear.h"
#include "softmax.h"
#include "dropout.h"

template<typename T, int DIM>
struct ScaleDotSelfAttentionForwardParameter{
    LinearParameter<T, DIM, DIM> q_param, k_param, v_param;
};
template<typename T, int DIM, int HEAD_SIZE>
struct MultiHeadAttentionForwardParameter{
    ScaleDotSelfAttentionForwardParameter<T, DIM> scaleDotSelfAttentionForwardParameter[HEAD_SIZE];
    LinearParameter<T, DIM*HEAD_SIZE, DIM> lp;
    T dr;
};
template<typename T, int DIM, int Q_LEN, int K_LEN>
void scaleDotSelfAttentionForward(T (&Q)[Q_LEN][DIM], T (&K)[K_LEN][DIM], T (&V)[K_LEN][DIM], T (&output)[Q_LEN][DIM],T scale, T dr, ScaleDotSelfAttentionForwardParameter<T, DIM> &param){
    T q_tmp_1[Q_LEN][DIM];
    T q_tmp[Q_LEN][DIM];
    T k_tmp[K_LEN][DIM];
    T v_tmp[K_LEN][DIM];
    linearForward(Q, q_tmp_1, param.q_param);
    linearForward(K, k_tmp, param.k_param);
    linearForward(V, v_tmp, param.v_param);
    for(int i = 0; i < Q_LEN; ++i){
        dropoutForward<T, DIM>(q_tmp_1[i], q_tmp[i], dr);
        for(int j = 0; j < DIM; ++j){
            q_tmp[i][j] *= scale;
        }
    }
    T nex_tmp[Q_LEN][K_LEN];
    for(int i = 0; i < Q_LEN; ++i){
        for(int j = 0; j < K_LEN; ++j){
            nex_tmp[i][j] = 0;
            for(int k = 0; k < DIM; ++k){
                nex_tmp[i][j] += q_tmp[i][k] * k_tmp[j][k];
            }
        }
    }
    T nex_tmp_2[Q_LEN][K_LEN];
    softmaxForward<T, K_LEN, Q_LEN>(nex_tmp, nex_tmp_2);
    for(int i = 0; i < Q_LEN; ++i){
        for(int j = 0; j < DIM; ++j){
            output[i][j] = 0;
            for(int k = 0; k < K_LEN; ++k){
                output[i][j] += nex_tmp_2[i][k] * v_tmp[k][j];
            }
        }
    }
}
template<typename  T, int DIM, int HEAD_SIZE, int Q_LEN, int K_LEN>
void multiHeadAttentionForward(T (&Q)[Q_LEN][DIM], T (&K)[K_LEN][DIM], T (&V)[K_LEN][DIM], T (&output)[Q_LEN][DIM], MultiHeadAttentionForwardParameter<T, DIM, HEAD_SIZE> &param){
    T scale = 1.0 / sqrt((double) DIM * 1.0 / HEAD_SIZE);
    T tmp[HEAD_SIZE][Q_LEN][DIM];
    for(int i = 0; i < HEAD_SIZE; ++i){
        scaleDotSelfAttentionForward<T, DIM, Q_LEN, K_LEN>(Q, K, V, tmp[i], scale, param.dr, param.scaleDotSelfAttentionForwardParameter[0]);
    }
    T fc_tmp[Q_LEN][DIM * HEAD_SIZE];
    for(int h = 0; h < HEAD_SIZE; ++h){
        for(int i = 0; i < Q_LEN; ++i){
            for(int j = 0; j < DIM; ++j){
                fc_tmp[i][h*HEAD_SIZE+j] = tmp[h][i][j];
            }
        }
    }
    linearForward<T, DIM * HEAD_SIZE, DIM, Q_LEN>(fc_tmp, output, param.lp);
}
#endif