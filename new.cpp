//
// Created by dianhsu on 2021/03/19.
//

#include <iostream>

#ifdef __linux__
#include <sys/time.h>
#else
#include <ctime>
#endif

#include "my_attention.h"

const int DIM = 64;
const int HEAD_SIZE = 8;
const int K_LEN = 20;
const int Q_LEN = 20;
typedef float T;
T Q[Q_LEN][DIM];
T K[K_LEN][DIM];
T V[K_LEN][DIM];
T output[Q_LEN][DIM];

int main() {

    int cnt[] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};
    auto param = new MyMultiHeadAttentionForwardParameter<T, DIM, HEAD_SIZE>();
    for(int item : cnt){
#ifdef __linux__
        struct timeval start{};
        gettimeofday(&start, nullptr);
#else
        auto start = clock();
#endif
        for(int j = 0; j < item; ++j){
            myMultiHeadAttentionForward<T, DIM, HEAD_SIZE, Q_LEN, K_LEN>(Q, K, V, output, *param);
        }
#ifdef __linux__
        struct timeval end{};
        gettimeofday(&end, nullptr);
        long long diff_ms = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))/1000;
#else
        auto end = clock();
        long long diff_ms = end - start;
#endif
        int data_size = sizeof(Q)*3;
        printf("[data size]: %d bytes, [throughput]: %.3f MB/s [batch size]: %d, [all time]: %lld ms, [avg time]: %.3f ms\n",
               data_size * item, data_size * (1e-3) * item / (double)diff_ms, item, diff_ms,
               (double)diff_ms / item);
    }
    return 0;
}
