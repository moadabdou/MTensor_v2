#include <MTensor/kernels.hpp>
#include <immintrin.h>
#include <cmath>
#include <omp.h>


KERNEL_API void fused_sgd_kernel_avx512( 
    float *param,
    float *grad,
    int64_t N,
    float lr,
    float weight_decay
){

    const int vec_width = 16; // 16 floats per __m512

    __m512 v_lr = _mm512_set1_ps(lr);
    __m512 v_weight_decay = _mm512_set1_ps(weight_decay);

    #pragma omp parallel for 
    for (int64_t i = 0; i <= N - vec_width; i+=vec_width){

        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 p_old = _mm512_loadu_ps(&param[i]);
        
        g = _mm512_add_ps(g, _mm512_mul_ps( v_weight_decay, p_old ));
        // Update param
        __m512 p_new = _mm512_sub_ps(p_old, _mm512_mul_ps(v_lr, g));

        // Store results
        _mm512_storeu_ps(&param[i], p_new); 

    }

    // Tail handling (scalar)
    int rem = N % vec_width;
    int start = N - rem;
    for (int i = start; i < N; ++i) {
        param[i] -= lr * (grad[i] + weight_decay*param[i]);
    }


}


KERNEL_API void fused_sgd_m_kernel_avx512( 
    float *param,
    float *grad,
    float *v, 
    int64_t N,
    float u,
    float lr,
    float weight_decay
){

    const int vec_width = 16; // 16 floats per __m512

    __m512 v_lr = _mm512_set1_ps(lr);
    __m512 v_u = _mm512_set1_ps(u);
    __m512 v_weight_decay = _mm512_set1_ps(weight_decay);

    #pragma omp parallel for 
    for (int64_t i = 0; i <= N - vec_width; i+=vec_width){

        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 v_old = _mm512_loadu_ps(&v[i]);
        __m512 p_old = _mm512_loadu_ps(&param[i]);

        g = _mm512_add_ps(g, _mm512_mul_ps( v_weight_decay, p_old ));

        // v = u * v + g
        __m512 v_new = _mm512_add_ps(
            _mm512_mul_ps(v_u, v_old),
            g
        );

        // Update param
        __m512 p_new = _mm512_sub_ps(p_old, _mm512_mul_ps(v_lr, v_new));

        // Store results
        _mm512_storeu_ps(&v[i], v_new);
        _mm512_storeu_ps(&param[i], p_new); 


    }

    // Tail handling (scalar)
    int rem = N % vec_width;
    int start = N - rem;
    for (int i = start; i < N; ++i) {
        float g = grad[i] + weight_decay*param[i];
        v[i] = u * v[i] + g;
        param[i] -= lr * v[i];
    }


}