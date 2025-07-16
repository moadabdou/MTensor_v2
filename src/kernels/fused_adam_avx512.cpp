#include <MTensor/kernels.hpp>
#include <immintrin.h>
#include <cmath>
#include <omp.h>


KERNEL_API void fused_adam_kernel_avx512( 
    float *param,
    float *grad,
    float *m, 
    float *v, 
    int64_t N,
    float beta1,
    float beta2,
    float *beta1_t,
    float *beta2_t,
    float eps,
    float lr
){

    const int vec_width = 16; // 16 floats per __m512

    *beta1_t *= beta1;
    *beta2_t *= beta2;

    __m512 v_beta1 = _mm512_set1_ps(beta1);
    __m512 v_beta2 = _mm512_set1_ps(beta2);
    __m512 v_one_minus_beta1 = _mm512_set1_ps(1.0f - beta1);
    __m512 v_one_minus_beta2 = _mm512_set1_ps(1.0f - beta2);
    __m512 v_lr = _mm512_set1_ps(lr);
    __m512 v_eps = _mm512_set1_ps(eps);
    __m512 v_bias_correction1 = _mm512_set1_ps(1.0f - *beta1_t);
    __m512 v_bias_correction2 = _mm512_set1_ps(1.0f - *beta2_t);

    #pragma omp parallel for 
    for (int64_t i = 0; i <= N - vec_width; i+=vec_width){

        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 m_old = _mm512_loadu_ps(&m[i]);
        __m512 v_old = _mm512_loadu_ps(&v[i]);
        __m512 p_old = _mm512_loadu_ps(&param[i]);

        // m = beta1 * m + (1 - beta1) * g
        __m512 m_new = _mm512_add_ps(
            _mm512_mul_ps(v_beta1, m_old),
            _mm512_mul_ps(v_one_minus_beta1, g)
        );

        // v = beta2 * v + (1 - beta2) * g^2
        __m512 g2 = _mm512_mul_ps(g, g);
        __m512 v_new = _mm512_add_ps(
            _mm512_mul_ps(v_beta2, v_old),
            _mm512_mul_ps(v_one_minus_beta2, g2)
        );

        // Bias correction
        __m512 m_hat = _mm512_div_ps(m_new, v_bias_correction1);
        __m512 v_hat = _mm512_div_ps(v_new, v_bias_correction2);

        // Update param
        __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), v_eps);
        __m512 update = _mm512_div_ps(m_hat, denom);
        __m512 p_new = _mm512_sub_ps(p_old, _mm512_mul_ps(v_lr, update));

        // Store results
        _mm512_storeu_ps(&m[i], m_new);
        _mm512_storeu_ps(&v[i], v_new);
        _mm512_storeu_ps(&param[i], p_new); 


    }

    // Tail handling (scalar)
    int rem = N % vec_width;
    int start = N - rem;
    for (int i = start; i < N; ++i) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        float m_hat = m[i] / (1.0f - *beta1_t);
        float v_hat = v[i] / (1.0f - *beta2_t);

        param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }


}

KERNEL_API void fused_adam_w_kernel_avx512( 
    float *param,
    float *grad,
    float *m, 
    float *v, 
    int64_t N,
    float beta1,
    float beta2,
    float *beta1_t,
    float *beta2_t,
    float eps,
    float lr,
    float weight_decay
){

    const int vec_width = 16; // 16 floats per __m512

    *beta1_t *= beta1;
    *beta2_t *= beta2;

    __m512 v_beta1 = _mm512_set1_ps(beta1);
    __m512 v_beta2 = _mm512_set1_ps(beta2);
    __m512 v_weight_decay = _mm512_set1_ps(weight_decay);
    __m512 v_one_minus_beta1 = _mm512_set1_ps(1.0f - beta1);
    __m512 v_one_minus_beta2 = _mm512_set1_ps(1.0f - beta2);
    __m512 v_lr = _mm512_set1_ps(lr);
    __m512 v_eps = _mm512_set1_ps(eps);
    __m512 v_bias_correction1 = _mm512_set1_ps(1.0f - *beta1_t);
    __m512 v_bias_correction2 = _mm512_set1_ps(1.0f - *beta2_t);

    #pragma omp parallel for 
    for (int64_t i = 0; i <= N - vec_width; i+=vec_width){

        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 m_old = _mm512_loadu_ps(&m[i]);
        __m512 v_old = _mm512_loadu_ps(&v[i]);
        __m512 p_old = _mm512_loadu_ps(&param[i]);

        // m = beta1 * m + (1 - beta1) * g
        __m512 m_new = _mm512_add_ps(
            _mm512_mul_ps(v_beta1, m_old),
            _mm512_mul_ps(v_one_minus_beta1, g)
        );

        // v = beta2 * v + (1 - beta2) * g^2
        __m512 g2 = _mm512_mul_ps(g, g);
        __m512 v_new = _mm512_add_ps(
            _mm512_mul_ps(v_beta2, v_old),
            _mm512_mul_ps(v_one_minus_beta2, g2)
        );

        // Bias correction
        __m512 m_hat = _mm512_div_ps(m_new, v_bias_correction1);
        __m512 v_hat = _mm512_div_ps(v_new, v_bias_correction2);

        // Update param
        __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), v_eps);
        __m512 update= _mm512_div_ps(m_hat, denom);
        __m512 decay = _mm512_mul_ps(p_old, v_weight_decay);
        __m512 update_w= _mm512_add_ps(decay, update);
        __m512 p_new = _mm512_sub_ps(p_old, _mm512_mul_ps(v_lr, update_w));

        // Store results
        _mm512_storeu_ps(&m[i], m_new);
        _mm512_storeu_ps(&v[i], v_new);
        _mm512_storeu_ps(&param[i], p_new); 


    }

    // Tail handling (scalar)
    int rem = N % vec_width;
    int start = N - rem;
    for (int i = start; i < N; ++i) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        float m_hat = m[i] / (1.0f - *beta1_t);
        float v_hat = v[i] / (1.0f - *beta2_t);

        param[i] -= lr * ( m_hat / (std::sqrt(v_hat) + eps) + param[i] * weight_decay);
    }


}


