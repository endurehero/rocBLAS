/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>
#include "rocblas_test.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_vector.hpp"
#include "rocblas_init.hpp"
#include "rocblas.hpp"
#include "rocblas_datatype2string.hpp"
#include "utility.hpp"
#include "cblas_interface.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "flops.hpp"


/* ============================================================================================ */

template <typename T>
void testing_dgemm_multi_stream(const Arguments& arg)
{
    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    T h_alpha;
    T h_beta = 0;
    if(std::is_same<T, rocblas_half>{})
    {
        h_alpha = float_to_half(arg.alpha);
    }
    else
    {
        h_alpha = arg.alpha;
    }

    double rocblas_error = 0.0;


    hipStream_t stream1, stream2;
    CHECK_HIP_ERROR(hipStreamCreate(&stream1));
    CHECK_HIP_ERROR(hipStreamCreate(&stream2));

    rocblas_local_handle handle1, handle2;
    ROCBLAS_CHECK_ERROR(rocblas_set_stream(handle1.rocblas_handle(), stream1));
    ROCBLAS_CHECK_ERROR(rocblas_set_stream(handle2.rocblas_handle(), stream2));
    

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M)
    {
        static const size_t safe_size = 100;
        device_vector<T> dA(safe_size);
        device_vector<T> dB(safe_size);
        
        device_vector<T> dC(safe_size);
        if(!dA || !dB || !dC)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemm<T>(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc),
            rocblas_status_invalid_size);
        return;
    }

    const auto size_A = static_cast<size_t>(lda) * static_cast<size_t>(A_col);
    const auto size_B = static_cast<size_t>(ldb) * static_cast<size_t>(B_col);
    const auto size_C = static_cast<size_t>(ldc) * static_cast<size_t>(N);

    // allocate memory on device
    device_vector<T> dA_1(size_A), dA_2(size_A);
    device_vector<T> dB_1(size_B), dB_2(size_B);
    device_vector<T> dC_1(size_C), dC_2(size_C);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    if(!dA_1 || !dB_1 || !dC_1 || !dA_2 || !dB_2 || !dC_2 || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hB(size_B);
    host_vector<T> hC_1(size_C);
    host_vector<T> hC_2(size_C);
    host_vector<T> hC_gold(size_C);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA, A_row, A_col, lda);
    rocblas_init_alternating_sign<T>(hB, B_row, B_col, ldb);
    rocblas_init<T>(hC_1, M, N, ldc);

    //  std::cout << "------------------------------------------------" << std::endl;
    //  for(int i = 0; i < size_A; i++){ cout << half_to_float(hA[i]) << "  "; }
    //  std::cout << std::endl << "------------------------------------------------" << std::endl;
    //  for(int i = 0; i < size_B; i++){ cout << half_to_float(hB[i]) << "  "; }
    //  std::cout << std::endl << "------------------------------------------------" << std::endl;
    //  for(int i = 0; i < size_C; i++){ cout << half_to_float(hC_1[i]) << "  "; }
    //  std::cout << std::endl << "------------------------------------------------" << std::endl;

    hC_2    = hC_1;
    hC_gold = hC_1;

    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // create 2 streams, labed as s1 and s2
    // Init s1: h2dCopyAsync
    // Loop:
    //       s1: dgemm        s2: h2dCopyAsync
    //       s1: d2hCopyAsync s2: dgemm
    //       s1: h2dCopyAsync s2: d2hCopyAsync
    
    // s1 h2dCopyAsync
    // copy data from CPU to device
    //CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy2DAsync(dA_1, sizeof(T) * size_A, hA, sizeof(T) * size_A, A_col * sizeof(T), A_row, hipMemcpyHostToDevice, stream1));
    
    //CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy2DAsync(dB_1, sizeof(T) * size_B, hB, sizeof(T) * size_B, B_col * sizeof(T), B_row, hipMemcpyHostToDevice, stream1));

    for(int i = 1; i < arg.iters; ++i){
        // s1 dgemm
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle1, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(
            handle1, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle1, rocblas_pointer_mode_device));

        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(
            handle1, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

        // s2 h2dCopyAsync
        // copy data from CPU to device
        //CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy2DAsync(dA_2, sizeof(T) * size_A, hA, sizeof(T) * size_A, A_col * sizeof(T), A_row, hipMemcpyHostToDevice, stream2));
        
        //CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy2DAsync(dB_2, sizeof(T) * size_B, hB, sizeof(T) * size_B, B_col * sizeof(T), B_row, hipMemcpyHostToDevice, stream2));
    
        // s1 d2hCopyAsync
        CHECK_HIP_ERROR(hipMemcpy2DAsync(hC_1, sizeof(T) * size_C, dC_1, sizeof(T) * size_C, C_col * sizeof(T), C_row, hipMemcpyDeviceToHost, stream1));

        // s2 dgemm
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle2, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(
            handle2, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle2, rocblas_pointer_mode_device));

        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(
            handle2, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));


        // s1 h2dCopyAsync
        // copy data from CPU to device
        //CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy2DAsync(dA_1, sizeof(T) * size_A, hA, sizeof(T) * size_A, A_col * sizeof(T), A_row, hipMemcpyHostToDevice, stream1));
        
        //CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy2DAsync(dB_1, sizeof(T) * size_B, hB, sizeof(T) * size_B, B_col * sizeof(T), B_row, hipMemcpyHostToDevice, stream1));
        
        //s2 d2hCopyAsync
        CHECK_HIP_ERROR(hipMemcpy2DAsync(hC_2, sizeof(T) * size_C, dC_2, sizeof(T) * size_C, C_col * sizeof(T), C_row, hipMemcpyDeviceToHost, stream2));
    
    }

    cblas_gemm<T, T>(transA, transB, M, N, K, h_alpha, hA, lda, hB, ldb, h_beta, hC_gold, ldc);

    if(arg.unit_check)
    {
        unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_1.data());
        unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_2.data());
    }

    if(arg.norm_check)
    {
        double error_hst_ptr =
            fabs(norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_1.data()));
        double error_dev_ptr =
            fabs(norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_2.data()));
        rocblas_error = error_hst_ptr > error_dev_ptr ? error_hst_ptr : error_dev_ptr;
    }
}
