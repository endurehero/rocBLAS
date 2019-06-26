/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "testing_gemm_multistream_asynccopy.hpp"
#include "type_dispatch.hpp"

//#include <omp.h>

/* =====================================================================
     BLAS-3 GEMM:
   =================================================================== */

namespace {

template <typename, typename = void>
struct multistream_asynccopy_gemm_testing: rocblas_test_invalid
{
};


template <typename T>
struct multistream_asynccopy_gemm_testing<
    T,
    typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
{
    explicit operator bool() {return true; }
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "gemm_multistream"))
            testing_dgemm_multi_stream<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: " << arg.function;
    }
};


// Multistream asynccopy gemm testing class
struct multistream_asynccopy_gemm : RocBLAS_Test<multistream_asynccopy_gemm, multistream_asynccopy_gemm_testing>
{
    // Filter for which types apply to this suite
    static bool type_filter(const Arguments& arg)
    {
        return rocblas_simple_dispatch<type_filter_functor>(arg);
    }

    // Filter for which functions apply to this suite
    static bool function_filter(const Arguments& arg)
    {
        return !strcmp(arg.function, "gemm_multistream");
    }

    // Google Test name suffix based on parameters
    static std::string name_suffix(const Arguments& arg)
    {
        return RocBLAS_TestName<multistream_asynccopy_gemm>{} << rocblas_datatype2string(arg.a_type) << '_'
                                        << (char)std::toupper(arg.transA) << '_' << arg.M << '_'
                                        << arg.N << '_' << arg.alpha << '_' << arg.lda << '_'
                                        << arg.incx << '_' << arg.beta << '_' << arg.incy;
    }
};

TEST_P(multistream_asynccopy_gemm, blas3){ rocblas_simple_dispatch<multistream_asynccopy_gemm_testing>(GetParam()); }
INSTANTIATE_TEST_CATEGORIES(multistream_asynccopy_gemm);

} // namespace
