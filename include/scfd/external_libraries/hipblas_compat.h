// Copyright (C) 2026 Nikolay Evstigneev, Oleg Ryabkov
// SPDX-License-Identifier: GPL-2.0-only

#ifndef __SCFD_HIPBLAS_COMPAT_H__
#define __SCFD_HIPBLAS_COMPAT_H__

#include <hip/hip_complex.h>
#include <hipblas/hipblas.h>
#include <hipblas/hipblas-version.h>

namespace scfd
{
namespace hipblas_compat
{

// hipBLAS 3 uses HIP complex types; version 2 also exposes them through these opt-ins.
#if hipblasVersionMajor >= 3 || defined( HIPBLAS_V2 ) || defined( ROCM_MATHLIBS_API_USE_HIP_COMPLEX )
using complex_float  = hipComplex;
using complex_double = hipDoubleComplex;
#else
using complex_float  = hipblasComplex;
using complex_double = hipblasDoubleComplex;
#endif

} // namespace hipblas_compat
} // namespace scfd

#endif
