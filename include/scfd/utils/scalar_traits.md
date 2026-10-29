<!--
Copyright (C) 2026 SCFD contributors

This file is part of SCFD.

SCFD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2 only of the License.

SCFD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SCFD.  If not, see <http://www.gnu.org/licenses/>.
-->

# Scalar Traits

`scalar_traits.h` provides arithmetic and `std::complex<T>` support.
Unsupported scalars have no default implementation. Include optional
specializations explicitly:

- `boost_multiprecision_scalar_traits.h`: host Boost real numbers.
- `thrust_complex_scalar_traits.h`: Thrust complex numbers, including device calls.
- `cucomplex_scalar_traits.h`: cuFloatComplex and cuDoubleComplex, also used by cuFFT.

Each specialization supplies `scalar_type`, `component_type`, `abs_type`,
`is_complex`, construction, conjugation, products, and finiteness checks.
`abs_type` is a concrete owning value type; custom/AD scalars must not be
implicitly narrowed to built-in floating-point types.

For `z = 3 + 4i`, `abs(z) = 5`, `abs_sq(z) = 25`, and
`asum_term(z) = 7`. BLAS complex ASUM uses the sum of absolute real and
imaginary components, not modulus, see:
<https://laplace.physics.ubc.ca/vnp4/intel/mkl/mklqref/blaslev1.htm>
<https://github.com/scijs/ndarray-blas-level1-complex>

`conj_mul(x,y)` conjugates the first operand.
Modulus uses hypot rather than squaring first, avoiding unnecessary overflow.
Compensated reductions must still retain their error-tracking algorithms.

`zero()` is a function so non-literal scalar types are supported; replace
historical `scalar_traits<T>::zero` value reads with `zero()`.
The existing real `sqr()` helper still means `x*x`, not squared complex modulus.

Run `make test-scalar-traits test-boost-scalar-traits` in `test/utils`.
These tests are also registered with CTest; Boost tests are enabled when Boost
headers are found. CUDA configurations register `scalar_traits_cuda_test`.
