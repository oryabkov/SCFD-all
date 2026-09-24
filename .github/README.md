# Local and GitHub CI

`run_ci.sh` selects usable machine configurations, builds with CMake, and runs
CTest. CMake registrations are the single list of test commands, arguments,
MPI ranks, timeouts, and working directories. There are no Python CI helpers;
handwritten Makefiles are unchanged.

## Run locally

```sh
# Preview configurations without probing or building.
bash ./run_ci.sh --list

# Require SERIAL/OpenMP; try available accelerators and MPI.
bash ./run_ci.sh

# Select configurations for this machine.
bash ./run_ci.sh --config build_configs/serial.cmake \
                 --config build_configs/cuda.cmake --jobs 4
```

Requirements: Bash 4+, CMake/CTest 3.14+, GNU `timeout`, Make, and the selected
compilers and SDKs. SYCL requires CMake 3.20+, oneDPL, and a usable SYCL GPU.
SERIAL does not require OpenMP; OMP and accelerator reference tests do. CUDA
includes cuSOLVER tests. HIP platform headers require hipBLAS, hipSOLVER, and
Thrust. Activate SDK environments before running CI.

No system packages are installed locally. CMake downloads GoogleTest sources
when needed. For offline use, set `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` in the
machine profile to an existing GoogleTest source directory.

## Platform selection

```sh
cmake -S . -B build/serial -DPLATFORM=SERIAL -DPLATFORM_MPI=OFF
cmake -S . -B build/omp-mpi -DPLATFORM=OMP -DPLATFORM_MPI=ON
cmake -S . -B build/cuda-mpi -DPLATFORM=CUDA -DPLATFORM_MPI=ON
cmake --build build/omp-mpi --parallel 2
(cd build/omp-mpi && ctest --output-on-failure)
```

`PLATFORM` is `SERIAL` (default), `OMP`, `CUDA`, `HIP`, or `SYCL`.
`PLATFORM_MPI` is `ON` or `OFF` (default). MPI dependencies/definitions belong
only to MPI targets; ordinary tests remain independent. `BUILD_TESTING=OFF`
disables the test build. The old `SCFD_WITH_*` switches are not supported.

Use separate build directories for different platforms/compilers. CMake finds
`nvcc` for CUDA; `CMAKE_CUDA_COMPILER` is needed only to override discovery.
HIP/SYCL select `hipcc`/`icpx` when no explicit C++ compiler is supplied.
Compiler paths, toolchain files, initial-cache files, and `CXX` can override
selection on the first configure.

These are CMake `-D` options. Existing Makefiles use assignments such as
`make PLATFORM=CUDA PLATFORM_MPI=ON`, where supported, not `make -D...`.

## Machine configurations

`build_configs/*.cmake` are trusted CMake **initial-cache files**, loaded with
`cmake -C` before compiler detection. They play the role of machine-specific
Make `.inc` files, but contain CMake commands, not shell or YAML.

| File | Platform | Required by local CI |
| --- | --- | --- |
| `serial.cmake` | SERIAL | Yes |
| `omp.cmake` | OMP | Yes |
| `cuda.cmake` | CUDA | No |
| `hip.cmake` | HIP | No |
| `sycl.cmake` | SYCL | No |

Profiles leave `PLATFORM_MPI` unset: the runner probes MPI and passes ON/OFF.
Set it explicitly in a custom profile to require or disable MPI. `AUTO` is a
runner policy, not a valid project option. Direct CMake use defaults MPI to OFF.

Copy a profile, or set overrides before including one. For example:

```cmake
set(CMAKE_CXX_COMPILER /path/to/compiler/bin/icpx CACHE FILEPATH "C++ compiler")
set(SCFD_SYCL_TARGET nvptx64-nvidia-cuda CACHE STRING "SYCL targets")
set(SCFD_SYCL_TARGET_BACKEND --cuda-gpu-arch=sm_75 CACHE STRING "Backend flags")
set(SCFD_ONEDPL_INCLUDE_DIR /path/to/oneDPL/include CACHE PATH "oneDPL headers")
set(PLATFORM_MPI ON CACHE BOOL "Require MPI")
set(SCFD_CI_REQUIRED ON CACHE BOOL "Require this platform")
include("/path/to/SCFD-all/build_configs/sycl.cmake")
```

Use the GPU's actual architecture. CUDA's `CMAKE_CUDA_ARCHITECTURES` requires
CMake 3.18+. HIP uses the active SDK; `SCFD_HIP_PLATFORM` can validate `amd` or
`nvidia`. Standard CMake compiler/linker variables remain available.

For a particular MPI installation, set matching `MPI_CXX_COMPILER` and
`MPIEXEC_EXECUTABLE`. `MPIEXEC_PREFLAGS`/`MPIEXEC_POSTFLAGS` are semicolon-separated
lists; CTest preserves argument boundaries and launcher ordering. The runner
does not enable root execution or change local oversubscription policy.
Tests use up to four ranks. GPU communication uses host staging, so GPU-aware
MPI is not required. Profiles default to Debug to keep assertions active.

## Selection and failures

The runner builds and executes small CTest probes using the machine profile.
Accelerator probes validate external SDK headers and execute a kernel; a compiler
alone is insufficient. The MPI probe launches four ranks and records the
matched wrapper and launcher for the main build.

- `SCFD_CI_REQUIRED=OFF`: unavailable platforms are SKIP, with probe logs.
  Set ON for platforms expected to work, so SDK failures cannot hide coverage.
- Explicit `PLATFORM_MPI=ON`: unavailable MPI is a failure. Local-only tests
  still run, but the overall result remains failed. OFF skips MPI probing.
  If omitted, unavailable MPI is a skip, not a failure.
- After probes pass, configure/build/test errors are failures, never skips.
  CTest continues independent cases; the runner continues other configurations.
  Any failure gives a nonzero final status. Invalid profiles also fail.
- A run without a usable configuration fails. `--list` previews configurations
  only and makes no availability claim.

CTest runs sequentially to avoid GPU oversubscription. `--jobs` sets build
parallelism; `--timeout` sets each test timeout (default 120 seconds).

## Adding a CI target

Add its executable and CTest registration in `test/<directory>/CMakeLists.txt`:

```cmake
add_executable(test_new_array.bin test_new_array.cpp)
target_link_libraries(test_new_array.bin PRIVATE gtest_main) # If needed.
scfd_add_test(test_new_array arrays serial)
```

The helper accepts case name, suite, platform label, then optional arguments.
It runs `${name}.bin` in an isolated directory with the configured timeout.
For MPI, follow communication examples using
`scfd_add_mpi_test(name target suite platform ranks [arguments...])` and link
`MPI::MPI_CXX`. Define bare `PLATFORM_MPI` only for platform-based MPI targets;
never define `PLATFORM_MPI=OFF` in C++, since it is a presence flag.

Use `if(PLATFORM STREQUAL "CUDA")` (or HIP/SYCL/OMP) and `if(PLATFORM_MPI)`
where needed. `scfd_target_platform(target)` configures the selected local
backend; `scfd_target_openmp(target)`/`scfd_target_sycl(target)` remain available.
Register new directories with `add_subdirectory(...)` in `test/CMakeLists.txt`.
New dependencies may require probe and hosted-install updates. Otherwise,
no shell manifest or workflow changes are needed.

```sh
# In a configured build:
(cd build/omp-mpi && ctest -N)
(cd build/omp-mpi && ctest -L communication --output-on-failure)
bash ./run_ci.sh --config build_configs/omp.cmake
```

## Coverage and diagnostics

Coverage remains limited to existing CMake portions of `arrays`, `static_vec`,
`static_mat`, `for_each`, `geometry`, `for_paper`, `communication`, `matmul`,
and `external_libraries`. Some examples have less checking than unit tests.
This does not add coverage for `backend`, `memory`, `utils`, `mgpu_reduce`,
`poisson_jacobi`, or every Make-only target inside included directories.

Generic host tests also run in accelerator builds. Platform communication
tests follow the selected backend; MPI adds platform and independent MPI
regressions. SERIAL trivial-platform checks remain explicitly non-MPI.
OpenMP examples run in OMP and accelerator configurations.

Results use timestamps, with a suffix for runs started in the same second:

```text
build/ci/run20260924153000/
  summary.txt
  configurations/01-serial/
    settings.log
    probe-backend/       # configure/build/run logs and probe build files
    probe-mpi/
    configure.log
    build.log
    ctest.log           # Per-case results and failure output
    build/
      Testing/Temporary/LastTest.log  # Output from every case
      test/arrays/cases/<case>/       # Isolated generated test data
```

`--build-root` changes the parent output directory. Probe/configure/build logs
remain after failures. The summary counts phases; CTest logs count tests.

## GitHub and future self-hosted runners

`.github/workflows/tests.yml` uses one `ubuntu-24.04` job: checkout once,
install dependencies once, run `--profile hosted`, and upload logs even after
failures. Hosted mode selects SERIAL/OMP, requires MPI, and excludes GPUs.
There is one GitHub check; CTest logs identify individual failing cases.

Installation is guarded by `runner.environment == 'github-hosted'`. Local and
future self-hosted runs use already installed dependencies. No self-hosted
workflow is enabled. Future GPU jobs should use explicit labels and profiles
with `--profile auto`, for trusted, reviewed revisions only. Do not route public
pull requests to persistent private runners: workflow approval alone does not
make arbitrary code safe.
