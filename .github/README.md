# Local and GitHub CI

`run_ci.sh` is the shared Linux/Bash runner. CMake builds the tests; the script
executes their binaries directly. It does not invoke CTest, parse workflow YAML,
use Python helpers, or change the handwritten Makefiles. Existing CTest
registrations remain available for manual use.

## Run locally

From the repository root:

```sh
# Preview candidate configurations and commands without compiling/running them.
bash ./run_ci.sh --list

# Require CPU/OpenMP; try MPI and each accelerator supported by the environment.
bash ./run_ci.sh

# Run only the configurations chosen for this machine.
bash ./run_ci.sh --config build_configs/cpu.cmake \
                 --config build_configs/cuda.cmake --jobs 4
```

Requirements are Bash 4+, CMake, GNU `timeout`, a build tool (`make`), and the
selected compilers and SDKs. The project requires CMake 3.14 or newer; SYCL
requires 3.20 or newer. CPU tests require OpenMP. SYCL tests require oneDPL and
a usable SYCL GPU. CUDA tests include cuSOLVER. The default configurations use
compiler names from `PATH`; activate the appropriate SDK environments first.

No system packages are installed locally. CMake's existing GoogleTest setup
downloads its sources when needed. For offline use, set
`FETCHCONTENT_SOURCE_DIR_GOOGLETEST` in the machine configuration to an existing
GoogleTest source directory.

## Machine configurations

`build_configs/*.cmake` are CMake **initial-cache files**, loaded with `cmake -C`
before compiler detection. They play the role of the machine-specific `.inc`
files used by a Make build, but contain CMake commands. They are not shell files
or YAML, and should only come from trusted sources.

The supplied configurations are:

| File | Local backend | MPI policy | Required |
| --- | --- | --- | --- |
| `cpu.cmake` | CPU/OpenMP | AUTO | Yes |
| `cuda.cmake` | CUDA | OFF | No |
| `hip.cmake` | HIP, using `hipcc` | OFF | No |
| `sycl.cmake` | SYCL, using `icpx` | OFF | No |
| `github_cpu.cmake` | CPU/OpenMP | ON | Yes |

Copy a configuration for a particular machine, or include a supplied one after
setting overrides. For example, a NVIDIA SYCL configuration could contain:

```cmake
set(CMAKE_CXX_COMPILER /path/to/compiler/bin/icpx CACHE STRING "C++ compiler")
set(SCFD_SYCL_TARGET nvptx64-nvidia-cuda CACHE STRING "SYCL targets")
set(SCFD_SYCL_TARGET_BACKEND --cuda-gpu-arch=sm_75 CACHE STRING "Backend flags")
set(SCFD_ONEDPL_INCLUDE_DIR /path/to/oneDPL/include CACHE PATH "oneDPL headers")
set(SCFD_CI_REQUIRED ON CACHE BOOL "This machine must pass SYCL tests")
include("/path/to/SCFD-all/build_configs/sycl.cmake")
```

Use the architecture appropriate to the actual GPU. For CUDA, set
`CMAKE_CUDA_ARCHITECTURES` (requires CMake 3.18+). HIP uses the active HIP SDK;
`SCFD_HIP_PLATFORM` may explicitly select `amd` or `nvidia` when necessary.
Compiler and linker flags can be supplied through the normal CMake cache
variables. For multiple backend options use separate configurations, not one
combined configuration.

For a particular MPI installation, set `MPI_CXX_COMPILER` and
`MPIEXEC_EXECUTABLE` to the matching wrapper and launcher. `MPIEXEC_PREFLAGS`
and `MPIEXEC_POSTFLAGS` are CMake lists, with semicolons separating arguments.
The runner preserves argument boundaries and uses CMake's launcher ordering.
It does not automatically enable MPI execution as root or change local MPI
oversubscription policy. Configure launcher flags for the machine if needed;
the current communication tests use up to four ranks.

Defaults are `Debug` builds so that assertions in standalone tests remain
active. Keep assertions enabled in custom CI configurations too. Each
configuration gets a fresh build directory, avoiding cached compiler changes.
Configurations can also be used independently of the runner:

```sh
cmake -S . -B build/my-cuda -C build_configs/cuda.cmake
cmake --build build/my-cuda --parallel 4
```

`SCFD_CI_MPI` and `SCFD_CI_REQUIRED` are runner policies, not project build
options. When using CMake directly, select MPI with `-DSCFD_WITH_MPI=ON`.

## Selection and failures

Before building SCFD, the runner configures, compiles and runs a small
capability probe using the same configuration. Accelerator probes execute a
kernel and check its result: finding a compiler alone is not enough. The MPI
probe launches four ranks using the discovered MPI installation.

- `SCFD_CI_REQUIRED=OFF`: an unavailable configuration is reported as SKIP,
  with probe logs explaining why. Set ON on machines expected to support it,
  so a broken SDK/device cannot silently reduce coverage.
- `SCFD_CI_MPI=AUTO`: enable MPI if its compile/link/run probe succeeds.
  ON makes unavailable MPI a failure; OFF disables it. A failed MPI probe does
  not prevent the remaining CPU tests from running.
- Once a capability is selected, project configuration, build and test errors
  are failures, never capability skips. Other configurations and independent
  test cases continue; any failure produces a nonzero final exit status.
- Invalid configuration files fail. A run with no usable configuration fails.
  `--list` only prints candidates; it makes no availability or test-success claim.

MPI is initially exercised in host builds only, with both SERIAL and OMP
platform selections. Accelerator/MPI combinations are not added in this pass.

## Adding a CI target

For an existing backend, update two places; the workflow YAML normally stays
unchanged.

1. Add the executable to `test/<directory>/CMakeLists.txt`, for example:

   ```cmake
   add_executable(test_new_array.bin test_new_array.cpp)
   target_link_libraries(test_new_array.bin PRIVATE gtest_main)
   ```

   Link GoogleTest only if needed. Follow existing backend guards
   (`CUDA_ENABLED`, `HIP_ENABLED`, `SYCL_ENABLED`, or `SCFD_WITH_MPI`) and use
   `scfd_target_openmp(target)`, `scfd_target_sycl(target)`, or `MPI::MPI_CXX`
   where appropriate. For a new directory, also add `add_subdirectory(...)`
   to `test/CMakeLists.txt`.

2. Register its command in `ci/tests.sh`: use `scfd_run_host_tests` for
   CPU/OpenMP, or the matching `scfd_run_cuda_tests`, `scfd_run_hip_tests`,
   `scfd_run_sycl_tests`, or `scfd_run_mpi_tests` function.

   ```bash
   # Suite, unique case name, binary path relative to the build directory, arguments.
   run_test arrays test_new_array test/arrays/test_new_array.bin

   # MPI additionally takes a rank count before the binary path.
   run_mpi_test communication test_new_mpi 4 test/communication/test_new_mpi.bin
   ```

   Logging, isolated working directories, timeouts, and failure reporting are
   automatic. Host tests also run in accelerator configurations. Adding only
   a CMake target builds the test but does not execute it in CI; `add_test(...)`
   registers it only for optional CTest use.

3. Preview the commands, then run the relevant configuration:

   ```sh
   bash ./run_ci.sh --list
   bash ./run_ci.sh --config build_configs/cpu.cmake
   ```

If a test needs a new dependency, update its CMake dependency checks and,
where necessary, the capability probe and hosted dependency-install step.

## Coverage and diagnostics

`ci/tests.sh` lists the direct invocations corresponding to existing CMake
targets. Keep it synchronized when adding or removing CMake targets. This
initial coverage is 22 host invocations, plus 20 MPI invocations when enabled.
Accelerator builds also run the host cases and add 25 CUDA, 6 HIP, or 3 SYCL
invocations. The existing OpenMP example is included even though it is not
registered with CTest. Some existing executables are examples with less
extensive checking than unit tests; CI does not add new assertions to them.

This covers the existing CMake portions of `arrays`, `static_vec`, `static_mat`,
`for_each`, `geometry`, `for_paper`, `communication`, `matmul`, and
`external_libraries`. The last two currently require an accelerator. It does
not add CMake coverage for `backend`, `memory`, `utils`, `mgpu_reduce`, or
`poisson_jacobi`, nor cover every Make-only target inside included directories.

Results use local timestamps, for example:

```text
build/ci/run20260924153000/
  summary.txt
  configurations/01-cpu/
    settings.log
    probe-backend/       # configure/build/run logs and probe build files
    probe-mpi/
    configure.log
    build.log
    build/              # generated CMake build tree
  test/arrays/01-cpu/test_tensor_host/run.log
  test/communication/01-cpu/test_mpi_binary_file/
    run.log
    test_file_write.txt
```

Runs started in the same second receive a numeric suffix. Each test has its
own working directory for generated data. `--build-root`, `--jobs`, and
`--timeout` customize the output location, build concurrency, and per-test
timeout. Probe, configure and build logs are retained even after failure.

## GitHub and future self-hosted runners

`.github/workflows/tests.yml` runs one job on `ubuntu-24.04`: checkout once,
install CPU/OpenMP/MPI dependencies once, invoke `run_ci.sh --profile hosted`,
then upload diagnostics even after test failures. The hosted profile requires
CPU/OpenMP/MPI and excludes accelerators even if their compilers happen to be
present. Results appear in the job summary and per-test log files; there is
one overall GitHub check, not a separate job for each directory.

Dependency installation is guarded by
`runner.environment == 'github-hosted'`, not `GITHUB_ACTIONS`. The shell runner
itself never installs packages, so local and future self-hosted runs use the
already installed environment.

No self-hosted job is enabled here. When adding one, use a separate,
trusted-code-only workflow with explicit runner labels and machine
configurations, calling `--profile auto`. Do not change the public
pull-request job to run on a persistent private machine; approving a workflow
does not make arbitrary pull-request code safe. GPU jobs should run only
reviewed, trusted revisions under an appropriate runner isolation policy.
