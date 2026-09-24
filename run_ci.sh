#!/usr/bin/env bash
# Select usable machine configurations; build with CMake and test with CTest.
set -uo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 2
profile=auto
jobs=2
test_timeout=120
build_root="$repo_dir/build/ci"
list_only=false
configs=()

usage() {
    printf '%s\n' \
        'Usage: bash run_ci.sh [options]' \
        '  --profile auto|hosted  Detect local capabilities, or require CPU/OpenMP/MPI.' \
        '  --config FILE          CMake initial-cache file; repeat for several configurations.' \
        '  --jobs N               Parallel build jobs (default: 2).' \
        '  --timeout SECONDS      Per-test timeout (default: 120).' \
        '  --build-root DIR       Results parent directory (default: build/ci).' \
        '  --list                 List candidate configurations without probing/building.' \
        '  --help                 Show this help.' \
        '' \
        'Without --config, auto tries serial/omp/cuda/hip/sycl; hosted uses serial/omp.' \
        'Compilers and SDK settings belong in build_configs/*.cmake, not in this script.'
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --profile|--config|--jobs|--timeout|--build-root)
            (($# >= 2)) || die "Missing value for $1"
            case "$1" in
                --profile) profile=$2 ;;
                --config) configs+=("$2") ;;
                --jobs) jobs=$2 ;;
                --timeout) test_timeout=$2 ;;
                --build-root) build_root=$2 ;;
            esac
            shift 2 ;;
        --list) list_only=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done
[[ $profile == auto || $profile == hosted ]] || die 'Profile must be auto or hosted'
[[ $jobs =~ ^[1-9][0-9]*$ ]] || die '--jobs must be a positive integer'
[[ $test_timeout =~ ^[1-9][0-9]*$ ]] || die '--timeout must be a positive integer'
((BASH_VERSINFO[0] >= 4)) || die 'Bash 4 or newer is required'
for tool in cmake ctest timeout; do
    command -v "$tool" >/dev/null || die "Required command not found: $tool"
done
if ((${#configs[@]} == 0)); then
    if [[ $profile == hosted ]]; then
        configs=("$repo_dir/build_configs/serial.cmake" "$repo_dir/build_configs/omp.cmake")
    else
        for platform in serial omp cuda hip sycl; do
            configs+=("$repo_dir/build_configs/$platform.cmake")
        done
    fi
fi
for index in "${!configs[@]}"; do
    config=${configs[index]}
    [[ -f $config ]] || die "Configuration not found: $config"
    configs[index]=$(cd -- "$(dirname -- "$config")" && printf '%s/%s' "$PWD" "$(basename -- "$config")")
done

mkdir -p -- "$build_root" || die "Cannot create $build_root"
build_root=$(cd -- "$build_root" && pwd -P) || exit 2
[[ $build_root != *$'\n'* ]] || die 'Output paths must not contain newlines'
run_base="$build_root/run$(date +%Y%m%d%H%M%S)"
run_dir=$run_base
suffix=1
while ! mkdir -- "$run_dir" 2>/dev/null; do
    [[ -d $run_dir ]] || die "Cannot create $run_dir"
    suffix=$((suffix + 1))
    run_dir="$run_base.$suffix"
done
printf 'Results: %s\n' "$run_dir"
if [[ -n ${GITHUB_OUTPUT:-} ]]; then
    printf 'run_dir=%s\n' "$run_dir" >> "$GITHUB_OUTPUT" || exit 2
fi
summary="$run_dir/summary.txt"
printf 'SCFD CI started: %s\nProfile: %s\n' "$(date -Iseconds)" "$profile" > "$summary" \
    || die "Cannot write $summary"
passed=0
failed=0
skipped=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OMP_DYNAMIC=${OMP_DYNAMIC:-FALSE}

report() {
    local state=$1 label=$2 detail=${3:-}
    printf '%-4s %-55s %s\n' "$state" "$label" "$detail" | tee -a "$summary" \
        || die "Cannot record results in $summary"
    case "$state" in
        PASS) passed=$((passed + 1)) ;;
        FAIL) failed=$((failed + 1)) ;;
        SKIP) skipped=$((skipped + 1)) ;;
    esac
}

finish() {
    local status=$?
    trap - EXIT
    if ((status != 0 && failed == 0)); then
        report FAIL runner "Interrupted or aborted (exit $status)"
    fi
    printf '\nChecks: %s passed, %s failed, %s skipped.\nFinished: %s\n' \
        "$passed" "$failed" "$skipped" "$(date -Iseconds)" | tee -a "$summary"
    if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
        printf '### SCFD test results\n\n```text\n' >> "$GITHUB_STEP_SUMMARY"
        cat "$summary" >> "$GITHUB_STEP_SUMMARY"
        printf '```\n' >> "$GITHUB_STEP_SUMMARY"
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# No eval: flags and commands always travel as arrays. CTest owns individual
# test timeouts and working directories; configure/build have outer timeouts.
logged_command() {
    local log=$1 working_dir=$2 limit=$3 status
    shift 3
    mkdir -p -- "$(dirname -- "$log")" "$working_dir" || return 2
    printf 'Command: ' > "$log" || return 2
    printf '%q ' "$@" >> "$log" || return 2
    printf '\nWorking directory: %s\n\n' "$working_dir" >> "$log" || return 2
    if [[ $limit == 0 ]]; then
        (cd -- "$working_dir" && "$@") >> "$log" 2>&1
    else
        (cd -- "$working_dir" && timeout --kill-after=10s "${limit}s" "$@") >> "$log" 2>&1
    fi
    status=$?
    printf '\nExit status: %s\n' "$status" >> "$log" || return 2
    return "$status"
}

probe() {
    local kind=$1 probe_dir="$config_dir/probe-$1"
    local extra=(-DPLATFORM_MPI=OFF)
    [[ $kind == mpi ]] && extra=(-DSCFD_CI_PROBE_MPI=ON -DPLATFORM_MPI=ON)
    probe_failure="configuration failed; see $probe_dir/configure.log"
    logged_command "$probe_dir/configure.log" "$config_dir" 300 \
        cmake -S "$repo_dir/ci/probe" -B "$probe_dir" -G 'Unix Makefiles' \
        -C "$config" "${extra[@]}" || return 1
    probe_failure="compilation failed; see $probe_dir/build.log"
    logged_command "$probe_dir/build.log" "$config_dir" 300 \
        cmake --build "$probe_dir" --parallel "$jobs" || return 1
    probe_failure="execution failed; see $probe_dir/run.log"
    logged_command "$probe_dir/run.log" "$probe_dir" 0 \
        ctest --output-on-failure --timeout "$test_timeout" --parallel 1
}

config_index=0
for config in "${configs[@]}"; do
    config_index=$((config_index + 1))
    config_name=$(basename -- "$config" .cmake)
    config_name=${config_name//[^a-zA-Z0-9_-]/_}
    printf -v config_id '%02d-%s' "$config_index" "$config_name"
    config_dir="$run_dir/configurations/$config_id"
    build_dir="$config_dir/build"
    metadata_dir="$config_dir/settings"
    if ! logged_command "$config_dir/settings.log" "$config_dir" 60 \
        cmake "-DCONFIG_FILE=$config" "-DOUTPUT_DIR=$metadata_dir" \
        -P "$repo_dir/ci/read_config.cmake"; then
        report FAIL "$config_id/settings" "Invalid configuration; see $config_dir/settings.log"
        continue
    fi
    platform=$(< "$metadata_dir/platform.txt")
    mpi_mode=$(< "$metadata_dir/mpi_mode.txt")
    required=$(< "$metadata_dir/required.txt")
    if [[ $profile == hosted ]]; then
        if [[ $platform != SERIAL && $platform != OMP ]]; then
            report SKIP "$config_id" 'Accelerators are disabled in the hosted profile'
            continue
        fi
        required=ON
        mpi_mode=ON
    fi
    if $list_only; then
        printf '\n%s: PLATFORM=%s required=%s MPI=%s (not probed)\n' "$config_id" "$platform" "$required" "$mpi_mode"
        continue
    fi
    printf '\nChecking %s (%s)\n' "$config_id" "$platform"
    if ! probe backend; then
        state=SKIP
        [[ $required == ON ]] && state=FAIL
        report "$state" "$config_id/capability" "Unavailable: $probe_failure"
        continue
    fi
    report PASS "$config_id/capability"
    mpi_enabled=OFF
    mpi_args=()
    if [[ $mpi_mode != OFF ]]; then
        if probe mpi; then
            mpi_enabled=ON
            mpi_args+=(-C "$config_dir/probe-mpi/mpi.cmake")
            report PASS "$config_id/mpi-capability"
        else
            state=SKIP
            [[ $mpi_mode == ON ]] && state=FAIL
            report "$state" "$config_id/mpi-capability" "Unavailable: $probe_failure. Local-only tests still run."
        fi
    fi
    if ! logged_command "$config_dir/configure.log" "$config_dir" 600 \
        cmake -S "$repo_dir" -B "$build_dir" -G 'Unix Makefiles' -C "$config" \
        "${mpi_args[@]}" -DBUILD_TESTING=ON "-DPLATFORM_MPI=$mpi_enabled" \
        "-DSCFD_TEST_TIMEOUT=$test_timeout"; then
        report FAIL "$config_id/configure" "See $config_dir/configure.log"
        continue
    fi
    if ! logged_command "$config_dir/build.log" "$config_dir" 1800 \
        cmake --build "$build_dir" --parallel "$jobs"; then
        report FAIL "$config_id/build" "See $config_dir/build.log"
        continue
    fi
    report PASS "$config_id/build"
    if logged_command "$config_dir/ctest.log" "$build_dir" 0 \
        ctest --output-on-failure --timeout "$test_timeout" --parallel 1; then
        report PASS "$config_id/tests" "See $config_dir/ctest.log"
    else
        report FAIL "$config_id/tests" "See $config_dir/ctest.log"
    fi
    cat "$config_dir/ctest.log"
done
if ! $list_only && ((passed == 0 && failed == 0)); then
    report FAIL runner 'No usable configuration was selected'
fi
((failed == 0))
