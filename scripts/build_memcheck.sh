#!/usr/bin/env bash

set -Eeuo pipefail

readonly TARGET_BIN="${TARGET_BIN:-./target/release/oxide-forge}"
readonly LOG_DIR="./log"
readonly BUILD_LOG="${LOG_DIR}/build-release.log"
readonly MEMCHECK_LOG="${LOG_DIR}/memcheck.log"

if [[ ! -f ./Cargo.toml ]]; then
    echo "Run this script from the project root: scripts/build_memcheck.sh" >&2
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo was not found." >&2
    exit 1
fi

if [[ -x /usr/local/cuda/bin/compute-sanitizer ]]; then
    readonly COMPUTE_SANITIZER=/usr/local/cuda/bin/compute-sanitizer
elif command -v compute-sanitizer >/dev/null 2>&1; then
    readonly COMPUTE_SANITIZER="$(command -v compute-sanitizer)"
else
    echo "compute-sanitizer was not found." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Building release target with CUDA line information..."
cargo oxide build --lineinfo -- --release 2>&1 | tee "${BUILD_LOG}"

if [[ ! -x "${TARGET_BIN}" ]]; then
    echo "Build completed without producing ${TARGET_BIN}." >&2
    exit 1
fi

echo
echo "Running Compute Sanitizer memcheck..."
"${COMPUTE_SANITIZER}" \
    --tool memcheck \
    --error-exitcode=86 \
    "${TARGET_BIN}" \
    2>&1 | tee "${MEMCHECK_LOG}"

echo
echo "Build and memcheck complete:"
echo "  Build log:    ./log/build-release.log"
echo "  Memcheck log: ./log/memcheck.log"
