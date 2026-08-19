#!/usr/bin/env bash

set -Eeuo pipefail

readonly TARGET_BIN="${TARGET_BIN:-./target/release/oxide-forge}"
readonly LOG_DIR="./log"
readonly REPORT_BASE="${LOG_DIR}/gemm-profile"
readonly KERNEL_FILTER='regex:^matrix_multiply$'
readonly GEMM_SKIP="${GEMM_SKIP:-1}"
readonly GEMM_COUNT="${GEMM_COUNT:-1}"

if [[ ! -f ./Cargo.toml ]]; then
    echo "Run this script from the project root: scripts/profile_gemm_ncu.sh" >&2
    exit 1
fi

if [[ ! -x "${TARGET_BIN}" ]]; then
    echo "Missing executable: ${TARGET_BIN}" >&2
    echo "Build it first: scripts/build_memcheck.sh" >&2
    exit 1
fi

if [[ ! ${GEMM_SKIP} =~ ^[0-9]+$ || ! ${GEMM_COUNT} =~ ^[1-9][0-9]*$ ]]; then
    echo "GEMM_SKIP must be non-negative and GEMM_COUNT must be positive." >&2
    exit 1
fi

if [[ -x /usr/local/cuda/bin/ncu ]]; then
    readonly NCU=/usr/local/cuda/bin/ncu
elif command -v ncu >/dev/null 2>&1; then
    readonly NCU="$(command -v ncu)"
else
    echo "ncu was not found." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

restore_report_owner() {
    if [[ -n ${SUDO_UID:-} && -n ${SUDO_GID:-} ]]; then
        chown -R "${SUDO_UID}:${SUDO_GID}" "${LOG_DIR}"
    fi
}
trap restore_report_owner EXIT

echo "Profiling matrix_multiply (skip=${GEMM_SKIP}, count=${GEMM_COUNT})..."
if "${NCU}" \
        --target-processes application-only \
        --kernel-name "${KERNEL_FILTER}" \
        --launch-skip "${GEMM_SKIP}" \
        --launch-count "${GEMM_COUNT}" \
        --replay-mode kernel \
        --set full \
        --apply-rules yes \
        --import-sass yes \
        --force-overwrite \
        --export "${REPORT_BASE}" \
        "${TARGET_BIN}" \
        >"${REPORT_BASE}-capture.log" 2>&1; then
    :
else
    status=$?
    echo "Nsight Compute capture failed. Last output:" >&2
    tail -n 40 "${REPORT_BASE}-capture.log" >&2
    if grep -q 'ERR_NVGPUCTRPERM' "${REPORT_BASE}-capture.log"; then
        echo >&2
        if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
            echo "Enable performance-counter access on the Windows host:" >&2
            echo "  NVIDIA App > System > Advanced > Developer" >&2
            echo "  > Manage GPU Performance Counters > Allow access to all users" >&2
            echo "Then rerun this script without sudo." >&2
        else
            echo "Enable NVIDIA GPU performance counters for this user or run as root." >&2
        fi
    fi
    exit "${status}"
fi

readonly REPORT_FILE="${REPORT_BASE}.ncu-rep"
if [[ ! -f "${REPORT_FILE}" ]]; then
    echo "Nsight Compute did not produce ${REPORT_FILE}." >&2
    echo "See ${REPORT_BASE}-capture.log for details." >&2
    exit 1
fi

{
    printf 'Kernel: matrix_multiply\n'
    printf 'Launch skip: %s\n' "${GEMM_SKIP}"
    printf 'Launch count: %s\n\n' "${GEMM_COUNT}"
    "${NCU}" \
        --import "${REPORT_FILE}" \
        --page details \
        --print-summary per-kernel
} >"${REPORT_BASE}-summary.txt"

"${NCU}" \
    --import "${REPORT_FILE}" \
    --page details \
    --print-details all \
    --print-rule-details \
    --print-summary per-kernel \
    >"${REPORT_BASE}-details.txt"

"${NCU}" \
    --import "${REPORT_FILE}" \
    --page raw \
    --csv \
    >"${REPORT_BASE}-metrics.csv"

"${NCU}" \
    --import "${REPORT_FILE}" \
    --page source \
    --print-source sass \
    >"${REPORT_BASE}-sass.txt"

echo "Profile complete:"
echo "  Summary: ./log/gemm-profile-summary.txt"
echo "  Details: ./log/gemm-profile-details.txt"
echo "  Metrics: ./log/gemm-profile-metrics.csv"
echo "  SASS:    ./log/gemm-profile-sass.txt"
echo "  Report:  ./log/gemm-profile.ncu-rep"
echo "  Capture: ./log/gemm-profile-capture.log"
