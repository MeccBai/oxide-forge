#!/usr/bin/env bash

set -Eeuo pipefail

if [[ ! -f ./Cargo.toml ]]; then
    echo "Run this script from the project root: scripts/clean_cuda_artifacts.sh" >&2
    exit 1
fi

shopt -s nullglob
artifacts=(./*.ll ./*.ptx)

if ((${#artifacts[@]} == 0)); then
    echo "No top-level LLVM IR or PTX artifacts to remove."
    exit 0
fi

printf 'Removing %d CUDA-Oxide artifacts:\n' "${#artifacts[@]}"
printf '  %s\n' "${artifacts[@]}"
rm -- "${artifacts[@]}"
