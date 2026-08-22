nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    -o ./log/oxide \
    ./target/release/oxide-forge