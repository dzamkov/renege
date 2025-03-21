# Runs all tests locally
cargo test --no-fail-fast 
cargo +nightly miri test --test test --no-fail-fast 
RUSTFLAGS="--cfg loom -C debug-assertions" cargo test --lib --release --no-fail-fast