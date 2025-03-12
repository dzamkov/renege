# Runs all tests locally
cargo test --no-fail-fast 
cargo +nightly miri test --test test --no-fail-fast 
LOOM_MAX_PREEMPTIONS=5 RUSTFLAGS="--cfg loom -C debug-assertions" cargo test --release --no-fail-fast --test loom