# `renege` 
[![](https://img.shields.io/crates/v/renege.svg)](https://crates.io/crates/renege)
[![](https://docs.rs/renege/badge.svg)](https://docs.rs/renege/)

**Tracking cache validity using fast concurrent invalidation propogation.**

## Background
Whenever a computer program needs to use the same data or computation multiple times, the simplest
and easiest optimization is usually to cache it. However, this introduces a notoriously tricky
problem: preventing stale data. The validity of a cache depends on a set of *conditions*. These
are statements that were true at the time the cache was populated and were assumed to be true
while building the cache, but may not necessarily be true in the future. For example:

* The last modification to the database was made at `2023-10-17T20:03:38`
* The user has the permission `view`
* The file `config.yaml` consists of a particular string
* The Young's modulus of the simulated material is `200 GPa`

If a cache depends on a condition that is no longer true, the cache is *invalid*. Using data from
an invalid cache can give different results from retreiving/computing the data directly, making the
optimization incorrect.

Solutions to this problem generally fall under two categories:

* **Pull-based:** When accessing the cache, the requestor re-checks all of the conditions that
the cache depends on. If any are false, the cache is invalid.

	* **Pros:** Simple and easy to implement

	* **Cons:** Tends to be slow, especially if there are many conditions and/or they are expensive
	to check. The performance penalty of the checks might even outweigh the benefits of the cache!

* **Push-based:** When a condition becomes false, caches that depend on it are marked as invalid.

	* **Pros:** Very fast. When accessing the cache, the requestor only needs to check if it is
	marked as invalid

	* **Cons:** Difficult to implement. Conditions need to be actively watched, and the caches
	depending on them need to be tracked. When caches depend on other caches, it becomes
	necessary to track the entire dependency graph to allow efficient invalidation.

## Our Approach

**Renege** simplifies the implementation of push-based cache invalidation by handling all of the
tracking and bookkeeping for you. The API is dead simple:

**[`Condition`](https://docs.rs/renege/latest/renege/struct.Condition.html):** A condition that a
cache can depend on. Is automatically invalidated when dropped.

**[`Token`](https://docs.rs/renege/latest/renege/struct.Token.html):** Tracks the validity of an
arbitrary set of conditions.

```rust
use renege::{Condition, Token};

// Create conditions
let pigs_cant_fly = Condition::new();
let water_is_wet = Condition::new();

// Use .token() to create a Token which tracks the validity of a single Condition
// Tokens can be combined using the & operator
let normality = pigs_cant_fly.token() & water_is_wet.token();
assert!(normality.is_valid());

// Conditions are invalidated when dropped
drop(water_is_wet);

// Use .is_valid() to check if all of the Conditions a Token depends on are still valid
assert!(!normality.is_valid());
```

## Features

* `Token::is_valid()` is extremely fast, requiring just a single atomic read, regardless of how
many conditions it's tracking.

* **Fully concurrent:** `Condition`s and `Token`s can freely be sent between threads. Invalidations
on one thread will propogate to all other threads. All common operations are
[lock-free](https://en.wikipedia.org/wiki/Non-blocking_algorithm).

* Aggressive deduplication ensures that at most one `Token` is created for a given set of
`Condition`s, regardless of how and where it is constructed. e.g. Assuming `a`, `b`, and `c` are
all tokens, one thread can build `(a & b) & (b & c)` while another thread builds `a & b & c`,
and they will both end up with the same `Token` sharing the same underlying storage.