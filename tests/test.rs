use renege::{Condition, Token};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

#[test]
fn test_3_cond() {
    let a = Condition::new();
    let b = Condition::new();
    let c = Condition::new();
    let b_c = b.token() & c.token();
    let a_b_c = a.token() & b_c;
    assert!(b_c.is_valid());
    assert!(a_b_c.is_valid());
    drop(a);
    assert!(b_c.is_valid());
    assert!(!a_b_c.is_valid());
    drop(c);
    assert!(!b_c.is_valid());
}

#[test]
fn test_invalid_at_construction() {
    let a = Condition::new();
    assert!(!(a.token() & Token::never()).is_valid());
    let b = Condition::new();
    let c = Condition::new();
    let b_token = b.token();
    drop(b);
    assert!(!(a.token() & b_token).is_valid());
    assert!(!(b_token & c.token()).is_valid());
}

#[test]
fn test_sensitivity_exhaustive() {
    #[cfg(miri)]
    const NUM_CONDS: usize = 4;
    #[cfg(not(miri))]
    const NUM_CONDS: usize = 10;
    for case in 0..NUM_CONDS {
        // Allocate the given number of conditions
        let conds: [Condition; NUM_CONDS] = core::array::from_fn(|_| Condition::new());

        // Create a token for each combination of conditions
        let tokens: [Token; 1 << NUM_CONDS] = core::array::from_fn(|i| {
            let mut token = Token::always();
            for (j, cond) in conds.iter().enumerate() {
                if i & (1 << j) != 0 {
                    token &= cond.token();
                }
            }
            token
        });

        // Invalidate the condition for the current case
        let mut conds = conds.map(Some);
        drop(conds[case].take());

        // Check that the appropriate tokens are still valid
        for (i, token) in tokens.iter().enumerate() {
            assert_eq!(token.is_valid(), i & (1 << case) == 0);
        }
    }
}

#[test]
fn test_debug() {
    let conds: [Condition; 20] = core::array::from_fn(|_| Condition::new());
    let mut token = Token::always();
    let mut cond_ids = Vec::new();
    assert_eq!(format!("{:?}", token), "Token([])");
    for cond in [&conds[0], &conds[7], &conds[11], &conds[17]] {
        let cond_debug = format!("{:?}", cond);
        let cond_id = cond_debug
            .strip_prefix("Condition(")
            .unwrap()
            .strip_suffix(')')
            .unwrap()
            .to_string();
        cond_ids.push(cond_id);
        token &= cond.token();
        assert_eq!(
            format!("{:?}", token),
            format!("Token([{}])", cond_ids.join(", "))
        );
    }
    assert_eq!(format!("{:?}", Token::never()), "Token(<invalid>)");
}

#[test]
fn test_dedup_always() {
    let a = Condition::new();
    let b = Condition::new();
    assert_eq!(Token::always() & Token::always(), Token::always());
    assert_eq!(Token::always() & a.token(), a.token());
    assert_eq!(a.token() & Token::always(), a.token());
    assert_eq!(
        Token::always() & (a.token() & b.token()),
        b.token() & a.token()
    );
}

#[test]
fn test_dedup_complex() {
    let a_cond = Condition::new();
    let b_cond = Condition::new();
    let c_cond = Condition::new();
    let d_cond = Condition::new();
    let e_cond = Condition::new();
    let a = a_cond.token();
    let b = b_cond.token();
    let c = c_cond.token();
    let d = d_cond.token();
    let e = e_cond.token();
    let t_0 = (a & d) & (b & (c & e));
    let t_1 = (d & e) & (a & (b & c));
    let t_2 = a & (c & (e & (b & d)));
    let t_3 = (((e & b) & c) & a) & d;
    assert_eq!(t_0, t_1);
    assert_eq!(t_0, t_2);
    assert_eq!(t_0, t_3);
}

#[test]
fn test_callback() {
    let step = Arc::new(AtomicUsize::new(0));
    let a = Condition::new();
    let b = Condition::new();
    let a_b_token = a.token() & b.token();
    a_b_token.on_invalid({
        let step = step.clone();
        move || {
            assert_eq!(step.load(Relaxed), 0);
            step.store(1, Relaxed);
            println!("(A & B) invalidated");
        }
    });
    a.invalidate_then({
        let step = step.clone();
        move || {
            assert_eq!(step.load(Relaxed), 1);
            step.store(2, Relaxed);
            println!("A invalidated");
        }
    });
    assert_eq!(step.load(Relaxed), 2);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_multithreaded_construct() {
    use rand::{Rng, SeedableRng};
    const NUM_CONDS: usize = 50;
    const NUM_DEPS: usize = 10;
    const NUM_THREADS: usize = 1000;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
    let conds = (0..NUM_CONDS).map(|_| Condition::new()).collect::<Box<_>>();
    let deps = (0..NUM_THREADS)
        .map(|_| {
            // This may cause a condition to appear multiple times, but that's okay because we
            // also want to test deduplication
            (0..NUM_DEPS)
                .map(|_| rng.random_range(0..NUM_CONDS))
                .collect::<Vec<_>>()
        })
        .collect::<Box<_>>();

    // Create combined tokens on separate threads
    let barrier = std::sync::Barrier::new(NUM_THREADS);
    let tokens = std::thread::scope(|s| {
        let barrier = &barrier;
        let res = deps
            .iter()
            .map(|deps| {
                let mut local_rng = rand::rngs::SmallRng::seed_from_u64(rng.random());
                let mut tokens = deps.iter().map(|&i| conds[i].token()).collect::<Vec<_>>();
                s.spawn(move || {
                    // Wait until all threads are ready
                    barrier.wait();

                    // Combine tokens randomly
                    while tokens.len() > 1 {
                        let a = tokens.swap_remove(local_rng.random_range(0..tokens.len()));
                        let b = tokens.swap_remove(local_rng.random_range(0..tokens.len()));
                        tokens.push(a & b);
                    }
                    (deps, tokens.pop().unwrap())
                })
            })
            .collect::<Vec<_>>();
        res.into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });

    // Verify the tokens were constructed correctly
    for (deps, act_token) in tokens {
        let exp_token = deps
            .iter()
            .fold(Token::always(), |acc, &i| acc & conds[i].token());
        assert_eq!(act_token, exp_token);
    }
}

#[test]
fn test_multithreaded_basic() {
    #[cfg(miri)]
    const NUM_ITERS: u64 = 10;
    #[cfg(not(miri))]
    const NUM_ITERS: u64 = 1000;
    std::thread::scope(|s| {
        let (tx, rx) = std::sync::mpsc::channel();
        for i in 0..NUM_ITERS {
            let tx = tx.clone();
            s.spawn(move || {
                println!("Thread {} starting", i);
                let value = std::sync::Arc::new(std::sync::Mutex::new("Good".to_string()));
                let cond = Condition::new();
                #[cfg(not(miri))]
                std::thread::sleep(std::time::Duration::from_millis(100 + i));
                tx.send((value.clone(), cond.token())).unwrap();
                println!("Thread {} sent payload", i);
                #[cfg(not(miri))]
                std::thread::sleep(std::time::Duration::from_millis(i % 10));
                cond.invalidate_immediately();
                *value.lock().unwrap() = format!("Bad({})", i);
                println!("Thread {} ending", i);
            });
        }
        drop(tx);
        while let Ok((value, token)) = rx.recv() {
            let value = value.lock().unwrap().clone();
            if token.is_valid() {
                assert_eq!(value, "Good");
                println!("Observed valid");
            } else {
                println!("Ignored invalid: {}", value);
            }
        }
    });
}

#[test]
fn test_multithreaded_combine() {
    #[cfg(miri)]
    const NUM_ITERS: u64 = 10;
    #[cfg(not(miri))]
    const NUM_ITERS: u64 = 1000;
    std::thread::scope(|s| {
        let any_valid = Condition::new();
        let any_valid_token = any_valid.token();
        let (tx, rx) = std::sync::mpsc::channel();
        for i in 0..NUM_ITERS {
            let tx = tx.clone();
            s.spawn(move || {
                println!("Thread {} starting", i);
                let value = std::sync::Arc::new(std::sync::Mutex::new("Good".to_string()));
                let cond = Condition::new();
                #[cfg(not(miri))]
                std::thread::sleep(std::time::Duration::from_millis(100 + i));
                tx.send((value.clone(), any_valid_token & cond.token()))
                    .unwrap();
                println!("Thread {} sent payload", i);
                #[cfg(not(miri))]
                std::thread::sleep(std::time::Duration::from_millis(i % 10));
                if i % 4 == 0 {
                    std::mem::forget(cond);
                } else {
                    cond.invalidate_immediately();
                    *value.lock().unwrap() = format!("Bad({})", i);
                }
                println!("Thread {} ending", i);
            });
        }
        drop(tx);
        let mut tokens = Vec::new();
        while let Ok((value, token)) = rx.recv() {
            let value = value.lock().unwrap().clone();
            if token.is_valid() {
                assert_eq!(value, "Good");
                println!("Observed valid");
                tokens.push(token);
            } else {
                println!("Ignored invalid: {}", value);
            }
        }
        println!("Invalidating all tokens");
        any_valid.invalidate_immediately();
        println!("Ensuring remaining {} tokens are invalid", tokens.len());
        for token in tokens {
            assert!(!token.is_valid());
        }
    });
}
