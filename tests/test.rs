use renege::{Condition, Token};

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
                tx.send((value.clone(), any_valid_token & cond.token())).unwrap();
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