use crate::{Token, Condition};
use crate::imp::token_eq;

#[test]
fn test_dedup_always() {
    let a = Condition::new();
    let b = Condition::new();
    assert!(token_eq(Token::always() & Token::always(), Token::always()));
    assert!(token_eq(Token::always() & a.token(), a.token()));
    assert!(token_eq(a.token() & Token::always(), a.token()));
    assert!(token_eq(
        Token::always() & (a.token() & b.token()),
        b.token() & a.token()
    ));
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
    assert!(token_eq(t_0, t_1));
    assert!(token_eq(t_0, t_2));
    assert!(token_eq(t_0, t_3));
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
        assert!(token_eq(act_token, exp_token));
    }
}
