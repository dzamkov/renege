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
