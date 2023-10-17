use crate::internal::*;
use crate::*;

#[test]
fn test_dep_range_between() {
    let scale_and_split = |dep: DependentRange| (dep.scale(), dep.split());
    assert_eq!(
        scale_and_split(DependentRange::between(TokenId::new(1), TokenId::new(2))),
        (1, TokenId::new(2))
    );
    assert_eq!(
        scale_and_split(DependentRange::between(TokenId::new(1), TokenId::new(3))),
        (1, TokenId::new(2))
    );
    assert_eq!(
        scale_and_split(DependentRange::between(TokenId::new(7), TokenId::new(9))),
        (3, TokenId::new(8))
    );
}

fn token_eq(a: Token, b: Token) -> bool {
    std::ptr::eq(a.block.as_ref(), b.block.as_ref()) && a.ext_id == b.ext_id
}

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