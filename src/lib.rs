#![doc = include_str!("../README.md")]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
mod atomic;
#[cfg(test)]
mod test;
mod imp;

pub use imp::{Condition, Token};