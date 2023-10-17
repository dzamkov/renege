#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
mod atomic;
mod internal;
#[cfg(test)]
mod test;
use std::ops::{BitAnd, BitAndAssign};


/// A condition that a cache can depend on. Is automatically invalidated when dropped.
///
/// ```
/// # use renege::Condition;
/// let cond = Condition::new();
/// let token = cond.token();
/// assert!(token.is_valid());
/// drop(cond);
/// assert!(!token.is_valid());
/// ```
pub struct Condition(Token);

impl Condition {
    /// Creates a new [`Condition`].
    pub fn new() -> Self {
        Self(internal::ThreadAllocator::with(internal::source))
    }

    /// Gets a [`Token`] which is valid for as long as this [`Condition`] is alive.
    pub fn token(&self) -> Token {
        self.0
    }
}

impl Drop for Condition {
    fn drop(&mut self) {
        let Token { block, ext_id } = self.0;
        internal::ThreadAllocator::with(|alloc| {
            internal::invalidate_source(alloc, block.as_ref(), ext_id)
        });
    }
}

impl Default for Condition {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks the valdity of an arbitrary set of [`Condition`]s.
#[derive(Clone, Copy)]
pub struct Token {
    block: internal::TokenBlockRef<'static>,
    ext_id: internal::ExtTokenId,
}

impl Token {
    /// Gets a token which is always valid. This is the [`Default`] token.
    ///
    /// ```
    /// # use renege::Token;
    /// assert!(Token::always().is_valid())
    /// ```
    pub fn always() -> Self {
        internal::always()
    }

    /// Gets a token which is never valid.
    ///
    /// ```
    /// # use renege::Token;
    /// assert!(!Token::never().is_valid())
    /// ```
    pub fn never() -> Self {
        internal::never()
    }

    /// Indicates whether this token is still valid. Once this returns `false`, it will never
    /// return `true` again.
    pub fn is_valid(&self) -> bool {
        internal::is_valid(*self)
    }
}

impl Default for Token {
    fn default() -> Self {
        Self::always()
    }
}

impl BitAnd for Token {
    type Output = Token;
    fn bitand(self, rhs: Self) -> Token {
        internal::ThreadAllocator::with(|alloc| internal::dependent(alloc, self, rhs))
    }
}

impl BitAndAssign for Token {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}
