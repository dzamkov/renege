#![doc = include_str!("../README.md")]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
pub mod alloc;
mod atomic;
mod imp;
#[cfg(test)]
mod test;

pub use global::{Condition, Token};

/// Contains specialized types that use the global allocator.
mod global {
    use crate::alloc;

    /// A condition that a cache can depend on. Is automatically invalidated when dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// # use renege::Condition;
    /// let cond = Condition::new();
    /// let token = cond.token();
    /// assert!(token.is_valid());
    /// drop(cond);
    /// assert!(!token.is_valid());
    /// ```
    ///
    /// Note that the effects of automatic invalidation of a [`Condition`] are not guaranteed to be
    /// immediately visible in a multi-threaded context. If you have particular requirements for
    /// synchronization/ordering of operations, use [`Condition::invalidate_immediately`] or
    /// [`Condition::invalidate_eventually`] instead.
    #[repr(transparent)]
    pub struct Condition(alloc::Condition<'static>);

    impl Condition {
        /// Creates a new [`Condition`].
        pub fn new() -> Self {
            Self(alloc::Global::with(|global| {
                let id = global.allocate_condition_id();
                alloc::Condition::new(global, id)
            }))
        }

        /// Invalidates this [`Condition`] "immediately".
        ///
        /// See [`alloc::Condition::invalidate_immediately`] for more details.
        pub fn invalidate_immediately(self) {
            let inner = alloc::Condition::from(self);
            alloc::Global::with(|alloc| inner.invalidate_immediately(alloc));
        }

        /// Invalidates this [`Condition`] "eventually".
        ///
        /// See [`alloc::Condition::invalidate_eventually`] for more details. For now, this is the type
        /// of invalidation done when a [`Condition`] is dropped.
        pub fn invalidate_eventually(self) {
            let inner = alloc::Condition::from(self);
            alloc::Global::with(|alloc| inner.invalidate_eventually(alloc));
        }

        /// Gets a [`Token`] which is valid for as long as this [`Condition`] is alive.
        pub fn token(&self) -> Token {
            let block = self.0.block;
            Token(alloc::Condition { block }.token())
        }
    }

    impl Default for Condition {
        fn default() -> Self {
            Self::new()
        }
    }

    impl From<alloc::Condition<'static>> for Condition {
        fn from(value: alloc::Condition<'static>) -> Self {
            Self(value)
        }
    }

    impl From<Condition> for alloc::Condition<'static> {
        fn from(value: Condition) -> Self {
            let block = value.0.block;
            // Disable invalidate-on-drop behavior
            std::mem::forget(value);
            alloc::Condition { block }
        }
    }

    impl std::ops::Drop for Condition {
        fn drop(&mut self) {
            let block = self.0.block;
            alloc::Global::with(|alloc| alloc::Condition { block }.invalidate_eventually(alloc));
        }
    }

    impl std::fmt::Debug for Condition {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    /// Tracks the validity of an arbitrary set of [`Condition`]s.
    #[derive(Clone, Copy)]
    pub struct Token(alloc::Token<'static>);

    impl Token {
        /// Gets a token which is always valid. This is the [`Default`] token.
        ///
        /// # Examples
        ///
        /// ```
        /// # use renege::Token;
        /// assert!(Token::always().is_valid())
        /// ```
        pub const fn always() -> Self {
            Self(alloc::Token::always())
        }

        /// Gets a token which is never valid.
        ///
        /// # Examples
        ///
        /// ```
        /// # use renege::Token;
        /// assert!(!Token::never().is_valid())
        /// ```
        pub const fn never() -> Self {
            Self(alloc::Token::never())
        }

        /// Indicates whether this token is still valid. Once this returns `false`, it will never
        /// return `true` again.
        pub fn is_valid(&self) -> bool {
            self.0.is_valid()
        }

        /// Indicates whether this token will always be valid.
        pub fn is_always_valid(&self) -> bool {
            self.0.is_always_valid()
        }
    }

    impl Default for Token {
        fn default() -> Self {
            Self::always()
        }
    }

    impl From<alloc::Token<'static>> for Token {
        fn from(value: alloc::Token<'static>) -> Self {
            Self(value)
        }
    }

    impl From<Token> for alloc::Token<'static> {
        fn from(value: Token) -> Self {
            value.0
        }
    }

    impl std::ops::BitAnd for Token {
        type Output = Token;
        fn bitand(self, rhs: Self) -> Token {
            alloc::Global::with(|alloc| Token(alloc::Token::combine(alloc, self.0, rhs.0)))
        }
    }

    impl std::ops::BitAndAssign for Token {
        fn bitand_assign(&mut self, rhs: Self) {
            *self = *self & rhs;
        }
    }

    impl std::fmt::Debug for Token {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }
}
