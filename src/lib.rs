#![doc = include_str!("../README.md")]
#![deny(missing_debug_implementations)]
#![cfg_attr(not(loom), deny(missing_docs))]
pub mod alloc;
mod atomic;
mod imp;
mod util;

#[cfg(not(loom))]
pub use global::{Condition, Token};

/// Contains specialized types that use the global allocator.
#[cfg(not(loom))]
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
    #[repr(transparent)]
    #[derive(PartialEq, Eq)]
    pub struct Condition(alloc::Condition<'static>);

    impl Condition {
        /// Creates a new [`Condition`].
        pub fn new() -> Self {
            Self(alloc::Global::with(|global| {
                let id = global.allocate_condition_id();
                alloc::Condition::new(global, id)
            }))
        }

        /// Invalidates this [`Condition`] "immediately". This has the same effect as dropping the
        /// [`Condition`].
        ///
        /// This may block the current thread. See [`alloc::Condition::invalidate_immediately`] for
        /// more details.
        pub fn invalidate_immediately(self) {
            let inner = alloc::Condition::from(self);
            alloc::Global::with(|alloc| inner.invalidate_immediately(alloc));
        }

        /// Invalidates this [`Condition`], and then calls the given function.
        ///
        /// This will never block the current thread. See [`alloc::Condition::invalidate_then`] for
        /// more details.
        pub fn invalidate_then(self, f: impl FnOnce() + Send + 'static) {
            let inner = alloc::Condition::from(self);
            alloc::Global::with(|alloc| inner.invalidate_then(alloc, f));
        }

        /// Invalidates this [`Condition`] "eventually".
        ///
        /// See [`alloc::Condition::invalidate_eventually`] for more details.
        pub fn invalidate_eventually(self) {
            let inner = alloc::Condition::from(self);
            alloc::Global::with(|alloc| inner.invalidate_eventually(alloc));
        }

        /// Sets this [`Condition`] to be invalidated "immediately" once `token` is no longer
        /// valid.
        ///
        /// Note that will cause a memory leak if `token` is [`Token::always`], since there would
        /// be no way to invalidate the condition.
        ///
        /// See [`alloc::Condition::invalidate_from_immediately`] for more details.
        pub fn invalidate_from_immediately(self, token: Token) {
            let inner = alloc::Condition::from(self);
            let token = alloc::Token::from(token);
            alloc::Global::with(|alloc| inner.invalidate_from_immediately(alloc, token));
        }

        /// Attempts to set this [`Condition`] to be invalidated "immediately" once `token` is no
        /// longer valid.
        ///
        /// This can only fail if `token` is already invalid, in which case it will return the
        /// condition unchanged. Unlike [`Condition::invalidate_from_immediately`], this will never
        /// block the current thread.
        ///
        /// Note that will cause a memory leak if `token` is [`Token::always`], since there would
        /// be no way to invalidate the condition.
        ///
        /// See [`alloc::Condition::try_invalidate_from`] for more details.
        pub fn try_invalidate_from(self, token: Token) -> Result<(), Self> {
            let inner = alloc::Condition::from(self);
            let token = alloc::Token::from(token);
            alloc::Global::with(|alloc| inner.try_invalidate_from(alloc, token))
                .map_err(Condition::from)
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
            if !std::thread::panicking() {
                let block = self.0.block;
                alloc::Global::with(|alloc| {
                    alloc::Condition { block }.invalidate_immediately(alloc)
                });
            }
        }
    }

    impl std::fmt::Debug for Condition {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    /// Tracks the validity of an arbitrary set of [`Condition`]s.
    /// 
    /// Tokens can be combined using the `&` operator. i.e. `a & b` is valid if and only if
    /// `a` and `b` are both valid.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use renege::{Condition, Token};
    /// let a = Condition::new();
    /// let b = Condition::new();
    /// let a_b = a.token() & b.token();
    /// assert!(a_b.is_valid());
    /// drop(a);
    /// assert!(!a_b.is_valid());
    /// ```
    #[derive(PartialEq, Eq, Clone, Copy)]
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

        /// Ensures that `f()` is called once this token is invalidated.
        ///
        /// This will never block the current thread. If this token is already invalid, the call to
        /// `f()` will happen immediately.  The call to `f()` will happen exactly once, and may 
        /// occur on any thread. It should not block the calling thread.
        /// 
        /// # Examples
        /// 
        /// ```
        /// # use renege::{Condition, Token};
        /// let cond = Condition::new();
        /// cond.token().on_invalid(|| println!("Token invalidated!"));
        /// drop(cond); // Prints "Token invalidated!" 
        /// ```
        pub fn on_invalid(self, f: impl FnOnce() + Send + 'static) {
            let inner = alloc::Token::from(self);
            alloc::Global::with(|alloc| inner.on_invalid(alloc, f));
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
