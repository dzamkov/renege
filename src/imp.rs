use crate::alloc::Allocator;
use crate::atomic::{Atomic, HasAtomic};
use crate::atomic::{AtomicBool, fence};
use crate::util::SafeTransmuteFrom;
use std::cell::UnsafeCell;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::mem::ManuallyDrop;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};

/// A condition that a cache can depend on.
///
/// The backing storage used by the condition has a lifetime of `'alloc` and is managed by
/// explicitly-provided [`Allocator`]s. Note that, unlike the top-level [`crate::Condition`], this
/// is not automatically invalidated when dropped because an [`Allocator`] is needed to perform
/// invalidation.
pub struct Condition<'alloc> {
    /// The block where the token for this condition is defined. The condition has the exclusive
    /// right to invalidate this block, so we can assume it is valid until the condition is dropped.
    pub(crate) block: &'alloc Block<'alloc>,
}

impl<'alloc> Condition<'alloc> {
    /// Creates a new [`Condition`] using the given [`Allocator`].
    ///
    /// There should not be any live [`Condition`]s with the same [`ConditionId`] and backing
    /// storage. For best performance, [`ConditionId`]s should be assigned in roughly increasing
    /// order, so the age of a [`Condition`] can be approximated by its [`ConditionId`].
    pub fn new<Alloc: Allocator<'alloc> + ?Sized>(alloc: &mut Alloc, id: ConditionId) -> Self {
        let block = alloc.allocate_block();
        block.verify_clean();

        // Relaxed memory order is sufficient here because the block won't be visible to
        // any other threads until the `Condition` is shared by the user, which would
        // require its own synchronization.
        block.range.store(ConditionRange::single(id), Relaxed);
        let mut tag = block.tag.load(Relaxed);
        debug_assert_eq!(tag.0 & !BlockTag::VERSION_MASK, 0);
        tag.0 |= BlockTag::TOKEN_FLAG;
        block.tag.store(tag, Relaxed);
        Self { block }
    }

    /// Invalidates this [`Condition`] "immediately".
    ///
    /// This call may block the current thread. If there are no requirements for when the
    /// invalidation should become visible, use [`Condition::invalidate_eventually`] instead.
    /// If blocking the current thread is not acceptable, use [`Condition::invalidate_then`]
    /// instead.
    ///
    /// # Memory Ordering
    ///
    /// Let `x` be any [`Token`] that was constructed from `self.token()`.
    ///
    /// All calls to `x.is_valid()` that
    /// [happen after](https://en.wikipedia.org/wiki/Happened-before)
    /// this call are guaranteed to return `false`.
    pub fn invalidate_immediately<Alloc: Allocator<'alloc> + ?Sized>(self, alloc: &mut Alloc) {
        #[cfg(loom)]
        use loom::thread::{Thread, current, park};
        #[cfg(not(loom))]
        use std::thread::{Thread, current, park};

        // Store information required to park/unpark the current thread. We can safely store this
        // on the stack since we won't return from this function until the callback is called
        let waiter = Waiter {
            thread: UnsafeCell::new(Some(current())),
            is_complete: AtomicBool::new(false),
        };
        let data = &waiter as *const _ as *mut ();
        if unsafe { !self.invalidate_then_raw(alloc, on_complete, data) } {
            while !waiter.is_complete.load(Relaxed) {
                park()
            }
        }

        /// Called when invalidation is complete.
        unsafe fn on_complete(data: *mut ()) {
            // SAFETY: `data` is a pointer to a `Waiter` which can't be dropped until after
            // `is_complete` is set to `true`.
            let waiter = unsafe { &*(data as *const Waiter) };
            // SAFETY: `waiter.thread` can only be accessed inside of this function, and this
            // function will be called at most once for a given `Waiter`, so `waiter.thread` should
            // be non-`None` at this point.
            let thread = unsafe { (*waiter.thread.get()).take().unwrap_unchecked() };
            waiter.is_complete.store(true, Relaxed);
            thread.unpark(); // Synchronizes with unpark
        }

        /// Information about a thread which may be parked.
        struct Waiter {
            thread: UnsafeCell<Option<Thread>>,
            is_complete: AtomicBool,
        }
    }

    /// Begins invalidating this [`Condition`], ensuring that `f()` is called once invalidation
    /// completes.
    ///
    /// This will never block the current thread.
    ///
    /// The call to `f()` will happen exactly once, and may occur on any thread that has access to
    /// an [`Allocator`] for `'alloc`. It should not block the calling thread.
    ///
    /// # Memory Ordering
    ///
    /// Let `x` be any [`Token`] that was constructed from `self.token()`.
    ///
    /// All calls to `x.is_valid()` that occur inside or
    /// [happen after](https://en.wikipedia.org/wiki/Happened-before) the call to `f()` are
    /// guaranteed to return `false`.
    pub fn invalidate_then<Alloc: Allocator<'alloc> + ?Sized>(
        self,
        alloc: &mut Alloc,
        f: impl FnOnce() + Send + 'alloc,
    ) {
        let (g, data) = crate::util::into_fn_ptr(f);

        // SAFETY: `g(data)` may be called on any thread at any time, assuming that `'alloc`
        // is valid for the duration of the call.
        unsafe {
            if self.invalidate_then_raw(alloc, g, data) {
                g(data)
            }
        }
    }

    /// Lower-level version of [`Condition::invalidate_then`].
    ///
    /// Begins invalidating this [`Condition`]. This will either return `true` to indicate that
    /// invalidation has been completed immediately, or return `false`, in which case `f(data)`
    /// will be called once invalidation completes.
    ///
    /// This will never block the current thread.
    ///
    /// If this returns `false`, the call to `f(data)` will happen exactly once, and may occur on
    /// any thread that has access to an [`Allocator`] for `'alloc`. It should not block the
    /// calling thread.
    ///
    /// # Memory Ordering
    ///
    /// Let `x` be any [`Token`] that was constructed from `self.token()`.
    ///
    /// If this returns `true`, all calls to `x.is_valid()` that
    /// [happen after](https://en.wikipedia.org/wiki/Happened-before) this call are guaranteed to
    /// return `false`.
    ///
    /// If this returns `false`, all calls to `x.is_valid()` that occur inside or
    /// [happen after](https://en.wikipedia.org/wiki/Happened-before) the call to `f(data)` are
    /// guaranteed to return `false`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `f(data)` is safe to call on any thread at any time. It may
    /// assume that `'alloc` is valid for the duration of the call, and that it is called at
    /// most once.
    pub unsafe fn invalidate_then_raw<Alloc: Allocator<'alloc> + ?Sized>(
        self,
        alloc: &mut Alloc,
        f: unsafe fn(*mut ()),
        data: *mut (),
    ) -> bool {
        let block = self.block;
        let block = match invalidate_condition::<TokenBlockHolding<'alloc>>(block) {
            Ok(block) => match finalize::<_, TokenBlockHolding<'alloc>>(alloc, block, None, None) {
                Ok(_) => return true,
                Err(block) => block,
            },
            Err(block) => block,
        };

        // Check if the block has no children
        if block.first_left_child().load(Relaxed).is_none()
            && block.right_tree().load(Relaxed).is_null()
        {
            // The block is invalid and has no children. No more children can be added, so the
            // next time it attempts finalization, it will be finalized. This means that if we
            // were to install a callback now, it will be called just before the block is finalized.
            // Perfect!

            // SAFETY: The caller must ensure that calling `f(data)` is safe
            unsafe { install_callback(alloc, &block, f, data) };
            unhold_finalize(alloc, block);
            false
        } else {
            // This case is trickier, the block may still have children and descendants that haven't
            // been invalidated. If we were to install a callback now, it would be called as
            // soon as it starts finalization, which is too early since that is well before we
            // can guarantee that all children have been finalized.

            // We need to make sure the callback is called *after* all children are finalized.
            // The simplest way to do this, without introducing any new flags/concepts, is to
            // create an artifical parent of `block` which is being held by `block`. The
            // parent can't be finalized until `block` finishes finalization, so if we install
            // the callback on this parent, it will be called at the right time.

            // There are other cases where we add a condition token block as a left child, so
            // we might as well continue that trend and use an artifical left parent.
            let parent = alloc.allocate_block();
            parent.verify_clean();
            let mut tag = parent.tag.load(Relaxed);
            debug_assert_eq!(tag.0 & !BlockTag::VERSION_MASK, 0);
            tag.0 |= BlockTag::TOKEN_FLAG;
            tag.0 |= BlockTag::INVALID_FLAG; // Ensure the parent will be finalized once unheld
            tag.0 += 2; // One holder from `block`, one created below
            parent.tag.store(tag, Relaxed);
            block.left_parent().store(Some(parent), Relaxed);
            let parent = TokenBlockHolding { block: parent };
            // SAFETY: The caller must ensure that calling `f(data)` is safe
            unsafe { install_callback(alloc, &parent, f, data) };
            let parent = unhold(parent);
            debug_assert!(parent.is_none());

            // Now set `block` to be holding `parent`
            block.tag.0.fetch_or(BlockTag::HOLDING_LEFT_FLAG, AcqRel); // Synchronizes with token unhold
            unhold_finalize(alloc, block);
            false
        }
    }

    /// Begins invalidating this [`Condition`].
    ///
    /// This will never block the current thread, but offers no guarantees about when the effects
    /// of the invalidation will be visible, even to the current thread. If stricter ordering
    /// is required, use [`Condition::invalidate_immediately`] or [`Condition::invalidate_then`].
    pub fn invalidate_eventually<Alloc: Allocator<'alloc> + ?Sized>(self, alloc: &mut Alloc) {
        let block = self.block;
        if let Ok(block) = invalidate_condition::<()>(block) {
            let _ = finalize::<_, ()>(alloc, block, None, None);
        }
    }

    /// Sets this [`Condition`] to be invalidated "immediately" once `token` is no longer valid.
    ///
    /// If `token` has already been invalidated, this is equivalent to
    /// [`Condition::invalidate_immediately`]. Note that this may block the current thread. If
    /// this is not acceptable, use [`Condition::try_invalidate_from`] instead.
    ///
    /// # Memory Ordering
    ///
    /// Let `x` be any [`Token`] that was constructed from `self.token()`.
    ///
    /// All calls to `x.is_valid()` that
    /// [happen after](https://en.wikipedia.org/wiki/Happened-before)
    /// this call are guaranteed to return `false` if `token.is_valid()` would return `false`.
    pub fn invalidate_from_immediately<Alloc: Allocator<'alloc> + ?Sized>(
        self,
        alloc: &mut Alloc,
        token: Token<'alloc>,
    ) {
        if let Err(err) = self.try_invalidate_from(alloc, token) {
            err.invalidate_immediately(alloc);
        }
    }

    /// Attempts to set this [`Condition`] to be invalidated "immediately" once `token` is no
    /// longer valid.
    ///
    /// This can only fail if `token` is already invalid, in which case it will return the
    /// condition unchanged. Unlike [`Condition::invalidate_from_immediately`], this will never
    /// block the current thread.
    ///
    /// # Memory Ordering
    ///
    /// Let `x` be any [`Token`] that was constructed from `self.token()`.
    ///
    /// If this call returns `Ok(())`, all calls to `x.is_valid()` that
    /// [happen after](https://en.wikipedia.org/wiki/Happened-before)
    /// this call are guaranteed to return `false` if `token.is_valid()` would return `false`.
    pub fn try_invalidate_from<Alloc: Allocator<'alloc> + ?Sized>(
        self,
        alloc: &mut Alloc,
        token: Token<'alloc>,
    ) -> Result<(), Self> {
        if let Some(block) = hold(token) {
            // Add this condition as a left child of the token block
            self.block.left_parent().store(Some(block.block), Relaxed);
            insert_left_child(alloc, &block, self.block);
            unhold_finalize(alloc, block);
            Ok(())
        } else {
            Err(self)
        }
    }

    /// Gets the [`ConditionId`] for this [`Condition`].
    pub fn id(&self) -> ConditionId {
        self.block.range.load(Relaxed).min()
    }

    /// Gets a [`Token`] which is valid for as long as this [`Condition`] is alive.
    pub fn token(&self) -> Token<'alloc> {
        self.block.token()
    }
}

impl PartialEq for Condition<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.block, other.block)
    }
}

impl Eq for Condition<'_> {}

impl std::fmt::Debug for Condition<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let range = self.block.range.load(Relaxed);
        f.debug_tuple("Condition").field(&range.min().0).finish()
    }
}

/// Tracks the validity of an arbitrary set of [`Condition`]s.
///
/// The backing storage used by the token has a lifetime of `'alloc` and is managed by
/// explicitly-provided [`Allocator`]s.
#[derive(Clone, Copy)]
pub struct Token<'alloc> {
    /// The block where this token is/was defined.
    block: &'alloc Block<'alloc>,

    /// The exclusive maximum value of [`Block::tag`] for this token to be considered valid.
    max_tag: BlockTag,
}

impl<'alloc> Token<'alloc> {
    /// Gets a token which is always valid. This is the [`Default`] token.
    ///
    /// # Examples
    ///
    /// ```
    /// # use renege::Token;
    /// assert!(Token::always().is_valid())
    /// ```
    #[cfg(not(loom))]
    pub const fn always() -> Self {
        Self::always_inner(&ALWAYS_NEVER)
    }

    #[cfg(loom)]
    pub fn always() -> Self {
        Self::always_inner(&ALWAYS_NEVER)
    }

    /// Implementation of [`Token::always`] given the [`ALWAYS_NEVER`] block.
    const fn always_inner(always_never: &'static Block<'static>) -> Self {
        // SAFETY: `ALWAYS_NEVER` will never be returned from `Allocator::allocate_block`,
        // so it is permanently free. Thus, all of its internal references will always be `None`
        // and it can be safely interpreted as having any `'alloc` lifetime.
        let block = unsafe {
            std::mem::transmute::<&'alloc Block<'static>, &'alloc Block<'alloc>>(always_never)
        };
        Self {
            block,
            max_tag: BlockTag::new(1, true).max_tag(),
        }
    }

    /// Gets a token which is never valid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use renege::Token;
    /// assert!(!Token::never().is_valid())
    /// ```
    #[cfg(not(loom))]
    pub const fn never() -> Self {
        Self::never_inner(&ALWAYS_NEVER)
    }

    #[cfg(loom)]
    pub fn never() -> Self {
        Self::never_inner(&ALWAYS_NEVER)
    }

    /// Implementation of [`Token::never`] given the [`ALWAYS_NEVER`] block.
    const fn never_inner(always_never: &'static Block<'static>) -> Self {
        // SAFETY: `ALWAYS_NEVER` will never be returned from `Allocator::allocate_block`,
        // so it is permanently free. Thus, all of its internal references will always be `None`
        // and it can be safely interpreted as having any `'alloc` lifetime.
        let block = unsafe {
            std::mem::transmute::<&'alloc Block<'static>, &'alloc Block<'alloc>>(always_never)
        };
        Self {
            block,
            max_tag: BlockTag::new(0, true).max_tag(),
        }
    }

    /// Gets a [`Token`] which is valid precisely when both of the given [`Token`]s are valid.
    pub fn combine<Alloc: Allocator<'alloc> + ?Sized>(
        alloc: &mut Alloc,
        a: Token<'alloc>,
        b: Token<'alloc>,
    ) -> Token<'alloc> {
        if a.is_always_valid() {
            b
        } else if b.is_always_valid() {
            a
        } else {
            Token::combine_non_always(alloc, a, b)
        }
    }

    /// Indicates whether this token is still valid. Once this returns `false`, it will never
    /// return `true` again.
    pub fn is_valid(&self) -> bool {
        self.block.tag.load(Relaxed) < self.max_tag
    }

    /// Indicates whether this token will always be valid.
    pub fn is_always_valid(&self) -> bool {
        self == &Self::always()
    }

    /// Ensures that `f()` is called once this token is invalidated.
    ///
    /// This will never block the current thread. If this token is already invalid, the call to
    /// `f()` will happen immediately.  The call to `f()` will happen exactly once, and may occur
    /// on any thread that has access to an [`Allocator`] for `'alloc`. It should not block the
    /// calling thread.
    pub fn on_invalid<Alloc: Allocator<'alloc> + ?Sized>(
        self,
        alloc: &mut Alloc,
        f: impl FnOnce() + Send + 'alloc,
    ) {
        let (g, data) = crate::util::into_fn_ptr(f);
        // SAFETY: `g(data)` may be called on any thread at any time, assuming that `'alloc`
        // is valid for the duration of the call.
        unsafe {
            if self.on_invalid_raw(alloc, g, data) {
                g(data)
            }
        }
    }

    /// Lower-level version of [`Token::on_invalid`].
    ///
    /// Either returns `true` if this token is invalid, or returns `false` and ensures `f(data)`
    /// will be called once this token is invalidated.
    ///
    /// This will never block the current thread. If this returns `false` the call to `f(data)`
    /// will happen exactly once, and may occur on any thread that has access to an
    /// [`Allocator`] for `'alloc`. It should not block the calling thread.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `f(data)` is safe to call on any thread at any time. It may
    /// assume that `'alloc` is valid for the duration of the call, and that it is called at
    /// most once.
    pub unsafe fn on_invalid_raw<Alloc: Allocator<'alloc> + ?Sized>(
        self,
        alloc: &mut Alloc,
        f: unsafe fn(*mut ()),
        data: *mut (),
    ) -> bool {
        if let Some(block) = hold(self) {
            // SAFETY: The caller must ensure that calling `f(data)` is safe
            unsafe { install_callback(alloc, &block, f, data) };
            unhold_finalize(alloc, block);
            false
        } else {
            true
        }
    }
}

impl Default for Token<'_> {
    fn default() -> Self {
        Self::always()
    }
}

impl PartialEq for Token<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.block, other.block) && self.max_tag == other.max_tag
    }
}

impl Eq for Token<'_> {}

impl std::fmt::Debug for Token<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_tuple("Token");
        let mut conds = Vec::new();
        if self.conditions(|c| conds.push(c.0)) {
            d.field(&conds);
        } else {
            d.field(&format_args!("<invalid>"));
        }
        d.finish()
    }
}

/// The primitive unit of data storage for the library.
///
/// Blocks are reusable units of data which provide the backing storage for [`Condition`]s,
/// [`Token`]s and several additional bookkeeping items.
#[repr(C)]
pub struct Block<'alloc> {
    /// Provides information about the type and lifecycle of a block.
    tag: Atomic<BlockTag>,

    /// Assuming that this is a [`BlockType::Token`] block, this field describes the approximate
    /// range of [`ConditionId`]s which the token may be sensitive to.
    ///
    /// This may not change while the block is live.
    range: Atomic<ConditionRange>,

    /// The "protected" data for this [`Block`].
    ///
    /// This may be accessed immutably if this block is a [`BlockType::Token`] block (and it
    /// can be guaranteed that it will remain one for the duration of the access).
    /// 
    /// This may be accessed mutably if this block is a [`BlockType::Callback`] block.
    ///
    /// [`UnsafeCell`] is required because there is no portable way of storing a function pointer
    /// in an [`Atomic`], as needed by `callback_fn`. To minimize [`Block`]s size, we store the
    /// mutually-exclusive field `first_left_child` in the same union.
    protected: UnsafeCell<Protected<'alloc>>,

    /// The [`BlockType::Token`] block which is the "right" parent of this block. The right parent
    /// of [`BlockType::Callback`] blocks is always [`None`].
    ///
    /// This may not change while the block is live. It must be set to [`None`] before
    /// incrementing the block's `VERSION` and freeing the block.
    ///
    /// The `range` of the parent tokens must not overlap, with all [`ConditionId`]s in the `range`
    /// of the left parent token being less than those in the `range` of the right parent token.
    right_parent: Atomic<Option<&'alloc Block<'alloc>>>,

    /// The [`Beta`] data for this block.
    beta: Beta<'alloc>,
}

/// The "protected" data for a [`Block`].
union Protected<'alloc> {
    /// Assuming that this is a [`BlockType::Token`] block, this field stores a reference to the
    /// first block in the doubly-linked list of token blocks which have this block as their
    /// left parent.
    pub first_left_child: ManuallyDrop<Atomic<Option<&'alloc Block<'alloc>>>>,

    /// Assuming that this is a [`BlockType::Callback`] block, holds the function pointer for the
    /// callback.
    pub callback_fn: unsafe fn(*mut ()),
}

/// An arbitrary union of mutually-exclusive data with compatible representations, intended to save
/// space in the [`Block`] struct.
///
/// Fields within this group may be targeted by `compare_exchange` operations at any time, even
/// after the block is freed.
union Beta<'alloc> {
    pub non_branch: ManuallyDrop<NonBranchBeta<'alloc>>,
    pub branch: ManuallyDrop<BranchBeta<'alloc>>,

    /// Assuming that this is a [`BlockType::Callback`] block, holds the data pointer for the
    /// callback.
    pub callback_data: ManuallyDrop<Atomic<*mut ()>>,
}

/// The [`Beta`] data for a non-[`BlockType::Branch`] block.
#[repr(C)]
struct NonBranchBeta<'alloc> {
    /// Assuming that this is a [`BlockType::Token`] block, holds a reference to the block at the
    /// root of the "right search tree" for this block.
    ///
    /// The right search tree contains all token blocks which have this block as their
    /// right parent. The tree is indexed by the address of the left parent of the token block.
    /// This allows efficiently searching for a token block which has a given left and right parent.
    ///
    /// While the block is live, modifications to this field are restricted. This may only
    /// be updated if any of the following are true:
    ///  * `right_tree` is [`None`]
    ///  * `right_tree` has a `right_parent` which is not this block
    ///  * This block is being finalized by the current thread
    pub right_tree: Atomic<Option<&'alloc Block<'alloc>>>,

    /// The [`BlockType::Token`] block which is the "left" parent of this block.
    /// 
    /// This may only change from [`None`] while the block is live. It must be set to [`None`]
    /// before incrementing the block's `VERSION` and freeing the block.
    ///
    /// The `range` of the parent tokens must not overlap, with all [`ConditionId`]s in the `range`
    /// of the left parent token being less than those in the `range` of the right parent token.
    pub left_parent: Atomic<Option<&'alloc Block<'alloc>>>,

    /// A reference to the next payload block in the doubly-linked list of token blocks which
    /// share the same left parent.
    pub next_left_sibling: Atomic<SiblingBlockRef<'alloc>>,

    /// A reference to the previous payload block in the doubly-linked list of token blocks which
    /// share the same left parent.
    pub prev_left_sibling: Atomic<Option<&'alloc Block<'alloc>>>,
}

/// The [`Beta`] data for a [`BlockType::Branch`] block.
#[repr(C)]
struct BranchBeta<'alloc> {
    /// The child nodes of branch block, organzied by the next few bits of their `left_parent`'s
    /// address.
    ///
    /// While the block is live, modifications to this field are restricted. A child may only be
    /// changed if any of the following are true:
    ///  * The child is [`None`]
    ///  * The child has a `right_parent` which does not match the root of the search tree
    ///  * The right parent of this block is being finalized by the current thread
    pub children: [Atomic<Option<&'alloc Block<'alloc>>>; BRANCH_SIZE],
}

/// Provides information about the type and lifecycle of a [`Block`].
///
/// The structure of the tag is as follows:
/// ```text
/// │ ...  20 │ 19 │ 18 │ 17 │ 16 │ 15 │ 14 │ 13 │ 12 │ 11 │ 10 │  9   ...   3 │  1 │  0 │
/// ├───────────────────────────────────────┬────┬────┬────┬────┬────────────────────────┤
/// │                VERSION                │ TK │  I │ HL │ HR │       NUM_HOLDERS      │
/// └───────────────────────────────────────┴────┴────┴────┴────┴────────────────────────┘
/// TK - TOKEN_FLAG
/// I  - INVALID_FLAG
/// HL - HOLDING_LEFT_FLAG
/// HR - HOLDING_RIGHT_FLAG
/// ```
///
/// # Invariants
///
/// For all blocks, the following rules apply:
///  * `VERSION` may never decrease
///  * Unless `TOKEN_FLAG` is set, `VERSION` is the only non-zero component of the tag
///  * `TOKEN_FLAG` may not be cleared unless `VERSION` increases
///  * `INVALID_FLAG` may not be cleared unless `VERSION` increases
///
/// [`BlockType::Branch`] and [`BlockType::Callback`] blocks don't make modifications to the tag:
/// they just store a version number. However, once a block is allocated as a [`BlockType::Token`]
/// block, its `TOKEN_FLAG` is set and the version number must be incremented before it can be
/// freed.
///
/// For [`BlockType::Token`] blocks, the following rules apply:
///  * `NUM_HOLDERS` provides the count of [`TokenBlockHolding`]s of the block.
///  * Whenever the tag changes to a state where the `INVALID_FLAG` is set, and `NUM_HOLDERS`
///    is zero, the thread that made the change obtains an exclusive [`TokenBlockFinalizing`] for
///    the block (i.e. it becomes the "finalizing" thread and is responsible for [`finalize`]ing
///    the block).
///  * When the `INVALID_FLAG` is set, additional [`TokenBlockHolding`]s may not be created
///    except through a [`TokenBlockFinalizing`].
///  * If the `HOLD_LEFT_FLAG` is set, [`TokenBlockFinalizing`] implicitly comes with a
///    [`TokenBlockHolding`] for the left parent.
///  * If the `HOLD_RIGHT_FLAG` is set, [`TokenBlockFinalizing`] implicitly comes with a
///    [`TokenBlockHolding`] for the right parent.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
struct BlockTag(usize);

impl HasAtomic for BlockTag {
    type Prim = usize;
}

unsafe impl SafeTransmuteFrom<BlockTag> for usize {}

impl BlockTag {
    /// A flag which indicates that a [`Block`] is a [`BlockType::Token`] block.
    ///
    /// There is no need to distinguish between [`BlockType::Branch`] and [`BlockType::Callback`]
    /// blocks, because they are never used in the same context: branch blocks can only appear
    /// in a right search tree, while callback blocks can only appear in a left child list.
    pub const TOKEN_FLAG: usize = 1 << 13;

    /// A flag which indicates that a [`BlockType::Token`] block is invalid.
    pub const INVALID_FLAG: usize = 1 << 12;

    /// A flag which indicates that a [`BlockType::Token`] block is holding its left parent.
    pub const HOLDING_LEFT_FLAG: usize = 1 << 11;

    /// A flag which indicates that a [`BlockType::Token`] block is holding its right parent.
    pub const HOLDING_RIGHT_FLAG: usize = 1 << 10;

    /// The bit mask which selects `NUM_HOLDERS` from tha tag.
    pub const NUM_HOLDERS_MASK: usize = (1 << 10) - 1;

    /// The shift applied to the version number in the tag.
    pub const VERSION_SHIFT: u32 = 14;

    /// A bit mask which selects the version number from the tag.
    pub const VERSION_MASK: usize = usize::MAX << Self::VERSION_SHIFT;

    /// The maximum allowed version number for a block.
    pub const MAX_VERSION: usize = Self::VERSION_MASK >> Self::VERSION_SHIFT;

    /// Constructs a [`BlockTag`] with the given version and token flag. All other bits are set to
    /// zero.
    pub const fn new(version: usize, is_token: bool) -> Self {
        assert!(version <= Self::MAX_VERSION);
        Self((version << Self::VERSION_SHIFT) | (is_token as usize * Self::TOKEN_FLAG))
    }

    /// Increments the version number of the tag and resets all other bits.
    pub const fn next(self) -> Self {
        Self(
            (self.0 | !Self::VERSION_MASK)
                .checked_add(1)
                .expect("block version number overflow"),
        )
    }

    /// Gets the `max_tag` for the [`Token`] defined by a block with this tag.
    pub const fn max_tag(self) -> Self {
        Self((self.0 & Self::VERSION_MASK) | Self::TOKEN_FLAG | Self::INVALID_FLAG)
    }

    /// Gets the version number of the tag.
    pub const fn version(&self) -> usize {
        self.0 >> Self::VERSION_SHIFT
    }

    /// Indicates whether the `TOKEN_FLAG` is set.
    pub const fn is_token(&self) -> bool {
        self.0 & Self::TOKEN_FLAG != 0
    }

    /// Indicates whether the `INVALID_FLAG` is set.
    pub const fn is_invalid(&self) -> bool {
        self.0 & Self::INVALID_FLAG != 0
    }

    /// Indicates whether the `HOLDING_LEFT_FLAG` is set.
    pub const fn is_holding_left(&self) -> bool {
        self.0 & Self::HOLDING_LEFT_FLAG != 0
    }

    /// Indicates whether the `HOLDING_RIGHT_FLAG` is set.
    pub const fn is_holding_right(&self) -> bool {
        self.0 & Self::HOLDING_RIGHT_FLAG != 0
    }

    /// Gets the current number of [`TokenBlockHolding`]s for this block.
    pub const fn num_holders(&self) -> usize {
        self.0 & Self::NUM_HOLDERS_MASK
    }
}

impl std::fmt::Debug for BlockTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockTag")
            .field("version", &self.version())
            .field("is_token", &self.is_token())
            .field("is_invalid", &self.is_invalid())
            .field("is_holding_left", &self.is_holding_left())
            .field("is_holding_right", &self.is_holding_right())
            .field("num_holders", &self.num_holders())
            .finish()
    }
}

/// Identifies a possible type of [`Block`].
///
/// This struct exists for documentation purposes only.
#[allow(dead_code)]
enum BlockType {
    /// A block which corresponds to a [`Token`].
    ///
    /// This is the only type of block which can have children. It keeps a list of its left
    /// children and a search tree for its right children. Before completing finalization, it
    /// must ensure that all of its children are finalized.
    Token,

    /// A block which indexes a subset of right children for the [`BlockType::Token`] block
    /// that is its `right_parent`.
    ///
    /// These blocks are "owned" by their right parent, and their lifecycle is inextricably
    /// linked to it. Thus, the only bits in [`BlockTag`] which are used are the version number
    /// (which remains constant).
    Branch,

    /// A block which can be inserted into the list of left children for a [`BlockType::Token`]
    /// token.
    /// 
    /// Upon being invalidated, instead of performing invalidation propogation, it will call
    /// `callback_fn(callback_data)`.
    /// 
    /// These blocks are "owned" by their left parent, and their lifecycle is inextricably
    /// linked to it. Thus, the only bits in [`BlockTag`] which are used are the version number
    /// (which remains constant).
    Callback,
}

/// A reference to a sibling block within a doubly-linked list of blocks.
///
/// This is basically a `Option<&'alloc Block>`, but it includes an extra bit of information
/// which says whether the current block is about to be removed from the list. This is required for
/// lock-free implementation of the doubly-linked list.
#[repr(transparent)]
#[derive(Clone, Copy)]
struct SiblingBlockRef<'alloc> {
    ptr: *mut Block<'alloc>,
    _marker: std::marker::PhantomData<&'alloc Block<'alloc>>,
}

impl<'alloc> HasAtomic for SiblingBlockRef<'alloc> {
    type Prim = *mut Block<'alloc>;
}

unsafe impl<'alloc> SafeTransmuteFrom<SiblingBlockRef<'alloc>> for *mut Block<'alloc> {}

impl<'alloc> SiblingBlockRef<'alloc> {
    /// A [`SiblingBlockRef`] corresponding to a value of [`None`].
    pub const NONE: Self = Self {
        ptr: std::ptr::null_mut(),
        _marker: std::marker::PhantomData,
    };

    /// A bit set on the pointer to indicate that the block is being removed from the list.
    const IS_REMOVING_BIT: usize = 0b1;

    /// Constructs a [`SiblingBlockRef`] from the given source reference and `is_removing` flag.
    pub fn new(source: Option<&'alloc Block<'alloc>>, is_removing: bool) -> Self {
        let ptr = HasAtomic::into_prim(source);
        let ptr = ptr.map_addr(|addr| addr | (Self::IS_REMOVING_BIT * usize::from(is_removing)));
        Self {
            ptr,
            _marker: std::marker::PhantomData,
        }
    }

    /// Gets the underlying `Option<&'alloc Block>` for this reference.
    pub fn source(self) -> Option<&'alloc Block<'alloc>> {
        unsafe { HasAtomic::from_prim(self.ptr.map_addr(|addr| addr & !Self::IS_REMOVING_BIT)) }
    }

    /// Gets the value of the `is_removing` flag for this reference.
    pub fn is_removing(self) -> bool {
        self.ptr.addr() & Self::IS_REMOVING_BIT != 0
    }
}

impl<'alloc> Atomic<SiblingBlockRef<'alloc>> {
    /// Sets the `is_removing` flag for this reference. Returns the underlying
    /// `Option<&'alloc Block>`.
    pub fn mark_removing(&self, order: std::sync::atomic::Ordering) -> Option<&'alloc Block> {
        // TODO: Use `fetch_or` once it is stabilized
        // https://doc.rust-lang.org/std/sync/atomic/struct.AtomicPtr.html#method.fetch_or
        let mut cur = self.0.load(Acquire);
        loop {
            if cur.addr() & SiblingBlockRef::IS_REMOVING_BIT != 0 {
                return unsafe {
                    HasAtomic::from_prim(
                        cur.map_addr(|addr| addr & !SiblingBlockRef::IS_REMOVING_BIT),
                    )
                };
            }
            let new = cur.map_addr(|addr| addr | SiblingBlockRef::IS_REMOVING_BIT);
            match self.0.compare_exchange_weak(cur, new, order, Acquire) {
                Ok(_) => return unsafe { HasAtomic::from_prim(cur) },
                Err(old) => cur = old,
            }
        }
    }
}

impl<'alloc> Block<'alloc> {
    /// Creates a new empty [`Block`].
    ///
    /// This can be used by an [`Allocator`] to create additional blocks.
    pub fn new() -> Self {
        Self {
            tag: Atomic::new(BlockTag::new(0, false)),
            range: Atomic::new(ConditionRange(0)),
            protected: UnsafeCell::new(Protected {
                first_left_child: ManuallyDrop::new(Atomic::new(None)),
            }),
            right_parent: Atomic::new(None),
            beta: Beta {
                branch: ManuallyDrop::new(BranchBeta {
                    children: [
                        Atomic::new(None),
                        Atomic::new(None),
                        Atomic::new(None),
                        Atomic::new(None),
                    ],
                }),
            },
        }
    }

    /// Assuming this is a [`Token`] block, gets the token it currently represents.
    ///
    /// This method must be used with caution, because it is very easy to unintentionally get
    /// the wrong token. Blocks may be invalidated, freed and reallocated as new tokens by another
    /// thread at any time.
    fn token(&'alloc self) -> Token<'alloc> {
        Token {
            block: self,
            max_tag: self.tag.load(Relaxed).max_tag(),
        }
    }

    /// Gets the [`Protected::first_left_child`] field of this block.
    ///
    /// # Safety
    /// The caller must ensure that this block is a [`BlockType::Token`] block for the duration
    /// that the returned reference is used.
    unsafe fn first_left_child(&self) -> &Atomic<Option<&'alloc Block<'alloc>>> {
        // SAFETY: The caller is responsible for ensuring that this is safe
        unsafe { &(*self.protected.get()).first_left_child }
    }

    /// Gets the [`Beta::callback_data`] field of this block.
    fn callback_data(&self) -> &Atomic<*mut ()> {
        // SAFETY: Regardless of what type of block this is, the `callback_data` field is always
        // populated with a valid `Atomic<*mut ()>`.
        unsafe { &self.beta.callback_data }
    }

    /// Gets the [`NonBranchBeta::right_tree`] field of this block.
    fn right_tree(&self) -> &Atomic<Option<&'alloc Block<'alloc>>, *mut Block<'alloc>> {
        // SAFETY: Regardless of what type of block this is, the `right_tree` field is always
        // populated with a valid `Atomic<*mut Block<'alloc>>`.
        unsafe { self.beta.non_branch.right_tree.cast() }
    }

    /// Gets the [`NonBranchBeta::left_parent`] field of this block.
    fn left_parent(&self) -> &Atomic<Option<&'alloc Block<'alloc>>> {
        // SAFETY: Regardless of what type of block this is, the `left_parent` field is
        // always occupied by a valid `Atomic<Option<&'alloc Block<'alloc>>>`.
        unsafe { &self.beta.non_branch.left_parent }
    }

    /// Gets the [`NonBranchBeta::next_left_sibling`] field of this block.
    fn next_left_sibling(&self) -> &Atomic<SiblingBlockRef<'alloc>> {
        // SAFETY: Regardless of what type of block this is, the `next_left_sibling` field is
        // always occupied by a valid `Atomic<SiblingBlockRef<'alloc>>`.
        unsafe { &self.beta.non_branch.next_left_sibling }
    }

    /// Gets the [`NonBranchBeta::prev_left_sibling`] field of this block.
    fn prev_left_sibling(&self) -> &Atomic<Option<&'alloc Block<'alloc>>> {
        // SAFETY: Regardless of what type of block this is, the `prev_left_sibling` field is
        // always occupied by a valid `Atomic<Option<&'alloc Block<'alloc>>>`.
        unsafe { &self.beta.non_branch.prev_left_sibling }
    }

    /// Gets an entry into [`BranchBeta::children`] for this block.
    fn child(&self, slot: usize) -> &Atomic<Option<&'alloc Block<'alloc>>, *mut Block<'alloc>> {
        // SAFETY: Regardless of what type of block this is, all child slots are always occupied by
        // a valid `Atomic<Option<&'alloc Block<'alloc>>, *mut Block<'alloc>>`
        unsafe { self.beta.branch.children[slot].cast() }
    }

    /// In debug builds, verifies that a block is in a "clean" state, as should be the case when
    /// obtained from or returned to an [`Allocator`].
    pub(crate) fn verify_clean(&self) {
        debug_assert_eq!(self.tag.load(Relaxed).0 & !BlockTag::VERSION_MASK, 0);
        debug_assert_eq!(
            unsafe { self.first_left_child().0.load(Relaxed) },
            std::ptr::null_mut()
        );
        debug_assert_eq!(self.right_parent.0.load(Relaxed), std::ptr::null_mut());
        for i in 0..BRANCH_SIZE {
            debug_assert_eq!(self.child(i).0.load(Relaxed), std::ptr::null_mut());
        }
    }
}

// SAFETY: The only field which is potentially not `Sync` is the `protected` field, which can only
// be accessed in the module. All of these accesses have their own safety documentation.
unsafe impl Sync for Block<'_> {}

impl Default for Block<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Block<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Block` is opaque to the user, so we don't have to expose any of its fields.
        f.debug_struct("Block").finish_non_exhaustive()
    }
}

/// A special [`Block`] used to implement the "always" and "never" tokens.
#[cfg(not(loom))]
static ALWAYS_NEVER: Block<'static> = Block {
    tag: Atomic::from_prim(BlockTag::new(1, true).0),
    // Use a high range so that this block will always be the "right parent" of a combined
    // token. The right parent is checked for validity before the left parent.
    range: Atomic::from_prim(ConditionRange::single(ConditionId::MAX).0),
    protected: UnsafeCell::new(Protected {
        first_left_child: ManuallyDrop::new(Atomic::null()),
    }),
    right_parent: Atomic::null(),
    beta: Beta {
        non_branch: std::mem::ManuallyDrop::new(NonBranchBeta {
            right_tree: Atomic::null(),
            left_parent: Atomic::null(),
            next_left_sibling: Atomic::null(),
            prev_left_sibling: Atomic::null(),
        }),
    },
};

#[cfg(loom)]
loom::lazy_static! {
    static ref ALWAYS_NEVER: Block<'static> = Block {
        tag: Atomic::new(BlockTag::new(1, true)),
        // Use a high range so that this block will always be the "right parent" of a combined
        // token. The right parent is checked for validity before the left parent.
        range: Atomic::new(ConditionRange::single(ConditionId::MAX)),
        protected: UnsafeCell::new(Protected {
            first_left_child: ManuallyDrop::new(Atomic::new(None)),
        }),
        right_parent: Atomic::new(None),
        beta: Beta {
            non_branch: std::mem::ManuallyDrop::new(NonBranchBeta {
                right_tree: Atomic::new(None),
                left_parent: Atomic::new(None),
                next_left_sibling: Atomic::new(SiblingBlockRef::NONE),
                prev_left_sibling: Atomic::new(None),
            }),
        },
    };
}

/// A unique identifier for a [`Condition`] within a certain context.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy, Hash)]
pub struct ConditionId(usize);

impl ConditionId {
    /// The maximum number of bits a [`ConditionId`] can occupy.
    const NUM_BITS: u32 = usize::BITS - usize::BITS.ilog2();

    /// The maximum index for a [`ConditionId`].
    pub const MAX_INDEX: usize = (1 << Self::NUM_BITS) - 1;

    /// The maximum [`ConditionId`].
    pub const MAX: ConditionId = Self(Self::MAX_INDEX);

    /// Constructs a [`ConditionId`] from the given index.
    ///
    /// The index must not be greater than  [`Self::MAX_INDEX`], or this will panic.
    pub const fn new(index: usize) -> Self {
        assert!(index <= Self::MAX_INDEX);
        Self(index)
    }

    /// Gets the index of this [`ConditionId`].
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Describes the approximate "range" of [`ConditionId`]s a [`Token`] may be sensitive to.
#[derive(PartialEq, Eq, Clone, Copy)]
struct ConditionRange(usize);

impl HasAtomic for ConditionRange {
    type Prim = usize;
}

unsafe impl SafeTransmuteFrom<ConditionRange> for usize {}

impl ConditionRange {
    /// Constructs a [`ConditionRange`] which contains only the given [`ConditionId`].
    pub const fn single(id: ConditionId) -> Self {
        Self(id.0)
    }

    /// Gets the smallest [`ConditionRange`] which fully contains both of the given ranges.
    pub fn combine(a: Self, b: Self) -> Self {
        let mask = (a.0 ^ b.0) & ConditionId::MAX_INDEX;
        let scale = usize::BITS - mask.leading_zeros();
        Self(
            (((scale as usize) << ConditionId::NUM_BITS) | (a.0 & ConditionId::MAX_INDEX))
                .max(a.0)
                .max(b.0)
                & !((1 << scale) - 1),
        )
    }

    /// Gets the "scale" of this range, i.e. the number of bits of [`ConditionId`] that vary
    /// within the range.
    pub const fn scale(self) -> u32 {
        (self.0 >> ConditionId::NUM_BITS) as u32
    }

    /// Gets the number of [`ConditionId`]s in this range.
    pub const fn size(self) -> usize {
        1 << self.scale()
    }

    /// Gets the smallest [`ConditionId`] in this range.
    pub const fn min(self) -> ConditionId {
        ConditionId(self.0 & ConditionId::MAX_INDEX)
    }

    /// Gets the [`ConditionRangeRelation`] between this range and another.
    #[inline]
    pub fn rel(self, other: Self) -> ConditionRangeRelation {
        match Ord::cmp(&self.min(), &other.min()) {
            Less => {
                let offset = other.min().0 - self.min().0;
                if offset < self.size() {
                    if offset < self.size() / 2 {
                        ConditionRangeRelation::ParentLeft
                    } else {
                        ConditionRangeRelation::ParentRight
                    }
                } else {
                    ConditionRangeRelation::DisjointLeft
                }
            }
            Equal => {
                // Since we know that the `min` component of the ranges are equal, we can compare
                // the raw values to compare the scales.
                match Ord::cmp(&self.0, &other.0) {
                    Less => ConditionRangeRelation::ChildLeft,
                    Equal => ConditionRangeRelation::Equal,
                    Greater => ConditionRangeRelation::ParentLeft,
                }
            }
            Greater => {
                let offset = self.min().0 - other.min().0;
                if offset < other.size() {
                    if offset < other.size() / 2 {
                        ConditionRangeRelation::ChildLeft
                    } else {
                        ConditionRangeRelation::ChildRight
                    }
                } else {
                    ConditionRangeRelation::DisjointRight
                }
            }
        }
    }
}

impl From<ConditionId> for ConditionRange {
    fn from(id: ConditionId) -> Self {
        Self(id.0)
    }
}

impl std::fmt::Debug for ConditionRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ConditionRange")
            .field(&(self.min().0..(self.min().0 + self.size())))
            .finish()
    }
}

/// Describes the relationship between two [`ConditionRange`]s.
#[derive(Debug)]
enum ConditionRangeRelation {
    /// All [`ConditionId`]s in the first range are less than those in the second range.
    DisjointLeft,

    /// All [`ConditionId`]s in the first range are greater than those in the second range.
    DisjointRight,

    /// All [`ConditionId`]s in the first range are in the first half of the second range.
    ChildLeft,

    /// All [`ConditionId`]s in the first range are in the second half of the second range.
    ChildRight,

    /// All [`ConditionId`]s in the second range are in the first half of the first range.
    ParentLeft,

    /// All [`ConditionId`]s in the second range are in the second half of the first range.
    ParentRight,

    /// The two ranges are equal.
    Equal,
}

impl<'alloc> Token<'alloc> {
    /// Gets the left and right parents of a token, or returns [`None`] if the token is invalid or
    /// doesn't have parents.
    ///
    /// If this returns [`Some`], then it is guranteed that, whenever this token is invalid, at
    /// least one of the returned parents is invalid.
    fn parents(&self) -> Option<(Token<'alloc>, Token<'alloc>)> {
        let left = self.block.left_parent().load(Relaxed)?.token();
        let right = self.block.right_parent.load(Relaxed)?.token();
        fence(Acquire); // Synchronizes with token invalidation
        if self.is_valid() {
            // How do we know the calls to `token()` above returned the right tokens? The parent
            // block's version number can't increase until all of its immediate children are
            // invalidated. `self` is an immediate child of both `left` and `right`, and we just
            // observed that it is valid, so we know that the version number of `left` and `right`
            // have not changed since `self` was created.
            Some((left, right))
        } else {
            None
        }
    }

    /// Calls `f` for each [`ConditionId`] that this token is sensitive to, or returns `false` if
    /// the token is invalid.
    ///
    /// This may return `false` even after `f` has been called.
    fn conditions(&self, mut f: impl FnMut(ConditionId)) -> bool {
        if self.is_always_valid() {
            return true;
        }
        return inner(*self, &mut f);
        fn inner(token: Token, f: &mut impl FnMut(ConditionId)) -> bool {
            let range = token.block.range.load(Relaxed);
            if range.scale() == 0 {
                f(range.min());
                fence(Acquire); // Synchronizes with token invalidation
                token.is_valid()
            } else if let Some((left, right)) = token.parents() {
                inner(left, f) && inner(right, f)
            } else {
                false
            }
        }
    }

    /// Like [`Token::combine`], but assumes that neither of the given tokens are always valid.
    fn combine_non_always<Alloc: Allocator<'alloc> + ?Sized>(
        alloc: &mut Alloc,
        a: Token<'alloc>,
        b: Token<'alloc>,
    ) -> Token<'alloc> {
        // These ranges may be incorrect if the associated tokens have been freed, but that
        // doesn't matter because in every branch, we will eventually check if the tokens are valid
        // and return `Token::never()` if not.
        let a_range = a.block.range.load(Relaxed);
        let b_range = b.block.range.load(Relaxed);
        match ConditionRange::rel(a_range, b_range) {
            ConditionRangeRelation::DisjointLeft => {
                Token::combine_exact(alloc, a, b, ConditionRange::combine(a_range, b_range))
            }
            ConditionRangeRelation::DisjointRight => {
                Token::combine_exact(alloc, b, a, ConditionRange::combine(a_range, b_range))
            }
            ConditionRangeRelation::ChildLeft => {
                let Some((b_left, b_right)) = b.parents() else {
                    return Token::never();
                };
                let n_b_left = Token::combine_non_always(alloc, a, b_left);
                if std::ptr::eq(n_b_left.block, b_left.block) {
                    b
                } else {
                    Token::combine_exact(alloc, n_b_left, b_right, b_range)
                }
            }
            ConditionRangeRelation::ChildRight => {
                let Some((b_left, b_right)) = b.parents() else {
                    return Token::never();
                };
                let n_b_right = Token::combine_non_always(alloc, a, b_right);
                if std::ptr::eq(n_b_right.block, b_right.block) {
                    b
                } else {
                    Token::combine_exact(alloc, b_left, n_b_right, b_range)
                }
            }
            ConditionRangeRelation::ParentLeft => {
                let Some((a_left, a_right)) = a.parents() else {
                    return Token::never();
                };
                let n_a_left = Token::combine_non_always(alloc, a_left, b);
                if std::ptr::eq(n_a_left.block, a_left.block) {
                    a
                } else {
                    Token::combine_exact(alloc, n_a_left, a_right, a_range)
                }
            }
            ConditionRangeRelation::ParentRight => {
                let Some((a_left, a_right)) = a.parents() else {
                    return Token::never();
                };
                let n_a_right = Token::combine_non_always(alloc, a_right, b);
                if std::ptr::eq(n_a_right.block, a_right.block) {
                    a
                } else {
                    Token::combine_exact(alloc, a_left, n_a_right, a_range)
                }
            }
            ConditionRangeRelation::Equal => {
                if std::ptr::eq(a.block, b.block) {
                    if a.max_tag == b.max_tag {
                        a
                    } else {
                        // The tokens have the same block, but different tags. That means at least
                        // one of them is invalid.
                        Token::never()
                    }
                } else {
                    let Some((a_left, a_right)) = a.parents() else {
                        return Token::never();
                    };
                    let Some((b_left, b_right)) = b.parents() else {
                        return Token::never();
                    };
                    let n_left = Token::combine_non_always(alloc, a_left, b_left);
                    let n_right = Token::combine_non_always(alloc, a_right, b_right);
                    Token::combine_exact(alloc, n_left, n_right, a_range)
                }
            }
        }
    }

    /// Gets or creates a [`Token`] which is dependent on *exactly* the two given tokens.
    /// The `range` of the tokens must not overlap, with all [`ConditionId`]s in the `range` of the
    /// left token being less than those in the `range` of the right token. The combined range is
    /// given by `range`.
    fn combine_exact<Alloc: Allocator<'alloc> + ?Sized>(
        alloc: &mut Alloc,
        left: Token<'alloc>,
        right: Token<'alloc>,
        range: ConditionRange,
    ) -> Token<'alloc> {
        // Perform a preliminary search for the desired token.
        let slot = match search_right_child(left.block, right) {
            Some(Ok(token)) => {
                // The search gave us a token which has the correct right parent token, and the
                // correct left parent *block*, but we still need to make sure it has the
                // correct left parent *token*. We can do this by simply checking that the
                // left parent token is still valid. Otherwise, it could be possible that
                // `left.block` was invalidated and reallocated as a new token, in which case
                // the returned token would erroneously be considered valid.
                return if left.is_valid() {
                    token
                } else {
                    Token::never()
                };
            }
            Some(Err(slot)) => slot,
            None => return Token::never(),
        };

        // Hold the parent blocks to keep them alive while we create the new token. This also
        // gives us one final check that the parent tokens are valid before we proceed.
        let Some(right) = hold(right) else {
            return Token::never();
        };
        let Some(left) = hold(left) else {
            unhold_finalize(alloc, right);
            return Token::never();
        };

        // Create new block for the token.
        let block = alloc.allocate_block();
        block.verify_clean();
        block.range.store(range, Relaxed);
        block.left_parent().store(Some(left.block), Relaxed);
        block.right_parent.store(Some(right.block), Relaxed);
        let tag = block.tag.load(Relaxed);
        debug_assert_eq!(tag.0 & !BlockTag::VERSION_MASK, 0);
        let mut n_tag = tag;
        n_tag.0 |= BlockTag::TOKEN_FLAG;
        block.tag.store(n_tag, Relaxed);

        // Try to insert the new token into the tree
        if let Err(token) = unsafe { insert_right_child(alloc, left.block, &right, slot, block) } {
            // Another thread created the token before us. Clean up and return the existing token.
            block.tag.store(tag, Relaxed);
            block.left_parent().store(None, Relaxed);
            block.right_parent.store(None, Relaxed);
            alloc.free_block(block);
            unhold_finalize(alloc, left);
            unhold_finalize(alloc, right);
            return token;
        }

        // Insert the new token into the left parent's list of children.
        insert_left_child(alloc, &left, block);

        // Clean up and return the new token
        unhold_finalize(alloc, left);
        unhold_finalize(alloc, right);
        Token {
            block,
            max_tag: tag.max_tag(),
        }
    }
}

/// The number of bits that are addressed by each branch node in a "right search tree".
const BRANCH_BITS: u32 = 2;

/// The number of slots in a branch node.
const BRANCH_SIZE: usize = 1 << BRANCH_BITS;

/// The number of least-significant bits of the left parent address that are ignored when
/// searching for a token in a "right search tree".
const SKIP_BITS: u32 = std::mem::size_of::<Block>().ilog2();

/// Searches the "right search tree" of `right_parent` for a child [`Token`] with the given
/// left parent.
///
/// Returns `Some(Ok(token))` if a token was found. In which case, it is guaranteed that
/// at least one of the following is true:
///  * The returned token is the only valid token with the given `left_parent` and `right_parent`.
///  * `left_parent` was invalid at some point during this call.
///  * Both the returned token and `right_parent` are invalid.
///
/// Returns `Some(Err(slot))` if no such token was found. In which case, returns the [`TreeSlot`]
/// where the token should be inserted.
///
/// Returns `None` if `right_parent` is known to be invalid.
fn search_right_child<'alloc>(
    left_parent: &'alloc Block<'alloc>,
    right_parent: Token<'alloc>,
) -> Option<Result<Token<'alloc>, TreeSlot<'alloc>>> {
    let mut bit_depth = SKIP_BITS;
    let mut slot = right_parent.block.right_tree();
    loop {
        let block = slot.load(Acquire); // Synchronizes with block publication
        if block.is_null() {
            return Some(Err(TreeSlot {
                ptr: slot,
                bit_depth,
            }));
        }
        if !right_parent.is_valid() {
            return None;
        }

        // SAFETY: Once a slot is present in a search tree, it will continue to be a valid slot
        // until its owner (`right_parent`) is finalized. We checked that `right_parent` was
        // valid after reading the contents of the slot, so `block` must be a valid reference
        // to a block.
        let block: &'alloc Block<'alloc> = unsafe { &*block };

        // Get the tag for the block before checking `block.right_parent` so we know that we
        // got the correct tag.
        let tag = block.tag.load(Acquire); // Synchronizes with version increment

        // Check whether the block actually belongs to this tree. It's possible for an old
        // member of the tree to be freed and reallocated as a new unrelated block.
        if !std::ptr::eq(
            HasAtomic::into_prim(block.right_parent.load(Relaxed)),
            right_parent.block,
        ) {
            return Some(Err(TreeSlot {
                ptr: slot,
                bit_depth,
            }));
        }

        // Check whether this is a branch block
        if !tag.is_token() {
            slot = block.child((left_parent as *const Block as usize >> bit_depth) % BRANCH_SIZE);
            bit_depth += BRANCH_BITS;
            continue;
        };

        // Check whether this is the token block we are looking for.
        let block_left_parent = block.left_parent().load(Relaxed);
        let token = Token {
            block,
            max_tag: tag.max_tag(),
        };
        if std::ptr::eq(HasAtomic::into_prim(block_left_parent), left_parent) && token.is_valid() {
            return Some(Ok(token));
        } else {
            return Some(Err(TreeSlot {
                ptr: slot,
                bit_depth,
            }));
        }
    }
}

/// Identifies a "slot" in a "right search tree" where a [`BlockType::Token`] block could be
/// inserted.
struct TreeSlot<'alloc> {
    /// The [`Atomic`] which determines the contents of this slot.
    ///
    /// If `owner` is valid, all slots within the tree must point to a valid block.
    ptr: &'alloc Atomic<Option<&'alloc Block<'alloc>>, *mut Block<'alloc>>,

    /// The index of the first bit of the `left_parent` address that is addressed by the
    /// subtree
    bit_depth: u32,
}

/// Attempts to insert a [`Token`] block into the tree rooted at this node. `right_parent` must
/// be the owner of the tree. If there is already a token with the given `left_parent` and
/// `right_parent`, this will return the existing token.
///
/// # Safety
/// The caller must ensure that `right_parent` is the owner of tree containing `slot`. Just don't
/// mix [`TreeSlot`]s from different trees - easy!
unsafe fn insert_right_child<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
    alloc: &mut Alloc,
    left_parent: &'alloc Block<'alloc>,
    right_parent: &TokenBlockHolding<'alloc>,
    slot: TreeSlot<'alloc>,
    target: &'alloc Block<'alloc>,
) -> Result<(), Token<'alloc>> {
    let mut occupant = std::ptr::null_mut();
    let mut bit_depth = slot.bit_depth;
    let mut slot = slot.ptr;
    'replace: loop {
        // We have already decided that we can remove `occupant`, so we will try to replace it
        // with `target` now.
        let Err(n_occupant) = slot.compare_exchange_weak(occupant, Some(target), Release, Acquire)
        else {
            return Ok(());
        };
        occupant = n_occupant;
        'occupied: loop {
            if occupant.is_null() {
                continue 'replace;
            }

            // SAFETY: Once a slot is present in a search tree, it will continue to be a valid slot
            // until its owner (`right_parent`) is finalized. We are holding `right_parent`, so
            // it can't be finalized during this call. Thus, every value read from a slot must
            // be a valid reference to a block.
            let block: &'alloc Block<'alloc> = unsafe { &*occupant };

            // Get the tag for the block before checking `block.right_parent` so we know that we
            // got the correct tag.
            let tag = block.tag.load(Acquire); // Synchronizes with version increment

            // Check whether the block actually belongs to this tree. It's possible for an old
            // member of the tree to be freed and reallocated as a new unrelated block.
            let block_right_parent = block.right_parent.load(Acquire); // Synchronizes with right parent removal
            if !std::ptr::eq(HasAtomic::into_prim(block_right_parent), right_parent.block) {
                continue 'replace;
            }

            // Check whether this is a branch block
            if !tag.is_token() {
                let target_bits = (left_parent as *const Block as usize) >> bit_depth;
                let target_slot_index = target_bits % BRANCH_SIZE;
                slot = block.child(target_slot_index);
                bit_depth += BRANCH_BITS;
                occupant = std::ptr::null_mut();
                continue 'replace;
            };

            // Check whether this is the token block we are looking for.
            let block_left_parent = block.left_parent().load(Acquire);
            let token = Token {
                block,
                max_tag: tag.max_tag(),
            };
            if !token.is_valid() {
                continue 'replace;
            }
            if std::ptr::eq(HasAtomic::into_prim(block_left_parent), left_parent) {
                return Err(token);
            }

            // SAFETY: At this point, we know `block` is a valid token block with a right
            // parent. All such blocks must also have a left parent.
            let block_left_parent = unsafe { block_left_parent.unwrap_unchecked() };

            // There is a valid token block in the slot, but it's not the one we're looking
            // for. We will have to create a branch block here so we can fit the existing
            // token, and the new token.
            loop {
                let block_bits = (block_left_parent as *const Block as usize) >> bit_depth;
                let target_bits = (left_parent as *const Block as usize) >> bit_depth;
                debug_assert_ne!(block_bits, target_bits);
                let block_slot_index = block_bits % BRANCH_SIZE;
                let target_slot_index = target_bits % BRANCH_SIZE;
                let branch = alloc.allocate_block();
                branch.verify_clean();
                branch.right_parent.store(Some(right_parent.block), Relaxed);
                branch.child(target_slot_index).store(Some(target), Relaxed);
                branch.child(block_slot_index).store(Some(block), Relaxed);

                // Attempt to insert branch
                match slot.compare_exchange(occupant, Some(branch), Release, Acquire) {
                    Ok(_) => {
                        // Check if we're done
                        if block_slot_index != target_slot_index {
                            return Ok(());
                        }

                        // We need to create more branches to distinguish between the two
                        // blocks.
                        slot = branch.child(block_slot_index);
                        bit_depth += BRANCH_BITS;
                        occupant = std::ptr::null_mut();
                        continue;
                    }
                    Err(n_occupant) => {
                        // Clean up and free the branch
                        branch.right_parent.store(None, Relaxed);
                        for i in 0..BRANCH_SIZE {
                            branch.child(i).store(None, Relaxed);
                        }
                        alloc.free_block(branch);

                        // Try again
                        occupant = n_occupant;
                        continue 'occupied;
                    }
                }
            }
        }
    }
}

/// Tries to "hold" the block associated with the given token, ensuring that it is not finalized
/// until it is unheld. Returns [`None`] if the token is invalid.
fn hold(token: Token) -> Option<TokenBlockHolding> {
    let mut tag = BlockTag(token.max_tag.0 & !BlockTag::INVALID_FLAG);
    loop {
        debug_assert!(tag.is_token());
        assert!(
            tag.num_holders() < BlockTag::NUM_HOLDERS_MASK,
            "maximum number of holders exceeded"
        );
        let mut n_tag = tag;
        n_tag.0 += 1;
        match token
            .block
            .tag
            .compare_exchange_weak(tag, n_tag, AcqRel, Relaxed) // Synchronizes with token unhold
        {
            Ok(_) => return Some(TokenBlockHolding { block: token.block }),
            Err(e_tag) => {
                if e_tag < token.max_tag {
                    tag = e_tag;
                } else {
                    // The token has already been invalidated.
                    return None;
                }
            }
        }
    }
}

/// "Unholds" the block associated with a given [`TokenBlockHolding`].
///
/// If there are no remaining holders for the block preventing finalization, this will
/// return a [`TokenBlockFinalizing`]. The calling thread is then responsible for [`finalize`]ing
/// the block.
#[must_use]
fn unhold(block: TokenBlockHolding) -> Option<TokenBlockFinalizing> {
    let guard = block;
    let block = guard.block;
    std::mem::forget(guard);
    let o_tag = BlockTag(block.tag.0.fetch_sub(1, AcqRel)); // Synchronizes with token unhold
    debug_assert!(o_tag.is_token());
    debug_assert!(o_tag.num_holders() > 0);
    if o_tag.0 & (BlockTag::INVALID_FLAG | BlockTag::NUM_HOLDERS_MASK)
        == (BlockTag::INVALID_FLAG + 1)
    {
        // The block has no remaining holders, and is invalid, so we should finalize it.
        Some(TokenBlockFinalizing { block })
    } else {
        None
    }
}

/// "Unholds" the block associated with a given [`TokenBlockHolding`]. If the block has no
/// remaining holders, this will [`finalize`] it.
fn unhold_finalize<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
    alloc: &mut Alloc,
    block: TokenBlockHolding<'alloc>,
) {
    if let Some(block) = unhold(block) {
        let _ = finalize::<_, ()>(alloc, block, None, None);
    }
}

/// Invalidates a [`Token`] block, assuming we have the exclusive right to do so.
///
/// If there are no [`TokenBlockHolding`]s on the block preventing finalization, this will
/// return a [`TokenBlockFinalizing`]. The calling thread is then responsible for [`finalize`]ing
/// the block. Otherwise, this will return an `Err`.
fn invalidate_condition<'alloc, Err: FinalizeError<'alloc>>(
    block: &'alloc Block<'alloc>,
) -> Result<TokenBlockFinalizing<'alloc>, Err> {
    Err::invalidate_condition(block)
}

/// Invalidates all children of a [`BlockType::Token`], attempts to finalize the children, and
/// if successful, frees the block.
///
/// If it is not possible to finish finalization without blocking the current thread (e.g. a
/// child has a [`TokenBlockHolding`] from another thread), this will "abort" the finalization
/// and return an `Err`. Finalization will be attempted again when all holders are removed.
/// It is guranteed to succeed on the second attempt because no new children can be added to
/// an invalid block.
///
/// If the block is holding its left or right parent, and the corresponding `left_num_holders` or
/// `right_num_holders` is not `None`, then this will prefer to decrement the given counter
/// rather than the parent block's `NUM_HOLDERS` tag. This is much quicker than an atomic
/// decrement and can prevent underflow in the case where a single block has many children that
/// are holding it.
fn finalize<'alloc, Alloc: Allocator<'alloc> + ?Sized, Err: FinalizeError<'alloc>>(
    alloc: &mut Alloc,
    mut block: TokenBlockFinalizing<'alloc>,
    left_num_holders: Option<&mut usize>,
    right_num_holders: Option<&mut usize>,
) -> Result<(), Err> {
    // Set number of holders to maximum to prevent underflow. Although we are guaranteed to have
    // zero holders at this point, other threads may still set the holding flag on this token,
    // so we still need an atomic operation here.
    let tag = BlockTag(block.tag.0.fetch_add(BlockTag::NUM_HOLDERS_MASK, Relaxed));
    debug_assert!(tag.is_token());
    debug_assert!(tag.is_invalid());
    debug_assert_eq!(tag.num_holders(), 0);

    // Invalidate children
    let mut num_holders = 0;
    start_finalize(alloc, &mut num_holders, &mut block);
    assert!(num_holders <= BlockTag::NUM_HOLDERS_MASK);

    // Remove extraneous holders and check if we end up with no holders remaining.
    let sub = BlockTag::NUM_HOLDERS_MASK - num_holders;
    match FinalizeError::unhold_many(block, sub) {
        Ok(block) => {
            finish_finalize(alloc, block, left_num_holders, right_num_holders);
            Ok(())
        }
        Err(err) => Err(err),
    }
}

/// A type of error that can be produced when we aren't able to finalize a token block.
///
/// There are effectively two useful instances of this trait:
///
/// `()`: Unholds the block and ensures that the block will be finalized at a later point once
/// all holders are removed.
///
/// [`TokenBlockHolding`]: Keeps a holder on the block, allowing a callback to be installed for
/// notification of when finalization completes.
trait FinalizeError<'alloc>: Sized {
    /// Removes `amount` holders from the given block.
    ///
    /// If no holders remain, this will return an [`Ok`], otherwise this will return an [`Err`]
    /// of this type.
    fn unhold_many(
        block: TokenBlockFinalizing<'alloc>,
        amount: usize,
    ) -> Result<TokenBlockFinalizing<'alloc>, Self>;

    /// Invalidates a [`Token`] block, assuming we have the exclusive right to do so.
    ///
    /// If no holders remain, this will return an [`Ok`], otherwise this will return an [`Err`]
    /// of this type.
    fn invalidate_condition(
        block: &'alloc Block<'alloc>,
    ) -> Result<TokenBlockFinalizing<'alloc>, Self>;
}

impl<'alloc> FinalizeError<'alloc> for () {
    fn unhold_many(
        block: TokenBlockFinalizing<'alloc>,
        amount: usize,
    ) -> Result<TokenBlockFinalizing<'alloc>, ()> {
        let o_tag = BlockTag(block.tag.0.fetch_sub(amount, AcqRel)); // Synchronizes with token unhold
        debug_assert!(
            o_tag.num_holders() >= amount,
            "attempted to unhold {} holders, but only {} were present",
            amount,
            o_tag.num_holders()
        );
        if o_tag.num_holders() == amount {
            Ok(block)
        } else {
            std::mem::forget(block);
            Err(())
        }
    }

    fn invalidate_condition(
        block: &'alloc Block<'alloc>,
    ) -> Result<TokenBlockFinalizing<'alloc>, Self> {
        let o_tag = BlockTag(block.tag.0.fetch_or(BlockTag::INVALID_FLAG, Acquire)); // Synchronizes with token unhold
        debug_assert!(o_tag.is_token());
        debug_assert!(!o_tag.is_invalid());
        if o_tag.num_holders() == 0 {
            Ok(TokenBlockFinalizing { block })
        } else {
            Err(())
        }
    }
}

impl<'alloc> FinalizeError<'alloc> for TokenBlockHolding<'alloc> {
    fn unhold_many(
        block: TokenBlockFinalizing<'alloc>,
        amount: usize,
    ) -> Result<TokenBlockFinalizing<'alloc>, TokenBlockHolding<'alloc>> {
        debug_assert!(amount > 0);

        // Try to remove all holders from the block. If this is not possible, leave one extra
        // holder to be returned as an error.
        let mut tag = block.tag.load(Relaxed);
        loop {
            let mut n_tag = tag;
            debug_assert!(tag.num_holders() >= amount);
            if tag.num_holders() == amount {
                n_tag.0 &= !BlockTag::NUM_HOLDERS_MASK;

                // Since there are no holders remaining, we don't even need to use atomic
                // operations here.
                block.tag.store(n_tag, Relaxed);
                return Ok(block);
            } else if amount > 1 {
                n_tag.0 -= amount - 1;
                match block.tag.compare_exchange_weak(tag, n_tag, AcqRel, Acquire) {
                    // Synchronizes with token unhold
                    Ok(_) => break,
                    Err(n_tag) => {
                        debug_assert_eq!(
                            tag.0 & !BlockTag::NUM_HOLDERS_MASK,
                            n_tag.0 & !BlockTag::NUM_HOLDERS_MASK
                        );
                        tag = n_tag;
                        continue;
                    }
                }
            } else {
                break;
            }
        }

        // We left an extra holder on the block, so now we can return a `TokenBlockHolding`.
        let guard = block;
        let block = guard.block;
        std::mem::forget(guard);
        Err(TokenBlockHolding { block })
    }

    fn invalidate_condition(
        block: &'alloc Block<'alloc>,
    ) -> Result<TokenBlockFinalizing<'alloc>, Self> {
        let mut tag = block.tag.load(Relaxed);
        loop {
            debug_assert!(tag.is_token());
            debug_assert!(!tag.is_invalid());
            let mut n_tag = tag;
            n_tag.0 |= BlockTag::INVALID_FLAG; // Set invalid flag
            if tag.num_holders() == 0 {
                match block.tag.compare_exchange_weak(tag, n_tag, AcqRel, Acquire) {
                    // Synchronizes with token unhold
                    Ok(_) => return Ok(TokenBlockFinalizing { block }),
                    Err(n_tag) => {
                        debug_assert_eq!(
                            tag.0 & !BlockTag::NUM_HOLDERS_MASK,
                            n_tag.0 & !BlockTag::NUM_HOLDERS_MASK
                        );
                        tag = n_tag;
                        continue;
                    }
                }
            } else {
                assert!(
                    tag.num_holders() < BlockTag::NUM_HOLDERS_MASK,
                    "maximum number of holders exceeded"
                );
                n_tag.0 += 1;
                match block.tag.compare_exchange_weak(tag, n_tag, AcqRel, Acquire) {
                    // Synchronizes with token unhold
                    Ok(_) => return Err(TokenBlockHolding { block }),
                    Err(n_tag) => {
                        debug_assert_eq!(
                            tag.0 & !BlockTag::NUM_HOLDERS_MASK,
                            n_tag.0 & !BlockTag::NUM_HOLDERS_MASK
                        );
                        tag = n_tag;
                        continue;
                    }
                }
            }
        }
    }
}

/// Invalidates all children of a [`BlockType::Token`] block.
///
/// This will attempt to finalize children that have no [`TokenBlockHolding`]s. For any child
/// that can not be completely finalized, this will set its [`BlockTag::HOLDING_LEFT_FLAG`]
/// or [`BlockTag::HOLDING_RIGHT_FLAG`] (depending on what kind of child it is) and increment
/// `num_holders`. Thus, at the end of the call, `num_holders` will be the total number of children
/// that are preventing finalization from completing.
///
/// Note that it is possible for children to be finalized while this is still running, in which
/// case they will decrement `NUM_HOLDERS` in `block.tag`. To prevent underflow, `NUM_HOLDERS`
/// should be set to its maximum value before calling this function, and then corrected afterwards
/// using `num_holders`.
///
/// It is guranteed that after this call, there will be no children in either the left child
/// list or the right search tree.
fn start_finalize<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
    alloc: &mut Alloc,
    num_holders: &mut usize,
    block: &mut TokenBlockFinalizing<'alloc>,
) {
    // Invalidate right children
    let right_tree = block.block.right_tree();
    invalidate_right(alloc, num_holders, block, right_tree);
    fn invalidate_right<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
        alloc: &mut Alloc,
        num_holders: &mut usize,
        right_parent: &mut TokenBlockFinalizing<'alloc>,
        slot: &Atomic<Option<&'alloc Block<'alloc>>, *mut Block<'alloc>>,
    ) {
        let block = slot.load(Acquire); // Synchronizes with block publication
        if block.is_null() {
            return;
        }

        // SAFETY: No other thread is allowed to modify the right tree, so we can guarantee that
        // all slots in the tree are valid references to a `Block`.
        let block: &'alloc Block<'alloc> = unsafe { &*block };
        slot.store(None, Relaxed);

        // Get the tag for the block before checking `block.right_parent` so we know that we
        // got the correct tag.
        let tag = block.tag.load(Acquire); // Synchronizes with version increment

        // Check whether the block actually belongs to this tree. It's possible for an old
        // member of the tree to be freed and reallocated as a new unrelated block.
        let block_right_parent = block.right_parent.load(Acquire); // Synchronizes with right parent removal
        if !std::ptr::eq(HasAtomic::into_prim(block_right_parent), right_parent.block) {
            return;
        }

        // If this is a branch block, invalidate its children recursively, then free it.
        if !tag.is_token() {
            for i in 0..BRANCH_SIZE {
                invalidate_right(alloc, num_holders, right_parent, block.child(i));
            }
            block.right_parent.store(None, Relaxed);
            alloc.free_block(block);
            return;
        }

        // Try to invalidate right child token
        if let Ok(block) = invalidate_token(BlockTag::HOLDING_RIGHT_FLAG, block, tag) {
            *num_holders += 1;
            if let Some(block) = block {
                let _ = finalize::<_, ()>(alloc, block, None, Some(num_holders));
            }
        }
    }

    // Invalidate left children
    let mut opt_first_child = block.first_left_child().load(Acquire); // Synchronizes with left child removal
    'process_child: while let Some(first_child) = opt_first_child {
        // Verify that this is still a left child of `block`.
        let tag = first_child.tag.load(Acquire); // Synchronizes with version increment
        let first_child_parent = first_child.left_parent().load(Relaxed);
        if !std::ptr::eq(HasAtomic::into_prim(first_child_parent), block.block) {
            // The child was very recently finalized by another thread and should have removed
            // itself from the list.
            fence(Acquire); // Synchronizes with left parent removal
            opt_first_child = block.first_left_child().load(Acquire); // Synchronizes with left child removal
            debug_assert!(!std::ptr::eq(
                HasAtomic::into_prim(opt_first_child),
                first_child
            ));
            continue 'process_child;
        }

        // Is this a callback block?
        if !tag.is_token() {
            // Read callback information and free the block
            let callback_fn = unsafe {
                let protected = &mut *first_child.protected.get();
                let callback_fn = protected.callback_fn;
                protected.first_left_child = ManuallyDrop::new(Atomic::new(None));
                callback_fn
            };
            let callback_data = first_child.callback_data().load(Relaxed);
            first_child
                .callback_data()
                .store(std::ptr::null_mut(), Relaxed);
            opt_first_child = force_remove_left_child(&*block, first_child);
            first_child.left_parent().store(None, Relaxed);
            first_child
                .next_left_sibling()
                .store(SiblingBlockRef::NONE, Relaxed);
            first_child.prev_left_sibling().store(None, Relaxed);
            alloc.free_block(first_child);

            // Call the callback
            unsafe { callback_fn(callback_data) };
            continue 'process_child;
        }

        // Try to invalidate left child token
        'finalize_child: {
            match invalidate_token(BlockTag::HOLDING_LEFT_FLAG, first_child, tag) {
                Ok(Some(first_child)) => {
                    *num_holders += 1;
                    if finalize::<_, ()>(alloc, first_child, Some(num_holders), None).is_err() {
                        break 'finalize_child;
                    }
                }
                Ok(None) => {
                    *num_holders += 1;
                    break 'finalize_child;
                }
                Err(_) => {
                    // The child was very recently finalized on another thread.
                }
            }

            // Finalization was recently completed. The child should have removed itself from the
            // left children list.
            let old_opt_first_child = opt_first_child;
            opt_first_child = block.first_left_child().load(Acquire); // Synchronizes with left child removal
            debug_assert!(!std::ptr::eq(
                HasAtomic::into_prim(opt_first_child),
                HasAtomic::into_prim(old_opt_first_child)
            ));
            continue 'process_child;
        }

        // We weren't able to finalize the child immediately. It will be finalized later
        // (possibly by another thread). Just get it out of the way for now. It has been
        // invalidated so there is no need to keep it in the list.
        opt_first_child = force_remove_left_child(&*block, first_child);
    }
}

/// Completes [`finalize`]ing a block after it is known that all of its children have been
/// finalized.
fn finish_finalize<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
    alloc: &mut Alloc,
    mut block: TokenBlockFinalizing<'alloc>,
    left_num_holders: Option<&mut usize>,
    right_num_holders: Option<&mut usize>,
) {
    // Clean up links to left parent
    let left_parent = block.left_parent().load(Relaxed);
    if let Some(left_parent) = left_parent {
        remove_left_child(left_parent, &mut block);
        block // Synchronizes with left parent removal
            .left_parent()
            .store(None, Release);
        block.prev_left_sibling().store(None, Relaxed);
        block // Synchronizes with left child removal
            .next_left_sibling()
            .store(SiblingBlockRef::NONE, Release);
    }

    // Clean up links to right parent
    let right_parent = block.right_parent.load(Relaxed);
    block.right_parent.store(None, Release); // Synchronizes with right parent removal

    // Increment version number
    let tag = block.tag.load(Relaxed);
    let tag = block.tag.swap(tag.next(), Release); // Synchronizes with version increment
    debug_assert!(tag.is_token());
    debug_assert!(tag.is_invalid());
    debug_assert_eq!(tag.num_holders(), 0);

    // Free the block
    let guard = block;
    let block = guard.block;
    std::mem::forget(guard);
    alloc.free_block(block);

    // Unhold parent blocks
    if tag.is_holding_left() {
        // SAFETY: The `HOLDING_LEFT_FLAG` can only be set by the left parent, so it must exist.
        let left_parent = unsafe { left_parent.unwrap_unchecked() };
        if let Some(left_num_holders) = left_num_holders {
            *left_num_holders -= 1;
        } else {
            unhold_finalize(alloc, TokenBlockHolding { block: left_parent });
        }
    }
    if tag.is_holding_right() {
        // SAFETY: The `HOLDING_RIGHT_FLAG` can only be set by the right parent, so it must exist.
        let right_parent = unsafe { right_parent.unwrap_unchecked() };
        if let Some(right_num_holders) = right_num_holders {
            *right_num_holders -= 1;
        } else {
            unhold_finalize(
                alloc,
                TokenBlockHolding {
                    block: right_parent,
                },
            );
        }
    }
}

/// Attempts to invalidate a [`Token`], returning `Err` if it has already been finalized.
///
/// If the block has not yet been finalized, this will set the `holding_flag` on its tag. If
/// there are no [`TokenBlockHolding`]s on the block preventing finalization, this will return
/// a [`TokenBlockFinalizing`].
fn invalidate_token<'alloc>(
    holding_flag: usize,
    block: &'alloc Block<'alloc>,
    mut tag: BlockTag,
) -> Result<Option<TokenBlockFinalizing<'alloc>>, ()> {
    debug_assert_eq!(
        holding_flag & !(BlockTag::HOLDING_LEFT_FLAG | BlockTag::HOLDING_RIGHT_FLAG),
        0
    );

    // We don't want to invalidate the token if the block has already been reallocated as a new
    // token, so we have to use `compare_exchange` instead of `fetch_or`.
    loop {
        let mut n_tag = tag;
        n_tag.0 |= BlockTag::INVALID_FLAG;
        n_tag.0 |= holding_flag;
        match block
            .tag
            .compare_exchange_weak(tag, n_tag, AcqRel, Acquire) // Synchronizes with token unhold and version increment
        {
            Ok(tag) => {
                // Are we responsible for finalizing the block?
                return Ok(if !tag.is_invalid() && tag.num_holders() == 0 {
                    Some(TokenBlockFinalizing {
                        block,
                    })
                } else {
                    None
                });
            }
            Err(e_tag) => {
                if e_tag.0 & BlockTag::VERSION_MASK <= tag.0 & BlockTag::VERSION_MASK {
                    debug_assert!(e_tag.is_token());
                    tag = e_tag;
                    continue;
                } else {
                    // The token has already been finalized.
                    return Err(());
                }
            }
        }
    }
}

/// Adds a [`BlockType::Callback`] block to the list of children of a token [`Block`].
/// This callback will be called when the token is finalized.
unsafe fn install_callback<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
    alloc: &mut Alloc,
    block: &TokenBlockHolding<'alloc>,
    callback_fn: unsafe fn(*mut ()),
    callback_data: *mut (),
) {
    let callback = alloc.allocate_block();
    callback.verify_clean();
    unsafe { (*callback.protected.get()).callback_fn = callback_fn }
    callback.callback_data().store(callback_data, Relaxed);
    callback.left_parent().store(Some(block.block), Relaxed);
    insert_left_child(alloc, block, callback);
}

/// Inserts a token [`Block`] into the list of children of its left parent.
fn insert_left_child<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
    alloc: &mut Alloc,
    left_parent: &TokenBlockHolding<'alloc>,
    block: &'alloc Block<'alloc>,
) {
    let first_child_slot = &left_parent.first_left_child();
    let mut opt_first_child = first_child_slot.load(Acquire); // Synchronizes with left child removal
    loop {
        block
            .next_left_sibling()
            .store(SiblingBlockRef::new(opt_first_child, false), Relaxed);
        if let Some(first_child) = opt_first_child {
            // Check whether `first_child` still belongs to `left_parent`.
            let tag = first_child.tag.load(Acquire); // Synchronizes with version increment
            let first_child_parent = first_child.left_parent().load(Relaxed);
            if !std::ptr::eq(HasAtomic::into_prim(first_child_parent), left_parent.block) {
                // The child was very recently finalized by another thread and should have removed
                // itself from the list.
                fence(Acquire); // Synchronizes with left parent removal
                opt_first_child = first_child_slot.load(Acquire); // Synchronizes with left child removal
                debug_assert!(!std::ptr::eq(
                    HasAtomic::into_prim(opt_first_child),
                    first_child
                ));
                continue;
            }

            // Is this a callback block?
            if !tag.is_token() {
                // Callback blocks won't be removed from the list until `left_parent` is
                // being finalized, and that can't happen while we're holding it.
                match first_child_slot.compare_exchange_weak(
                    opt_first_child,
                    Some(block),
                    AcqRel, // Synchronizes with left child removal
                    Acquire,
                ) {
                    Ok(_) => {
                        first_child.prev_left_sibling().store(Some(block), Relaxed);
                        break;
                    }
                    Err(n_opt_first_child) => {
                        opt_first_child = n_opt_first_child;
                        continue;
                    }
                }
            }

            // Before inserting a new child at the front of the list, we will hold the current
            // first child. This greatly simplifies the logic for lock-free removal of an
            // arbitrary child token in the list because we won't have to worry about
            // `prev_left_sibling` being out of date.
            let token = Token {
                block: first_child,
                max_tag: tag.max_tag(),
            };
            if let Some(first_child) = hold(token) {
                match first_child_slot.compare_exchange_weak(
                    opt_first_child,
                    Some(block),
                    AcqRel, // Synchronizes with left child removal
                    Acquire,
                ) {
                    Ok(_) => {
                        first_child.prev_left_sibling().store(Some(block), Relaxed);
                        unhold_finalize(alloc, first_child);
                        break;
                    }
                    Err(n_opt_first_child) => {
                        unhold_finalize(alloc, first_child);
                        opt_first_child = n_opt_first_child;
                        continue;
                    }
                }
            } else {
                // Remove invalid child from list
                opt_first_child = force_remove_left_child(left_parent, first_child);
                continue;
            }
        } else {
            // Try to insert block as first child
            match first_child_slot.compare_exchange_weak(
                None,
                Some(block),
                AcqRel, // Synchronizes with left child removal
                Acquire,
            ) {
                Ok(_) => break,
                Err(n_opt_next_sibling) => {
                    opt_first_child = n_opt_next_sibling;
                    continue;
                }
            }
        }
    }
}

/// Removes a [`Token`] block from the list of children of its left parent.
fn remove_left_child<'alloc>(
    left_parent: &'alloc Block<'alloc>,
    block: &mut TokenBlockFinalizing<'alloc>,
) {
    // First, we mark `next_left_sibling` as removing so that if `to_sibling` is itself being
    // removed, it will not try to modify our `next_left_sibling`, which we will no longer read.
    let to_sibling = block.block.next_left_sibling().mark_removing(AcqRel); // Synchronizes with left child removal

    // Ensure that `block` is no longer in the list of children.
    let between_siblings = BlockList {
        head: block.block,
        tail: None,
    };
    loop {
        let prev_sibling = block.prev_left_sibling().load(Relaxed);
        if update(
            left_parent,
            prev_sibling,
            &between_siblings,
            block.block,
            to_sibling,
        ) {
            return;
        }
    }

    /// Attempts to update a "next"/"first" reference from either `opt_from_sibling`, some
    /// sibling preceding it, or the `first_left_child` of the list, to point to `to_sibling`.
    /// If successful, this will return `true`.
    ///
    /// If it is discovered that `before_to_sibling` has already been removed from the list, this
    /// will also return `true`.
    ///
    /// Prior to returning `true`, this will update the `prev_left_sibling` of `to_sibling` to
    /// ensure that `before_to_sibling` is not reachable by traversing the list backwards.
    ///
    /// `between_siblings` should be a list of all siblings after `opt_from_sibling` and before
    /// `to_sibling`. Note that it should include `before_to_sibling`.
    ///
    /// If it is discovered there were changes made to the list by another thread while this
    /// was executing, this will return `false`. The caller should retry to entire operation.
    fn update<'alloc>(
        left_parent: &'alloc Block<'alloc>,
        opt_from_sibling: Option<&'alloc Block<'alloc>>,
        between_siblings: &BlockList<'_, 'alloc>,
        before_to_sibling: &'alloc Block<'alloc>,
        to_sibling: Option<&'alloc Block<'alloc>>,
    ) -> bool {
        if let Some(from_sibling) = opt_from_sibling {
            match from_sibling.next_left_sibling().compare_exchange_weak(
                SiblingBlockRef::new(Some(before_to_sibling), false),
                SiblingBlockRef::new(to_sibling, false),
                AcqRel, // Synchronizes with left child removal
                Acquire,
            ) {
                Ok(_) => (),
                Err(e_next_sibling) => {
                    // Ensure that `from_sibling` is still in the list of children.
                    let from_sibling_parent = from_sibling.left_parent().load(Acquire); // Synchronizes with left parent removal
                    if !std::ptr::eq(HasAtomic::into_prim(from_sibling_parent), left_parent) {
                        return false;
                    }

                    // If `from_sibling` is currently being removed, help it out by looking
                    // further back in the list.
                    if e_next_sibling.is_removing() {
                        let prev_from_sibling = from_sibling.prev_left_sibling().load(Relaxed);
                        let between_siblings = BlockList {
                            head: from_sibling,
                            tail: Some(between_siblings),
                        };
                        return update(
                            left_parent,
                            prev_from_sibling,
                            &between_siblings,
                            before_to_sibling,
                            to_sibling,
                        );
                    } else if let Some(e_next_sibling) = e_next_sibling.source() {
                        if contains(between_siblings, e_next_sibling) {
                            if from_sibling
                                .next_left_sibling()
                                .compare_exchange_weak(
                                    SiblingBlockRef::new(Some(e_next_sibling), false),
                                    SiblingBlockRef::new(to_sibling, false),
                                    AcqRel, // Synchronizes with left child removal
                                    Acquire,
                                )
                                .is_err()
                            {
                                return false;
                            }
                        } else {
                            // Since `from_sibling.next_left_sibling` points to a block that is
                            // not between `from_sibling` and `to_sibling`, we know that
                            // `before_to_sibling` has already been removed from the list.
                        }
                    } else {
                        // All siblings after `from_sibling` have been removed from the list,
                        // so `before_to_sibling` is not in the list.
                    }
                }
            }
        } else {
            let first_left_child = unsafe { left_parent.first_left_child() };
            match first_left_child.compare_exchange_weak(
                Some(before_to_sibling),
                to_sibling,
                AcqRel, // Synchronizes with left child removal
                Acquire,
            ) {
                Ok(_) => (),
                Err(None) => {
                    // The list is empty, so `before_to_sibling` is not in the list.
                }
                Err(Some(e_first_sibling)) => {
                    if contains(between_siblings, e_first_sibling) {
                        if first_left_child
                            .compare_exchange_weak(
                                Some(e_first_sibling),
                                to_sibling,
                                AcqRel, // Synchronizes with left child removal
                                Acquire,
                            )
                            .is_err()
                        {
                            return false;
                        }
                    } else {
                        // Since `first_left_child` points to a block that is not before
                        // `to_sibling`, we know that `before_to_sibling` has already been
                        // removed from the list.
                    }
                }
            }
        }

        // Before we return `true`, we need to make sure there is no way to reach
        // `before_to_sibling` by traversing the list backwards (through `prev_left_sibling`) from
        // a live child block. The only such block for which this is possible now is
        // `to_sibling`, so we will try traversing from it to see if we reach `before_to_sibling`.
        fix_prev(opt_from_sibling, before_to_sibling, to_sibling);
        true
    }

    /// A linked list of [`Block`]s which can be stored on the stack while making recursive calls.
    struct BlockList<'stack, 'alloc> {
        head: &'alloc Block<'alloc>,
        tail: Option<&'stack BlockList<'stack, 'alloc>>,
    }

    /// Determines whether `list` contains the given block.
    fn contains<'alloc>(mut list: &BlockList<'_, 'alloc>, block: &'alloc Block<'alloc>) -> bool {
        loop {
            if std::ptr::eq(list.head, block) {
                return true;
            }
            if let Some(tail) = list.tail {
                list = tail;
            } else {
                return false;
            }
        }
    }
}

/// Removes a block from the beginning of the list of children of its left parent.
/// Unlike [`remove_left_child`], this does not require a [`TokenBlockFinalizing`].
///
/// Returns the next child in the list, or [`None`] if there are no more children.
fn force_remove_left_child<'alloc>(
    left_parent: &TokenBlockHolding<'alloc>,
    block: &'alloc Block<'alloc>,
) -> Option<&'alloc Block<'alloc>> {
    let mut next_sibling = block.next_left_sibling().load(Acquire); // Synchronizes with left child removal
    loop {
        let block_parent = block.left_parent().load(Acquire); // Synchronizes with left parent removal
        if !std::ptr::eq(HasAtomic::into_prim(block_parent), left_parent.block) {
            // `block` has already removed itself from the list of children. Now the first child
            // should be some other block.
            let first_child = left_parent.first_left_child().load(Relaxed);
            debug_assert!(!std::ptr::eq(HasAtomic::into_prim(first_child), block));
            return first_child;
        }

        // Mark `block` as removing
        if !next_sibling.is_removing() {
            if let Err(n_next_sibling) = block.next_left_sibling().compare_exchange_weak(
                next_sibling,
                SiblingBlockRef::new(next_sibling.source(), true),
                AcqRel, // Synchronizes with left child removal
                Acquire,
            ) {
                next_sibling = n_next_sibling;
                continue;
            }
        }
        let next_sibling = next_sibling.source();

        // Attempt to remove `block` from the list of children.
        match left_parent.first_left_child().compare_exchange_weak(
            Some(block),
            next_sibling,
            AcqRel, // Synchronizes with left child removal
            Acquire,
        ) {
            Ok(_) => {
                fix_prev(None, block, next_sibling);
                return next_sibling;
            }
            Err(n_block) => {
                if std::ptr::eq(HasAtomic::into_prim(n_block), block) {
                    // Retry due to spurious failure
                    continue;
                } else {
                    fix_prev(None, block, next_sibling);
                    return n_block;
                }
            }
        }
    }
}

/// Checks whether `before_to_sibling` is reachable from `to_sibling` by traversing the child
/// list backwards through `prev_left_sibling`. If so, it will update `to_sibling.prev_left_sibling`
/// to point to `opt_from_sibling`.
///
/// This assumes that `opt_from_sibling`, `before_to_sibling`, and `to_sibling` are in order
/// in the child list. It also assumes that there are no valid blocks from
/// `before_to_sibling` (inclusive) to `to_sibling` (exclusive).
fn fix_prev<'alloc>(
    opt_from_sibling: Option<&'alloc Block<'alloc>>,
    before_to_sibling: &'alloc Block<'alloc>,
    to_sibling: Option<&'alloc Block<'alloc>>,
) {
    let Some(to_sibling) = to_sibling else {
        return;
    };
    let mut prev_to_sibling = before_to_sibling;
    'retry: loop {
        match to_sibling.prev_left_sibling().compare_exchange_weak(
            Some(prev_to_sibling),
            opt_from_sibling,
            Relaxed,
            Relaxed,
        ) {
            Ok(_) | Err(None) => return,
            Err(Some(n_prev_to_sibling)) => {
                // Check if `before_to_sibling` is reachable from `n_prev_to_sibling`.
                prev_to_sibling = n_prev_to_sibling;
                let mut test_sibling = n_prev_to_sibling;
                loop {
                    if std::ptr::eq(test_sibling, before_to_sibling) {
                        continue 'retry;
                    } else if !test_sibling.tag.load(Relaxed).is_invalid() {
                        // There are no valid blocks between `before_to_sibling` and `to_sibling`,
                        // so if `test_sibling` is valid, it must already be before
                        // `before_to_sibling`.
                        return;
                    } else if let Some(n_test_sibling) =
                        test_sibling.prev_left_sibling().load(Relaxed)
                    {
                        test_sibling = n_test_sibling;
                    } else {
                        break;
                    }
                }
            }
        }
    }
}

/// A wrapper over a `&'alloc Block<'alloc>` for a [`BlockType::Token`] [`Block`] that is being
/// "held".
///
/// Blocks may not be finalized until all of their holders are removed. The last holder of an
/// invalidated block is responsible for finalizing it.
#[repr(transparent)]
struct TokenBlockHolding<'alloc> {
    block: &'alloc Block<'alloc>,
}

impl<'alloc> TokenBlockHolding<'alloc> {
    /// Gets the [`Protected::first_left_child`] field of this block.
    fn first_left_child(&self) -> &Atomic<Option<&'alloc Block<'alloc>>> {
        // SAFETY: `block` must be a token block while we are holding a `TokenBlockHolding`.
        unsafe { self.block.first_left_child() }
    }
}

impl<'alloc> std::ops::Deref for TokenBlockHolding<'alloc> {
    type Target = Block<'alloc>;
    fn deref(&self) -> &Self::Target {
        self.block
    }
}

impl std::ops::Drop for TokenBlockHolding<'_> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        if !std::thread::panicking() {
            panic!("block was not unheld");
        }
    }
}

/// A wrapper over a `&'alloc Block<'alloc>` for a [`BlockType::Token`] [`Block`] that the owner
/// is responsible for finalizing.
///
/// This is an exclusive status that permits additional operations necessary for finalization.
#[repr(transparent)]
struct TokenBlockFinalizing<'alloc> {
    block: &'alloc Block<'alloc>,
}

impl<'alloc> std::ops::Deref for TokenBlockFinalizing<'alloc> {
    type Target = TokenBlockHolding<'alloc>;
    fn deref(&self) -> &Self::Target {
        unsafe { std::mem::transmute(self) }
    }
}

impl std::ops::Drop for TokenBlockFinalizing<'_> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        if !std::thread::panicking() {
            panic!("block was not freed");
        }
    }
}
