use crate::alloc::Allocator;
use crate::atomic::{Atomic, HasAtomic};
use crate::atomic::{AtomicPtr, AtomicUsize, fence};
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

        // Relaxed memory order is sufficient here because the block won't be visible to
        // any other threads until the `Condition` is shared by the user, which would
        // require its own synchronization.
        block.range.store(ConditionRange::single(id), Relaxed);
        debug_assert!(block.left_parent.load(Relaxed).is_none());
        debug_assert!(block.right_parent.load(Relaxed).is_none());
        debug_assert!(
            block
                .token_footer()
                .first_left_child
                .load(Relaxed)
                .is_none()
        );
        debug_assert!(block.token_footer().right_tree.load(Relaxed).is_none());
        debug_assert!(
            block
                .token_footer()
                .next_left_sibling
                .load(Relaxed)
                .is_none()
        );
        debug_assert!(
            block
                .token_footer()
                .prev_left_sibling
                .load(Relaxed)
                .is_none()
        );
        let tag = block.tag.load(Relaxed);
        block.tag.store(tag.next(), Release);
        Self { block }
    }

    /// Invalidates this [`Condition`] "immediately".
    ///
    /// All calls to [`Token::is_valid`] on a token that was constructed from `self.token()` that
    /// [happen after](https://en.wikipedia.org/wiki/Happened-before) this call are guaranteed to
    /// return `false`.
    ///
    /// This call may block the current thread and is generally slower than
    /// [`Condition::invalidate_eventually`].
    pub fn invalidate_immediately<Alloc: Allocator<'alloc> + ?Sized>(self, alloc: &mut Alloc) {
        // TODO: Real implementation - block thread until invalidation propogation completes
        self.invalidate_eventually(alloc);
    }

    /// Invalidates this [`Condition`] "eventually".
    ///
    /// This will never block the current thread.
    pub fn invalidate_eventually<Alloc: Allocator<'alloc> + ?Sized>(self, alloc: &mut Alloc) {
        let block = self.block;
        invalidate_condition(alloc, block);
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
        // SAFETY: All of the references inside of `ALWAYS_NEVER` will always be `None`, so
        // it can safely be interpreted as having any `'alloc` lifetime.
        let block = unsafe {
            std::mem::transmute::<&'alloc Block<'static>, &'alloc Block<'alloc>>(always_never)
        };
        Self {
            block,
            max_tag: BlockTag::new(1, true, 0),
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
        // SAFETY: All of the references inside of `ALWAYS_NEVER` will always be `None`, so
        // it can safely be interpreted as having any `'alloc` lifetime.
        let block = unsafe {
            std::mem::transmute::<&'alloc Block<'static>, &'alloc Block<'alloc>>(always_never)
        };
        Self {
            block,
            max_tag: BlockTag::new(0, true, 0),
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
        let always = Self::always();
        std::ptr::eq(self.block, always.block) && self.max_tag == always.max_tag
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
#[repr(C)]
pub struct Block<'alloc> {
    /// The tag for this block, consisting of the version number, invalid bit, and number of
    /// "holders".
    ///
    /// It is guranteed that:
    ///  * The version number will never decrease.
    ///  * The invalid bit can't be cleared unless the version number increases.
    ///  * The number of holders can't increase while the invalid bit is set.
    ///  * If this is a [`Token`] block, the version number can't increase until all of this
    ///    block's immediate children are invalidated.
    ///
    /// A block is "live" if either the invalid bit is cleared, or the number of holders is
    /// greater than zero. Certain invariants must hold while the block is live, and this is key
    /// to the lock-free design of the library. These invariants will be documented on the field
    /// they apply to.
    tag: Atomic<BlockTag>,

    /// Assuming that this is a [`Token`] block, this field describes the approximate range of
    /// [`ConditionId`]s which the token may be sensitive to.
    ///
    /// This may not change while the block is live.
    range: Atomic<ConditionRange>,

    /// Assuming that this is a [`Token`] block for a token created by combining two parent tokens,
    /// this field points to the block for the "left" parent token.
    ///
    /// This may not change while the block is live. Regardless of whether the block is live, this
    /// can never be updated to a dead block.
    ///
    /// The `range` of the parent tokens must not overlap, with all [`ConditionId`]s in the `range`
    /// of the left parent token being less than those in the `range` of the right parent token.
    left_parent: Atomic<Option<&'alloc Block<'alloc>>>,

    /// Assuming that this is a [`Token`] block for a token created by combining two parent tokens,
    /// this field points to the block for the "right" parent token.
    ///
    /// This may not change while the block is live. Regardless of whether the block is live, this
    /// can never be updated to a dead block.
    ///
    /// The `range` of the parent tokens must not overlap, with all [`ConditionId`]s in the `range`
    /// of the left parent token being less than those in the `range` of the right parent token.
    ///
    /// If a block has a `right_parent`, but not a `left_parent`, then it is a "branch" block
    /// for the "right search tree" of `right_parent`.
    right_parent: Atomic<Option<&'alloc Block<'alloc>>>,

    /// The footer for this block, which contains type-specific data.
    footer: BlockFooter<'alloc>,
}

/// The footer for a [`Block`].
///
/// This contains different information depending on the block's type.
union BlockFooter<'alloc> {
    pub token: ManuallyDrop<TokenBlockFooter<'alloc>>,
    pub branch: ManuallyDrop<BranchBlockFooter<'alloc>>,
}

/// The footer for a [`Token`] block.
#[repr(C)]
struct TokenBlockFooter<'alloc> {
    /// A reference to the first [`Token`] block in the doubly-linked list of token blocks which
    /// have this block as their left parent.
    pub first_left_child: Atomic<Option<&'alloc Block<'alloc>>>,

    /// A reference to the block at the root of the "right search tree" for this block.
    ///
    /// The right search tree contains all [`Token`] blocks which have this block as their
    /// right parent. The tree is indexed by the address of the left parent of the token block.
    /// This allows efficiently searching for a token block which has a given left and right parent.
    ///
    /// While the block is live, modifications to this field are restricted. It may only be changed
    /// if any of the following are true:
    ///  * `right_tree` is [`None`]
    ///  * `right_tree` has its invalid bit set
    ///  * `right_tree` has a `right_parent` which does not match this block
    pub right_tree: Atomic<Option<&'alloc Block<'alloc>>>,

    /// A reference to the next [`Token`] block in the doubly-linked list of token blocks which
    /// share the same left parent.
    pub next_left_sibling: Atomic<SiblingBlockRef<'alloc>>,

    /// A reference to the previous [`Token`] block in the doubly-linked list of token blocks which
    /// share the same left parent.
    pub prev_left_sibling: Atomic<Option<&'alloc Block<'alloc>>>,
}

/// The footer for a "tree" block.
#[repr(C)]
struct BranchBlockFooter<'alloc> {
    /// The child nodes of tree block, organzied by the next few bits of their `left_parent`'s
    /// address.
    ///
    /// While the block is live, modifications to this field are restricted. A child may only be
    /// changed if any of the following are true:
    ///  * The child is [`None`]
    ///  * The child has its invalid bit set
    ///  * The child has a `right_parent` which does not match the root of the search tree
    pub children: [Atomic<Option<&'alloc Block<'alloc>>>; TreeNode::BRANCH_SIZE],
}

/// The "tag" for a [`Block`].
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
struct BlockTag(usize);

unsafe impl HasAtomic for BlockTag {
    type Prim = usize;
    type Atomic = AtomicUsize;
}

impl BlockTag {
    /// A bit which indicates that a block is invalid.
    pub const INVALID_BIT: usize = 1 << Self::HOLD_BITS;

    /// The number of bits used to store the number of "holder" threads this block has.
    pub const HOLD_BITS: u32 = u8::BITS;

    /// A mask which selects the number of holders from the tag.
    pub const HOLD_MASK: usize = (1 << Self::HOLD_BITS) - 1;

    /// The shift applied to the version number in the tag.
    pub const VERSION_SHIFT: u32 = Self::HOLD_BITS + 1;

    /// A mask which selects the version number from the tag.
    pub const VERSION_MASK: usize = usize::MAX << Self::VERSION_SHIFT;

    /// The maximum allowed version number for a block.
    pub const MAX_VERSION: usize = Self::VERSION_MASK >> Self::VERSION_SHIFT;

    /// Constructs a [`BlockTag`] from its components.
    pub const fn new(version: usize, invalid: bool, holders: u8) -> Self {
        assert!(version <= Self::MAX_VERSION);
        Self(
            (version << Self::VERSION_SHIFT)
                | (invalid as usize * Self::INVALID_BIT)
                | (holders as usize),
        )
    }

    /// Increments the version number of the tag and resets all other bits.
    pub const fn next(self) -> Self {
        Self(
            (self.0 | !Self::VERSION_MASK)
                .checked_add(1)
                .expect("block version number overflow"),
        )
    }
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

unsafe impl<'alloc> HasAtomic for SiblingBlockRef<'alloc> {
    type Prim = *mut Block<'alloc>;
    type Atomic = AtomicPtr<Block<'alloc>>;
}

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

    /// Indicates whether this is [`Self::NONE`].
    pub const fn is_none(self) -> bool {
        self.ptr.is_null()
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

impl<'a> Atomic<SiblingBlockRef<'a>> {
    /// Sets the `is_removing` flag for this reference. Returns the underlying
    /// `Option<&'alloc Block>`.
    pub fn mark_removing(&self, order: std::sync::atomic::Ordering) -> Option<&'a Block> {
        // TODO: Use `fetch_or` once it is stabilized
        // https://doc.rust-lang.org/std/sync/atomic/struct.AtomicPtr.html#method.fetch_or
        let mut cur = self.0.load(order);
        loop {
            if cur.addr() & SiblingBlockRef::IS_REMOVING_BIT != 0 {
                return unsafe {
                    HasAtomic::from_prim(
                        cur.map_addr(|addr| addr & !SiblingBlockRef::IS_REMOVING_BIT),
                    )
                };
            }
            let new = cur.map_addr(|addr| addr | SiblingBlockRef::IS_REMOVING_BIT);
            match self.0.compare_exchange_weak(cur, new, Relaxed, order) {
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
            tag: Atomic::new(BlockTag::new(0, true, 0)),
            range: Atomic::new(ConditionRange(0)),
            left_parent: Atomic::new(None),
            right_parent: Atomic::new(None),
            footer: BlockFooter {
                branch: ManuallyDrop::new(BranchBlockFooter {
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

    /// Assuming this is a [`Token`] block, gets the footer for the block.
    fn token_footer(&self) -> &TokenBlockFooter<'alloc> {
        // SAFETY: All of the `footer` variants have the same field layout, so this is safe
        // regardless of the actual type of the block.
        unsafe { &self.footer.token }
    }

    /// Assuming this is a "branch" block, gets the footer for the block.
    fn branch_footer(&self) -> &BranchBlockFooter<'alloc> {
        // SAFETY: All of the `footer` variants have the same field layout, so this is safe
        // regardless of the actual type of the block.
        unsafe { &self.footer.branch }
    }

    /// Assuming this is a [`Token`] block, gets the token it currently represents.
    ///
    /// This method must be used with caution, because it is very easy to unintentionally get
    /// the wrong token. Blocks may be invalidated, freed and reallocated as new tokens by another
    /// thread at any time, unless the current thread actively prevents this from happening,
    /// such as by holding a [`BlockGuard`].
    fn token(&'alloc self) -> Token<'alloc> {
        Token {
            block: self,
            max_tag: BlockTag(
                (self.tag.load(Relaxed).0 & BlockTag::VERSION_MASK) | BlockTag::INVALID_BIT,
            ),
        }
    }
}

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

/// A unique identifier for a [`Condition`].
///
/// Even after a condition is invalidated, its identifier will never be re-used.
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

unsafe impl HasAtomic for ConditionRange {
    type Prim = usize;
    type Atomic = AtomicUsize;
}

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

/// A special [`Block`] used to implement the "always" and "never" tokens.
#[cfg(not(loom))]
static ALWAYS_NEVER: Block<'static> = Block {
    tag: Atomic::from_prim(BlockTag::new(1, false, 0).0),
    // Use a high range so that this block will always be the "right parent" of a combined
    // token. The right parent is checked for validity before the left parent.
    range: Atomic::from_prim(ConditionRange::single(ConditionId::MAX).0),
    left_parent: Atomic::null(),
    right_parent: Atomic::null(),
    footer: BlockFooter {
        token: std::mem::ManuallyDrop::new(TokenBlockFooter {
            first_left_child: Atomic::null(),
            right_tree: Atomic::null(),
            next_left_sibling: Atomic::null(),
            prev_left_sibling: Atomic::null(),
        }),
    },
};

#[cfg(loom)]
loom::lazy_static! {
    static ref ALWAYS_NEVER: Block<'static> = Block {
        tag: Atomic::new(BlockTag::new(1, false, 0)),
        // Use a high range so that this block will always be the "right parent" of a combined
        // token. The right parent is checked for validity before the left parent.
        range: Atomic::new(ConditionRange::single(ConditionId::MAX)),
        left_parent: Atomic::new(None),
        right_parent: Atomic::new(None),
        footer: BlockFooter {
            token: std::mem::ManuallyDrop::new(TokenBlockFooter {
                first_left_child: Atomic::new(None),
                right_tree: Atomic::new(None),
                next_left_sibling: Atomic::new(SiblingBlockRef::NONE),
                prev_left_sibling: Atomic::new(None),
            }),
        },
    };
}

impl<'alloc> Token<'alloc> {
    /// Gets the left and right parents of a token, or returns [`None`] if the token is invalid or
    /// doesn't have parents.
    ///
    /// If this returns [`Some`], then it is guranteed that, whenever this token is invalid, at
    /// least one of the returned parents is invalid.
    fn parents(&self) -> Option<(Token<'alloc>, Token<'alloc>)> {
        let left = self.block.left_parent.load(Relaxed)?.token();
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
        // These ranges may be incorrect if the associated tokens have been invalidated, but that
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
        let node = match TreeNode::root(right.block).search(left.block, right.block) {
            Ok(token) => {
                // The search gave us a token which has the correct parent *blocks*, but we still
                // need to make sure it has the correct parent *tokens*. We do this by checking that
                // the parent tokens are still valid. Otherwise, it could be possible that either
                // `left.block` or `right.block` was invalidated and reallocated as a new token, in
                // which case the returned token would erroneously be considered valid.
                fence(Acquire); // Synchronizes with token invalidation
                return if right.is_valid() && left.is_valid() {
                    token
                } else {
                    Token::never()
                };
            }
            Err(node) => node,
        };

        // Hold the parent blocks to keep them alive while we create the new token. This also
        // gives us one final check that the parent tokens are valid before we proceed.
        let Some(right_guard) = hold(right) else {
            return Token::never();
        };
        let Some(left_guard) = hold(left) else {
            unhold(alloc, right_guard);
            return Token::never();
        };

        // Create new block for the token.
        let block = alloc.allocate_block();
        block.range.store(range, Relaxed);
        block.left_parent.store(Some(left.block), Relaxed);
        block.right_parent.store(Some(right.block), Relaxed);
        debug_assert!(
            block
                .token_footer()
                .first_left_child
                .load(Relaxed)
                .is_none()
        );
        debug_assert!(block.token_footer().right_tree.load(Relaxed).is_none());
        debug_assert!(
            block
                .token_footer()
                .next_left_sibling
                .load(Relaxed)
                .is_none()
        );
        debug_assert!(
            block
                .token_footer()
                .prev_left_sibling
                .load(Relaxed)
                .is_none()
        );
        fence(Release); // Synchronizes with token creation
        let tag = block.tag.load(Relaxed).next();
        block.tag.store(tag, Release);

        // Try to insert the new token into the tree
        if let Err(token) = node.insert(alloc, left.block, &right_guard, block) {
            // Another thread created the token before us. Clean up and return the existing token.
            block
                .tag
                .store(BlockTag(tag.0 | BlockTag::INVALID_BIT), Relaxed);
            block.left_parent.store(None, Relaxed);
            block.right_parent.store(None, Relaxed);
            alloc.free_block(block);
            unhold(alloc, left_guard);
            unhold(alloc, right_guard);
            return token;
        }

        // Insert the new token into the left parent's list of children.
        insert_left_child(alloc, &left_guard, block);

        // Clean up and return the new token
        unhold(alloc, left_guard);
        unhold(alloc, right_guard);
        Token {
            block,
            max_tag: BlockTag(tag.0 | BlockTag::INVALID_BIT),
        }
    }
}

/// Represents a node of a "right search tree".
struct TreeNode<'alloc> {
    /// The current contents of this node.
    ///
    /// Rather than directly referencing the block, this references the [`Atomic`] which defines
    /// the conents for this node, allowing the node to be updated.
    slot: &'alloc Atomic<Option<&'alloc Block<'alloc>>>,

    /// The index of the first bit of the `left_parent` address that is considered by this node.
    bit_depth: u32,
}

impl<'alloc> TreeNode<'alloc> {
    /// The number of bits that are addressed by each branch node in a "right search tree".
    const BRANCH_BITS: u32 = 2;

    /// The number of slots in a branch node.
    const BRANCH_SIZE: usize = 1 << Self::BRANCH_BITS;

    /// The number of least-significant bits of the left parent address that are ignored when
    /// searching for a token in a "right search tree".
    const SKIP_BITS: u32 = std::mem::size_of::<Block>().ilog2();

    /// Gets the root [`TreeNode`] for the right search tree of the given block.
    pub fn root(right_parent: &'alloc Block<'alloc>) -> Self {
        let right_tree = &right_parent.token_footer().right_tree;
        Self {
            slot: right_tree,
            bit_depth: Self::SKIP_BITS,
        }
    }

    /// Searches the tree rooted at this node for a [`Token`] with the given left parent.
    /// `right_parent` must be the owner of the tree. If the requested token is not found, this
    /// returns the [`TreeNode`] where it should be inserted.
    ///
    /// If a token is returned, it is guaranteed that one of the following is true:
    ///  * The returned token is the only valid token with the given `left_parent` and
    ///    `right_parent`.
    ///  * Either `left_parent` or `right_parent` were invalid at some point during this call.
    pub fn search(
        mut self,
        left_parent: &'alloc Block<'alloc>,
        right_parent: &'alloc Block<'alloc>,
    ) -> Result<Token<'alloc>, TreeNode<'alloc>> {
        loop {
            // Check if a block is in the slot.
            let Some(block) = self.slot.load(Acquire) else {
                return Err(self);
            };

            // Assuming that this block is a token block (which might not be the case), grab a
            // `Token` for it now so that we can verify if it was invalidated later.
            let token = block.token();
            fence(Acquire); // Synchronizes with token creation

            // Check whether the block actually belongs to this tree. It's possible for an old
            // member of the tree to be freed and reallocated as a new unrelated block.
            if !std::ptr::eq(block.right_parent.0.load(Relaxed), right_parent) {
                return Err(self);
            }

            // Check whether this is a branch block
            let block_left_parent = block.left_parent.load(Relaxed);
            let Some(block_left_parent) = block_left_parent else {
                self.slot = &block.branch_footer().children
                    [(left_parent as *const Block as usize >> self.bit_depth) % Self::BRANCH_SIZE];
                self.bit_depth += Self::BRANCH_BITS;
                continue;
            };

            // Check whether this is the token block we are looking for.
            fence(Acquire); // Synchronizes with token invalidation
            if token.is_valid() && std::ptr::eq(block_left_parent, left_parent) {
                return Ok(token);
            } else {
                return Err(self);
            }
        }
    }

    /// Attempts to insert a [`Token`] block into the tree rooted at this node. `right_parent` must
    /// be the owner of the tree. If there is already a token with the given `left_parent` and
    /// `right_parent`, this will return the existing token.
    pub fn insert<Alloc: Allocator<'alloc> + ?Sized>(
        mut self,
        alloc: &mut Alloc,
        left_parent: &'alloc Block<'alloc>,
        right_guard: &BlockGuard<'alloc>,
        target: &'alloc Block<'alloc>,
    ) -> Result<(), Token<'alloc>> {
        let right_parent = right_guard.block;
        let mut occupant = None;
        'replace: loop {
            // We have already decided that we can remove `occupant`, so we will try to replace it
            // with `target` now.
            let Err(n_occupant) =
                self.slot
                    .compare_exchange_weak(occupant, Some(target), Release, Acquire)
            else {
                return Ok(());
            };
            occupant = n_occupant;
            'occupied: loop {
                let Some(block) = occupant else {
                    continue 'replace;
                };

                // Assuming that this block is a token block (which might not be the case), grab a
                // `Token` for it now so that we can verify if it was invalidated later.
                let token = block.token();
                fence(Acquire); // Synchronizes with token creation

                // Check whether the block actually belongs to this tree. It's possible for an old
                // member of the tree to be freed and reallocated as a new unrelated block.
                if !std::ptr::eq(block.right_parent.0.load(Relaxed), right_parent) {
                    continue 'replace;
                }

                // Check whether this is a branch block
                let block_left_parent = block.left_parent.load(Relaxed);
                let Some(block_left_parent) = block_left_parent else {
                    self.slot = &block.branch_footer().children[(left_parent as *const Block
                        as usize
                        >> self.bit_depth)
                        % Self::BRANCH_SIZE];
                    self.bit_depth += Self::BRANCH_BITS;
                    occupant = None;
                    continue 'replace;
                };

                // Check whether this is the token block we are looking for.
                fence(Acquire); // Synchronizes with token invalidation
                if !token.is_valid() {
                    continue 'replace;
                }
                if std::ptr::eq(block_left_parent, left_parent) {
                    return Err(token);
                }

                // There is a valid token block in the slot, but it's not the one we're looking
                // for. We will have to create a branch block here so we can fit the existing
                // token, and the new token.
                loop {
                    let block_bits = (block_left_parent as *const Block as usize) >> self.bit_depth;
                    let target_bits = (left_parent as *const Block as usize) >> self.bit_depth;
                    debug_assert_ne!(block_bits, target_bits);
                    let block_slot_index = block_bits % Self::BRANCH_SIZE;
                    let target_slot_index = target_bits % Self::BRANCH_SIZE;
                    let branch = alloc.allocate_block();
                    debug_assert!(branch.left_parent.load(Relaxed).is_none());
                    branch.right_parent.store(Some(right_parent), Relaxed);
                    for child in branch.branch_footer().children.iter() {
                        debug_assert!(child.load(Relaxed).is_none());
                    }
                    branch.branch_footer().children[target_slot_index].store(Some(target), Relaxed);
                    branch.branch_footer().children[block_slot_index].store(Some(block), Relaxed);

                    // Attempt to insert branch
                    match self
                        .slot
                        .compare_exchange(occupant, Some(branch), Release, Acquire)
                    {
                        Ok(_) => {
                            // Check if we're done
                            if block_slot_index != target_slot_index {
                                return Ok(());
                            }

                            // We need to create more branches to distinguish between the two
                            // blocks.
                            self.slot = &branch.branch_footer().children[block_slot_index];
                            self.bit_depth += Self::BRANCH_BITS;
                            occupant = None;
                            continue;
                        }
                        Err(n_occupant) => {
                            // Clean up and free the branch
                            branch.right_parent.store(None, Relaxed);
                            for child in branch.branch_footer().children.iter() {
                                child.store(None, Relaxed);
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
}

/// A reference to a [`Block`] which signals that it is being "held" by the current thread.
///
/// This gurantees that the block is "live" for as long as the guard is held.
///
/// This *won't* automatically "unhold" the block when dropped, but it will panic if the block was
/// not unheld. That's because unholding a block might require performing invalidation propogation,
/// which requires access to an allocator and could be expensive.
struct BlockGuard<'alloc> {
    block: &'alloc Block<'alloc>,
}

impl std::ops::Drop for BlockGuard<'_> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        panic!("block was not unheld");
    }
}

/// Tries to "hold" the block associated with the given token, ensuring that it remains "live" for
/// as long as the guard is held. Returns [`None`] if the token is invalid.
fn hold(token: Token) -> Option<BlockGuard> {
    let mut tag = token.max_tag.0 & BlockTag::VERSION_MASK;
    loop {
        let n_tag = tag + 1;
        assert!(
            n_tag < token.max_tag.0,
            "maximum number of holders exceeded"
        );
        match token
            .block
            .tag
            .0
            .compare_exchange_weak(tag, n_tag, Relaxed, Relaxed)
        {
            Ok(_) => return Some(BlockGuard { block: token.block }),
            Err(e_tag) => {
                if e_tag < token.max_tag.0 {
                    tag = e_tag;
                } else {
                    // The token has already been invalidated.
                    return None;
                }
            }
        }
    }
}

/// "Unholds" the block associated with a given [`BlockGuard`].
fn unhold<'alloc, Alloc: Allocator<'alloc> + ?Sized>(alloc: &mut Alloc, guard: BlockGuard<'alloc>) {
    let block = guard.block;
    std::mem::forget(guard);
    let o_tag = block.tag.0.fetch_sub(1, AcqRel); // Synchronizes with token unhold
    debug_assert!(o_tag & BlockTag::HOLD_MASK > 0);
    if o_tag & !BlockTag::VERSION_MASK == (BlockTag::INVALID_BIT + 1) {
        fence(Release); // Synchronizes with token invalidation
        // We were the last holder of an invalid block, so it's our responsibility to finalize it.
        finalize(alloc, block);
    }
}

/// Invalidates a [`Token`] if it has not already been invalidated. Returns `true` if the block
/// was [`finalize`]d.
fn invalidate_token<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
    alloc: &mut Alloc,
    token: Token<'alloc>,
) -> bool {
    // We don't want to invalidate the token if the block has already been reallocated as a new
    // token, so we have to use `compare_exchange` instead of `fetch_or`.
    let mut tag = token.max_tag.0 & BlockTag::VERSION_MASK;
    loop {
        let n_tag = tag | BlockTag::INVALID_BIT;
        match token
            .block
            .tag
            .0
            .compare_exchange_weak(tag, n_tag, Acquire, Relaxed) // Synchronizes with token unhold
        {
            Ok(_) => {
                if tag & BlockTag::HOLD_MASK == 0 {
                    fence(Release); // Synchronizes with token invalidation
                    // There are no holders on the block, so we can finalize it.
                    finalize(alloc, token.block);
                    return true;
                } else {
                    return false;
                }
            }
            Err(e_tag) => {
                if e_tag < token.max_tag.0 {
                    tag = e_tag;
                } else {
                    // The token has already been invalidated.
                    return false;
                }
            }
        }
    }
}

/// Invalidates a [`Token`] block that we have exclusive ownership of.
///
/// This is the internal implementation of [`Condition::invalidate_eventually`].
fn invalidate_condition<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
    alloc: &mut Alloc,
    block: &'alloc Block<'alloc>,
) {
    let o_tag = block.tag.0.fetch_or(BlockTag::INVALID_BIT, Acquire); // Synchronizes with token unhold
    debug_assert!(o_tag & BlockTag::INVALID_BIT == 0);
    if o_tag & !BlockTag::VERSION_MASK == 0 {
        fence(Release); // Synchronizes with token invalidation
        // There are no holders of this block, so we can finalize it ourselves.
        finalize(alloc, block);
    }
}

/// Performs invalidation propogation on a [`Token`] block and frees it.
///
/// The block must be not be live, and the current thread must have been the thread which last
/// set its tag.
fn finalize<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
    alloc: &mut Alloc,
    block: &'alloc Block<'alloc>,
) {
    // Remove from left parent's list of children
    if let Some(left_parent) = block.left_parent.load(Relaxed) {
        remove_left_child(left_parent, block);

        // Clean up links to left parent
        block // Synchronizes with left parent removal
            .left_parent
            .store(None, Release);
        block.token_footer().prev_left_sibling.store(None, Relaxed);
        block // Synchronizes with left sibling removal
            .token_footer()
            .next_left_sibling
            .store(SiblingBlockRef::NONE, Release);
    }

    // Invalidate right children
    invalidate_right(alloc, block, &block.token_footer().right_tree);
    fn invalidate_right<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
        alloc: &mut Alloc,
        right_parent: &'alloc Block<'alloc>,
        slot: &Atomic<Option<&'alloc Block<'alloc>>>,
    ) {
        // No other thread is allowed to modify the slot, so we can safely read it without
        // synchronization.
        let Some(block) = slot.load(Acquire) else {
            return;
        };
        slot.store(None, Relaxed);

        // Assuming that this block is a token block (which might not be the case), grab a `Token`
        // for it now so that we can verify if it was invalidated later.
        let token = block.token();

        // Check whether the block actually belongs to this tree. It's possible for an old
        // member of the tree to be freed and reallocated as a new unrelated block.
        if !std::ptr::eq(block.right_parent.0.load(Relaxed), right_parent) {
            return;
        }

        // Check whether this is a branch block
        if block.left_parent.load(Relaxed).is_some() {
            fence(Acquire); // Synchronizes with token invalidation
            invalidate_token(alloc, token);
        } else {
            for slot in block.branch_footer().children.iter() {
                invalidate_right(alloc, right_parent, slot);
            }
            block.right_parent.store(None, Relaxed);
            alloc.free_block(block);
        }
    }

    // Invalidate left children
    let mut opt_first_child = block.token_footer().first_left_child.load(Relaxed);
    while let Some(first_child) = opt_first_child {
        // Verify that this is still a left child of `block`.
        let first_child = first_child.token();
        let first_child_prev_sibling = first_child
            .block
            .token_footer()
            .prev_left_sibling
            .load(Relaxed);
        let first_child_parent = first_child // Synchronizes with left parent removal
            .block
            .left_parent
            .load(Acquire);
        if !std::ptr::eq(HasAtomic::into_prim(first_child_parent), block) {
            // Before `left_child` was invalidated, it must have removed itself from the
            // left children list.
            opt_first_child = block.token_footer().first_left_child.load(Relaxed);
            debug_assert!(
                !std::ptr::eq(HasAtomic::into_prim(opt_first_child), first_child.block),
                "parent is none: {}",
                first_child_parent.is_none()
            );
            continue;
        }

        // Since we removed every child before `first_child`, it should not have any previous
        // siblings.
        debug_assert!(first_child_prev_sibling.is_none());

        // Try to invalidate the child
        if invalidate_token(alloc, first_child) {
            // Since we just ran finalization on this thread, the child should have removed
            // itself from the left children list.
            opt_first_child = block.token_footer().first_left_child.load(Relaxed);
            debug_assert!(!std::ptr::eq(
                HasAtomic::into_prim(block.token_footer().first_left_child.load(Relaxed)),
                first_child.block
            ));
        } else {
            // This child is already scheduled to be invalidated by another thread, so we can
            // safely remove it from the list.
            opt_first_child = force_remove_left_child(block, first_child.block);
        }
    }

    // Free block
    block.right_parent.store(None, Relaxed);
    alloc.free_block(block);
}

/// Inserts a token [`Block`] into the list of children of its left parent.
fn insert_left_child<'alloc, Alloc: Allocator<'alloc> + ?Sized>(
    alloc: &mut Alloc,
    left_guard: &BlockGuard<'alloc>,
    block: &'alloc Block<'alloc>,
) {
    let first_child_slot = &left_guard.block.token_footer().first_left_child;
    let mut opt_first_child = first_child_slot.load(Acquire);
    loop {
        block
            .token_footer()
            .next_left_sibling
            .store(SiblingBlockRef::new(opt_first_child, false), Relaxed);
        if let Some(first_child) = opt_first_child {
            // Before inserting a new child at the front of the list, we will hold the current
            // first child. This greatly simplifies the logic for lock-free removal of an
            // arbitrary child in the list because we won't have to worry about
            // `prev_left_sibling` being out of date.
            if let Some(first_child_guard) = hold(first_child.token()) {
                match first_child_slot.compare_exchange_weak(
                    opt_first_child,
                    Some(block),
                    Release,
                    Acquire,
                ) {
                    Ok(_) => {
                        first_child
                            .token_footer()
                            .prev_left_sibling
                            .store(Some(block), Release);
                        unhold(alloc, first_child_guard);
                        break;
                    }
                    Err(n_opt_first_child) => {
                        unhold(alloc, first_child_guard);
                        opt_first_child = n_opt_first_child;
                        continue;
                    }
                }
            } else {
                // Remove invalid child from list
                opt_first_child = force_remove_left_child(left_guard.block, first_child);
                continue;
            }
        } else {
            // Try to insert block as first child
            match first_child_slot.compare_exchange_weak(None, Some(block), Release, Acquire) {
                Ok(_) => break,
                Err(n_opt_next_sibling) => {
                    opt_first_child = n_opt_next_sibling;
                    continue;
                }
            }
        }
    }
}

/// Removes a [`Token`] block from the list of children of its left parent. This assumes that
/// `block` is not live and the current thread owns it.
fn remove_left_child<'alloc>(left_parent: &'alloc Block<'alloc>, block: &'alloc Block<'alloc>) {
    // First, we mark `next_left_sibling` as removing so that if `to_sibling` is itself being
    // removed, it will not try to modify our `next_left_sibling`, which we will no longer read.
    let to_sibling = block
        .token_footer()
        .next_left_sibling
        .mark_removing(Relaxed);

    // Ensure that `block` is no longer in the list of children.
    let between_siblings = BlockList {
        head: block,
        tail: None,
    };
    loop {
        let prev_sibling = block.token_footer().prev_left_sibling.load(Relaxed);
        if update(
            left_parent,
            prev_sibling,
            &between_siblings,
            block,
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
            match from_sibling
                .token_footer()
                .next_left_sibling
                .compare_exchange_weak(
                    SiblingBlockRef::new(Some(before_to_sibling), false),
                    SiblingBlockRef::new(to_sibling, false),
                    Relaxed,
                    Acquire, // Synchronizes with left sibling removal
                ) {
                Ok(_) => (),
                Err(e_next_sibling) => {
                    // Ensure that `from_sibling` is still in the list of children.
                    if !std::ptr::eq(
                        HasAtomic::into_prim(from_sibling.left_parent.load(Relaxed)),
                        left_parent,
                    ) {
                        return false;
                    }

                    // If `from_sibling` is currently being removed, help it out by looking
                    // further back in the list.
                    if e_next_sibling.is_removing() {
                        let prev_from_sibling =
                            from_sibling.token_footer().prev_left_sibling.load(Relaxed);
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
                                .token_footer()
                                .next_left_sibling
                                .compare_exchange_weak(
                                    SiblingBlockRef::new(Some(e_next_sibling), false),
                                    SiblingBlockRef::new(to_sibling, false),
                                    Relaxed,
                                    Relaxed,
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
            match left_parent
                .token_footer()
                .first_left_child
                .compare_exchange_weak(Some(before_to_sibling), to_sibling, Relaxed, Relaxed)
            {
                Ok(_) => (),
                Err(None) => {
                    // The list is empty, so `before_to_sibling` is not in the list.
                }
                Err(Some(e_first_sibling)) => {
                    if contains(between_siblings, e_first_sibling) {
                        if left_parent
                            .token_footer()
                            .first_left_child
                            .compare_exchange_weak(
                                Some(e_first_sibling),
                                to_sibling,
                                Relaxed,
                                Relaxed,
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

/// Removes a [`Token`] block from the beginning of the list of children of its left parent. This
/// assumes that `block` is invalid, but unlike [`remove_left_child`], it does not require that
/// the current thread owns it.
///
/// Returns the next child in the list, or [`None`] if there are no more children.
fn force_remove_left_child<'alloc>(
    left_parent: &'alloc Block<'alloc>,
    block: &'alloc Block<'alloc>,
) -> Option<&'alloc Block<'alloc>> {
    let mut next_sibling = block.token_footer().next_left_sibling.load(Relaxed);
    loop {
        let block_parent = block.left_parent.load(Acquire); // Synchronizes with left parent removal
        if !std::ptr::eq(HasAtomic::into_prim(block_parent), left_parent) {
            // `block` has already removed itself from the list of children. Now the first child
            // should be some other block.
            let first_child = left_parent.token_footer().first_left_child.load(Relaxed);
            debug_assert!(!std::ptr::eq(HasAtomic::into_prim(first_child), block));
            return first_child;
        }

        // Mark `block` as removing
        if !next_sibling.is_removing() {
            if let Err(n_next_sibling) = block
                .token_footer()
                .next_left_sibling
                .compare_exchange_weak(
                    next_sibling,
                    SiblingBlockRef::new(next_sibling.source(), true),
                    Relaxed,
                    Relaxed,
                )
            {
                next_sibling = n_next_sibling;
                continue;
            }
        }
        let next_sibling = next_sibling.source();

        // Attempt to remove `block` from the list of children.
        match left_parent
            .token_footer()
            .first_left_child
            .compare_exchange_weak(Some(block), next_sibling, Relaxed, Relaxed)
        {
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
        match to_sibling
            .token_footer()
            .prev_left_sibling
            .compare_exchange_weak(Some(prev_to_sibling), opt_from_sibling, Relaxed, Relaxed)
        {
            Ok(_) | Err(None) => return,
            Err(Some(n_prev_to_sibling)) => {
                // Check if `before_to_sibling` is reachable from `n_prev_to_sibling`.
                prev_to_sibling = n_prev_to_sibling;
                let mut test_sibling = n_prev_to_sibling;
                loop {
                    if std::ptr::eq(test_sibling, before_to_sibling) {
                        continue 'retry;
                    } else if test_sibling.token().is_valid() {
                        // There are no valid blocks between `before_to_sibling` and `to_sibling`,
                        // so if `test_sibling` is valid, it must already be before
                        // `before_to_sibling`.
                        return;
                    } else if let Some(n_test_sibling) =
                        test_sibling.token_footer().prev_left_sibling.load(Relaxed)
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
