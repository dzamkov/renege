use super::Token;
use crate::atomic::{Atomic, PrimRepr};
use std::cell::RefCell;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use std::sync::atomic::{fence, AtomicU64};

/// A unique identifier for a [`Token`]. Even after a token is invalidated, its identifier will
/// never be re-used.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct TokenId(u64);

impl TokenId {
    /// The [`TokenId`] for the unique token which is always valid.
    pub const ALWAYS: TokenId = TokenId(0);

    /// The [`TokenId`] for the unique token which is never valid.
    // Use a high `TokenId` so that this will always be considered as the secondary parent, which
    // is checked for validity before the primary parent.
    pub const NEVER: TokenId = TokenId((1 << Self::NUM_BITS) - 1);

    /// The maximum number of bits a [`TokenId`] can occupy.
    const NUM_BITS: u8 = 58;

    /// Constructs a [`TokenId`] with the given index.
    #[cfg(test)]
    pub fn new(index: u64) -> Self {
        assert!(index < (1 << Self::NUM_BITS));
        Self(index)
    }

    /// Gets the index of this [`TokenId`].
    pub fn index(&self) -> u64 {
        self.0
    }
}

unsafe impl PrimRepr for TokenId {
    type Prim = u64;
}

/// A [`TokenId`] paired with a bit which indicates whether it is for a dependent token.
#[repr(transparent)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct ExtTokenId(u64);

impl ExtTokenId {
    /// The [`ExtTokenId`] for the unique token which is always valid.
    pub const ALWAYS: ExtTokenId = ExtTokenId(TokenId::ALWAYS.0);

    /// The [`ExtTokenId`] for the unique token which is never valid.
    pub const NEVER: ExtTokenId = ExtTokenId(TokenId::NEVER.0);

    /// The maximum number of bits an [`ExtTokenId`] can occupy.
    const NUM_BITS: u8 = TokenId::NUM_BITS + 1;

    /// Constructs an [`ExtTokenId`] from the given [`TokenId`] and `is_dependent` value.
    pub const fn new(id: TokenId, is_dependent: bool) -> Self {
        ExtTokenId(id.0 | (is_dependent as u64) << TokenId::NUM_BITS)
    }

    /// Indicates whether this [`ExtTokenId`] is for a dependent token.
    pub fn is_dependent(&self) -> bool {
        self.0 & (1 << TokenId::NUM_BITS) != 0
    }
}

impl From<ExtTokenId> for TokenId {
    fn from(value: ExtTokenId) -> Self {
        Self(value.0 & ((1 << Self::NUM_BITS) - 1))
    }
}

/// Describes the approximate "range" of [`TokenId`]s a dependent [`Token`] may be sensitive to.
///
/// Due to the requirements of how tokens are combined, this also determines a "splitting"
/// `TokenId` (see `DependentRange::split`).
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct DependentRange(u64);

impl DependentRange {
    /// Constructs a new [`DependentRange`] with the given splitting [`TokenId`] and scale.
    pub fn new(split: TokenId, scale: u8) -> Self {
        debug_assert!(scale < TokenId::NUM_BITS);
        debug_assert!(split.0 & ((1 << scale) - 1) == 0);
        Self((u64::from(scale) << TokenId::NUM_BITS) | split.0)
    }

    /// Constructs the smallest [`DependentRange`] which contains the given [`TokenId`]s.
    pub fn between(low: TokenId, high: TokenId) -> Self {
        debug_assert!(low < high);
        let scale = (low.0 ^ high.0).ilog2() as u8;
        let split = high.0 & !((1 << scale) - 1);
        Self::new(TokenId(split), scale)
    }

    /// Gets the splitting [`TokenId`] for this range. The "left" parent of a dependent token
    /// with this range is only sensitive to `TokenId`s less than the splitting `TokenId` and the
    /// "right" parent is only sensitive to `TokenId`s greater than or equal to the splitting
    /// `TokenId`.
    pub fn split(&self) -> TokenId {
        TokenId(self.0 & ((1 << TokenId::NUM_BITS) - 1))
    }

    /// Gets the "scale" of this range. This is the log-base-2 of the number of [`TokenId`]s in
    /// this range on either side of the splitting [`TokenId`], which is also one less than the
    /// log-base-2 of the total number of [`TokenId`]s in this range.
    pub fn scale(&self) -> u8 {
        (self.0 >> TokenId::NUM_BITS) as u8
    }

    /// Gets the smallest [`TokenId`] which is in this range.
    pub fn min_bound(&self) -> TokenId {
        TokenId(self.split().0 - (1 << self.scale()))
    }

    /// Gets the largest [`TokenId`] which is in this range.
    pub fn max_bound(&self) -> TokenId {
        TokenId(self.split().0 + (1 << self.scale()) - 1)
    }
}

unsafe impl PrimRepr for DependentRange {
    type Prim = u64;
}

/// The header for a [`TokenBlock`].
#[repr(transparent)]
#[derive(PartialEq, Eq, Clone, Copy)]
struct TokenBlockHeader(u64);

/// Describes the overall state of a [`TokenBlock`]
#[repr(u8)]
enum TokenBlockState {
    InvalidPrimary = 0,
    InvalidSecondary = 1,
    InvalidatingSecondary = 2,
    Locked = 3,
    Unlocked = 4,
}

impl TokenBlockHeader {
    /// A header indicating a source token is invalid.
    pub const INVALID_SOURCE: TokenBlockHeader =
        TokenBlockHeader((TokenBlockState::InvalidPrimary as u64) << ExtTokenId::NUM_BITS);

    /// A header indicating a dependent token has been invalidated from the primary dependency
    /// list.
    pub const INVALID_DEP_PRIMARY: TokenBlockHeader = TokenBlockHeader(
        (TokenBlockState::InvalidPrimary as u64) << ExtTokenId::NUM_BITS | 1 << TokenId::NUM_BITS,
    );

    /// A header indicating a dependent token has been invalidated from the secondary dependency
    /// tree, but not the primary dependency list.
    pub const INVALID_DEP_SECONDARY: TokenBlockHeader = TokenBlockHeader(
        (TokenBlockState::InvalidSecondary as u64) << ExtTokenId::NUM_BITS | 1 << TokenId::NUM_BITS,
    );

    /// A header indicating a dependent token is being invalidated from the secondary dependency
    /// tree, but not the primary dependency list.
    pub const INVALIDATING_DEP_SECONDARY: TokenBlockHeader = TokenBlockHeader(
        (TokenBlockState::InvalidatingSecondary as u64) << ExtTokenId::NUM_BITS
            | 1 << TokenId::NUM_BITS,
    );

    /// Constructs a header for a token that is unlocked and valid.
    pub const fn unlocked(ext_id: ExtTokenId) -> Self {
        TokenBlockHeader(((TokenBlockState::Unlocked as u64) << ExtTokenId::NUM_BITS) | ext_id.0)
    }

    /// Constructs a header for a token that is locked and valid.
    pub const fn locked(ext_id: ExtTokenId) -> Self {
        TokenBlockHeader(((TokenBlockState::Locked as u64) << ExtTokenId::NUM_BITS) | ext_id.0)
    }

    /// Gets the [`TokenBlockState`] for this header.
    pub fn state(&self) -> TokenBlockState {
        unsafe { std::mem::transmute((self.0 >> ExtTokenId::NUM_BITS) as u8) }
    }

    /// Gets the [`TokenId`] associated with the block.
    pub fn id(&self) -> TokenId {
        TokenId(self.0 & ((1 << TokenId::NUM_BITS) - 1))
    }

    /// Gets the [`ExtTokenId`] associated with the block.
    pub fn ext_id(&self) -> ExtTokenId {
        ExtTokenId(self.0 & ((1 << ExtTokenId::NUM_BITS) - 1))
    }

    /// Determines whether this is a header for a [`DependentTokenBlock`].
    pub fn is_dependent(&self) -> bool {
        self.ext_id().is_dependent()
    }

    /// Determines whether this the block describes a valid token.
    pub fn is_valid(&self) -> bool {
        matches!(
            self.state(),
            TokenBlockState::Locked | TokenBlockState::Unlocked
        )
    }
}

unsafe impl PrimRepr for TokenBlockHeader {
    type Prim = u64;
}

/// A data block which defines a [`Token`].
#[repr(C)]
pub struct TokenBlock {
    /// Contains the current [`TokenId`] for this block, as well as a few extra flags. If the
    /// `DEP_FLAG` is set, this [`TokenBlock`] must be the `base` of a [`DependentTokenBlock`].
    header: Atomic<TokenBlockHeader>,
    secondary_child_id: Atomic<TokenId>,
    primary_child: Atomic<Option<&'static DependentTokenBlock>>,
    secondary_child_block: Atomic<Option<DependentBlockRef<'static>>>,
}

/// Extends [`TokenBlock`] for a token constructed as the dependent of two [`Token`]s.
#[repr(C)]
struct DependentTokenBlock {
    base: TokenBlock,

    /// Describes the range of source [`TokenId`]s this token is sensitive to.
    dep_range: Atomic<DependentRange>,

    /// A reference to the [`TokenBlock`] which describes the "left" parent of this token. The
    /// left parent is only sensitive to source [`TokenId`]s which are less than the splitting
    /// `TokenId` of `dep_range`.
    left_parent: Atomic<TokenBlockRef<'static>>,

    /// A reference to the [`TokenBlock`] which describes the "right" parent of this token. The
    /// right parent is sensitive to source [`TokenId`]s which are greater than or equal to the
    /// splitting `TokenId` of `dep_range`.
    right_parent: Atomic<TokenBlockRef<'static>>,

    /// The next [`DependentTokenBlock`] in the dependency list of the primary parent for this
    /// token, or [`None`] if this is the last block in the list.
    next_primary_sibling: Atomic<Option<&'static DependentTokenBlock>>,
}

/// A data block which acts as a branch node in the secondary dependency tree.
#[repr(C)]
struct DependentTreeBlock {
    ids: [Atomic<TokenId>; 1 << Self::SEARCH_BITS],
    blocks: [Atomic<Option<DependentBlockRef<'static>>>; 1 << Self::SEARCH_BITS],
}

impl Default for TokenBlock {
    fn default() -> Self {
        Self {
            header: Atomic::new(TokenBlockHeader::INVALID_SOURCE),
            secondary_child_id: Atomic::new(TokenId::ALWAYS),
            primary_child: Atomic::new(None),
            secondary_child_block: Atomic::new(None),
        }
    }
}

impl Default for DependentTokenBlock {
    fn default() -> Self {
        Self {
            base: Default::default(),
            dep_range: Atomic::new(DependentRange(0)),
            left_parent: Atomic::new(TokenBlockRef::from_ref(&ALWAYS_NEVER)),
            right_parent: Atomic::new(TokenBlockRef::from_ref(&ALWAYS_NEVER)),
            next_primary_sibling: Atomic::new(None),
        }
    }
}

impl Default for DependentTreeBlock {
    fn default() -> Self {
        Self {
            ids: core::array::from_fn(|_| Atomic::new(TokenId::ALWAYS)),
            blocks: core::array::from_fn(|_| Atomic::new(None)),
        }
    }
}

impl DependentTreeBlock {
    /// The number of bits of the search key consumed by each [`DependentTreeBlock`].
    const SEARCH_BITS: u8 = 2;

    /// A mask used to convert a search key into an index into a [`DependentTreeBlock`].
    const SEARCH_MASK: u64 = (1 << Self::SEARCH_BITS) - 1;
}

/// A reference to a [`TokenBlock`] or a [`DependentTokenBlock`].
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct TokenBlockRef<'a> {
    _marker: PhantomData<&'a ()>,
    ptr: NonNull<TokenBlock>,
}

unsafe impl Sync for TokenBlockRef<'_> {}

unsafe impl Send for TokenBlockRef<'_> {}

unsafe impl PrimRepr for TokenBlockRef<'_> {
    type Prim = NonNull<TokenBlock>;
}

impl<'a> From<&'a TokenBlock> for TokenBlockRef<'a> {
    fn from(value: &'a TokenBlock) -> Self {
        debug_assert!(!value.header.load(Relaxed).is_dependent());
        Self::from_ref(value)
    }
}

impl<'a> From<&'a DependentTokenBlock> for TokenBlockRef<'a> {
    fn from(value: &'a DependentTokenBlock) -> Self {
        debug_assert!(value.base.header.load(Relaxed).is_dependent());
        unsafe { std::mem::transmute(value) }
    }
}

impl<'a> TokenBlockRef<'a> {
    /// Constructs a [`TokenBlockRef`].
    pub const fn from_ref(source: &'a TokenBlock) -> Self {
        unsafe { std::mem::transmute(source) }
    }

    /// Gets the underlying [`TokenBlock`] for this reference.
    pub fn as_ref(&self) -> &'a TokenBlock {
        unsafe { self.ptr.as_ref() }
    }
}

impl TokenBlockRef<'static> {
    /// Gets the [`Token`] currently represented by this block, or [`None`] if it is invalid.
    pub fn token(&self) -> Option<Token> {
        let header = self.as_ref().header.load(Relaxed);
        if header.is_valid() {
            Some(Token {
                block: *self,
                ext_id: header.ext_id(),
            })
        } else {
            None
        }
    }
}

/// A reference to a [`DependentTokenBlock`] or a [`DependentTreeBlock`].
#[repr(transparent)]
#[derive(Clone, Copy)]
struct DependentBlockRef<'a> {
    _marker: PhantomData<&'a ()>,
    ptr: NonNull<Atomic<TokenBlockHeader>>,
}

unsafe impl Sync for DependentBlockRef<'_> {}

unsafe impl Send for DependentBlockRef<'_> {}

unsafe impl PrimRepr for DependentBlockRef<'_> {
    type Prim = NonNull<Atomic<TokenBlockHeader>>;
}

impl<'a> From<&'a DependentTokenBlock> for DependentBlockRef<'a> {
    fn from(value: &'a DependentTokenBlock) -> Self {
        debug_assert!(value.base.header.load(Relaxed).is_dependent());
        unsafe { std::mem::transmute(value) }
    }
}

impl<'a> From<&'a DependentTreeBlock> for DependentBlockRef<'a> {
    fn from(value: &'a DependentTreeBlock) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl<'a> DependentBlockRef<'a> {
    /// Disambiguates a [`DependentBlockRef`] as a reference to a [`DependentTokenBlock`] or a
    /// [`DependentTreeBlock`].
    pub fn as_token_or_block(&self) -> Result<&'a DependentTokenBlock, &'a DependentTreeBlock> {
        let header = unsafe { self.ptr.as_ref() }.load(Relaxed);
        if header.ext_id().is_dependent() {
            // SAFETY: A `DependentTokenBlock` always has a header with the dependent flag set.
            // A `DependentTreeBlock` never does. Thus, we know this is a `DependentTokenBlock`.
            Ok(unsafe { self.ptr.cast::<DependentTokenBlock>().as_ref() })
        } else {
            // SAFETY: A `DependentTokenBlock` always has a header with the dependent flag set.
            // A `DependentTreeBlock` never does. Thus, we know this is a `DependentTreeBlock`.
            Err(unsafe { self.ptr.cast::<DependentTreeBlock>().as_ref() })
        }
    }
}

impl Token {
    /// Gets the [`DependentTokenBlock`] for this token, or returns [`None`] if it is not a
    /// dependent token.
    fn dependent_block(&self) -> Option<&'static DependentTokenBlock> {
        if self.ext_id.is_dependent() {
            // SAFETY: Since the dependent flag is set, the token must be dependent. Thus,
            // `self.block` is the `base` for a `DependentTokenBlock`. Due to the layout required
            // by `repr(C)`, this should be at the same location as the outer block.
            Some(unsafe { std::mem::transmute(self.block) })
        } else {
            None
        }
    }
}

impl DependentTokenBlock {
    /// Attempts to get the [`DependentInfo`] for a token using the block. Returns [`None`] if
    /// the token is invalid.
    fn dependent_info(&self, id: TokenId) -> Option<DependentInfo> {
        let left_parent = self.left_parent.load(Relaxed);
        let right_parent = self.right_parent.load(Relaxed);
        let dep_range = self.dep_range.load(Relaxed);
        fence(Acquire);
        if self.base.header.load(Relaxed).id() == id {
            Some(DependentInfo {
                left_parent,
                right_parent,
                dep_range,
            })
        } else {
            None
        }
    }
}

/// Provides dependent-specific information about a dependent [`Token`].
struct DependentInfo {
    left_parent: TokenBlockRef<'static>,
    right_parent: TokenBlockRef<'static>,
    dep_range: DependentRange,
}

/// The [`TokenBlock`] used to implement the "always" and "never" tokens.
static ALWAYS_NEVER: TokenBlock = TokenBlock {
    header: Atomic::new(TokenBlockHeader::unlocked(ExtTokenId::ALWAYS)),
    secondary_child_id: Atomic::new(TokenId::ALWAYS), // Unused
    primary_child: Atomic::new(None),
    secondary_child_block: Atomic::new(None),
};

/// Gets a [`Token`] which is always valid.
pub fn always() -> Token {
    Token {
        block: TokenBlockRef::from_ref(&ALWAYS_NEVER),
        ext_id: ExtTokenId::ALWAYS,
    }
}

/// Gets a [`Token`] which is never valid.
pub fn never() -> Token {
    Token {
        block: TokenBlockRef::from_ref(&ALWAYS_NEVER),
        ext_id: ExtTokenId::NEVER,
    }
}

/// Checks whether a [`Token`] is valid.
pub fn is_valid(token: Token) -> bool {
    token.block.as_ref().header.load(Relaxed).ext_id() == token.ext_id
}

/// Creates a new source [`Token`].
pub fn source(alloc: &mut ThreadAllocator) -> Token {
    let id = alloc.alloc_token_id();
    let block = alloc.alloc_source_token_block();
    let ext_id = ExtTokenId::new(id, false);
    debug_assert!(block.header.load(Relaxed) == TokenBlockHeader::INVALID_SOURCE);
    debug_assert!(block.primary_child.load(Relaxed).is_none());
    debug_assert!(block.secondary_child_block.load(Relaxed).is_none());
    let n_header = TokenBlockHeader::unlocked(ext_id);
    block.header.store(n_header, Relaxed);
    Token {
        block: block.into(),
        ext_id,
    }
}

/// Gets or creates a [`Token`] which is valid precisely when both of the given [`Token`]s are
/// valid.
pub fn dependent(alloc: &mut ThreadAllocator, a: Token, b: Token) -> Token {
    if a.ext_id == ExtTokenId::ALWAYS {
        b
    } else if b.ext_id == ExtTokenId::ALWAYS {
        a
    } else {
        dependent_non_always(alloc, a, b)
    }
}

/// Gets or creates a [`Token`] which is dependent on the given [`Token`]s that are known not
/// to be [`always`].
fn dependent_non_always(alloc: &mut ThreadAllocator, a: Token, b: Token) -> Token {
    let a_id = TokenId::from(a.ext_id);
    if let Some(a_block) = a.dependent_block() {
        if let Some(a_info) = a_block.dependent_info(a_id) {
            dependent_dep(alloc, a_block, a_info, a_id, b)
        } else {
            never()
        }
    } else {
        dependent_source(alloc, a.block.as_ref(), a_id, b)
    }
}

/// Gets or creates a [`Token`] which is dependent on a source [`Token`] and some other [`Token`].
fn dependent_source(
    alloc: &mut ThreadAllocator,
    a_block: &'static TokenBlock,
    a_id: TokenId,
    b: Token,
) -> Token {
    if let Some(b_block) = b.dependent_block() {
        let b_id = b.ext_id.into();
        if let Some(b_info) = b_block.dependent_info(b_id) {
            dependent_dep_source(alloc, b_block, b_info, b.ext_id.into(), a_block, a_id)
        } else {
            never()
        }
    } else {
        let b_id = TokenId::from(b.ext_id);
        match a_id.cmp(&b_id) {
            Less => dependent_exact(
                alloc,
                a_block.into(),
                ExtTokenId::new(a_id, false),
                b.block,
                b.ext_id,
                DependentRange::between(a_id, b_id),
            ),
            Equal => b,
            Greater => dependent_exact(
                alloc,
                b.block,
                b.ext_id,
                a_block.into(),
                ExtTokenId::new(a_id, false),
                DependentRange::between(b_id, a_id),
            ),
        }
    }
}

/// Gets or creates a [`Token`] which is dependent on a dependent [`Token`] and some other
/// [`Token`].
fn dependent_dep(
    alloc: &mut ThreadAllocator,
    a_block: &'static DependentTokenBlock,
    a_info: DependentInfo,
    a_id: TokenId,
    b: Token,
) -> Token {
    if let Some(b_block) = b.dependent_block() {
        let b_id = TokenId::from(b.ext_id);
        let Some(b_info) = b_block.dependent_info(b_id) else {
            return never();
        };
        let left_block;
        let left_info;
        let left_id;
        let right_block;
        let right_info;
        let right_id;
        match a_info.dep_range.split().cmp(&b_info.dep_range.split()) {
            Less => {
                left_block = a_block;
                left_info = a_info;
                left_id = a_id;
                right_block = b_block;
                right_info = b_info;
                right_id = b_id;
            }
            Equal => {
                let Some(a_left_parent) = a_info.left_parent.token() else {
                    return never();
                };
                let Some(a_right_parent) = a_info.right_parent.token() else {
                    return never();
                };
                let Some(b_left_parent) = b_info.left_parent.token() else {
                    return never();
                };
                let Some(b_right_parent) = b_info.right_parent.token() else {
                    return never();
                };
                let left = dependent_non_always(alloc, a_left_parent, b_left_parent);
                let right = dependent_non_always(alloc, a_right_parent, b_right_parent);
                return dependent_exact(
                    alloc,
                    left.block,
                    left.ext_id,
                    right.block,
                    right.ext_id,
                    DependentRange::new(
                        a_info.dep_range.split(),
                        u8::max(a_info.dep_range.scale(), b_info.dep_range.scale()),
                    ),
                );
            }
            Greater => {
                left_block = b_block;
                left_info = b_info;
                left_id = b_id;
                right_block = a_block;
                right_info = a_info;
                right_id = a_id;
            }
        }
        if left_info.dep_range.max_bound() < right_info.dep_range.min_bound() {
            dependent_exact(
                alloc,
                left_block.into(),
                ExtTokenId::new(left_id, true),
                right_block.into(),
                ExtTokenId::new(right_id, true),
                DependentRange::between(
                    left_info.dep_range.min_bound(),
                    right_info.dep_range.max_bound(),
                ),
            )
        } else if left_info.dep_range.max_bound() < right_info.dep_range.split() {
            debug_assert!(right_info.dep_range.min_bound() <= left_info.dep_range.min_bound());
            let Some(right_left) = right_info.left_parent.token() else {
                return never();
            };
            let Some(right_right) = right_info.right_parent.token() else {
                return never();
            };
            let left = dependent_dep(alloc, left_block, left_info, left_id, right_left);
            dependent_exact(
                alloc,
                left.block,
                left.ext_id,
                right_right.block,
                right_right.ext_id,
                right_info.dep_range,
            )
        } else {
            debug_assert!(left_info.dep_range.split() <= right_info.dep_range.min_bound());
            debug_assert!(right_info.dep_range.max_bound() <= left_info.dep_range.max_bound());
            let Some(left_left) = left_info.left_parent.token() else {
                return never();
            };
            let Some(left_right) = left_info.right_parent.token() else {
                return never();
            };
            let right = dependent_dep(alloc, right_block, right_info, right_id, left_right);
            dependent_exact(
                alloc,
                left_left.block,
                left_left.ext_id,
                right.block,
                right.ext_id,
                left_info.dep_range,
            )
        }
    } else {
        dependent_dep_source(
            alloc,
            a_block,
            a_info,
            a_id,
            b.block.as_ref(),
            b.ext_id.into(),
        )
    }
}

/// Gets or creates a [`Token`] which is dependent on a dependent [`Token`] and a source [`Token`].
fn dependent_dep_source(
    alloc: &mut ThreadAllocator,
    a_block: &'static DependentTokenBlock,
    a_info: DependentInfo,
    a_id: TokenId,
    b_block: &'static TokenBlock,
    b_id: TokenId,
) -> Token {
    #[allow(clippy::collapsible_else_if)]
    if a_info.dep_range.split() <= b_id {
        if b_id <= a_info.dep_range.max_bound() {
            let Some(a_left) = a_info.left_parent.token() else {
                return never();
            };
            let Some(a_right) = a_info.right_parent.token() else {
                return never();
            };
            let right = dependent_source(alloc, b_block, b_id, a_right);
            dependent_exact(
                alloc,
                a_left.block,
                a_left.ext_id,
                right.block,
                right.ext_id,
                a_info.dep_range,
            )
        } else {
            dependent_exact(
                alloc,
                a_block.into(),
                ExtTokenId::new(a_id, true),
                b_block.into(),
                ExtTokenId::new(b_id, false),
                DependentRange::between(a_info.dep_range.min_bound(), b_id),
            )
        }
    } else {
        if b_id >= a_info.dep_range.min_bound() {
            let Some(a_left) = a_info.left_parent.token() else {
                return never();
            };
            let Some(a_right) = a_info.right_parent.token() else {
                return never();
            };
            let left = dependent_source(alloc, b_block, b_id, a_left);
            dependent_exact(
                alloc,
                left.block,
                left.ext_id,
                a_right.block,
                a_right.ext_id,
                a_info.dep_range,
            )
        } else {
            dependent_exact(
                alloc,
                b_block.into(),
                ExtTokenId::new(b_id, false),
                a_block.into(),
                ExtTokenId::new(a_id, true),
                DependentRange::between(b_id, a_info.dep_range.max_bound()),
            )
        }
    }
}

/// Gets or creates a [`Token`] which is dependent on *exactly* the two given tokens. The tokens
/// must be provided in order such that the [`DependentRange`] of the `left` token is less than,
/// and not overlapping with, the [`DependentRange`] of the `right` token.
fn dependent_exact(
    alloc: &mut ThreadAllocator,
    left_block: TokenBlockRef<'static>,
    left_ext_id: ExtTokenId,
    right_block: TokenBlockRef<'static>,
    right_ext_id: ExtTokenId,
    dep_range: DependentRange,
) -> Token {
    let primary_block = left_block.as_ref();
    let primary_ext_id = left_ext_id;
    let secondary_block_ref = right_block;
    let secondary_block = secondary_block_ref.as_ref();
    let secondary_ext_id = right_ext_id;

    // Check whether such a token already exists by searching the secondary dependency tree
    if let Ok(res) = search_secondary(primary_ext_id.into(), secondary_block) {
        // Make sure to validate secondary token before returning. The prior search could return
        // invalid results otherwise.
        fence(Acquire);
        if secondary_block.header.load(Relaxed).ext_id() == secondary_ext_id {
            return res;
        } else {
            return never();
        }
    }

    // Lock the secondary token to add a new dependency
    loop {
        let header = secondary_block.header.compare_exchange_weak(
            TokenBlockHeader::unlocked(secondary_ext_id),
            TokenBlockHeader::locked(secondary_ext_id),
            Acquire,
            Relaxed,
        );
        match header {
            Ok(_) => break,
            Err(header) => {
                if header.ext_id() == secondary_ext_id {
                    // Token could be locked by another thread. Try again
                    std::hint::spin_loop();
                    continue;
                } else {
                    // Token has been invalidated
                    return never();
                }
            }
        }
    }

    // Find where the the new token should go in the dependency tree
    let slot = match search_secondary(primary_ext_id.into(), secondary_block) {
        Ok(res) => {
            // The token was created by another thread while we were locking the secondary token.
            todo!()
        }
        Err(slot) => slot,
    };

    // Lock the primary token to add a new dependency
    loop {
        let header = primary_block.header.compare_exchange_weak(
            TokenBlockHeader::unlocked(primary_ext_id),
            TokenBlockHeader::locked(primary_ext_id),
            Acquire,
            Relaxed,
        );
        match header {
            Ok(_) => break,
            Err(header) => {
                if header.ext_id() == primary_ext_id {
                    // Token could be locked by another thread. Try again
                    std::hint::spin_loop();
                    continue;
                } else {
                    // Token has been invalidated
                    unlock(alloc, secondary_block_ref, secondary_ext_id);
                    return never();
                }
            }
        }
    }

    // Create new token
    let res_id = alloc.alloc_token_id();
    let res_ext_id = ExtTokenId::new(res_id, true);
    let res_block = alloc.alloc_dependent_token_block();
    res_block
        .base
        .header
        .store(TokenBlockHeader::unlocked(res_ext_id), Relaxed);
    debug_assert!(res_block.base.primary_child.load(Relaxed).is_none());
    debug_assert!(res_block.base.secondary_child_block.load(Relaxed).is_none());
    res_block.dep_range.store(dep_range, Relaxed);
    res_block.left_parent.store(left_block, Relaxed);
    res_block.right_parent.store(right_block, Relaxed);

    // Add token to the primary dependency list
    let old_primary_child = primary_block.primary_child.swap(Some(res_block), Relaxed);
    res_block
        .next_primary_sibling
        .store(old_primary_child, Relaxed);

    // Unlock the primary token
    let header = primary_block.header.compare_exchange(
        TokenBlockHeader::locked(primary_ext_id),
        TokenBlockHeader::unlocked(primary_ext_id),
        Release,
        Relaxed,
    );
    if let Err(header) = header {
        debug_assert!(header.ext_id() != primary_ext_id);

        // Primary token has been invalidated while we were adding to the dependency list.
        // We need to propogate invalidation from the primary token before returning.
        todo!();
        return never();
    }

    // Add token to secondary dependency tree
    slot.insert(alloc, res_block, res_id);

    // Unlock the secondary token
    let header = secondary_block.header.compare_exchange(
        TokenBlockHeader::locked(secondary_ext_id),
        TokenBlockHeader::unlocked(secondary_ext_id),
        Release,
        Relaxed,
    );
    if let Err(header) = header {
        debug_assert!(header.ext_id() != secondary_ext_id);

        // Secondary token has been invalidated while we were adding to the dependency tree.
        // We need to propogate invalidation from the secondary token before returning.
        todo!();
        return never();
    }

    // Return the new token
    Token {
        block: res_block.into(),
        ext_id: res_ext_id,
    }
}

/// Searches through a secondary dependent tree to find a dependent token with a particular primary
/// parent. Returns either the token, or the slot where it should be inserted.
fn search_secondary(
    primary_parent_id: TokenId,
    secondary_parent_block: &'static TokenBlock,
) -> Result<Token, TokenSlot> {
    let mut slot_id = &secondary_parent_block.secondary_child_id;
    let mut slot_block = &secondary_parent_block.secondary_child_block;
    let mut search_bit_depth = 0;
    loop {
        if let Some(cur_block) = slot_block.load(Relaxed) {
            match cur_block.as_token_or_block() {
                Ok(cur_block) => {
                    let cur_id = slot_id.load(Relaxed);
                    let primary_parent = cur_block.left_parent.load(Relaxed);
                    let primary_parent_header = primary_parent.as_ref().header.load(Relaxed);
                    fence(Acquire);
                    let cur_header = cur_block.base.header.load(Relaxed);

                    // Verify that the block is still valid
                    if cur_header.id() == cur_id {
                        // Verify that the block has the expected parent
                        let cur_primary_parent_id = primary_parent_header.id();
                        if cur_primary_parent_id == primary_parent_id {
                            return Ok(Token {
                                block: cur_block.into(),
                                ext_id: ExtTokenId::new(cur_id, true),
                            });
                        } else {
                            return Err(TokenSlot::Occupied {
                                block: slot_block,
                                cur_id,
                                cur_block,
                                cur_search_suffix: cur_primary_parent_id.0 >> search_bit_depth,
                                new_search_suffix: primary_parent_id.0 >> search_bit_depth,
                            });
                        }
                    } else {
                        return Err(TokenSlot::Vacant {
                            id: slot_id,
                            block: slot_block,
                        });
                    }
                }
                Err(cur_block) => {
                    let search_index = ((primary_parent_id.0 >> search_bit_depth)
                        & DependentTreeBlock::SEARCH_MASK)
                        as usize;
                    slot_id = &cur_block.ids[search_index];
                    slot_block = &cur_block.blocks[search_index];
                    search_bit_depth += DependentTreeBlock::SEARCH_BITS;
                    continue;
                }
            }
        } else {
            return Err(TokenSlot::Vacant {
                id: slot_id,
                block: slot_block,
            });
        }
    }
}

/// Unlocks a [`TokenBlock`], propogating invalidation if needed.
fn unlock(alloc: &mut ThreadAllocator, block: TokenBlockRef<'static>, ext_id: ExtTokenId) {
    let header = block.as_ref().header.compare_exchange(
        TokenBlockHeader::locked(ext_id),
        TokenBlockHeader::unlocked(ext_id),
        Release,
        Relaxed,
    );
    match header {
        Ok(_) => (),
        Err(header) => {
            debug_assert!(header.ext_id() != ext_id);
            todo!()
        }
    }
}

/// Describes where a [`Token`] should be inserted into a secondary dependency tree.
enum TokenSlot {
    /// The slot is empty.
    Vacant {
        /// The location where the identifier is stored for the slot.
        id: &'static Atomic<TokenId>,

        /// The location where the reference is stored for the slot.
        block: &'static Atomic<Option<DependentBlockRef<'static>>>,
    },

    /// The slot is occupied by a [`DependentTokenBlock`].
    Occupied {
        /// The location where the reference is stored for the slot.
        block: &'static Atomic<Option<DependentBlockRef<'static>>>,

        /// The [`TokenId`] for the token currently occupying the slot.
        cur_id: TokenId,

        /// The block currently occupying the slot.
        cur_block: &'static DependentTokenBlock,

        /// The suffix of the current token's primary parent's [`TokenId`] which has not yet
        /// been considered in the search tree.
        cur_search_suffix: u64,

        /// The suffix of the requested token's primary parent's [`TokenId`] which has not yet
        /// been considered in the search tree.
        new_search_suffix: u64,
    },
}

impl TokenSlot {
    /// Inserts a dependent [`Token`] into this slot.
    pub fn insert(
        self,
        alloc: &mut ThreadAllocator,
        n_block: &'static DependentTokenBlock,
        n_id: TokenId,
    ) {
        match self {
            TokenSlot::Vacant { id, block } => {
                id.store(n_id, Relaxed);
                block.store(Some(n_block.into()), Relaxed);
            }
            TokenSlot::Occupied {
                mut block,
                cur_id,
                cur_block,
                mut cur_search_suffix,
                mut new_search_suffix,
            } => loop {
                let tree_block = alloc.alloc_dependent_tree_block();
                for i in 0..(1 << DependentTreeBlock::SEARCH_BITS) {
                    debug_assert!(tree_block.blocks[i].load(Relaxed).is_none());
                }
                block.store(Some(tree_block.into()), Relaxed);
                let cur_index = (cur_search_suffix & DependentTreeBlock::SEARCH_MASK) as usize;
                let new_index = (new_search_suffix & DependentTreeBlock::SEARCH_MASK) as usize;
                if cur_index == new_index {
                    block = &tree_block.blocks[cur_index];
                    cur_search_suffix >>= DependentTreeBlock::SEARCH_BITS;
                    new_search_suffix >>= DependentTreeBlock::SEARCH_BITS;
                } else {
                    tree_block.ids[cur_index].store(cur_id, Relaxed);
                    tree_block.blocks[cur_index].store(Some(cur_block.into()), Relaxed);
                    tree_block.ids[new_index].store(n_id, Relaxed);
                    tree_block.blocks[new_index].store(Some(n_block.into()), Relaxed);
                    break;
                }
            },
        }
    }
}

/// Ensures that a source [`Token`] has been invalidated.
pub fn invalidate_source(
    alloc: &mut ThreadAllocator,
    block: &'static TokenBlock,
    ext_id: ExtTokenId,
) {
    // Lock for invalidation
    let mut exp_header = TokenBlockHeader::unlocked(ext_id);
    loop {
        let header = block.header.compare_exchange_weak(
            exp_header,
            TokenBlockHeader::INVALID_SOURCE,
            AcqRel,
            Relaxed,
        );
        match header {
            Ok(_) => break,
            Err(header) => {
                if header.ext_id() == ext_id {
                    // Try again (if the block is locked, set it to invalid anyway to signal the
                    // holder to invalidate it)
                    exp_header = header;
                    continue;
                } else {
                    // The token has already been invalidated
                    return;
                }
            }
        }
    }

    // Invalidate children
    invalidate_children(alloc, block);

    // Return to allocator
    alloc.free_source_token_block(block);
}

/// Assuming that a [`TokenBlock`] has been locked, invalidates all children in both the primary
/// dependency list and secondary dependency tree.
fn invalidate_children(alloc: &mut ThreadAllocator, block: &'static TokenBlock) {
    if let Some(primary_child) = block.primary_child.load(Relaxed) {
        block.primary_child.store(None, Relaxed);
        invalidate_primary(alloc, primary_child);
    }
    if let Some(secondary_child_block) = block.secondary_child_block.load(Relaxed) {
        block.secondary_child_block.store(None, Relaxed);
        invalidate_secondary(alloc, secondary_child_block, &block.secondary_child_id);
    }
}

/// Invalidates all dependencies in a primary dependency list, starting from the given block.
fn invalidate_primary(alloc: &mut ThreadAllocator, mut block: &'static DependentTokenBlock) {
    loop {
        // Lock for invalidation
        let header = block
            .base
            .header
            .swap(TokenBlockHeader::INVALID_DEP_PRIMARY, Release);
        match header.state() {
            TokenBlockState::InvalidPrimary => {
                // Should not be possible. A token may only be invalidated from the primary list
                // once.
                debug_assert!(false);
                unsafe { std::hint::unreachable_unchecked() }
            }
            TokenBlockState::InvalidSecondary => {
                // The token was already invalidated from the secondary dependency tree, but
                // we still need to traverse the rest of the primary dependency list.
            }
            TokenBlockState::InvalidatingSecondary | TokenBlockState::Locked => {
                // The token was already locked by another thread. The holder of the lock is now
                // responsible for continuing the invalidation.
                return;
            }
            TokenBlockState::Unlocked => {
                // Invalidate children
                invalidate_children(alloc, &block.base);
            }
        }

        // Free the block
        let next = block.next_primary_sibling.load(Relaxed);
        block.next_primary_sibling.store(None, Relaxed);
        alloc.free_dependent_token_block(block);

        // Continue traversing the primary dependency list
        if let Some(next) = next {
            block = next;
            continue;
        } else {
            return;
        }
    }
}

/// Invalidates all dependencies in a secondary dependency tree.
fn invalidate_secondary(
    alloc: &mut ThreadAllocator,
    block: DependentBlockRef<'static>,
    id: &Atomic<TokenId>,
) {
    match block.as_token_or_block() {
        Ok(block) => {
            let id = id.load(Relaxed);
            let ext_id = ExtTokenId::new(id, true);

            // Lock for invalidation
            let mut exp_header = TokenBlockHeader::unlocked(ext_id);
            loop {
                let header = block.base.header.compare_exchange_weak(
                    exp_header,
                    TokenBlockHeader::INVALIDATING_DEP_SECONDARY,
                    Release,
                    Relaxed,
                );
                match header {
                    Ok(header) => {
                        if let TokenBlockState::Locked = header.state() {
                            // The token was already locked by another thread. The holder of the
                            // lock is now responsible for continuing the invalidation.
                            return;
                        } else {
                            break;
                        }
                    }
                    Err(header) => {
                        if header.ext_id() == ext_id {
                            // Try again (if the block is locked, set it to invalid anyway to
                            // signal the holder to invalidate it)
                            exp_header = header;
                            continue;
                        } else {
                            // The token has already been invalidated
                            return;
                        }
                    }
                }
            }

            // Invalidate children
            invalidate_children(alloc, &block.base);

            // Unlock
            let header = block.base.header.compare_exchange(
                TokenBlockHeader::INVALIDATING_DEP_SECONDARY,
                TokenBlockHeader::INVALID_DEP_SECONDARY,
                Relaxed,
                Relaxed,
            );
            match header {
                Ok(_) => {
                    // We don't actually free the block here because it may still be referenced in
                    // the primary dependency list.
                }
                Err(header) => {
                    debug_assert!(header == TokenBlockHeader::INVALID_DEP_PRIMARY);

                    // While we were doing an invalidation from the secondary parent, the token
                    // was also invalidated from the primary parent, and we are now responsible
                    // for propagating that invalidation.
                    todo!()
                }
            }
        }
        Err(block) => {
            // Invalidate children
            for i in 0..(1 << DependentTreeBlock::SEARCH_BITS) {
                if let Some(child) = block.blocks[i].load(Relaxed) {
                    block.blocks[i].store(None, Relaxed);
                    invalidate_secondary(alloc, child, &block.ids[i]);
                }
            }

            // Free the block
            alloc.free_dependent_tree_block(block);
        }
    }
}

/// Iterates over all source [`Token`]s that a given [`Token`] is dependent on, or returns
/// `false` if the token is invalid. This may return `false` even after some tokens have been
/// returned.
pub fn dependencies(token: Token, mut f: impl FnMut(Token)) -> bool {
    return if token.ext_id == ExtTokenId::ALWAYS {
        true
    } else {
        inner(token, &mut f)
    };
    fn inner(token: Token, f: &mut impl FnMut(Token)) -> bool {
        if let Some(block) = token.dependent_block() {
            if let Some(info) = block.dependent_info(token.ext_id.into()) {
                if let Some(left) = info.left_parent.token() {
                    if !inner(left, f) {
                        return false;
                    }
                }
                if let Some(right) = info.right_parent.token() {
                    if !inner(right, f) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        } else if is_valid(token) {
            f(token);
            true
        } else {
            false
        }
    }
}

/// Contains thread-specific information used to allocate and free [`Block`] and [`Token`]s.
pub struct ThreadAllocator {
    next_token_id: u64,
    last_reserved_token_id: u64,
    free_source_token_blocks: Vec<&'static TokenBlock>,
    free_dependent_token_blocks: Vec<&'static DependentTokenBlock>,
    free_dependent_tree_blocks: Vec<&'static DependentTreeBlock>,
}

thread_local! {
    static ALLOC: RefCell<ThreadAllocator> = RefCell::new(ThreadAllocator {
        next_token_id: 0,
        last_reserved_token_id: 0,
        free_source_token_blocks: Vec::new(),
        free_dependent_token_blocks: Vec::new(),
        free_dependent_tree_blocks: Vec::new(),
    });
}

impl ThreadAllocator {
    /// Calls the given function with the [`ThreadAllocator`] for this thread.
    pub fn with<R>(f: impl FnOnce(&mut ThreadAllocator) -> R) -> R {
        ALLOC.with(|cell| f(&mut cell.borrow_mut()))
    }

    /// Gets an unused [`TokenId`].
    pub fn alloc_token_id(&mut self) -> TokenId {
        if self.next_token_id < self.last_reserved_token_id {
            let res = self.next_token_id;
            self.next_token_id += 1;
            TokenId(res)
        } else {
            let res = NEXT_TOKEN_ID.fetch_add(NUM_RESERVED_TOKEN_IDS, Relaxed);
            self.next_token_id = res + 1;
            self.last_reserved_token_id = res + NUM_RESERVED_TOKEN_IDS;
            TokenId(res)
        }
    }

    /// Gets an unused non-dependent [`TokenBlock`].
    fn alloc_source_token_block(&mut self) -> &'static TokenBlock {
        alloc_block(128, &mut self.free_source_token_blocks)
    }

    /// Reclaims an unused non-dependent [`TokenBlock`].
    fn free_source_token_block(&mut self, block: &'static TokenBlock) {
        self.free_source_token_blocks.push(block);
    }

    /// Gets an unused [`DependentTokenBlock`].
    fn alloc_dependent_token_block(&mut self) -> &'static DependentTokenBlock {
        alloc_block(64, &mut self.free_dependent_token_blocks)
    }

    /// Reclaims an unused  [`DependentTokenBlock`].
    fn free_dependent_token_block(&mut self, block: &'static DependentTokenBlock) {
        self.free_dependent_token_blocks.push(block);
    }

    /// Gets an unused [`DependentTreeBlock`].
    fn alloc_dependent_tree_block(&mut self) -> &'static DependentTreeBlock {
        alloc_block(64, &mut self.free_dependent_tree_blocks)
    }

    /// Reclaims an unused [`DependentTreeBlock`].
    fn free_dependent_tree_block(&mut self, block: &'static DependentTreeBlock) {
        self.free_dependent_tree_blocks.push(block);
    }
}

/// The number of [`TokenId`]s that should be reserved by a [`ThreadAllocator`] at a time.
const NUM_RESERVED_TOKEN_IDS: u64 = 32;

/// The global next unused [`TokenId`].
static NEXT_TOKEN_ID: AtomicU64 = AtomicU64::new(2);

/// Allocates a block of a given type.
fn alloc_block<T: Default>(chunk_size: usize, free_blocks: &mut Vec<&'static T>) -> &'static T {
    if let Some(res) = free_blocks.pop() {
        res
    } else {
        let chunk = std::iter::repeat_with(|| T::default())
            .take(chunk_size)
            .collect::<Vec<_>>()
            .leak();

        // Tell miri that the leak is intentional
        #[cfg(miri)]
        {
            extern "Rust" {
                fn miri_static_root(ptr: *const u8);
            }
            unsafe {
                miri_static_root(chunk.as_ptr().cast::<_>());
            }
        }
        free_blocks.extend(chunk.iter());
        free_blocks.pop().unwrap()
    }
}
