//! This module defines alternatives to [`Token`] and [`Condition`] which can use custom allocators
//! and backing storage.
pub use crate::imp::{Block, Condition, ConditionId, Token};
pub use global::Global;

/// An allocator for [`Block`]s used by [`Token`]s and [`Condition`]s.
///
/// An individual [`Allocator`] will only be used by a single thread, but there may be multiple
/// co-existing [`Allocator`]s which allocate to the same backing storage. The `'alloc` lifetime
/// specifies the lifetime of that storage.
///
/// [`Block`]s may contain arbitrary references between each other (even after
/// [`Allocator::free_block`] is called), so the backing storage can't shrink: it must be
/// dropped all at once, along with all [`Token`]s and [`Condition`]s which use it.
pub trait Allocator<'alloc> {
    /// Allocates a new [`Block`].
    fn allocate_block(&mut self) -> &'alloc Block<'alloc>;

    /// Frees a [`Block`].
    ///
    /// This allows the block to be returned by a following call to [`Allocator::allocate_block`].
    fn free_block(&mut self, block: &'alloc Block<'alloc>);
}

/// Contains functionality related to the global allocator.
mod global {
    use crate::alloc::{Allocator, Block, ConditionId};
    #[allow(unused_imports)]
    use crate::{Condition, Token};
    use std::cell::RefCell;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::Relaxed;

    /// The global [`Allocator`] used by [`Token`]s and [`Condition`]s.
    ///
    /// There is actually a different instance of [`Global`] for each thread, all sharing the same
    /// underlying storage.
    pub struct Global {
        next_cond_index: usize,
        last_reserved_cond_index: usize,
        free_blocks: Vec<&'static Block<'static>>,
    }

    thread_local! {
        static ALLOC: RefCell<Global> = const {
            RefCell::new(Global {
                next_cond_index: 0,
                last_reserved_cond_index: 0,
                free_blocks: Vec::new(),
            })
        };
    }

    /// The next unused [`ConditionId`] index.
    static NEXT_COND_INDEX: AtomicUsize = AtomicUsize::new(0);

    /// The number of [`ConditionId`]s that should be reserved by a [`Global`] at a time.
    const NUM_RESERVED_COND_IDS: usize = 32;

    /// The number of [`Block`]s that should be allocated at a time.
    const CHUNK_SIZE: usize = 64;

    impl Global {
        /// Calls the given function with the [`Global`] for this thread.
        pub fn with<R>(f: impl FnOnce(&mut Global) -> R) -> R {
            ALLOC.with(|cell| f(&mut cell.borrow_mut()))
        }

        /// Gets an unused [`ConditionId`].
        pub fn allocate_condition_id(&mut self) -> ConditionId {
            if self.next_cond_index < self.last_reserved_cond_index {
                let res = self.next_cond_index;
                self.next_cond_index += 1;
                ConditionId::new(res)
            } else {
                let res = NEXT_COND_INDEX.fetch_add(NUM_RESERVED_COND_IDS, Relaxed);
                self.next_cond_index = res + 1;
                self.last_reserved_cond_index = res + NUM_RESERVED_COND_IDS;
                ConditionId::new(res)
            }
        }
    }

    impl Allocator<'static> for Global {
        /// Gets an unused [`Block`].
        fn allocate_block(&mut self) -> &'static Block<'static> {
            if let Some(res) = self.free_blocks.pop() {
                res
            } else {
                let chunk = std::iter::repeat_with(Block::default)
                    .take(CHUNK_SIZE)
                    .collect::<Vec<_>>()
                    .leak();

                // Tell miri that the leak is intentional
                #[cfg(miri)]
                {
                    unsafe extern "Rust" {
                        unsafe fn miri_static_root(ptr: *const u8);
                    }
                    unsafe {
                        miri_static_root(chunk.as_ptr().cast::<_>());
                    }
                }
                self.free_blocks.extend(chunk.iter());
                self.free_blocks.pop().unwrap()
            }
        }

        /// Reclaims an unused [`Block`].
        fn free_block(&mut self, block: &'static Block<'static>) {
            self.free_blocks.push(block);
        }
    }

    impl std::fmt::Debug for Global {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Global").finish_non_exhaustive()
        }
    }
}
