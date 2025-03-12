#![cfg(loom)]
use renege::alloc::{Allocator, Block, Condition, ConditionId, Token};

/// The backing storage used by the allocator.
///
/// We need to set a high alignment to ensure deterministic behavior, since the lowest bits of
/// the block address are used for indexing.
#[repr(C)]
#[repr(align(1024))]
struct Storage<'alloc> {
    blocks: [Block<'alloc>; 16],
    in_use: loom::sync::Mutex<u64>,
}

impl<'alloc> Storage<'alloc> {
    /// Creates a new [`Storage`].
    pub fn new() -> Self {
        Self {
            blocks: std::array::from_fn(|_| Block::new()),
            in_use: loom::sync::Mutex::new(0),
        }
    }

    /// Allocates a [`Block`] from this storage.
    pub fn allocate_block(&'alloc self) -> &'alloc Block<'alloc> {
        let mut in_use = self.in_use.lock().unwrap();
        let index = in_use.trailing_ones() as usize;
        *in_use |= 1 << index;
        &self.blocks[index]
    }

    /// Frees a [`Block`] from this storage.
    pub fn free_block(&'alloc self, block: &'alloc Block<'alloc>) {
        let index =
            unsafe { (block as *const Block<'alloc>).offset_from(self.blocks.as_ptr()) } as usize;
        let mut in_use = self.in_use.lock().unwrap();
        *in_use &= !(1 << index);
    }
}

/// A [`Storage`] which can be shared between threads.
///
/// This is needed to work around the fact that loom has no support for scoped threads, so we have
/// to use `Arc` to share the storage between threads.
#[derive(Clone)]
struct SharedStorage(loom::sync::Arc<Storage<'static>>);

impl SharedStorage {
    /// Creates new shared storage.
    pub fn new() -> Self {
        Self(loom::sync::Arc::new(Storage::new()))
    }

    /// Calls a function with the underlying storage.
    pub fn with<R>(&self, f: impl for<'alloc> FnOnce(&'alloc Storage<'alloc>) -> R) -> R {
        // SAFETY: The storage lifetime isn't actually `'static`, but there is no other named
        // lifetime we could use here. The storage actually has a consistent lifetime, tied to
        // the `Arc` it is stored in.
        let inner = unsafe {
            std::mem::transmute::<&Storage<'static>, &'static Storage<'static>>(&*self.0)
        };
        f(inner)
    }

    /// Calls a function with an [`Allocator`] derived from the underlying storage.
    fn with_static_alloc<R>(&self, f: impl FnOnce(&mut dyn Allocator<'static>) -> R) -> R {
        self.with(|storage| {
            // SAFETY: It is the responsibility of the caller to ensure that the products of
            // the allocator do not outlive the storage.
            let mut storage: &'static Storage<'static> = unsafe { std::mem::transmute(storage) };
            f(&mut storage)
        })
    }
}

impl<'alloc> Allocator<'alloc> for &'alloc Storage<'alloc> {
    fn allocate_block(&mut self) -> &'alloc Block<'alloc> {
        (**self).allocate_block()
    }

    fn free_block(&mut self, block: &'alloc Block<'alloc>) {
        (**self).free_block(block)
    }
}

#[test]
fn test_construct() {
    loom::model(|| {
        let storage = SharedStorage::new();
        storage.with_static_alloc(|alloc| {
            let a = Condition::new(alloc, ConditionId::new(0));
            let a_token = a.token();
            let b_handle = {
                let storage = storage.clone();
                loom::thread::spawn(move || storage.with_static_alloc(|alloc| {
                    let b = Condition::new(alloc, ConditionId::new(1));
                    let a_b_token = Token::combine(alloc, a_token, b.token());
                    (b, a_b_token)
                }))
            };
            let c_handle = {
                let storage = storage.clone();
                loom::thread::spawn(move || storage.with_static_alloc(|alloc| {
                    let c = Condition::new(alloc, ConditionId::new(2));
                    let a_c_token = Token::combine(alloc, a_token, c.token());
                    (c, a_c_token)
                }))
            };
            let (b, a_b_token) = b_handle.join().unwrap();
            let (c, a_c_token) = c_handle.join().unwrap();
            assert_eq!(a_b_token, Token::combine(alloc, a_token, b.token()));
            assert_eq!(a_c_token, Token::combine(alloc, a_token, c.token()));
        });
    });
}

#[test]
fn test_invalidate_simple() {
    loom::model(|| {
        let storage = SharedStorage::new();
        storage.with_static_alloc(|alloc| {
            let a = Condition::new(alloc, ConditionId::new(0));
            let a_token = a.token();
            let b_handle = {
                let storage = storage.clone();
                loom::thread::spawn(move || storage.with_static_alloc(|alloc| {
                    let b = Condition::new(alloc, ConditionId::new(1));
                    let a_b_token = Token::combine(alloc, a_token, b.token());
                    (b, a_b_token)
                }))
            };
            let c_handle = {
                let storage = storage.clone();
                loom::thread::spawn(move || storage.with_static_alloc(|alloc| {
                    let c = Condition::new(alloc, ConditionId::new(2));
                    let a_c_token = Token::combine(alloc, a_token, c.token());
                    (c, a_c_token)
                }))
            };
            a.invalidate_eventually(alloc);
            let (_, a_b_token) = b_handle.join().unwrap();
            let (_, a_c_token) = c_handle.join().unwrap();
            // At this point, we are on the only remaining thread, so invalidation propogation
            // should be complete.
            assert!(!a_b_token.is_valid());
            assert!(!a_c_token.is_valid());
        })
    })
}

#[test]
fn test_invalidate_ordering() {
    loom::model(|| {
        let storage = SharedStorage::new();
        storage.with_static_alloc(|alloc| {
            let a = Condition::new(alloc, ConditionId::new(0));
            let b = Condition::new(alloc, ConditionId::new(1));
            let a_token = a.token();
            let c_handle = {
                let storage = storage.clone();
                loom::thread::spawn(move || storage.with_static_alloc(|alloc| {
                    let c = Condition::new(alloc, ConditionId::new(2));
                    let a_c_token = Token::combine(alloc, a_token, c.token());
                    (c, a_c_token)
                }))
            };
            let a_b_token = Token::combine(alloc, a.token(), b.token());
            assert!(a_b_token.is_valid());
            a.invalidate_immediately(alloc);
            assert!(!a_b_token.is_valid());
            let (_, a_c_token) = c_handle.join().unwrap();
            assert!(!a_c_token.is_valid())
        })
    })
}