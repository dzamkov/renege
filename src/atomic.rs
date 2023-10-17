//! This module contains helper types and functions for working with atomic values.
use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

/// A wrapper over a value of type `T` which permits interior mutability using atomic operations.
#[repr(transparent)]
pub struct Atomic<T>(UnsafeCell<T>);

unsafe impl<T: Sync> Sync for Atomic<T> {}

unsafe impl<T: Send> Send for Atomic<T> {}

impl<T> Atomic<T> {
    /// Constructs an [`Atomic`] wrapper over the given value.
    pub const fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }
}

impl<T: PrimRepr<Prim = Prim>, Prim: HasAtomic> Atomic<T> {
    /// Loads a value from the [`Atomic`].
    pub fn load(&self, order: Ordering) -> T {
        T::from_prim(unsafe { Prim::atomic_load(self.0.get() as *mut Prim, order) })
    }

    /// Stores a value into the [`Atomic`].
    pub fn store(&self, val: T, order: Ordering) {
        unsafe { Prim::atomic_store(self.0.get() as *mut Prim, val.into_prim(), order) }
    }
}

impl<T: PrimRepr<Prim = u64>> Atomic<T> {
    /// Stores a value into the [`Atomic`] if the current value is the same as `current`.
    pub fn compare_exchange(
        &self,
        current: T,
        new: T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<T, T> {
        unsafe { &*(self.0.get() as *mut AtomicU64) }
            .compare_exchange(current.into_prim(), new.into_prim(), success, failure)
            .map(T::from_prim)
            .map_err(T::from_prim)
    }

    /// Stores a value into the [`Atomic`] if the current value is the same as `current`. May
    /// spuriously fail.
    pub fn compare_exchange_weak(
        &self,
        current: T,
        new: T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<T, T> {
        unsafe { &*(self.0.get() as *mut AtomicU64) }
            .compare_exchange_weak(current.into_prim(), new.into_prim(), success, failure)
            .map(T::from_prim)
            .map_err(T::from_prim)
    }

    /// Stores a value into the [`Atomic`], returning the previous value.
    pub fn swap(&self, val: T, order: Ordering) -> T {
        T::from_prim(unsafe { &*(self.0.get() as *mut AtomicU64) }.swap(val.into_prim(), order))
    }
}

impl<Ptr: PrimRepr<Prim = NonNull<T>>, T> Atomic<Option<Ptr>> {
    /// Loads a value from the [`Atomic`].
    pub fn load(&self, order: Ordering) -> Option<Ptr> {
        let ptr = unsafe { &*(self.0.get() as *mut AtomicPtr<T>) }.load(order);
        NonNull::new(ptr).map(Ptr::from_prim)
    }

    /// Stores a value into the [`Atomic`].
    pub fn store(&self, val: Option<Ptr>, order: Ordering) {
        unsafe { &*(self.0.get() as *mut AtomicPtr<T>) }.store(
            if let Some(val) = val {
                Ptr::into_prim(val).as_ptr()
            } else {
                std::ptr::null_mut()
            },
            order,
        )
    }

    /// Stores a value into the [`Atomic`], returning the previous value.
    pub fn swap(&self, val: Option<Ptr>, order: Ordering) -> Option<Ptr> {
        let ptr = unsafe { &*(self.0.get() as *mut AtomicPtr<T>) }.swap(
            if let Some(val) = val {
                Ptr::into_prim(val).as_ptr()
            } else {
                std::ptr::null_mut()
            },
            order,
        );
        NonNull::new(ptr).map(Ptr::from_prim)
    }
}

/// Indicates that a primitive type has an [`Atomic`] representation.
pub trait HasAtomic {
    unsafe fn atomic_load(ptr: *mut Self, order: Ordering) -> Self;
    unsafe fn atomic_store(ptr: *mut Self, val: Self, order: Ordering);
}

impl HasAtomic for u64 {
    unsafe fn atomic_load(ptr: *mut u64, order: Ordering) -> u64 {
        (*(ptr as *mut AtomicU64)).load(order)
    }

    unsafe fn atomic_store(ptr: *mut u64, val: u64, order: Ordering) {
        (*(ptr as *mut AtomicU64)).store(val, order)
    }
}

impl<T> HasAtomic for NonNull<T> {
    unsafe fn atomic_load(ptr: *mut Self, order: Ordering) -> Self {
        NonNull::new((*(ptr as *mut AtomicPtr<T>)).load(order)).unwrap_unchecked()
    }

    unsafe fn atomic_store(ptr: *mut Self, val: Self, order: Ordering) {
        (*(ptr as *mut AtomicPtr<T>)).store(val.as_ptr(), order)
    }
}

/// Indicates that the internal representation of a type matches some primitive.
///
/// # Safety
/// The type must be marked as `#[repr(transparent)]` and have `Self::Prim` as its only
/// non-zero-sized field.
pub unsafe trait PrimRepr: Copy {
    type Prim;

    /// Gets the internal representation of a value of this type.
    fn into_prim(self) -> Self::Prim {
        unsafe { std::mem::transmute_copy(&self) }
    }

    /// Constructs a value of this type from its internal representation.
    fn from_prim(value: Self::Prim) -> Self {
        unsafe { std::mem::transmute_copy(&value) }
    }
}

unsafe impl<T> PrimRepr for &'_ T {
    type Prim = NonNull<T>;
}
