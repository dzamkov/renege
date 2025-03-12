//! This module contains helper types and functions for working with atomic values.
#[cfg(not(loom))]
pub use std::sync::atomic::{AtomicPtr, AtomicUsize, AtomicBool, fence};
#[cfg(loom)]
pub use loom::sync::atomic::{AtomicPtr, AtomicUsize, AtomicBool, fence};
use std::sync::atomic::Ordering;

/// A wrapper over a value of type `T` which permits interior mutability using atomic operations.
#[repr(transparent)]
pub struct Atomic<T: HasAtomic>(pub T::Atomic);

/// Indicates that the internal representation of a type matches some primitive which has an
/// atomic counterpart.
///
/// # Safety
/// The type must have the same size and alignment as `Prim`.
pub unsafe trait HasAtomic: Copy {
    /// The primitive representation of this type.
    type Prim: HasAtomic<Prim = Self::Prim, Atomic = Self::Atomic>;

    /// The atomic counterpart to `Prim`.
    type Atomic: IsAtomic<Prim = Self::Prim>;

    /// Converts a value of this type into its primitive representation.
    fn into_prim(value: Self) -> Self::Prim {
        unsafe { std::mem::transmute_copy(&value) }
    }

    /// Gets a value of this type from its primitive representation.
    unsafe fn from_prim(value: Self::Prim) -> Self {
        unsafe { std::mem::transmute_copy(&value) }
    }
}

unsafe impl HasAtomic for usize {
    type Prim = usize;
    type Atomic = AtomicUsize;
}

unsafe impl<T> HasAtomic for *mut T {
    type Prim = *mut T;
    type Atomic = AtomicPtr<T>;
}

unsafe impl<T> HasAtomic for Option<&'_ T> {
    type Prim = *mut T;
    type Atomic = AtomicPtr<T>;
}

/// An atomic primitive type.
pub trait IsAtomic {
    type Prim: HasAtomic<Prim = Self::Prim, Atomic = Self>;
    fn new(value: Self::Prim) -> Self;
    fn load(&self, order: Ordering) -> Self::Prim;
    fn store(&self, val: Self::Prim, order: Ordering);
    fn compare_exchange(
        &self,
        current: Self::Prim,
        new: Self::Prim,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::Prim, Self::Prim>;
    fn compare_exchange_weak(
        &self,
        current: Self::Prim,
        new: Self::Prim,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::Prim, Self::Prim>;
}

impl IsAtomic for AtomicUsize {
    type Prim = usize;
    fn new(value: Self::Prim) -> Self {
        Self::new(value)
    }
    fn load(&self, order: Ordering) -> usize {
        self.load(order)
    }
    fn store(&self, val: Self::Prim, order: Ordering) {
        self.store(val, order)
    }
    fn compare_exchange(
        &self,
        current: Self::Prim,
        new: Self::Prim,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::Prim, Self::Prim> {
        self.compare_exchange(current, new, success, failure)
    }
    fn compare_exchange_weak(
        &self,
        current: Self::Prim,
        new: Self::Prim,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::Prim, Self::Prim> {
        self.compare_exchange_weak(current, new, success, failure)
    }
}

impl<T> IsAtomic for AtomicPtr<T> {
    type Prim = *mut T;
    fn new(value: Self::Prim) -> Self {
        Self::new(value)
    }
    fn load(&self, order: Ordering) -> Self::Prim {
        self.load(order)
    }
    fn store(&self, val: Self::Prim, order: Ordering) {
        self.store(val, order)
    }
    fn compare_exchange(
        &self,
        current: Self::Prim,
        new: Self::Prim,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::Prim, Self::Prim> {
        self.compare_exchange(current, new, success, failure)
    }
    fn compare_exchange_weak(
        &self,
        current: Self::Prim,
        new: Self::Prim,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::Prim, Self::Prim> {
        self.compare_exchange_weak(current, new, success, failure)
    }
}

impl<T: HasAtomic> Atomic<T> {
    /// Constructs an [`Atomic`] wrapper over the given value.
    pub fn new(value: T) -> Self {
        Self(T::Atomic::new(T::into_prim(value)))
    }

    /// Loads a value from the [`Atomic`].
    pub fn load(&self, order: Ordering) -> T {
        let prim = self.0.load(order);
        unsafe { T::from_prim(prim) }
    }

    /// Stores a value into the [`Atomic`].
    pub fn store(&self, val: T, order: Ordering) {
        self.0.store(T::into_prim(val), order);
    }

    /// Stores a value into the [`Atomic`] if the current value is the same as `current`.
    pub fn compare_exchange(
        &self,
        current: T,
        new: T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<T, T> {
        self.0
            .compare_exchange(T::into_prim(current), T::into_prim(new), success, failure)
            .map(|prim| unsafe { T::from_prim(prim) })
            .map_err(|prim| unsafe { T::from_prim(prim) })
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
        self.0
            .compare_exchange_weak(T::into_prim(current), T::into_prim(new), success, failure)
            .map(|prim| unsafe { T::from_prim(prim) })
            .map_err(|prim| unsafe { T::from_prim(prim) })
    }
}

#[cfg(not(loom))]
impl<T: HasAtomic<Prim = usize, Atomic = AtomicUsize>> Atomic<T> {
    /// Constructs an [`Atomic`] wrapper over the given primitive value.
    /// 
    /// Unlike [`Atomic::new`], this is `const`.
    pub const fn from_prim(value: usize) -> Self {
        Self(AtomicUsize::new(value))
    }
}

#[cfg(not(loom))]
impl<Ptr: HasAtomic<Prim = *mut T, Atomic = AtomicPtr<T>>, T> Atomic<Ptr> {
    /// Constructs an [`Atomic`] wrapper over the null pointer.
    pub const fn null() -> Self {
        Self(AtomicPtr::new(std::ptr::null_mut()))
    }
}