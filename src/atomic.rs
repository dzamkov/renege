//! This module contains helper types and functions for working with atomic values.
use crate::util::SafeTransmuteFrom;
#[cfg(all(test, loom))]
pub use loom::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, fence};
use std::sync::atomic::Ordering;
#[cfg(not(all(test, loom)))]
pub use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, fence};

/// A wrapper over an atomic type which can be loaded as a `Load` and stored as a `Store`.
///
/// `Load` and `Store` will usually be the same, but may differ when the same atomic location is
/// used to store different types of values. In this case, `Load` will be the common supertype of
/// all the `Store` types that can be stored in the atomic.
#[repr(transparent)]
pub struct Atomic<
    Store: HasAtomic,
    Load: HasAtomic<Prim = Store::Prim> + SafeTransmuteFrom<Store> = Store,
>(
    pub <Store::Prim as IsPrim>::Atomic,
    std::marker::PhantomData<<Load::Prim as IsPrim>::Atomic>,
);

/// Indicates that the internal representation of a type matches some primitive which has an
/// atomic counterpart.
pub trait HasAtomic: Copy {
    /// The primitive representation of this type.
    type Prim: IsPrim + SafeTransmuteFrom<Self>;

    /// Converts a value of this type into its primitive representation.
    fn into_prim(value: Self) -> Self::Prim {
        SafeTransmuteFrom::transmute_from(value)
    }

    /// Gets a value of this type from its primitive representation.
    ///
    /// # Safety
    /// The caller must ensure that `value` has a bit pattern which can soundly be interpreted as
    /// having the type `Self`.
    unsafe fn from_prim(value: Self::Prim) -> Self {
        // SAFETY: The caller must assure this is valid
        unsafe { std::mem::transmute_copy(&value) }
    }
}

impl HasAtomic for usize {
    type Prim = usize;
}

impl<T> HasAtomic for *mut T {
    type Prim = *mut T;
}

impl<T> HasAtomic for Option<&'_ T> {
    type Prim = *mut T;
}

/// A primitive type which has an atomic counterpart.
pub trait IsPrim: Copy {
    type Atomic: IsAtomic<Prim = Self>;
}

impl IsPrim for usize {
    type Atomic = AtomicUsize;
}

impl<T> IsPrim for *mut T {
    type Atomic = AtomicPtr<T>;
}

/// An atomic primitive type.
pub trait IsAtomic {
    type Prim: IsPrim<Atomic = Self>;
    fn new(value: Self::Prim) -> Self;
    fn load(&self, order: Ordering) -> Self::Prim;
    fn store(&self, val: Self::Prim, order: Ordering);
    fn swap(&self, val: Self::Prim, order: Ordering) -> Self::Prim;
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
    fn swap(&self, val: Self::Prim, order: Ordering) -> Self::Prim {
        self.swap(val, order)
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
    fn swap(&self, val: Self::Prim, order: Ordering) -> Self::Prim {
        self.swap(val, order)
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
        Self(
            <T::Prim as IsPrim>::Atomic::new(T::into_prim(value)),
            std::marker::PhantomData,
        )
    }
}

impl<Store: HasAtomic, Load: HasAtomic<Prim = Store::Prim> + SafeTransmuteFrom<Store>>
    Atomic<Store, Load>
{
    /// Loads a value from the [`Atomic`].
    pub fn load(&self, order: Ordering) -> Load {
        let prim = self.0.load(order);
        unsafe { Load::from_prim(prim) }
    }

    /// Stores a value into the [`Atomic`].
    pub fn store(&self, val: Store, order: Ordering) {
        self.0.store(Store::into_prim(val), order);
    }

    /// Stores a value into the [`Atomic`], returning the previous value.
    pub fn swap(&self, val: Store, order: Ordering) -> Load {
        let res = self.0.swap(Store::into_prim(val), order);
        unsafe { Load::from_prim(res) }
    }

    /// Stores a value into the [`Atomic`] if the current value is the same as `current`.
    pub fn compare_exchange(
        &self,
        current: Load,
        new: Store,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Load, Load> {
        self.0
            .compare_exchange(
                Load::into_prim(current),
                Store::into_prim(new),
                success,
                failure,
            )
            .map(|prim| unsafe { Load::from_prim(prim) })
            .map_err(|prim| unsafe { Load::from_prim(prim) })
    }

    /// Stores a value into the [`Atomic`] if the current value is the same as `current`. May
    /// spuriously fail.
    pub fn compare_exchange_weak(
        &self,
        current: Load,
        new: Store,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Load, Load> {
        self.0
            .compare_exchange_weak(
                Load::into_prim(current),
                Store::into_prim(new),
                success,
                failure,
            )
            .map(|prim| unsafe { Load::from_prim(prim) })
            .map_err(|prim| unsafe { Load::from_prim(prim) })
    }

    /// Restricts the `Store` or loosens the `Load` type of an [`Atomic`] reference.
    pub fn cast<NStore, NLoad>(&self) -> &Atomic<NStore, NLoad>
    where
        NStore: HasAtomic<Prim = Store::Prim>,
        Store: SafeTransmuteFrom<NStore>,
        NLoad: HasAtomic<Prim = Store::Prim> + SafeTransmuteFrom<NStore> + SafeTransmuteFrom<Load>,
    {
        // SAFETY: Both `Self` and `Atomic<NStore, NLoad>` have the same internal representation
        // of `<Store::Prim as IsPrim>::Atomic`, so the question is whether loads and stores
        // through the returned reference are safe. Since `Store: SafeTransmuteFrom<NStore>`,
        // stores of type `NStore` are also valid stores of type `Store`. Similarly, since
        // `NLoad: SafeTransmuteFrom<Load>`, loads of type `NLoad` are valid where a load
        // of type `Load` would be valid.
        unsafe { std::mem::transmute(self) }
    }
}

#[cfg(not(all(test, loom)))]
impl<T: HasAtomic<Prim = usize>> Atomic<T> {
    /// Constructs an [`Atomic`] wrapper over the given primitive value.
    ///
    /// Unlike [`Atomic::new`], this is `const`.
    pub const fn from_prim(value: usize) -> Self {
        Self(AtomicUsize::new(value), std::marker::PhantomData)
    }
}

#[cfg(not(all(test, loom)))]
impl<Ptr: HasAtomic<Prim = *mut T>, T> Atomic<Ptr> {
    /// Constructs an [`Atomic`] wrapper over the null pointer.
    pub const fn null() -> Self {
        Self(
            AtomicPtr::new(std::ptr::null_mut()),
            std::marker::PhantomData,
        )
    }
}
