/// Converts a closure, `f()`, into an equivalent call `g(data)` where `g` is a function pointer.
///
/// `g` must be called with the returned data pointer, and it may only be called once. The
/// thread safety and lifetime properties of `g` and `data` are inherited from `f`.
pub fn into_fn_ptr<F: FnOnce() -> T, T>(f: F) -> (unsafe fn(*mut ()) -> T, *mut ()) {
    // TODO: There's probably a clever way of packing smaller types directly into the pointer,
    // but that opens up a whole can of soundness issues that I don't want to worry about right
    // now.
    let ptr = Box::into_raw(Box::new(f)) as *mut ();
    unsafe fn g<F: FnOnce() -> T, T>(ptr: *mut ()) -> T {
        // SAFETY: The caller must ensure that `ptr` matches what we just created above
        let f: Box<F> = unsafe { Box::from_raw(ptr as *mut F) };
        f()
    }
    (g::<F, T>, ptr)
}