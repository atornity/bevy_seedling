use std::sync::{Arc, Mutex};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Shared<T: ?Sized>(Arc<T>);

impl<T: Send + Sync + 'static> Shared<T> {
    pub fn new(value: T) -> Self {
        let value = Self(Arc::new(value));

        REGISTRY.lock().unwrap().push(Box::new(value.clone()));

        value
    }
}

impl<T: ?Sized + Send + Sync + 'static> Shared<T> {
    pub fn new_unsized(f: impl FnOnce() -> Arc<T>) -> Self {
        let value = Self(f());

        REGISTRY.lock().unwrap().push(Box::new(value.clone()));

        value
    }
}

impl<T: ?Sized> core::ops::Deref for Shared<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

trait StrongCount: Send + Sync {
    fn count(&self) -> usize;
}

impl<T: Send + Sync + ?Sized> StrongCount for Shared<T> {
    fn count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

static REGISTRY: Mutex<Vec<Box<dyn StrongCount + 'static>>> = Mutex::new(Vec::new());

pub fn collect_shared() {
    let mut registry = REGISTRY.lock().unwrap();

    registry.retain(|ptr| ptr.count() > 1);
}

impl<T: ?Sized> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn registry_size() -> usize {
        REGISTRY.lock().unwrap().len()
    }

    fn test_drop_works() {
        assert_eq!(registry_size(), 0);

        let value = Shared::new(1);

        assert_eq!(registry_size(), 1);

        collect_shared();

        assert_eq!(registry_size(), 1);

        drop(value);

        // Even though we've dropped the "last reference,"
        // the inner drop won't be called until we do garbage
        // collection.
        assert_eq!(registry_size(), 1);

        collect_shared();

        assert_eq!(registry_size(), 0);
    }

    fn test_unsized_works() {
        assert_eq!(registry_size(), 0);

        let value = Shared::new_unsized(|| Arc::<[i32]>::from([1, 2, 3]));

        assert_eq!(registry_size(), 1);

        collect_shared();

        assert_eq!(registry_size(), 1);

        drop(value);

        assert_eq!(registry_size(), 1);

        collect_shared();

        assert_eq!(registry_size(), 0);
    }

    // These have to be grouped into one test because
    // they all access a global context.
    //
    // This still isn't very robust -- no other tests
    // in this crate can use `Shared` types.
    #[test]
    fn test_shared() {
        test_drop_works();
        test_unsized_works();
    }
}
