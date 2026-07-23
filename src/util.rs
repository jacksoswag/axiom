//! Tiny helpers with no natural owning module, shared to avoid duplicate copies.

pub(crate) fn finite(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}
