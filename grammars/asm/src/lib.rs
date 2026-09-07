/// Stable entry point used by the hips grammar loader.
#[no_mangle]
pub unsafe extern "C" fn hips_language() -> *const () {
    tree_sitter_asm::LANGUAGE.into_raw()()
}
