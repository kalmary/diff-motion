include!("../../build_support.rs");

fn main() {
    pyo3_build_config::add_extension_module_link_args();
    add_conda_library_path();
}
