use std::{env, path::Path};

fn add_conda_library_path() {
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    if env::var("HOST").ok() != env::var("TARGET").ok() {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if !matches!(target_os.as_str(), "linux" | "macos") {
        return;
    }

    if let Ok(prefix) = env::var("CONDA_PREFIX") {
        let library_path = Path::new(&prefix).join("lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library_path.display());
    }
}
