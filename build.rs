//! Compiles PureTD's C move generator and network forward pass into this binary.
//!
//! The engine is Alexander Strehl's C, unmodified — `bg_engine.c` generates the
//! legal plays and encodes the network's inputs, `nn_eval.c` runs the forward
//! pass over weights exported by `export_weights.py`. PyTorch produces those
//! weights on a build machine and never ships.
//!
//! The sources come from our fork rather than being vendored, so upstream fixes
//! arrive by pulling the fork instead of by hand-copying files that then drift.
//! PURETD_SRC overrides the location for CI, where the checkout is elsewhere.

use std::path::PathBuf;

fn main() {
    let src = std::env::var("PURETD_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../backgammon-ai-engine"));

    let engine = src.join("c_engine");
    let inference = src.join("c_inference");

    for file in ["c_engine/bg_engine.c", "c_inference/nn_eval.c"] {
        let path = src.join(file);
        assert!(
            path.exists(),
            "PureTD source not found at {}. Clone customation/backgammon-ai-engine \
             beside this repository, or set PURETD_SRC to its checkout.",
            path.display()
        );
        println!("cargo:rerun-if-changed={}", path.display());
    }

    cc::Build::new()
        .file(engine.join("bg_engine.c"))
        .file(inference.join("nn_eval.c"))
        .include(&engine)
        .include(&inference)
        .opt_level(2)
        // Upstream's own warnings are upstream's business; we compile their
        // code as they wrote it and do not patch it to silence anything.
        .warnings(false)
        .compile("puretd");

    println!("cargo:rerun-if-env-changed=PURETD_SRC");
}
