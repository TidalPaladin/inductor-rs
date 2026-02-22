//! Command-line interface for inductor-rs.

use clap::Parser;
use std::path::PathBuf;

/// Run AOT-compiled PyTorch Inductor models from Rust.
#[derive(Parser, Debug)]
#[command(name = "inductor-rs")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the .pt2 model package.
    ///
    /// Required for inference mode.
    #[arg(short, long)]
    pub model: Option<PathBuf>,

    /// Device to run on (cpu, cuda:0, cuda:1, etc).
    #[arg(short, long, default_value = "cpu")]
    pub device: String,

    /// Path to input data file (JSON with tensor data).
    ///
    /// Required for inference mode.
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Output format (json, pretty).
    #[arg(short, long, default_value = "json")]
    pub format: String,

    /// Path to optional config file (template only; not applied unless wired in).
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Run health checks instead of normal inference.
    ///
    /// Health checks include device diagnostics, build/runtime metadata,
    /// optional model load validation, and optional inference smoke test
    /// when both --model and --input are supplied.
    #[arg(long)]
    pub check: bool,
}

impl Cli {
    /// Parse command line arguments.
    pub fn parse_args() -> Self {
        Self::parse()
    }
}
