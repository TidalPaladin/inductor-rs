//! Command-line interface for inductor-rs.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Run AOT-compiled PyTorch Inductor models from Rust.
#[derive(Parser, Debug)]
#[command(name = "inductor-rs")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run inference on a model.
    Infer {
        /// Path to the .pt2 model package.
        #[arg(short, long)]
        model: PathBuf,

        /// Device to run inference on (cpu, cuda:0, cuda:1, etc).
        #[arg(short, long, default_value = "cpu")]
        device: String,

        /// Path to input data file (JSON with tensor data).
        #[arg(short, long)]
        input: PathBuf,

        /// Output format (json, pretty).
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Path to optional config file (template only; not applied unless wired in).
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Show model information.
    Info {
        /// Path to the .pt2 model package.
        #[arg(short, long)]
        model: PathBuf,

        /// Device to load model on.
        #[arg(short, long, default_value = "cpu")]
        device: String,
    },

    /// Test device capabilities.
    TestDevice {
        /// Device to test (cpu, cuda:0, cuda:1, etc).
        #[arg(short, long, default_value = "cpu")]
        device: String,
    },
}

impl Cli {
    /// Parse command line arguments.
    pub fn parse_args() -> Self {
        Self::parse()
    }
}
