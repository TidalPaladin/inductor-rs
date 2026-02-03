//! CLI entry point for inductor-rs.

use anyhow::{bail, Context, Result};
use ndarray::ArrayD;
use serde_json::Value;
use std::fs;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use inductor_rs::cli::{Cli, Commands};
use inductor_rs::config::Config;
use inductor_rs::inference::{test_device, AotModel};

/// Get the libtorch version from the build script.
fn libtorch_version() -> &'static str {
    option_env!("LIBTORCH_VERSION").unwrap_or("unknown")
}

/// Get the enabled features.
fn enabled_features() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "rocm") {
        "rocm"
    } else {
        "cpu"
    }
}

fn main() -> Result<()> {
    // Initialize logging
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    let cli = Cli::parse_args();

    match cli.command {
        Commands::Infer {
            model,
            device,
            input,
            format,
            config,
        } => {
            // Load optional config
            let _config = if let Some(config_path) = config {
                Config::from_yaml_file(&config_path)
                    .with_context(|| format!("Failed to load config: {}", config_path.display()))?
            } else {
                Config::default()
            };

            // Parse device
            let device = device.parse()?;
            info!("Using device: {}", device);

            // Load model
            info!("Loading model: {}", model.display());
            let aot_model = AotModel::load(&model, device)?;
            info!("Model loaded successfully");

            // Load input data
            info!("Loading input: {}", input.display());
            let input_json: Value = serde_json::from_str(
                &fs::read_to_string(&input)
                    .with_context(|| format!("Failed to read input: {}", input.display()))?,
            )?;

            // Parse input tensor from JSON
            // Expected format: { "data": [...], "shape": [B, C, H, W] }
            let data_array = input_json["data"]
                .as_array()
                .context("Input must have 'data' array")?;
            let mut data = Vec::with_capacity(data_array.len());
            for (i, v) in data_array.iter().enumerate() {
                let value = v
                    .as_f64()
                    .with_context(|| format!("data[{i}] must be a number"))?;
                data.push(value as f32);
            }

            let shape_array = input_json["shape"]
                .as_array()
                .context("Input must have 'shape' array")?;
            let mut shape = Vec::with_capacity(shape_array.len());
            for (i, v) in shape_array.iter().enumerate() {
                let dim = v
                    .as_i64()
                    .or_else(|| v.as_u64().map(|u| u as i64))
                    .or_else(|| {
                        v.as_f64().and_then(|f| {
                            if f.fract() == 0.0 {
                                Some(f as i64)
                            } else {
                                None
                            }
                        })
                    })
                    .with_context(|| format!("shape[{i}] must be an integer"))?;
                if dim < 0 {
                    bail!("shape[{i}] must be >= 0, got {dim}");
                }
                shape.push(dim as usize);
            }

            let expected_len = shape.iter().try_fold(1usize, |acc, &d| {
                acc.checked_mul(d).context("Input shape product overflow")
            })?;
            if data.len() != expected_len {
                bail!(
                    "Input data length {} does not match shape product {}",
                    data.len(),
                    expected_len
                );
            }

            let input_tensor = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
                .context("Failed to create input tensor")?;

            // Run inference
            info!("Running inference...");
            let result = aot_model.infer(&input_tensor)?;

            info!(
                "Inference complete: {} outputs in {:.2}ms",
                result.outputs.len(),
                result.latency_ms
            );

            // Format output
            let output = serde_json::json!({
                "num_outputs": result.outputs.len(),
                "latency_ms": result.latency_ms,
                "memory_bytes": result.memory_bytes,
                "outputs": result.outputs.iter().enumerate().map(|(i, t)| {
                    serde_json::json!({
                        "index": i,
                        "shape": t.shape(),
                        "numel": t.len(),
                    })
                }).collect::<Vec<_>>()
            });

            if format == "pretty" {
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("{}", serde_json::to_string(&output)?);
            }
        }

        Commands::Info { model, device } => {
            let device = device.parse()?;

            println!("inductor-rs v{}", env!("CARGO_PKG_VERSION"));
            println!("libtorch: {}", libtorch_version());
            println!("features: {}", enabled_features());
            println!();
            println!("Model: {}", model.display());
            println!("Device: {}", device);

            // Try to load the model to verify it works
            info!("Loading model...");
            let _model = AotModel::load(&model, device)?;
            println!("Status: OK (model loaded successfully)");
        }

        Commands::TestDevice { device } => {
            let device = device.parse()?;
            println!("Testing device: {}", device);
            println!();

            match test_device(&device) {
                Ok(results) => {
                    println!("{}", results);
                }
                Err(e) => {
                    eprintln!("Device test failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}
