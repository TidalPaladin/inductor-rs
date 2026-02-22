//! CLI entry point for inductor-rs.

use anyhow::{bail, Context, Result};
use ndarray::ArrayD;
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use inductor_rs::cli::Cli;
use inductor_rs::config::Config;
use inductor_rs::inference::{test_device, AotModel, Device, InferenceResult};

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

#[derive(Debug, Serialize)]
struct OutputSummary {
    index: usize,
    shape: Vec<usize>,
    numel: usize,
}

#[derive(Debug, Serialize)]
struct DeviceCheck {
    ok: bool,
    details: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelCheck {
    provided: bool,
    path: Option<String>,
    ok: Option<bool>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct InferenceSmoke {
    ran: bool,
    input: Option<String>,
    ok: Option<bool>,
    num_outputs: Option<usize>,
    latency_ms: Option<f64>,
    memory_bytes: Option<usize>,
    outputs: Option<Vec<OutputSummary>>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct HealthCheckReport {
    mode: &'static str,
    ok: bool,
    version: &'static str,
    libtorch: &'static str,
    features: &'static str,
    device: String,
    device_check: DeviceCheck,
    model_check: ModelCheck,
    inference_smoke: InferenceSmoke,
}

fn parse_input_tensor(input: &Path) -> Result<ArrayD<f32>> {
    let input_json: Value = serde_json::from_str(
        &fs::read_to_string(input)
            .with_context(|| format!("Failed to read input: {}", input.display()))?,
    )?;

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
    Ok(input_tensor)
}

fn summarize_outputs(result: &InferenceResult) -> Vec<OutputSummary> {
    result
        .outputs
        .iter()
        .enumerate()
        .map(|(i, t)| OutputSummary {
            index: i,
            shape: t.shape().to_vec(),
            numel: t.len(),
        })
        .collect()
}

fn print_value(value: &Value, format: &str) -> Result<()> {
    match format {
        "pretty" => println!("{}", serde_json::to_string_pretty(value)?),
        _ => println!("{}", serde_json::to_string(value)?),
    }
    Ok(())
}

fn load_optional_config(config_path: Option<&Path>) -> Result<Config> {
    if let Some(config_path) = config_path {
        Config::from_yaml_file(config_path)
            .with_context(|| format!("Failed to load config: {}", config_path.display()))
    } else {
        Ok(Config::default())
    }
}

fn require_infer_args(cli: &Cli) -> Result<(&Path, &Path)> {
    let model_path = cli
        .model
        .as_deref()
        .context("Inference mode requires --model <path/to/model.pt2>")?;
    let input_path = cli
        .input
        .as_deref()
        .context("Inference mode requires --input <path/to/input.json>")?;
    Ok((model_path, input_path))
}

fn run_device_check(device: &Device) -> (DeviceCheck, bool) {
    match test_device(device) {
        Ok(details) => (
            DeviceCheck {
                ok: true,
                details: Some(details),
                error: None,
            },
            true,
        ),
        Err(e) => (
            DeviceCheck {
                ok: false,
                details: None,
                error: Some(e.to_string()),
            },
            false,
        ),
    }
}

fn run_model_check(
    model_path: Option<&Path>,
    device: &Device,
) -> (ModelCheck, Option<AotModel>, bool) {
    let mut model_check = ModelCheck {
        provided: model_path.is_some(),
        path: model_path.map(|p| p.display().to_string()),
        ok: None,
        error: None,
    };

    if let Some(path) = model_path {
        match AotModel::load(path, device.clone()) {
            Ok(model) => {
                model_check.ok = Some(true);
                return (model_check, Some(model), true);
            }
            Err(e) => {
                model_check.ok = Some(false);
                model_check.error = Some(e.to_string());
                return (model_check, None, false);
            }
        }
    }

    (model_check, None, true)
}

fn run_inference_smoke_check(
    input_path: Option<&Path>,
    model: Option<&AotModel>,
) -> (InferenceSmoke, bool) {
    let should_run = input_path.is_some() && model.is_some();
    let mut smoke = InferenceSmoke {
        ran: should_run,
        input: input_path.map(|p| p.display().to_string()),
        ok: None,
        num_outputs: None,
        latency_ms: None,
        memory_bytes: None,
        outputs: None,
        error: None,
    };

    let Some(path) = input_path else {
        return (smoke, true);
    };

    let Some(model) = model else {
        // Inference smoke is optional and only runs when both --model and --input are provided.
        return (smoke, true);
    };

    match parse_input_tensor(path)
        .and_then(|input| model.infer(&input).map_err(anyhow::Error::from))
    {
        Ok(result) => {
            smoke.ok = Some(true);
            smoke.num_outputs = Some(result.outputs.len());
            smoke.latency_ms = Some(result.latency_ms);
            smoke.memory_bytes = Some(result.memory_bytes);
            smoke.outputs = Some(summarize_outputs(&result));
            (smoke, true)
        }
        Err(e) => {
            smoke.ok = Some(false);
            smoke.error = Some(e.to_string());
            (smoke, false)
        }
    }
}

fn run_infer(cli: &Cli, device: Device) -> Result<()> {
    let (model_path, input_path) = require_infer_args(cli)?;

    info!("Using device: {}", device);
    info!("Loading model: {}", model_path.display());
    let aot_model = AotModel::load(model_path, device)?;
    info!("Model loaded successfully");

    info!("Loading input: {}", input_path.display());
    let input_tensor = parse_input_tensor(input_path)?;

    info!("Running inference...");
    let result = aot_model.infer(&input_tensor)?;
    info!(
        "Inference complete: {} outputs in {:.2}ms",
        result.outputs.len(),
        result.latency_ms
    );

    let output = serde_json::json!({
        "mode": "infer",
        "num_outputs": result.outputs.len(),
        "latency_ms": result.latency_ms,
        "memory_bytes": result.memory_bytes,
        "outputs": summarize_outputs(&result),
    });
    print_value(&output, &cli.format)
}

fn run_check(cli: &Cli, device: Device) -> Result<()> {
    info!("Running health check on device: {}", device);

    let (device_check, device_ok) = run_device_check(&device);
    let (model_check, loaded_model, model_ok) = run_model_check(cli.model.as_deref(), &device);
    let (inference_smoke, smoke_ok) =
        run_inference_smoke_check(cli.input.as_deref(), loaded_model.as_ref());
    let overall_ok = device_ok && model_ok && smoke_ok;

    let report = HealthCheckReport {
        mode: "check",
        ok: overall_ok,
        version: env!("CARGO_PKG_VERSION"),
        libtorch: libtorch_version(),
        features: enabled_features(),
        device: device.to_string(),
        device_check,
        model_check,
        inference_smoke,
    };

    let report_value = serde_json::to_value(&report)?;
    print_value(&report_value, &cli.format)?;

    if !report.ok {
        bail!("health check failed");
    }
    Ok(())
}

fn main() -> Result<()> {
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    let cli = Cli::parse_args();
    load_optional_config(cli.config.as_deref())?;
    let device: Device = cli.device.parse()?;

    if cli.check {
        run_check(&cli, device)
    } else {
        run_infer(&cli, device)
    }
}
