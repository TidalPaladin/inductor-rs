use anyhow::{bail, Context, Result};
use approx::assert_abs_diff_eq;
use inductor_rs::inference::test_device;
use inductor_rs::{AotModel, Device};
use ndarray::Array4;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("lock poisoned")
}

fn fixture_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("../tests/fixtures/dummy_model.pt2")
}

fn require_fixture() -> Result<PathBuf> {
    let model_path = fixture_path();
    if !model_path.exists() {
        bail!(
            "Missing test fixture at {}. Run `make test-fixtures` first.",
            model_path.display()
        );
    }
    Ok(model_path)
}

#[test]
fn infer_dummy_model_outputs() -> Result<()> {
    let _guard = test_lock();
    let model_path = require_fixture()?;

    let model =
        AotModel::load(&model_path, Device::cpu()).context("Failed to load dummy model fixture")?;

    let input = Array4::<f32>::zeros((1, 1, 224, 224));
    let result = model.infer(&input)?;

    assert_eq!(result.outputs.len(), 3, "expected 3 outputs");

    let out0 = &result.outputs[0].data;
    assert_eq!(out0.shape(), &[1]);
    let out0_slice = out0.as_slice().context("output_0 not contiguous")?;
    assert_abs_diff_eq!(out0_slice[0], 0.42, epsilon = 1e-5);

    let out1 = &result.outputs[1].data;
    assert_eq!(out1.shape(), &[1, 224, 224]);
    let out1_slice = out1.as_slice().context("output_1 not contiguous")?;
    for &v in out1_slice {
        assert_abs_diff_eq!(v, 0.1, epsilon = 1e-5);
    }

    let out2 = &result.outputs[2].data;
    assert_eq!(out2.shape(), &[1, 5]);
    let out2_slice = out2.as_slice().context("output_2 not contiguous")?;
    let expected = [0.1, 0.2, 0.3, 0.4, 0.5];
    assert_eq!(out2_slice.len(), expected.len());
    for (v, exp) in out2_slice.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*v, *exp, epsilon = 1e-5);
    }

    Ok(())
}

#[test]
fn model_load_invalid_path_returns_error() {
    let _guard = test_lock();
    let result = AotModel::load("/nonexistent/model.pt2", Device::cpu());
    assert!(result.is_err(), "expected loading an invalid path to fail");
}

#[test]
fn infer_dummy_model_multiple_calls_are_stable() -> Result<()> {
    let _guard = test_lock();
    let model_path = require_fixture()?;
    let model = AotModel::load(&model_path, Device::cpu())?;
    let input = Array4::<f32>::zeros((1, 1, 224, 224));

    for _ in 0..3 {
        let result = model.infer(&input)?;
        assert_eq!(result.outputs.len(), 3);
        let out0 = &result.outputs[0].data;
        let out0_slice = out0.as_slice().context("output_0 not contiguous")?;
        assert_abs_diff_eq!(out0_slice[0], 0.42, epsilon = 1e-5);
    }

    Ok(())
}

#[test]
fn infer_dummy_model_reports_timing_and_cpu_memory() -> Result<()> {
    let _guard = test_lock();
    let model_path = require_fixture()?;
    let model = AotModel::load(&model_path, Device::cpu())?;
    let input = Array4::<f32>::zeros((1, 1, 224, 224));

    let result = model.infer(&input)?;
    assert!(
        result.latency_ms > 0.0,
        "latency should be > 0, got {}",
        result.latency_ms
    );
    assert_eq!(
        result.memory_bytes, 0,
        "CPU memory_bytes should be 0, got {}",
        result.memory_bytes
    );

    Ok(())
}

#[test]
fn test_device_cpu_reports_basic_info() -> Result<()> {
    let _guard = test_lock();
    let output = test_device(&Device::cpu())?;
    assert!(
        output.contains("Device type: CPU"),
        "expected CPU device info in output, got: {}",
        output
    );
    Ok(())
}
