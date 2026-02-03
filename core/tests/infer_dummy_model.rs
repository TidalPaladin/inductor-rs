use anyhow::{bail, Context, Result};
use approx::assert_abs_diff_eq;
use inductor_rs::{AotModel, Device};
use ndarray::Array4;
use std::path::PathBuf;

#[test]
fn infer_dummy_model_outputs() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_path = manifest_dir.join("../tests/fixtures/dummy_model.pt2");
    if !model_path.exists() {
        bail!(
            "Missing test fixture at {}. Run `make test-fixtures` first.",
            model_path.display()
        );
    }

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
