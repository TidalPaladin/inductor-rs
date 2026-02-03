#!/usr/bin/env python3
"""Generate a dummy model compiled with AOTInductor for testing.

This script creates a minimal PyTorch model with deterministic outputs,
then compiles it with AOTInductor to produce a .pt2 package that can be
used for testing the Rust FFI bindings.

Usage:
    python generate_test_model.py -o tests/fixtures/dummy_model.pt2
    python generate_test_model.py --height 512 --width 256 -o model.pt2 --verify
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


class DummyModel(nn.Module):
    """Minimal model with deterministic outputs for testing.

    Returns three output tensors with known values:
    - output_0: scalar 0.42 per batch item
    - output_1: spatial tensor with 0.1 values
    - output_2: feature vector with [0.1, 0.2, 0.3, 0.4, 0.5] per batch item
    """

    def __init__(self, height: int = 224, width: int = 224):
        super().__init__()
        self.h = height
        self.w = width
        self.register_buffer("_height", torch.tensor(height))
        self.register_buffer("_width", torch.tensor(width))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning 3 output tensors.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Tuple of 3 tensors with deterministic values.
        """
        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Output 0: scalar per batch (like a classification score)
        output_0 = torch.full((B,), 0.42, dtype=dtype, device=device)

        # Output 1: spatial tensor (like a heatmap)
        output_1 = torch.full((B, self.h, self.w), 0.1, dtype=dtype, device=device)

        # Output 2: feature vector
        output_2 = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5]] * B,
            dtype=dtype,
            device=device,
        )

        return output_0, output_1, output_2


def generate(
    output_path: str,
    height: int = 224,
    width: int = 224,
    device: str = "cpu",
    batch_size: int = 1,
) -> str:
    """Generate a compiled AOTInductor model package.

    Args:
        output_path: Path to save the compiled .pt2 package.
        height: Input/output height.
        width: Input/output width.
        device: Target device ("cpu" or "cuda:N").
        batch_size: Example batch size for tracing.

    Returns:
        Path to the generated .pt2 package.
    """
    # Create model
    model = DummyModel(height, width).eval()

    # Create example input
    example_input = torch.randn(batch_size, 1, height, width)

    # Move to device if needed
    if device != "cpu":
        model = model.to(device)
        example_input = example_input.to(device)

    # Export the model
    print(f"Exporting model with input shape {example_input.shape}...")
    exported = torch.export.export(model, (example_input,))

    # Compile with AOTInductor
    print(f"Compiling with AOTInductor to {output_path}...")

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use aoti_compile_and_package
    so_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=output_path,
    )

    print(f"Generated: {so_path}")
    return so_path


def verify(model_path: str, height: int, width: int, device: str = "cpu", batch_size: int = 1) -> bool:
    """Verify the compiled model works correctly.

    Args:
        model_path: Path to the compiled .pt2 package.
        height: Expected height.
        width: Expected width.
        device: Device to run on.

    Returns:
        True if verification passed.
    """
    print(f"\nVerifying model at {model_path}...")

    # Load the compiled model
    from torch._inductor import aoti_load_package

    device_index = -1
    if device != "cpu":
        if ":" in device:
            device_index = int(device.split(":")[1])
        else:
            device_index = 0

    runner = aoti_load_package(model_path, device_index=device_index)

    def run_runner(runner_obj, inputs):
        if hasattr(runner_obj, "run"):
            return runner_obj.run(inputs)
        if hasattr(runner_obj, "forward"):
            try:
                return runner_obj.forward(*inputs)
            except TypeError:
                return runner_obj.forward(inputs)
        if callable(runner_obj):
            try:
                return runner_obj(*inputs)
            except TypeError:
                return runner_obj(inputs)
        raise RuntimeError(f"Unsupported runner type: {type(runner_obj)}")

    # Create test input
    test_input = torch.randn(batch_size, 1, height, width)
    if device != "cpu":
        test_input = test_input.to(device)

    # Run inference
    outputs = run_runner(runner, [test_input])

    if isinstance(outputs, tuple):
        outputs = list(outputs)
    elif not isinstance(outputs, list):
        outputs = [outputs]

    # Verify output count
    if len(outputs) != 3:
        print(f"ERROR: Expected 3 outputs, got {len(outputs)}")
        return False

    # Verify shapes
    expected_shapes = [
        (batch_size,),  # output_0: scalar per batch
        (batch_size, height, width),  # output_1: spatial
        (batch_size, 5),  # output_2: feature vector
    ]

    names = ["output_0", "output_1", "output_2"]

    for output, expected, name in zip(outputs, expected_shapes, names, strict=False):
        if tuple(output.shape) != expected:
            print(f"ERROR: {name} shape mismatch: expected {expected}, got {tuple(output.shape)}")
            return False
        print(f"  {name}: shape={tuple(output.shape)}, mean={output.float().mean().item():.4f}")

    # Verify deterministic values
    output_0 = outputs[0]
    if not torch.allclose(output_0, torch.full_like(output_0, 0.42), atol=1e-5):
        print(f"ERROR: output_0 values incorrect, expected 0.42, got {output_0}")
        return False

    print("\nVerification PASSED!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate a dummy model compiled with AOTInductor for testing")
    parser.add_argument(
        "-o",
        "--output",
        default="tests/fixtures/dummy_model.pt2",
        help="Output path for the compiled .pt2 package",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Input/output height (default: 224)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Input/output width (default: 224)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Target device (default: cpu)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the generated model after compilation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Example batch size for tracing (default: 1)",
    )

    args = parser.parse_args()

    # Check PyTorch version
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version < (2, 1):
        print(f"ERROR: PyTorch >= 2.1 required for AOTInductor, got {torch.__version__}")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Generate the model
    try:
        so_path = generate(
            args.output,
            height=args.height,
            width=args.width,
            device=args.device,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"ERROR: Failed to generate model: {e}")
        sys.exit(1)

    # Verify if requested
    if args.verify:
        try:
            if not verify(so_path, args.height, args.width, args.device, args.batch_size):
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
