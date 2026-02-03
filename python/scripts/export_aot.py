#!/usr/bin/env python
"""Export a PyTorch model to AOTInductor format for Rust inference.

This is a template script that demonstrates how to export PyTorch models
using torch.export and AOTInductor to create .pt2 packages that can be
loaded from C++/Rust.

IMPORTANT: Set TORCH_COMPILE_DISABLE=1 to avoid FakeTensorMode conflicts
between @torch.compile decorators and torch.export tracing.

Example usage:
    TORCH_COMPILE_DISABLE=1 python scripts/export_aot.py \
        --output model.pt2 \
        --device cpu \
        --verify

The exported model can then be loaded using the Rust CLI:
    inductor-rs infer --model model.pt2 --device cpu --input input.json

Note: The model is exported with a static batch size (default: 1). Dynamic
batch sizes may or may not work depending on the model.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch._dynamo
import torch.nn as nn


def load_model() -> nn.Module:
    """Load your model here.

    CUSTOMIZE: Replace this with your actual model loading code.

    Returns:
        A PyTorch model ready for export.

    Example:
        # Load from checkpoint
        model = MyModel()
        model.load_state_dict(torch.load("checkpoint.pt"))
        return model

        # Or load from safetensors
        from safetensors.torch import load_model as load_safetensors
        model = MyModel()
        load_safetensors(model, "weights.safetensors")
        return model
    """

    # Example: Simple dummy model for testing
    class DummyModel(nn.Module):
        """Simple model that returns sum and mean of input."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 2)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning two outputs.

            Args:
                x: Input tensor of shape (B, 1, H, W)

            Returns:
                Tuple of (logits, features)
            """
            features = self.conv(x)
            pooled = self.pool(features).flatten(1)
            logits = self.fc(pooled)
            return logits.sigmoid(), pooled

    return DummyModel()


def get_example_inputs(
    batch_size: int = 1,
    channels: int = 1,
    height: int = 224,
    width: int = 224,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """Create example inputs for tracing.

    CUSTOMIZE: Modify this to match your model's expected input shape and type.

    Args:
        batch_size: Batch size for tracing.
        channels: Number of input channels.
        height: Input height.
        width: Input width.
        device: Target device.
        dtype: Input data type.

    Returns:
        Tuple of example input tensors.
    """
    x = torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)
    return (x,)


def verify(
    model_path: str,
    example_inputs: tuple[torch.Tensor, ...],
    device: str = "cpu",
) -> bool:
    """Verify the compiled model works correctly.

    Args:
        model_path: Path to the compiled .pt2 file.
        example_inputs: Example inputs for testing.
        device: Device to run on.

    Returns:
        True if verification passed.
    """
    print(f"\nVerifying model at {model_path}...")

    # Load the compiled model
    from torch._inductor import aoti_load_package

    device_index = -1
    if device != "cpu" and ":" in device:
        device_index = int(device.split(":")[1])

    runner = aoti_load_package(model_path, device_index=device_index)

    # Run inference
    outputs = runner(*example_inputs)

    # Handle single output vs tuple
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

    print(f"  Number of outputs: {len(outputs)}")

    for i, output in enumerate(outputs):
        if isinstance(output, torch.Tensor):
            print(
                f"  Output {i}: shape={tuple(output.shape)}, "
                f"mean={output.float().mean().item():.4f}, "
                f"min={output.float().min().item():.4f}, "
                f"max={output.float().max().item():.4f}"
            )
        else:
            print(f"  Output {i}: {type(output)}")

    print("\nVerification PASSED!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to AOTInductor format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path for .pt2 package",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export (cpu or cuda, default: cpu)",
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
        help="Batch size for exported model (default: 1)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Input height (default: 224)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Input width (default: 224)",
    )
    args = parser.parse_args()

    # Check environment
    if os.environ.get("TORCH_COMPILE_DISABLE") != "1":
        print("WARNING: TORCH_COMPILE_DISABLE=1 is not set. This may cause export failures.")
        print("Consider running with: TORCH_COMPILE_DISABLE=1 python scripts/export_aot.py ...")

    # Check PyTorch version
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version < (2, 1):
        print(f"ERROR: PyTorch >= 2.1 required for AOTInductor, got {torch.__version__}")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    device = torch.device(args.device)

    # Log GPU information for CUDA/ROCm exports
    if device.type == "cuda":
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            print(f"Target GPU: {props.name}")
            if hasattr(props, "gcnArchName"):
                print(f"  Architecture: {props.gcnArchName}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("WARNING: CUDA device requested but torch.cuda.is_available() is False")

    # Load model
    print("\nLoading model...")
    model = load_model()
    model.to(device)
    model.eval()

    # Create example input
    print(f"Creating example input: batch_size={args.batch_size}, height={args.height}, width={args.width}")
    example_inputs = get_example_inputs(
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        device=args.device,
    )

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        outputs = model(*example_inputs)
    if isinstance(outputs, (list, tuple)):
        print(f"  Output shapes: {[tuple(o.shape) for o in outputs]}")
    else:
        print(f"  Output shape: {tuple(outputs.shape)}")

    # Reset dynamo cache
    print("Resetting dynamo cache...")
    torch._dynamo.reset()

    # Re-create example input after reset
    example_inputs = get_example_inputs(
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        device=args.device,
    )

    # Export using torch.export
    print("Exporting model with torch.export...")
    exported = torch.export.export(model, example_inputs)

    # Compile with AOTInductor
    print(f"Compiling with AOTInductor to {args.output}...")

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Use aoti_compile_and_package to create a .pt2 package
    so_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(args.output),
    )
    print(f"Successfully exported to {so_path}")

    # Print summary
    print("\nExport Summary:")
    print(f"  Output: {args.output}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Input shape: ({args.batch_size}, 1, {args.height}, {args.width})")

    # Verify if requested
    if args.verify:
        try:
            if not verify(str(so_path), example_inputs, args.device):
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
