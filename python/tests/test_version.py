def test_torch_version_string():
    import torch

    assert isinstance(torch.__version__, str)
    assert torch.__version__.strip() != ""
