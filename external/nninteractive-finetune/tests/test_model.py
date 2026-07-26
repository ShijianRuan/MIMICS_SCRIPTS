import torch

from nninteractive_finetune.model import configure_trainable_parameters


class ToyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.stem = torch.nn.Sequential(
            torch.nn.Conv3d(8, 4, 3, padding=1),
            torch.nn.InstanceNorm3d(4, affine=True),
        )
        self.encoder.stages = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv3d(4, 4, 3, padding=1),
                    torch.nn.InstanceNorm3d(4, affine=True),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv3d(4, 4, 3, padding=1),
                    torch.nn.InstanceNorm3d(4, affine=True),
                ),
            ]
        )
        self.decoder = torch.nn.Module()
        self.decoder.stages = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Conv3d(4, 4, 3, padding=1)),
                torch.nn.Sequential(torch.nn.Conv3d(4, 4, 3, padding=1)),
            ]
        )
        self.decoder.seg_layers = torch.nn.ModuleList(
            [torch.nn.Conv3d(4, 2, 1), torch.nn.Conv3d(4, 2, 1)]
        )


def test_clopa_in_only_selects_instance_norm_affine():
    network = ToyNetwork()
    result = configure_trainable_parameters(network, "clopa_in")
    assert result["names"]
    assert all(
        name.endswith(".weight") or name.endswith(".bias") for name in result["names"]
    )
    assert all("encoder" in name for name in result["names"])
    assert not any("conv" in name for name in result["names"])


def test_clopa_conv_selects_boundary_convolutions_not_middle_encoder():
    network = ToyNetwork()
    result = configure_trainable_parameters(network, "clopa_conv")
    names = result["names"]
    assert "encoder.stem.0.weight" in names
    assert "encoder.stages.0.0.weight" in names
    assert "encoder.stages.1.0.weight" not in names
    assert "decoder.stages.1.0.weight" in names
    assert "decoder.seg_layers.1.weight" in names
