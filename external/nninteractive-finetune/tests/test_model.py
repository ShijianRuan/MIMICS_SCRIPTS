import torch

from nninteractive_finetune.model import (
    configure_trainable_parameters,
    network_parameter_fingerprint,
    state_dict_loaded_parameter_fingerprint,
)


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


class SharedAliasNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shared = torch.nn.Linear(2, 2, bias=False)
        self.primary = shared
        self.aliases = torch.nn.ModuleList([shared])


def test_full_state_dict_preserves_shared_parameter_update_on_reload():
    baseline = SharedAliasNetwork()
    baseline_state = {
        name: value.detach().clone()
        for name, value in baseline.state_dict().items()
    }

    # This reproduces the legacy registration bug: updating only the first
    # state-dict alias is silently overwritten by the later alias on load.
    partial = dict(baseline_state)
    partial["primary.weight"] = partial["primary.weight"] + 3.0
    assert state_dict_loaded_parameter_fingerprint(baseline, partial) == (
        network_parameter_fingerprint(baseline)
    )
    legacy_loaded = SharedAliasNetwork()
    legacy_loaded.load_state_dict(partial)
    assert network_parameter_fingerprint(legacy_loaded) == network_parameter_fingerprint(
        baseline
    )

    # Updating the real parameter makes every state-dict alias agree, so the
    # effective network survives the same load path used by nnInteractive.
    adapted = SharedAliasNetwork()
    adapted.load_state_dict(baseline_state)
    adapted.primary.weight.data.add_(3.0)
    complete = {
        name: value.detach().clone()
        for name, value in adapted.state_dict().items()
    }
    assert state_dict_loaded_parameter_fingerprint(adapted, complete) == (
        network_parameter_fingerprint(adapted)
    )
    reloaded = SharedAliasNetwork()
    reloaded.load_state_dict(complete)
    assert network_parameter_fingerprint(reloaded) == network_parameter_fingerprint(
        adapted
    )
    assert network_parameter_fingerprint(reloaded) != network_parameter_fingerprint(
        baseline
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
