"""Offline smoke tests for the flexict-finetune standalone repo.

Run with the host python_env:
    python_env/python.exe integrations/flexict-finetune/tests/test_flexict_pkg.py

Covers:
  - flexict/ package imports and backbone construction (2D/3D dual mode)
  - Primus decoder forward on fp32 dummy tensors (no NaN)
  - pretrained weight loading (missing==unexpected==0) when weights present
  - uncertainty.run_batch on synthetic 2-mask disagreement data + ranking
  - trainer file: env-var contract, no import needed (nnU-Net absent ok)

Weights (~576MB x2) are NOT required: weight tests skip when absent.
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, os.path.join(REPO, "flexict")):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestFlexictPackage(unittest.TestCase):
    def test_package_import(self):
        import flexict.models as m
        self.assertTrue(callable(m.flexi_ct_backbone_base))

    @staticmethod
    def _build_backbone():
        from flexict.models import flexi_ct_backbone_base
        model = flexi_ct_backbone_base(
            patch_size=8, in_chans=1, n_storage_tokens=4,
            qkv_bias=False, mask_k_bias=True,
            drop_path_rate=0.2, layerscale_init=1.0e-05)
        # cls/storage/mask tokens are torch.empty — initialize before use,
        # otherwise garbage (possibly NaN) propagates into features.
        model.init_weights()
        return model

    def test_backbone_build_2d_input(self):
        """flexi_ct_backbone_base branches on input rank: 4D -> 2D path."""
        import torch
        model = self._build_backbone()
        model.eval()
        with torch.no_grad():
            feats = model.get_intermediate_layers(
                torch.zeros(1, 1, 64, 64), n=[3], reshape=True)
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].shape[0], 1)
        self.assertEqual(feats[0].shape[1], 864)
        self.assertEqual(feats[0].shape[2], 64 // 8)
        self.assertEqual(feats[0].shape[3], 64 // 8)
        self.assertTrue(torch.isfinite(feats[0]).all())

    def test_backbone_build_3d_input(self):
        """5D input takes the 3D patch-embed branch and produces D,H,W features."""
        import torch
        model = self._build_backbone()
        model.eval()
        with torch.no_grad():
            feats = model.get_intermediate_layers(
                torch.zeros(1, 1, 8, 64, 64), n=[3], reshape=True)
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].shape[1], 864)
        self.assertEqual(feats[0].shape[2], 8 // 8)
        self.assertTrue(torch.isfinite(feats[0]).all())


class TestPrimusDecoder(unittest.TestCase):
    def test_primus2d_forward_fp32(self):
        """Primus 2D decoder forward on an fp32 dummy tensor: finite output."""
        import torch
        from flexict.flexict_primus import FlexiCTPrimus2D
        model = TestFlexictPackage._build_backbone()
        net = FlexiCTPrimus2D(
            embed_dim=864, patch_size=8, num_classes=2,
            dino_encoder=model, interaction_indices=[3, 7, 11, 15])
        net.eval()
        with torch.no_grad():
            out = net(torch.zeros(1, 1, 64, 64))
        self.assertEqual(out.shape[:2], (1, 2))
        self.assertTrue(torch.isfinite(out).all())


class TestPretrainedWeights(unittest.TestCase):
    W2D = os.path.join(REPO, "weights", "flexict_2d", "model.safetensors")
    W3D = os.path.join(REPO, "weights", "flexict_3d", "model.safetensors")

    @classmethod
    def setUpClass(cls):
        # Weight tests are heavyweight (~576MB each); skip entirely when
        # weights are not installed (e.g. CI or fresh clones).
        if not (os.path.isfile(cls.W2D) and os.path.isfile(cls.W3D)):
            raise unittest.SkipTest(
                "pretrained weights not present (weights/ is gitignored); "
                "copy from the distribution share first")

    def test_flexict2d_weight_load(self):
        from safetensors.torch import load_file
        from flexict.models import flexi_ct_backbone_base
        sd = load_file(self.W2D)
        self.assertTrue(all(k.startswith("backbone.") for k in sd))
        model = flexi_ct_backbone_base(
            patch_size=8, in_chans=1, n_storage_tokens=4,
            qkv_bias=False, mask_k_bias=True,
            drop_path_rate=0.2, layerscale_init=1.0e-05)
        sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)

    def test_flexict3d_weight_load(self):
        from safetensors.torch import load_file
        from flexict.models import flexi_ct_backbone_base
        sd = load_file(self.W3D)
        self.assertTrue(all(k.startswith("backbone.") for k in sd))
        model = flexi_ct_backbone_base(
            patch_size=8, in_chans=1, n_storage_tokens=4,
            qkv_bias=False, mask_k_bias=True,
            drop_path_rate=0.2, layerscale_init=1.0e-05)
        sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)


class TestUncertainty(unittest.TestCase):
    def _make_case(self, mask_dirs, case, shape=(8, 8, 4)):
        import numpy as np
        import nibabel as nib
        import os
        rng = np.random.RandomState(42)
        base = rng.rand(*shape) > 0.5
        for d, frac in zip(mask_dirs, (1.0, 0.6)):
            arr = base.copy()
            if frac < 1.0:
                flip = rng.rand(*shape) < frac
                arr = np.where(flip, ~arr, arr)
            nib.save(nib.Nifti1Image(arr.astype(np.uint8), np.eye(4)),
                     os.path.join(d, case + ".nii.gz"))

    def test_run_batch_disagreement_and_ranking(self):
        import numpy as np
        import nibabel as nib
        from uncertainty import uncertainty as unc
        with tempfile.TemporaryDirectory() as tmp:
            d1 = os.path.join(tmp, "alg_a"); os.makedirs(d1)
            d2 = os.path.join(tmp, "alg_b"); os.makedirs(d2)
            out = os.path.join(tmp, "out")
            self._make_case([d1, d2], "case001")
            self._make_case([d1, d2], "case002")
            with open(os.path.join(tmp, "cases.txt"), "w") as fh:
                fh.write("case001\ncase002\n")
            unc.run_batch(
                mask_dirs=[d1, d2],
                out_dir=out,
                case_list_txt=os.path.join(tmp, "cases.txt"),
                filename_template="{case}.nii.gz",
                method="disagreement",
                target_labels=[1],
                min_masks=2,
            )
            self.assertTrue(os.path.isfile(
                os.path.join(out, "case001_uncertainty_disagreement.nii.gz")))
            self.assertTrue(os.path.isfile(os.path.join(out, "case001_consensus.nii.gz")))
            csv_path = os.path.join(out, "ranking.csv")
            rows = unc.analyze_uncertainty_dir(
                out_dir=out, method="disagreement",
                sort_key="integrated", report_csv=csv_path)
            self.assertTrue(rows)
            self.assertTrue(os.path.isfile(csv_path))
            with open(csv_path) as fh:
                header = fh.readline().strip().split(",")
            self.assertIn("case", header)


class TestTrainerContract(unittest.TestCase):
    """The trainer needs nnU-Net installed to import, so test the file as text:
    env-var contract and class names that the Mimics pipeline relies on."""

    TRAINER = os.path.join(REPO, "trainers", "flexict_trainer.py")

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(cls.TRAINER):
            raise unittest.SkipTest("trainer file not found")

    def test_env_var_names(self):
        with open(self.TRAINER, encoding="utf-8") as fh:
            src = fh.read()
        for var in ("FLEXICT_EXT_DIR", "FLEXICT2D_CKPT", "FLEXICT3D_CKPT",
                    "NUM_EPOCHS", "MIRROR_DISABLE_AXES"):
            self.assertIn(var, src)

    def test_trainer_class_names(self):
        with open(self.TRAINER, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("class flexict2d_Trainer", src)
        self.assertIn("class flexict3d_Trainer", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
