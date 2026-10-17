"""Bounded model checks, including real approved-source migration."""
from pathlib import Path
import json
import tempfile
import time
import unittest

import torch

try:
    from . import model as v2
except ImportError:
    import model as v2

PROJECT = Path(__file__).resolve().parents[1]
RING = PROJECT / "ring_training/colab_results/20261017_111703_294103_b07af1/best_model.pt"
NEEDLE = PROJECT / "rotation_dataset/audits/20261017_100006_266720_f43a79f6_results/extracted/best_model.pt"


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        torch.manual_seed(42)

    def test_transfer_source_outputs_and_frozen_needle_after_update(self):
        model = v2.AugmentedRingModel()
        origin = v2.load_transfer_checkpoints(model, RING, NEEDLE)
        self.assertEqual((origin["source_ring_epoch"], origin["source_legacy_epoch"]), (33, 19))
        self.assertTrue(model.freeze_needle)
        image = torch.randn(1, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            ring_expected = model.ring_branch(image)
            needle_expected = model.needle_branch(image)
            initial = model(image)
        for name in ("arc_logits", "speed"):
            self.assertTrue(torch.equal(initial[name], ring_expected[name]), name)
        for name in ("points", "heatmap_logits"):
            self.assertTrue(torch.equal(initial[name], needle_expected[name]), name)
        self.assertEqual(tuple(initial["tick_heatmap_logits"].shape), (1, 6, 16, 16))
        self.assertEqual(tuple(initial["tick_points"].shape), (1, 6, 2))
        self.assertEqual(tuple(initial["tick_visibility_logits"].shape), (1, 6))
        old_needle = {k: v.clone() for k, v in model.needle_branch.state_dict().items()}
        old_tick = model.tick_decoder.heatmap_head.weight.detach().clone()
        model.configure_stage("finetune")
        model.train()
        self.assertFalse(model.needle_branch.training)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        output = model(image)
        loss = (output["tick_points"] - .2).square().mean() + output["arc_logits"].square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.needle_branch.parameters()))
        optimizer.step()
        for name, tensor in model.needle_branch.state_dict().items():
            self.assertTrue(torch.equal(tensor, old_needle[name]), name)
        model.eval()
        with torch.no_grad():
            updated = model(image)
        self.assertTrue(torch.equal(initial["points"], updated["points"]))
        self.assertFalse(torch.equal(old_tick, model.tick_decoder.heatmap_head.weight))

    def test_stage_groups_and_genuine_fresh_policy(self):
        model = v2.AugmentedRingModel()
        self.assertFalse(model.freeze_needle)
        model.configure_stage("new_heads")
        self.assertFalse(any(p.requires_grad for p in model.ring_branch.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.tick_decoder.parameters()))
        self.assertTrue(model.needle_branch.heatmap_head.weight.requires_grad)
        model.configure_stage("finetune", unfreeze_all_encoder=True)
        self.assertTrue(all(p.requires_grad for p in model.ring_branch.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.needle_branch.features.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.ring_branch.tick_head.parameters()))
        groups = {v2.parameter_group(n) for n, _ in model.named_parameters()}
        self.assertEqual(groups, {None, "new", "shared", "backbone", "needle"})
        model.train()
        for branch in (model.ring_branch, model.needle_branch):
            self.assertTrue(all(not m.training for m in branch.features.modules()
                                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)))
        result = model(torch.randn(1, 3, 64, 64))
        result["heatmap_logits"].square().mean().backward()
        self.assertGreater(float(model.needle_branch.features[0][0].weight.grad.abs().sum()), 0)

    def test_balanced_visibility_and_hidden_tick_mask(self):
        output = {"speed": torch.tensor([50.], requires_grad=True),
                  "points": torch.full((1, 2, 2), .5, requires_grad=True),
                  "heatmap_logits": torch.zeros(1, 2, 8, 8, requires_grad=True),
                  "tick_points": torch.full((1, 6, 2), .5, requires_grad=True),
                  "tick_heatmap_logits": torch.zeros(1, 6, 8, 8, requires_grad=True),
                  "arc_logits": torch.zeros(1, 1, 32, 32, requires_grad=True),
                  "tick_visibility_logits": torch.zeros(1, 6, requires_grad=True)}
        batch = {"speed": torch.tensor([50.]), "points": torch.full((1, 2, 2), .5),
                 "tick_points": torch.full((1, 6, 2), .2),
                 "tick_valid": torch.tensor([[0., 1., 1., 1., 1., 1.]]),
                 "arc_mask": torch.zeros(1, 1, 32, 32), "arc_valid": torch.zeros(1, 1, 32, 32)}
        loss, components = v2.training_loss(output, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(float(output["tick_points"].grad[0, 0].abs().sum()), 0.)
        self.assertEqual(float(output["tick_heatmap_logits"].grad[0, 0].abs().sum()), 0.)
        self.assertEqual(float(output["arc_logits"].grad.abs().sum()), 0.)
        gradients = output["tick_visibility_logits"].grad[0]
        self.assertAlmostEqual(float(gradients[0] / -gradients[1]), 19., places=5)
        self.assertGreater(components["visibility_loss"], 0.)

    def test_reject_wrong_source_before_either_branch_mutates(self):
        model = v2.AugmentedRingModel()
        old = model.ring_branch.features[0][0].weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            v2.load_transfer_checkpoints(model, RING, NEEDLE, expected_ring_sha256="0" * 64)
        payload = torch.load(NEEDLE, weights_only=True, map_location="cpu")
        payload["keypoint_order"] = ["tip", "pivot"]
        with tempfile.TemporaryDirectory(prefix="ringv2_") as directory:
            bad = Path(directory) / "bad.pt"
            torch.save(payload, bad)
            with self.assertRaisesRegex(ValueError, "output"):
                v2.load_transfer_checkpoints(model, RING, bad, expected_legacy_sha256=None)
        self.assertTrue(torch.equal(old, model.ring_branch.features[0][0].weight))


def run_tiny_fit(steps=80):
    """Diagnostic only: four training images, cached frozen encoder, no saved weights."""
    from ring_training.ring_data import RingDataset
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = v2.AugmentedRingModel()
    origin = v2.load_transfer_checkpoints(model, RING, NEEDLE)
    model.configure_stage("new_heads")
    dataset = RingDataset(PROJECT / "ring_geometry_pilot/review20")
    indices = [0, 1, 2, 5]
    samples = [dataset[index] for index in indices]
    image = torch.stack([sample["image"] for sample in samples])
    targets = torch.stack([sample["tick_points"] for sample in samples])
    valid = torch.stack([sample["tick_valid"] for sample in samples])
    model.eval()
    with torch.no_grad():
        skips, deep = {}, image
        for index, layer in enumerate(model.ring_branch.features):
            deep = layer(deep)
            if index in (1, 3, 8):
                skips[index] = deep
    height, width = skips[1].shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    centers = targets * torch.tensor([width - 1, height - 1])
    squared = ((xx[None, None] - centers[:, :, 0, None, None]).square()
               + (yy[None, None] - centers[:, :, 1, None, None]).square())
    density = torch.exp(-squared / (2 * 1.5**2)).flatten(2)
    density = density / density.sum(-1, keepdim=True).clamp_min(1e-12)
    optimizer = torch.optim.AdamW(model.tick_decoder.parameters(), lr=.001, weight_decay=.0001)

    def measure():
        logits, _ = model.tick_decoder(deep, skips)
        points = v2.legacy.heatmap_coordinates(logits)
        ce = (-(density * logits.flatten(2).log_softmax(-1)).sum(-1) * valid).sum() / valid.sum()
        coordinate = (torch.nn.functional.smooth_l1_loss(points, targets, beta=.02,
                                                        reduction="none").mean(-1) * valid).sum() / valid.sum()
        errors = (torch.linalg.vector_norm(points - targets, dim=-1) * 511)[valid.bool()]
        loss = ce + 5 * coordinate
        return loss, {"tick_heatmap_loss": float(ce.detach()), "tick_coordinate_loss": float(coordinate.detach()),
                      "visible_mean_error_px": float(errors.detach().mean()),
                      "visible_p95_error_px": float(torch.quantile(errors.detach(), .95))}

    history, start = [], time.perf_counter()
    for step in range(steps + 1):
        if step == 0 or step % 10 == 0 or step == steps:
            with torch.no_grad():
                _, metrics = measure()
            history.append({"step": step, **metrics})
            print(json.dumps(history[-1]), flush=True)
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss, _ = measure()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.tick_decoder.parameters(), 5.)
        optimizer.step()
    report = {"purpose": "four-training-image tick-decoder fit; no held-out accuracy claim",
              "source": origin, "sample_ids": [dataset.records[i]["sample_id"] for i in indices],
              "input_size": 512, "updates": steps, "lr": .001, "encoder": "frozen cached epoch33 features",
              "optimizer": "AdamW", "trained": "tick_decoder only", "weights_saved": False,
              "history": history, "elapsed_seconds": time.perf_counter() - start}
    folder = PROJECT / "ring_training_v2/checks"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "four_image_tick_fit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    import sys
    if "--tinyfit" in sys.argv:
        run_tiny_fit()
    else:
        unittest.main(verbosity=2)
