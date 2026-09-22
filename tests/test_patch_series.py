"""Bounded diagnostic removal: inspect the integration patch's resulting hunks.

Full-tree application/compile is scripts/verify-tree.py (CI static tier and
the image build), not emulated here. Removed lines are excluded so old
upstream comments are not mistaken for retained APIs. Check balanced
functional-call edits by default; GLM_STAGE (path to a fully applied sglang
tree) additionally checks the calls in the applied source.

Ported from the eval suite:test_sglang_patch_subset.py to the new layout.
"""
from pathlib import Path
import re
import os
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "patches"


def postimages():
    result = {}
    for section in re.split(r"(?=^diff --git )", (PATCHES / "sglang-glm53-integration.patch").read_text(), flags=re.M):
        if not section:
            continue
        path = re.search(r"^\+\+\+ b/(.+)$", section, re.M).group(1)
        result[path] = "\n".join(line[1:] for line in section.splitlines()
                                 if line.startswith(" ") or (line.startswith("+") and not line.startswith("+++")))
    return result


class PatchSubsetTests(unittest.TestCase):
    def test_diagnostics_and_standalone_benchmark_are_absent(self):
        images = postimages()
        text = "\n".join(images) + "\n" + "\n".join(images.values())
        for name in ("extend_mem_profile", "mem_forensics", "memory_forensics",
                     "SGLANG_EXTEND_MEM_PROFILE", "SGLANG_MEM_FORENSICS",
                     "bench_kda_verify_sweep", "_execute_eager", "_forward_cuda_impl"):
            with self.subTest(name=name):
                self.assertNotIn(name, text)

    def test_unwrapped_operations_remain(self):
        # Most unwrapped operations now match pinned upstream and disappear
        # from the diff entirely. Reject unbalanced deletions in patch mode;
        # applied-source mode additionally checks their actual order.
        patch = (PATCHES / "sglang-glm53-integration.patch").read_text()
        stage = os.environ.get("GLM_STAGE")
        srt = "python/sglang/srt/"
        expected = {
            "layers/attention/dsa/dsa_indexer.py": ["def forward_cuda("],
            "layers/attention/dsa_backend.py": ["set_mla_kv_buffer(", "self._forward_flashinfer_sparse_mla("],
            "layers/attention/linear/kda_backend.py": ["causal_conv1d_fn(", "self.kernel_dispatcher.extend("],
            "managers/mm_utils.py": ["embedding, mask, input_ids = get_embedding_and_mask(", "_offload_items_to_host(items)"],
            "model_executor/model_runner.py": ["self._prepare_eager_forward_batch(forward_batch)",
                                               "self._maybe_execute_deferred_mamba_cow_and_clear(forward_batch)",
                                               "ret = self.eager_runner.execute(",
                                               "forward_batch.post_forward_mlp_sync_batch(ret)"],
        }
        for path, calls in expected.items():
            with self.subTest(path=path):
                section = patch.split(f"+++ b/{srt}{path}\n")[1].split("diff --git ")[0]
                added = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+"))
                removed = "\n".join(line[1:] for line in section.splitlines() if line.startswith("-"))
                for call in calls:
                    self.assertGreaterEqual(added.count(call), removed.count(call), call)
                if stage:
                    text = (Path(stage) / srt / path).read_text()
                    positions = [text.index(call) for call in calls]
                    self.assertEqual(positions, sorted(positions))

    def test_functional_mm_tests_remain_without_profiler_spies(self):
        text = postimages()["test/registered/unit/managers/test_mm_embed_mapped_embedder.py"]
        for name in ("test_mapped_embedders_for_every_modality",
                     "test_mixed_mapped_and_model_fallback_embedders",
                     "test_missing_embedder_still_asserts_by_modality",
                     "torch.testing.assert_close", "pytest.raises(AssertionError"):
            self.assertIn(name, text)
        self.assertEqual(text.count("_assert_scattered(input_embeds, embedding, input_ids, image, audio)"), 3)
        self.assertNotIn("_PhaseSpy", text)
        self.assertNotIn("spy.tags", text)


if __name__ == "__main__":
    unittest.main()
