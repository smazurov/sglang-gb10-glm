"""GLM loader/NextN route guards and the ModelOpt metadata canonicaliser.

The guards execute exact added code from the active MXFP8 patch by default:
    python3 -m unittest tests.test_speculation_guards
    GLM_STAGE=<tree with the full patch series applied> \
        python3 -m unittest tests.test_speculation_guards
The scoped default supplies only the NextN base-constructor test double; it
never imports SGLang or carries a whole-file overlay.
"""
import ast
import importlib.util
import os
from pathlib import Path
import textwrap
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MXFP8_PATCH = ROOT / "patches" / "sglang-glm53-spark-mxfp8.patch"

norm_spec = importlib.util.spec_from_file_location(
    "normalize", ROOT / "gates" / "normalize_modelopt.py")
normalize = importlib.util.module_from_spec(norm_spec)
norm_spec.loader.exec_module(normalize)


def added_guard(flat_name: str, anchor: str) -> str:
    """Extract one contiguous addition, fail closed if its patch scope changes."""
    text = MXFP8_PATCH.read_text()
    header = f"+++ b/python/sglang/srt/models/{flat_name}\n"
    section = text.split(header)[1].split("--- a/")[0]
    assert section.count(anchor) == 1, anchor
    lines = section[section.index(anchor):].splitlines()
    added = []
    for line in lines:
        if not line.startswith("+"):
            break
        added.append(line[1:])
    source = textwrap.dedent("\n".join(added))
    ast.parse(source)  # No copied implementation or partial guard accepted.
    return source


def overlay_source(flat_name: str) -> str:
    stage = os.environ.get("GLM_STAGE")
    if stage:
        return (Path(stage) / "python/sglang/srt/models" / flat_name).read_text()
    if flat_name == "glm5_next.py":
        guard = added_guard(flat_name, "+    @staticmethod")
        helper = added_guard(flat_name, "+def _normalize_modelopt_mxfp8_scale_name")
        return "class Glm5NextForConditionalGeneration:\n" + textwrap.indent(guard, "    ") + "\n" + helper
    assert flat_name == "glm5_next_nextn.py"
    guard = added_guard(flat_name, '+        if quant_config is not None and quant_config.get_name() == "modelopt_mixed":')
    # Only the base call is a stub in scoped mode. Applied-tree mode above
    # exercises the real constructor as before; the guard itself is exact.
    return ('class Glm5NextForConditionalGenerationNextN:\n'
            '    def __init__(self, config, quant_config=None, prefix: str = ""):\n'
            + textwrap.indent(guard, "        ")
            + '\n        super().__init__(config, quant_config=quant_config, prefix=prefix)\n')


class NormalizeTests(unittest.TestCase):
    def test_metadata_normalizes_quant_routes_without_changing_formats(self):
        original = {"architectures": ["Glm5NextForConditionalGeneration"],
                    "text_config": {"first_k_dense_replace": 3, "num_hidden_layers": 4},
                    "quantization_config": {"quant_algo": "MIXED_PRECISION", "quantized_layers": {
                        "model.language_model.layers.3.mlp.experts": {"quant_algo": "NVFP4"},
                        "model.language_model.layers.4.mlp.experts": {"quant_algo": "W4A16_NVFP4"}}}}
        fixed = normalize.normalize(original)
        self.assertEqual(fixed["quantization_config"]["quantized_layers"]["model.layers.3.mlp.experts"], {"quant_algo": "NVFP4"})
        self.assertIn("model.language_model.layers.3.mlp.experts", original["quantization_config"]["quantized_layers"])
        self.assertNotIn("kv_cache_quant_algo", original["quantization_config"])
        del original["quantization_config"]["quantized_layers"]["model.language_model.layers.3.mlp.experts"]
        with self.assertRaisesRegex(ValueError, "Missing NVFP4"):
            normalize.normalize(original)


class GuardTests(unittest.TestCase):
    def test_loader_guard_and_scale_alias_without_torch(self):
        source = ast.parse(overlay_source("glm5_next.py"))
        model = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Glm5NextForConditionalGeneration")
        guard = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_mixed_expert_routes")
        helper = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "_normalize_modelopt_mxfp8_scale_name")
        guard.decorator_list = []
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[guard, helper], type_ignores=[])), "guard-and-alias", "exec"), ns)
        routes = {"model.layers.3.mlp.experts": "NVFP4", "model.layers.0.self_attn.q_proj": "MXFP8",
                  "model.decoder.self_attn.q_proj": "MXFP8"}
        qc = SimpleNamespace(get_name=lambda: "modelopt_mixed", _resolve_quant_algo=routes.get)
        cfg = SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=4)
        ns["_validate_mixed_expert_routes"](cfg, qc, "")
        alias = ns["_normalize_modelopt_mxfp8_scale_name"]
        self.assertEqual(alias("model.layers.0.self_attn.q_proj.weight_scale", qc), "model.layers.0.self_attn.q_proj.weight_scale_inv")
        self.assertEqual(alias("model.decoder.self_attn.q_proj.weight_scale", qc), "model.decoder.self_attn.q_proj.weight_scale_inv")
        self.assertEqual(alias("model.layers.3.mlp.experts.weight_scale", qc), "model.layers.3.mlp.experts.weight_scale")
        routes.clear()
        with self.assertRaisesRegex(RuntimeError, "Refusing unquantized"):
            ns["_validate_mixed_expert_routes"](cfg, qc, "")

    def test_nextn_guard_rejects_unmapped_draft_before_construction(self):
        tree = ast.parse(overlay_source("glm5_next_nextn.py"))
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Glm5NextForConditionalGenerationNextN")
        constructor = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        class Base:
            constructed = False
            def __init__(self, *args, **kwargs):
                Base.constructed = True
        ns = {"Base": Base}
        cls = ast.ClassDef(name="Draft", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[constructor], decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "nextn-guard", "exec"), ns)
        routes = {}
        qc = SimpleNamespace(get_name=lambda: "modelopt_mixed", _resolve_quant_algo=routes.get)
        with self.assertRaisesRegex(RuntimeError, "Invalid mixed NextN route"):
            ns["Draft"](SimpleNamespace(), qc)
        self.assertFalse(Base.constructed)
        routes.update({"model.decoder.mlp.experts": "W4A16_NVFP4",
                       "model.decoder.self_attn.fused_qkv_a_proj_with_mqa": "MXFP8",
                       "model.decoder.mlp.shared_experts.gate_up_proj": "MXFP8"})
        ns["Draft"](SimpleNamespace(), qc)
        self.assertTrue(Base.constructed)
        # Every draft route is required, not just the first expert route.
        for name in list(routes):
            expected = routes.pop(name)
            for invalid in (None, "BF16"):
                if invalid is not None:
                    routes[name] = invalid
                Base.constructed = False
                with self.subTest(route=name, invalid=invalid):
                    with self.assertRaisesRegex(RuntimeError, "Invalid mixed NextN route"):
                        ns["Draft"](SimpleNamespace(), qc)
                    self.assertFalse(Base.constructed)
            routes[name] = expected


if __name__ == "__main__":
    unittest.main()
