#!/usr/bin/env python3
"""CPU-side regression tests for the glm-flash-v2 image gate (gate-pr33391.py).

The gate's CUDA sections can only run on a GB10 host, but their STRUCTURE is
checkable anywhere. Original blocker: the wrapper gate used to assert a 2-D
prefix shape against combine's 4-D [1, seq, heads, d_v] output and printed
PASS before validating — a successful CUDA run would have failed the gate.
These tests pin the fixed structure with AST, no torch needed.
"""
import ast
import unittest
from pathlib import Path

GATE = Path(__file__).resolve().parent.parent / "gates" / "gate-pr33391.py"


def wrapper_function(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "wrapper_main":
            return node
    raise AssertionError("wrapper_main not found")


class TestWrapperShapeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(GATE.read_text())

    def test_wrapper_asserts_full_batched_shape(self):
        """combine returns [1, seq_len, heads, d_v] unchanged; the gate must
        assert that complete shape, not a squeezed 2-D prefix."""
        fn = wrapper_function(self.tree)
        src = ast.unparse(fn)
        self.assertIn("(1, SEQ_LEN, NUM_HEADS, D_V)", src)
        self.assertNotIn("out.shape[:2]", src)

    def test_pass_printed_only_after_validation(self):
        """The PASS message must come after the shape assert, so a shape
        regression fails the gate instead of logging PASS first."""
        fn = wrapper_function(self.tree)
        # The wrapper body is a try block; flatten its statement list in order.
        def flat(nodes):
            out = []
            for n in nodes:
                if isinstance(n, (ast.Try, ast.TryStar)):
                    out.extend(flat(n.body) + flat(n.orelse) + flat(n.finalbody))
                elif isinstance(n, (ast.If, ast.With)):
                    out.extend(flat(n.body))
                else:
                    out.append(n)
            return out
        printed, asserted = None, None
        for index, node in enumerate(flat(fn.body)):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                name = getattr(call.func, "id", "")
                if name == "print" and "PASS" in ast.unparse(call):
                    printed = index
            if isinstance(node, ast.Assert):
                if "wrapper shape" in ast.unparse(node):
                    asserted = index
        self.assertIsNotNone(printed, "wrapper PASS print missing")
        self.assertIsNotNone(asserted, "wrapper shape assert missing")
        self.assertLess(asserted, printed, "PASS printed before the shape assert")

    def test_route_main_resolves_real_mixed_routes(self):
        """The gate must resolve routes through the real config class and the
        real NextN mapper (canonical model.* namespace + model.decoder.*), and
        exercise the NVFP4 quantization kernel + MoE scale helpers."""
        names = {node.name for node in ast.walk(self.tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIn("route_main", names)
        src = ast.unparse(self.tree)
        for token in (
            "ModelOptMixedPrecisionConfig.from_config",
            "apply_weight_name_mapper",
            "Glm5NextForConditionalGenerationNextN",
            "model.decoder.mlp.experts",
            "group_size",
            "fp4_quantize",
            "_compute_gemm1_alphas",
        ):
            self.assertIn(token, src, f"route gate missing {token}")

    def test_gate_refuses_wrong_wrapper_shape_string(self):
        """Guard the exact regression that shipped in round-2 review."""
        src = ast.unparse(self.tree)
        self.assertNotIn("out.shape[:2]", src)
        self.assertNotIn("out.shape[-1]", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
