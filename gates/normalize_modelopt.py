#!/usr/bin/env python3
"""Derive SGLang GLM5 metadata; preserve quant formats and all tensor payloads."""
import copy
import json
import sys


def normalize(config):
    if config.get("architectures") != ["Glm5NextForConditionalGeneration"]:
        raise ValueError("This metadata adapter is specific to Glm5NextForConditionalGeneration")
    out = copy.deepcopy(config)
    quant = out["quantization_config"]
    if quant.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("Expected MIXED_PRECISION")
    if quant.get("kv_cache_quant_algo") is not None or quant.get("kv_cache_scheme") is not None:
        raise ValueError("Refusing to replace checkpoint-provided KV quantization metadata")
    quant["kv_cache_quant_algo"] = None
    original = quant["quantized_layers"]
    if not original:
        raise ValueError("Empty quantization map")
    mapped = {}
    for name, value in original.items():
        # Mirrors Glm5Next.load_weights' removal of language_model. from names.
        name = name.replace("language_model.", "")
        if name in mapped:
            raise ValueError(f"Quantization-name collision: {name}")
        mapped[name] = value
    quant["quantized_layers"] = mapped
    for group in quant.get("config_groups", {}).values():
        if "targets" in group:
            group["targets"] = [name.replace("language_model.", "") for name in group["targets"]]
    text = out["text_config"]
    for i in range(text["first_k_dense_replace"], text["num_hidden_layers"]):
        name = f"model.layers.{i}.mlp.experts"
        if mapped.get(name, {}).get("quant_algo") != "NVFP4":
            raise ValueError(f"Missing NVFP4 expert route: {name}")
    # The built-in NextN mapper changes this canonical prefix to model.decoder.
    mtp = f"model.layers.{text['num_hidden_layers']}.mlp.experts"
    if mapped.get(mtp, {}).get("quant_algo") != "W4A16_NVFP4":
        raise ValueError("Missing W4A16_NVFP4 MTP expert route")
    # NextN's class-level packed mapping is not the target's mapping. Explicit
    # aliases prevent fused draft linears from silently becoming unquantized.
    base = f"model.layers.{text['num_hidden_layers']}"
    for fused, shards in [
        ("self_attn.fused_qkv_a_proj_with_mqa", ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"]),
        ("mlp.shared_experts.gate_up_proj", ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"]),
    ]:
        values = [mapped.get(f"{base}.{name}") for name in shards]
        if not any(values):
            continue
        if not all(values) or values[0] != values[1] or values[0].get("quant_algo") != "MXFP8":
            raise ValueError(f"Incomplete or heterogeneous MTP fusion: {fused}")
        mapped[f"{base}.{fused}"] = copy.deepcopy(values[0])
    return out


if __name__ == "__main__":
    json.dump(normalize(json.load(sys.stdin)), sys.stdout, indent=2)
    print()
