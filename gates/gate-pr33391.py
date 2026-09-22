"""Trace-gate for the fp8 DSA partial kernel on a NoPE (rope-less) MLA model.

GLM-5.3-Flash sets qk_rope_head_dim=0, so d_tail = dim - d_v = 0. The upstream
fp8 partial kernel emits the rope-tail GEMM unconditionally; with d_tail=0 that
is a K=0 tile and TileLang's MMA emitter dies with "Unsupported k_dim 0" the
first time prefill touches it -- roughly 8 minutes into a cold boot.

Compiling the kernel directly costs seconds and fails in the same place, so run
this before shipping an image rather than discovering it from a crash-looping
container. Exits non-zero on any failure.

Shapes follow the deployed config: TP=2 (64 heads -> 32/rank), kv_lora_rank=512,
index_topk=2048, GB10 tiles block_I=32 / threads=128.
"""

import sys
import traceback

import torch

from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
    _pick_inner_iter,
    sparse_mla_fwd_decode_partial_fp8,
)

NUM_HEADS = 32  # 64 attention heads at TP=2
D_V = 512  # kv_lora_rank
D_TAIL = 0  # qk_rope_head_dim -- the whole point of this gate
TOPK = 2048  # index_topk
BLOCK_I = 32  # GB10 retune
THREADS = 128  # GB10 retune
SEQ_LEN = 1
NUM_PAGES = 4096


def dsa_main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1

    ni = TOPK // BLOCK_I
    cu = torch.cuda.get_device_properties(0).multi_processor_count
    inner_iter = _pick_inner_iter(SEQ_LEN, ni, cu, 1)
    n_groups = TOPK // (BLOCK_I * inner_iter)
    print(
        f"SMs={cu} ni={ni} inner_iter={inner_iter} n_groups={n_groups} "
        f"d_tail={D_TAIL}"
    )
    if n_groups < 1:
        print("FAIL: n_groups < 1")
        return 1

    try:
        kernel = sparse_mla_fwd_decode_partial_fp8(
            NUM_HEADS,
            D_V,
            D_TAIL,
            TOPK,
            sm_scale=1.0 / (D_V**0.5),
            block_I=BLOCK_I,
            inner_iter=inner_iter,
            threads=THREADS,
        )
    except Exception:
        traceback.print_exc()
        print("FAIL: kernel construction raised")
        return 1

    fp8 = torch.float8_e4m3fn
    dev = "cuda"
    q = torch.zeros(1, SEQ_LEN, NUM_HEADS, D_V + D_TAIL, device=dev).to(fp8)
    kv = torch.zeros(1, NUM_PAGES, 1, D_V + D_TAIL, device=dev).to(fp8)
    idx = torch.zeros(1, SEQ_LEN, 1, TOPK, dtype=torch.int32, device=dev)

    try:
        # Compilation (and the MMA layout inference that raised "Unsupported
        # k_dim 0") happens here, on first invocation.
        out, lse = kernel(q, kv, idx)
        torch.cuda.synchronize()
    except Exception:
        traceback.print_exc()
        print("FAIL: kernel compile/launch raised")
        return 1

    want_o = (1, SEQ_LEN, n_groups, NUM_HEADS, D_V)
    want_lse = (1, SEQ_LEN, n_groups, NUM_HEADS)
    if tuple(out.shape) != want_o or tuple(lse.shape) != want_lse:
        print(f"FAIL: shapes {tuple(out.shape)} {tuple(lse.shape)}")
        return 1
    if not torch.isfinite(lse).all():
        print("FAIL: non-finite LSE")
        return 1

    print(f"PASS: traced and ran, partial_o={want_o}")
    return 0


def wrapper_main() -> int:
    # Production entry point: the DSA backend dispatches through
    # tilelang_sparse_fwd, not the partial kernel directly. This exercises
    # the wrapper's CUDA fp8 branch (block_I=32/threads=128 tile selection
    # via _cuda_sm_count) plus the partial+combine path the direct call
    # bypasses.
    try:
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            tilelang_sparse_fwd,
        )
        fp8 = torch.float8_e4m3fn
        q = torch.zeros(SEQ_LEN, NUM_HEADS, D_V + D_TAIL, device="cuda").to(fp8)
        kv = torch.zeros(NUM_PAGES, 1, D_V + D_TAIL, device="cuda").to(fp8)
        idx = torch.zeros(SEQ_LEN, 1, TOPK, dtype=torch.int32, device="cuda")
        out = tilelang_sparse_fwd(q, kv, idx, sm_scale=1.0 / (D_V**0.5))
        torch.cuda.synchronize()
        # combine returns the batched [1, seq, heads, d_v] tensor unchanged.
        want = (1, SEQ_LEN, NUM_HEADS, D_V)
        assert tuple(out.shape) == want, f"wrapper shape {tuple(out.shape)} != {want}"
        print(f"PASS: tilelang_sparse_fwd wrapper, out={tuple(out.shape)}")
        return 0
    except Exception:
        traceback.print_exc()
        print("FAIL: wrapper dispatch through tilelang_sparse_fwd")
        return 1


def dependency_main() -> int:
    """Fail closed on the candidate's prebuilt ARM64/CUDA13 ABI and KV layout."""
    import builtins
    import platform
    from importlib.metadata import version
    from types import SimpleNamespace
    from unittest.mock import patch

    from packaging.version import Version
    from sglang.srt.arg_groups.overrides import _check_tilelang_dsa_fp8_kv
    from sglang.srt.mem_cache import kv_cache_configurator as kv_config
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import precompile_w4a16_prefill_routes
    from sglang.srt.layers.quantization.unquant import Bf16GemmBackend, should_enable_bf16_splitk_gemm

    assert platform.machine() in ("aarch64", "arm64"), platform.machine()
    assert torch.version.cuda == "13.0", torch.version.cuda
    assert torch.cuda.get_device_capability() == (12, 1), "gate requires the serving GB10"
    for package, required in (("torch", "2.13.0"), ("tilelang", "0.1.12"),
                              ("flashinfer-python", "0.6.18"), ("sglang-kernel", "0.4.6.post1")):
        assert Version(version(package)).public == required, (package, version(package))

    # On SM121 the vendor's SM120-only precompile returns before importing
    # its patched route-pack API. SM100 Split-K is also not selected by AUTO.
    assert not precompile_w4a16_prefill_routes(device=torch.device("cuda"), num_experts=256, top_k=8)
    assert not should_enable_bf16_splitk_gemm(Bf16GemmBackend.AUTO)
    _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=False)
    for prefill, decode in (("tilelang", "flashinfer_sparse_mla"), ("flashinfer_sparse_mla", "tilelang")):
        try:
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", prefill, decode, hip=False)
        except ValueError:
            pass
        else:
            raise AssertionError("mixed raw/scaled KV backend accepted")

    # Execute the real layout calculator, not a source-marker approximation.
    # Both target and draft use the raw 512-byte FP8 row; MXFP8 describes
    # checkpoint weights, never a KV dtype. Forbid reaching native NoPE imports.
    model = SimpleNamespace(hf_config=SimpleNamespace(), kv_lora_rank=512, qk_rope_head_dim=0)
    kernel = SimpleNamespace(dsa_prefill_backend="tilelang", dsa_decode_backend="tilelang")
    original_import = builtins.__import__

    def no_native_mla(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("flashinfer.mla") or (name == "flashinfer" and "mla" in fromlist):
            raise RuntimeError("native NoPE dependency reached")
        return original_import(name, globals, locals, fromlist, level)

    with patch.object(kv_config, "get_exec", return_value=SimpleNamespace(kernel=kernel)), \
         patch.object(kv_config, "get_disagg", return_value=SimpleNamespace(disaggregation_mode="null")), \
         patch.object(kv_config, "is_deepseek_dsa", return_value=True), \
         patch.object(kv_config, "_is_hip", False), \
         patch("builtins.__import__", side_effect=no_native_mla):
        for role in ("target", "draft"):
            assert kv_config.calculate_mla_kv_cache_dim(model_config=model, kv_cache_dtype=torch.float8_e4m3fn) == 512, role
        # Negative control: changing both backends must trip the native guard.
        kernel.dsa_prefill_backend = kernel.dsa_decode_backend = "flashinfer_sparse_mla"
        try:
            kv_config.calculate_mla_kv_cache_dim(model_config=model, kv_cache_dtype=torch.float8_e4m3fn)
        except RuntimeError as exc:
            assert str(exc) == "native NoPE dependency reached"
        else:
            raise AssertionError("native dependency negative control did not fire")
    print("PASS: ARM64/CUDA13 exact ABI, TileLang raw KV, native-only dependency exclusions", flush=True)
    return 0


def route_main() -> int:
    """Resolve the REAL mixed-quant routes this checkpoint serves, then exercise
    the NVFP4 MoE-path kernels numerically before full loading.

    The served checkpoint is canonicalized by the sglang_speculation role's
    normalize_modelopt.py: quantized_layers keys carry the tree namespace
    (model.layers.*), the MTP expert route is W4A16_NVFP4, and the draft's
    fused projections get explicit aliases. The server then builds
    ModelOptMixedPrecisionConfig with the model class's packed_modules_mapping
    (loader.py injects it) and, for the draft runner, applies the NextN
    name mapper (model.layers.<N> -> model.decoder) onto the quant map. This
    gate replicates exactly that construction on a synthetic map shaped like
    the deployed checkpoint and pins the runtime contract round-1 left as
    static rationale: routed experts NVFP4 at group_size 16 (never the W4A16
    K/32 CuTe b12x path, which requires group_size 32), shared experts and
    fused attention MXFP8 via the packed mapping, target-side route guard
    executed directly including its refusal path, and the NextN required map
    resolved through the real mapper. Full FusedMoE dispatch and per-kernel
    numerics stay with the startup probe + on-cluster spec bench; this gate
    covers everything constructible without a full model allocation.
    """
    try:
        from types import SimpleNamespace

        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import _compute_g1_scale_c
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptMixedPrecisionConfig,
            ModelOptNvFp4FusedMoEMethod,
            _compute_gemm1_alphas,
        )
        from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
        from sglang.srt.models.glm5_next_nextn import (
            Glm5NextForConditionalGenerationNextN,
        )
        from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
    except Exception:
        traceback.print_exc()
        print("FAIL: route gate imports")
        return 1

    # Canonical namespace, exactly what normalize_modelopt.py writes and what
    # the server's ModelOptMixedPrecisionConfig receives (plus the model
    # class's packed_modules_mapping, injected by loader.get_quant_config).
    # All 42 routed layers carry entries, as in the real checkpoint, so the
    # target-side guard's full loop is exercised, not just two layers.
    mtp = "model.layers.45"
    expert_routes = {
        f"model.layers.{i}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}
        for i in range(3, 45)
    }
    checkpoint_quant = {
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": None,
        "quantized_layers": {
            "model.layers.0.self_attn.q_a_proj": {"quant_algo": "MXFP8", "group_size": 32},
            "model.layers.0.self_attn.kv_a_proj_with_mqa": {"quant_algo": "MXFP8", "group_size": 32},
            **expert_routes,
            "model.layers.3.mlp.shared_experts.gate_proj": {"quant_algo": "MXFP8", "group_size": 32},
            "model.layers.3.mlp.shared_experts.up_proj": {"quant_algo": "MXFP8", "group_size": 32},
            f"{mtp}.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            f"{mtp}.self_attn.fused_qkv_a_proj_with_mqa": {"quant_algo": "MXFP8", "group_size": 32},
            f"{mtp}.mlp.shared_experts.gate_up_proj": {"quant_algo": "MXFP8", "group_size": 32},
        },
        "ignore": [],
        "packed_modules_mapping": {
            "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        },
    }
    try:
        qc = ModelOptMixedPrecisionConfig.from_config(checkpoint_quant)
        assert qc.get_name() == "modelopt_mixed", qc.get_name()
        assert qc.nvfp4_config.group_size == 16, f"checkpoint group_size {qc.nvfp4_config.group_size} != 16"
        routed = qc._resolve_quant_algo("model.layers.3.mlp.experts")
        assert routed == "NVFP4", f"routed route {routed!r}"
        shared = qc._resolve_quant_algo(
            "model.layers.3.mlp.shared_experts.gate_up_proj"
        )
        assert shared == "MXFP8", f"shared route {shared!r}"
        fused_attn = qc._resolve_quant_algo(
            "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa"
        )
        assert fused_attn == "MXFP8", f"fused attention route {fused_attn!r}"
        # The vendor CuTe b12x W4A16 path requires K/32, whereas both our
        # target NVFP4 and draft W4A16_NVFP4 use K/16. Force the CUTLASS
        # predicate true so an uninitialized runner cannot make this pass
        # vacuously; the K/32 positive control proves the guard is exercised.
        from unittest.mock import PropertyMock, patch

        assert routed != "W4A16_NVFP4", "target must stay NVFP4"
        with patch.object(ModelOptNvFp4FusedMoEMethod, "enable_flashinfer_cutlass_moe",
                          new_callable=PropertyMock, return_value=True):
            for quant_config in (qc.nvfp4_config, qc.nvfp4a16_config):
                method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
                method.quant_config = quant_config
                assert not method._use_flashinfer_b12x_w4a16, "vendor K/32 route active for K/16"
            method.quant_config = SimpleNamespace(is_w4a16=True, group_size=32)
            assert method._use_flashinfer_b12x_w4a16, "K/32 route positive control failed"
        print(f"route PASS group_size={qc.nvfp4_config.group_size} routed={routed} shared={shared} fused={fused_attn}", flush=True)

        # Target-side guard, executed directly: valid map passes, a checkpoint
        # that drops the routed-expert entries must raise (the BF16 fallback
        # the 2026-09-06 incident proved costs 283.5 GiB/rank at TP2).
        cfg = SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=45)
        Glm5NextForConditionalGeneration._validate_mixed_expert_routes(cfg, qc, "")
        tampered_map = {k: v for k, v in checkpoint_quant["quantized_layers"].items()
                        if not k.endswith(".mlp.experts")}
        tampered = ModelOptMixedPrecisionConfig.from_config(
            {**checkpoint_quant, "quantized_layers": tampered_map}
        )
        try:
            Glm5NextForConditionalGeneration._validate_mixed_expert_routes(cfg, tampered, "")
        except RuntimeError:
            print("route PASS refusal path", flush=True)
        else:
            raise AssertionError("route guard accepted unquantized experts")

        # Stock 0.6.18 satisfies upstream's exact version, not the vendor's
        # patched W4A16/NoPE contracts. Those are excluded separately below.
        # Probe active imports as well: version strings alone do not prove ABI.
        import flashinfer
        import flashinfer.fused_moe as _fmoe
        import flashinfer.fused_moe.core as _fmoec
        import flashinfer.fp4_quantization as _fp4q
        from flashinfer.utils import device_support_pdl  # noqa: F401
        _missing = []
        for holder, names in (
            (flashinfer, ("fp4_quantize", "nvfp4_quantize", "SfLayout", "top_k",
                          "shuffle_matrix_a", "shuffle_matrix_sf_a",
                          "reorder_rows_for_gated_act_gemm")),
            (_fmoe, ("cutlass_fused_moe", "Fp8QuantizationType")),
            (_fmoec, ("ActivationType", "Fp8QuantizationType")),
            (_fp4q, ("block_scale_interleave",)),
        ):
            _missing += [f"{holder.__name__}.{n}" for n in names if not hasattr(holder, n)]
        assert not _missing, f"flashinfer active-path symbols missing: {_missing}"
        print(f"route PASS flashinfer {flashinfer.__version__} active-path symbols", flush=True)

        # NextN block: the draft runner applies the NextN name mapper onto the
        # quant map (loader.py, model_class.get_hf_to_sglang_mapper), turning
        # the canonical model.layers.45.* keys into model.decoder.*. Resolve
        # the guard's required map through the REAL mapper against the REAL
        # config class.
        mapper = Glm5NextForConditionalGenerationNextN.get_hf_to_sglang_mapper(cfg)
        assert qc.quantized_layers.get(f"{mtp}.mlp.experts", {}).get("quant_algo") == "W4A16_NVFP4"
        draft_qc = ModelOptMixedPrecisionConfig.from_config(checkpoint_quant)
        draft_qc.apply_weight_name_mapper(mapper)
        nextn_required = {
            "model.decoder.mlp.experts": "W4A16_NVFP4",
            "model.decoder.self_attn.fused_qkv_a_proj_with_mqa": "MXFP8",
            "model.decoder.mlp.shared_experts.gate_up_proj": "MXFP8",
        }
        for name, expected in nextn_required.items():
            actual = draft_qc._resolve_quant_algo(name)
            assert actual == expected, f"NextN route {name}: {actual!r}, expected {expected}"
        print("route PASS NextN required map through the real mapper", flush=True)

        # NVFP4 MoE-path kernels on the real activation format.
        with torch.device("cuda"), torch.no_grad():
            x = torch.randn(256, 7168, dtype=torch.bfloat16)
            gscale = (x.abs().max() / 448.0).float().reciprocal()
            x_q, sf = fp4_quantize(x, gscale, sf_vec_size=16)
            assert x_q.dtype == torch.uint8 and tuple(x_q.shape)[-1] == 7168 // 2
            assert sf.dtype == torch.uint8 and sf.numel() > 0
            torch.cuda.synchronize()
            print(f"NVFP4 quantize PASS x_q={tuple(x_q.shape)} sf={sf.numel()}B", flush=True)

            # Fused w13 gate/up alpha split + TRT-LLM up-half scale, with
            # realistic expert counts (same helpers the integration pins on CPU).
            for num_experts, is_gated in ((256, True), (8, True), (128, False)):
                # Gated layers carry a [E, 2] gate/up scale; non-gated layers
                # (Relu2) carry one scale per expert, as the real checkpoints do.
                scale2 = (torch.rand(num_experts, 2, dtype=torch.float32) + 0.5
                          if is_gated else
                          torch.rand(num_experts, dtype=torch.float32) + 0.5)
                input_scale = torch.rand(num_experts, dtype=torch.float32) + 0.5
                g1, g1_up = _compute_gemm1_alphas(scale2, input_scale, is_gated)
                assert g1.shape == (num_experts,) and g1_up.shape == (num_experts,)
                if is_gated:
                    assert torch.allclose(g1, input_scale * scale2[:, 0])
                    assert torch.allclose(g1_up, input_scale * scale2[:, 1])
                else:
                    assert torch.allclose(g1, input_scale * scale2)
                    assert torch.allclose(g1_up, g1)
                c = _compute_g1_scale_c(
                    torch.rand(num_experts, dtype=torch.float32) + 0.5,
                    g1, g1_up, is_gated,
                )
                assert c.shape == (num_experts,) and torch.isfinite(c).all()
            torch.cuda.synchronize()
            print("NVFP4 MoE scale helpers PASS", flush=True)
    except Exception:
        traceback.print_exc()
        print("FAIL: mixed-quant route/NVFP4 kernel gate")
        return 1
    return 0


def pr33391_main(integration_only=False) -> int:
    """Run the PR's CPU tests unchanged, then exercise the newer native IPC path."""
    import asyncio
    import pickle
    import subprocess
    from unittest.mock import patch

    import zmq
    import zmq.asyncio
    import sglang.srt.managers.io_struct as io
    from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem, MultimodalProcessorOutput

    if not integration_only:
        for test in (
            'test/registered/unit/managers/test_mm_utils_split.py',
            'test/registered/unit/managers/test_scheduler_chunked_req_gate.py',
            'test/registered/unit/managers/test_scheduler_hybrid_ssm_admission_latch.py',
            'test/registered/unit/managers/test_scheduler_hybrid_ssm_admission_retry.py',
        ):
            result = subprocess.run([sys.executable, '/sgl-workspace/sglang/' + test])
            if result.returncode:
                print(f'FAIL: native CPU regressions {test}', flush=True)
                return 1
        # This suite uses pytest-parametrized functions, not unittest.main().
        # Plain file execution collects zero tests; retain pytest's nonzero
        # status, including its no-tests-collected failure.
        result = subprocess.run([
            sys.executable, '-m', 'pytest', '-q',
            '/sgl-workspace/sglang/test/registered/unit/layers/attention/test_kda_extend_host_lengths.py',
        ])
        if result.returncode:
            print('FAIL: native KDA host-length/tracking regressions', flush=True)
            return 1

    def request(rid):
        n, rows, width = 10, 32, 1176
        features = torch.arange(n * rows * width, dtype=torch.float32).reshape(n * rows, width)
        grids = torch.tensor([[1, 1, rows]] * n, dtype=torch.int64)
        bundled = MultimodalDataItem(modality=Modality.IMAGE, feature=features,
            offsets=[(i, i) for i in range(n)], model_specific_data={'image_grid_thw': grids})
        items = get_new_expanded_mm_items([bundled])
        items[0].precomputed_embeddings = items[0].feature
        for item in items:
            item.model_specific_data['nested_gate'] = [item.feature, (item.model_specific_data['image_grid_thw'],)]
        obj = io.TokenizedGenerateReqInput(rid=rid, input_text=None, input_ids=[1, 2, 3], input_embeds=None,
            mm_inputs=MultimodalProcessorOutput(mm_items=items), token_type_ids=None,
            sampling_params=None, return_logprob=False, logprob_start_len=-1,
            top_logprobs_num=0, token_ids_logprob=None, stream=False)
        return obj

    def entries(obj):
        return obj.batch if isinstance(obj, io.BatchTokenizedGenerateReqInput) else [obj]

    def check(before, after):
        for original, decoded in zip(entries(before), entries(after)):
            old_items, items = original.mm_inputs.mm_items, decoded.mm_inputs.mm_items
            assert len(old_items) == len(items)
            assert items[0].feature is items[0].precomputed_embeddings
            for old, item in zip(old_items, items):
                grid = item.model_specific_data['image_grid_thw']
                assert item.feature.untyped_storage().nbytes() == item.feature.numel() * item.feature.element_size(), 'native IPC left oversized feature storage'
                assert grid.untyped_storage().nbytes() == grid.numel() * grid.element_size(), 'native IPC left oversized metadata storage'
                torch.testing.assert_close(item.feature, old.feature, rtol=0, atol=0)
                torch.testing.assert_close(grid, old.model_specific_data['image_grid_thw'], rtol=0, atol=0)
                nested = item.model_specific_data['nested_gate']
                assert nested[0] is item.feature and nested[1][0] is grid

    async def async_roundtrip(obj):
        ctx = zmq.asyncio.Context()
        tx, rx = ctx.socket(zmq.PAIR), ctx.socket(zmq.PAIR)
        try:
            tx.setsockopt(zmq.SNDTIMEO, 5000); rx.setsockopt(zmq.RCVTIMEO, 5000)
            tx.bind('inproc://pr33391-async'); rx.connect('inproc://pr33391-async')
            await io.async_sock_send(tx, obj)
            wire = await rx.recv()
            await tx.send(wire)
            return wire, await io.async_sock_recv(rx)
        finally:
            tx.close(linger=0); rx.close(linger=0); ctx.term()

    try:
        with patch.object(io, '_USE_PICKLE_IPC', True):
            for asynchronous in (False, True):
                for batched in (False, True):
                    obj = io.BatchTokenizedGenerateReqInput(batch=[request('a'), request('b')]) if batched else request('a')
                    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
                    logical = sum(item.feature.numel() * item.feature.element_size() for req in entries(obj) for item in req.mm_inputs.mm_items)
                    assert len(raw) > 9 * logical, 'negative fixture did not reproduce amplification'
                    before = pickle.loads(raw)
                    if asynchronous:
                        wire, decoded = asyncio.run(async_roundtrip(obj))
                    else:
                        ctx = zmq.Context(); tx, rx = ctx.socket(zmq.PAIR), ctx.socket(zmq.PAIR)
                        try:
                            tx.setsockopt(zmq.SNDTIMEO, 5000); rx.setsockopt(zmq.RCVTIMEO, 5000)
                            tx.bind('inproc://pr33391-sync'); rx.connect('inproc://pr33391-sync')
                            io.sock_send(tx, obj); wire = rx.recv()
                            tx.send(wire); decoded = io.sock_recv(rx)
                        finally:
                            tx.close(linger=0); rx.close(linger=0); ctx.term()
                    check(before, decoded)
                    assert len(wire) < logical + 128 * 1024 * len(entries(obj)), 'native IPC size regression'
                    second = pickle.dumps(decoded, protocol=pickle.HIGHEST_PROTOCOL)
                    check(before, pickle.loads(second))
                    assert len(second) < logical + 128 * 1024 * len(entries(obj))
                    print(f'PR33391 IPC PASS async={asynchronous} batch={batched} logical={logical} before={len(raw)} after={len(wire)} second={len(second)}', flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    print('PASS: PR CPU regressions and native sync/async single/batch IPC', flush=True)
    return 0


def main() -> int:
    import inspect
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
    from sglang.srt.utils.common import broadcast_pyobj
    assert 'compact_cpu_views' not in inspect.getsource(BaseMultimodalProcessor._prepare_mm_items_for_transport)
    assert 'serialized_data = bytes(tensor_data.cpu().numpy())' in inspect.getsource(broadcast_pyobj)
    if dependency_main() != 0:
        return 1
    if pr33391_main() != 0:
        return 1
    if dsa_main() != 0:
        return 1
    if wrapper_main() != 0:
        return 1
    try:
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",
            weight_block_size=[1, 32], use_mxfp8=True,
        )
        with torch.device("cuda"), torch.no_grad():
            # Include b_proj's real N=32 TP2 shape, which cannot use the
            # DeepGEMM N%64 path without a working fallback. Largest weight
            # tested here is 8 MiB; this gate runs before full model allocation.
            for n, k in ((128, 128), (16, 4096), (32, 4096), (2048, 4096)):
                method = Fp8LinearMethod(config)
                layer = torch.nn.Module()
                method.create_weights(
                    layer, input_size_per_partition=k,
                    output_partition_sizes=[n], input_size=k, output_size=n,
                    params_dtype=torch.bfloat16, skip_block_quant_check=True,
                )
                layer.weight.data.fill_(0.125)
                # UE8M0 exponent 128 encodes scale 2.0.
                layer.weight_scale_inv.data.fill_(128)
                method.process_weights_after_loading(layer)
                for rows in (1, 5, 64):
                    x = torch.full((rows, k), 0.25, dtype=torch.bfloat16, device="cuda")
                    out = method.apply(layer, x, bias=None)
                    torch.cuda.synchronize()
                    assert tuple(out.shape) == (rows, n), f"Output padding leaked: {out.shape} != {(rows, n)}"
                    torch.testing.assert_close(out.float(), torch.full_like(out.float(), k / 16), rtol=0.02, atol=0.02)
                    print(f"MXFP8 GPU PASS M={rows} N={n} K={k} backend={method.mxfp8_dense_backend}", flush=True)
                del method, layer, x, out

            # Exercise the real MLA post-loader, not just its dequantization
            # formula: mixed configs used to enter the per-tensor FP8 branch
            # and crash on the absent weight_scale attribute.
            from types import SimpleNamespace
            from sglang.srt.models.deepseek_common.deepseek_weight_loader import DeepseekV2WeightLoaderMixin
            for is_nextn in (False, True):
                projection = torch.nn.Module()
                projection.weight = torch.nn.Parameter(torch.full((320, 128), 0.125, dtype=torch.float8_e4m3fn), requires_grad=False)
                projection.weight_scale_inv = torch.nn.Parameter(torch.full((320, 4), 114, dtype=torch.uint8), requires_grad=False)
                projection.quant_method = SimpleNamespace(use_mxfp8=True)
                attention = SimpleNamespace(kv_b_proj=projection, qk_nope_head_dim=64, v_head_dim=96, w_kc=None, w_vc=None, w_scale=None)
                decoder = SimpleNamespace(self_attn=attention)
                model = SimpleNamespace(start_layer=0, end_layer=1, layers=[decoder], decoder=decoder)
                owner = SimpleNamespace(model=model, config=SimpleNamespace(num_hidden_layers=1), quant_config=SimpleNamespace(weight_block_size=None))
                DeepseekV2WeightLoaderMixin.post_load_weights(owner, is_nextn=is_nextn)
                # Key absorption maps q_nope[64] -> latent[128]. The
                # loader's double transpose preserves [heads,64,128] while
                # making the key-dimension axis contiguous, not the last one.
                assert attention.w_kc.shape == (2, 64, 128), attention.w_kc.shape
                assert attention.w_kc.stride(-2) == 1, attention.w_kc.stride()
                assert attention.w_vc.shape == (2, 128, 96), attention.w_vc.shape
                assert attention.w_scale is None
                for absorbed in (attention.w_kc, attention.w_vc):
                    assert absorbed.dtype == torch.bfloat16
                    torch.testing.assert_close(absorbed, torch.full_like(absorbed, 0.125 * 2 ** -13), rtol=0, atol=0)
                assert projection.weight.dtype == torch.float8_e4m3fn
                assert projection.weight_scale_inv.dtype == torch.uint8
                print(f"MXFP8 MLA post-load GPU PASS nextn={is_nextn}", flush=True)
    except Exception:
        traceback.print_exc()
        print("FAIL: MXFP8 kernel gate; do not load the full model")
        return 1
    if route_main() != 0:
        return 1
    print("PASS: PR33391 CPU/native IPC, DSA, native MXFP8 kernels, MLA post-loading, mixed routes, NVFP4 MoE kernels")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--integration-only", action="store_true")
    args = parser.parse_args()
    sys.exit(pr33391_main(integration_only=args.integration_only) if args.cpu_only or args.integration_only else main())
