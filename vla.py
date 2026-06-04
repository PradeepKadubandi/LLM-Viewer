"""VLA (Vision-Language-Action) multi-phase roofline orchestration.

A single VLA inference step on an edge device composes three phases:

    1. vision encode : patch_embed (once) + ViT transformer over image tokens
    2. projector     : map vision features into the LLM token space
    3. LLM prefill   : backbone forward over [image tokens + text tokens]

(The action-generation phase -- autoregressive token decode vs diffusion /
flow-matching chunk -- is added in a later step; see
design_docs/vla_roofline_analyzer.md.)

Each phase reuses the golden-locked op cost machinery: the ViT and LLM are
analyzed by ModelAnalyzer (prefill stage), and the patch-embed / projector ops
go through op_handlers + roofline_model.evaluate_op directly. The orchestrator
also produces a memory-fit verdict against the device's memory_capacity --
the headline question for edge deployment.
"""

import importlib
from dataclasses import dataclass

import configs.vit as vit_cfg
from model_analyzer import ModelAnalyzer
from op_handlers import OpContext, OP_HANDLERS
from roofline_model import evaluate_op
from hardwares.hardware_params import hardware_params


@dataclass
class VLAConfig:
    """A VLA = one vision encoder + one LLM backbone (+ projector)."""

    name: str
    vision_model_id: str           # key in model_params/vision_encoders.py
    llm_model_id: str              # HF id or model_params key
    llm_source: str = "huggingface"
    llm_config_file: str = None    # None -> ModelAnalyzer auto-search


def _phase(time, OPs, weight=0.0, **extra):
    d = {"time": time, "OPs": OPs, "weight": weight}
    d.update(extra)
    return d


def analyze_vla(
    vla_cfg,
    hardware,
    num_text_tokens=16,
    batchsize=1,
    w_bit=16,
    a_bit=16,
    kv_bit=None,
    use_flashattention=False,
):
    """Analyze one observe->prefill VLA step. Returns a phase/total breakdown
    plus a memory-fit verdict for the given hardware."""
    if kv_bit is None:
        kv_bit = a_bit
    w_byte, a_byte, kv_byte = w_bit / 8, a_bit / 8, kv_bit / 8

    # --- Vision encoder ---------------------------------------------------
    vmp = importlib.import_module("model_params.vision_encoders").model_params[vla_cfg.vision_model_id]
    n_img = vit_cfg.get_num_image_tokens(vmp)
    n_patches = n_img - (1 if getattr(vmp, "has_cls_token", False) else 0)
    patch_dim = vit_cfg.get_patch_dim(vmp)
    vision_hidden = vit_cfg.get_hidden_size(vmp)
    vision_heads = vit_cfg.get_num_attention_heads(vmp)

    v_an = ModelAnalyzer(vla_cfg.vision_model_id, hardware, "configs/vit.py", source="vision_encoders")
    v_res = v_an.analyze(
        seqlen=n_img, batchsize=batchsize, w_bit=w_bit, a_bit=a_bit, kv_bit=kv_bit,
        use_flashattention=use_flashattention,
    )
    v_pf = v_res["total_results"]["prefill"]
    bandwidth, max_OPS, _ = v_an.get_hardware_info()

    # patch embedding runs once per image (not per layer)
    v_ctx = OpContext(
        batchsize, a_byte, w_byte, kv_byte,
        hidden_size=vision_hidden, num_attention_heads=vision_heads,
        num_key_value_heads=vision_heads, head_size=vision_hidden // vision_heads,
        onchip_buffer=0,
    )
    pe = OP_HANDLERS["patch_embed"](v_ctx, None, None, num_patches=n_patches, patch_dim=patch_dim)
    pe_eval = evaluate_op(**pe, bandwidth=bandwidth, max_OPS=max_OPS)

    # --- LLM backbone prefill over [image tokens + text tokens] -----------
    l_an = ModelAnalyzer(vla_cfg.llm_model_id, hardware, vla_cfg.llm_config_file, source=vla_cfg.llm_source)
    llm_hidden = l_an.config.get_hidden_size(l_an.model_params)
    prefix_len = n_img + num_text_tokens
    l_res = l_an.analyze(
        seqlen=prefix_len, batchsize=batchsize, w_bit=w_bit, a_bit=a_bit, kv_bit=kv_bit,
        use_flashattention=use_flashattention,
    )
    l_pf = l_res["total_results"]["prefill"]

    # --- Projector: vision_hidden -> llm_hidden, per image token ----------
    p_ctx = OpContext(
        batchsize, a_byte, w_byte, kv_byte,
        hidden_size=llm_hidden, num_attention_heads=1, num_key_value_heads=1,
        head_size=1, onchip_buffer=0,
    )
    proj = OP_HANDLERS["linear"](p_ctx, n_img, n_img, ic=vision_hidden, oc=llm_hidden, is_kv_proj=False)
    proj_eval = evaluate_op(**proj, bandwidth=bandwidth, max_OPS=max_OPS)

    # --- Aggregate phases -------------------------------------------------
    phases = {
        "patch_embed": _phase(pe_eval["inference_time"], pe["OPs"], pe["load_weight"]),
        "vision_encoder": _phase(v_pf["inference_time"], v_pf["OPs"], v_pf["memory_consumption_weight"]),
        "projector": _phase(proj_eval["inference_time"], proj["OPs"], proj["load_weight"]),
        "llm_prefill": _phase(
            l_pf["inference_time"], l_pf["OPs"], l_pf["memory_consumption_weight"],
            kv_cache=l_pf["memory_consumption_kv_cache"],
        ),
    }
    total_time = sum(p["time"] for p in phases.values())
    total_weight = sum(p["weight"] for p in phases.values())
    kv_cache = phases["llm_prefill"]["kv_cache"]
    # phases run sequentially -> activation memory is reused, so peak is the max
    act_peak = max(v_pf["memory_consumption_tmp_act"], l_pf["memory_consumption_tmp_act"])
    peak_memory = total_weight + kv_cache + act_peak

    capacity = hardware_params[hardware].get("memory_capacity")
    fits = (peak_memory <= capacity) if capacity else None

    return {
        "config": vla_cfg.name,
        "hardware": hardware,
        "num_image_tokens": n_img,
        "prefix_len": prefix_len,
        "phases": phases,
        "total_time": total_time,
        "total_weight": total_weight,
        "kv_cache": kv_cache,
        "act_peak": act_peak,
        "peak_memory": peak_memory,
        "memory_capacity": capacity,
        "fits_in_memory": fits,
    }
