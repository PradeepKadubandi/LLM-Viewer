"""VLA model presets: a vision encoder + an LLM backbone.

Referenced by analyze_vla_cli.py. The `tinyvla_demo` preset uses a non-gated
backbone (TinyLlama) so it runs end-to-end without HF auth; `openvla_7b`
documents the real OpenVLA scale (Llama-2-7b is gated -- loading it needs HF
access). OpenVLA actually uses a *dual* SigLIP+DINOv2 encoder; this single-
encoder approximation captures the dominant cost and will be extended later.
"""

from vla import VLAConfig


VLA_MODELS = {
    # Runnable demo: CLIP-L vision + TinyLlama 1.1B backbone (no gating).
    "tinyvla_demo": VLAConfig(
        name="tinyvla_demo",
        vision_model_id="clip_vit_large_p14_336",
        llm_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ),
    # OpenVLA scale: DINOv2-L vision + Llama-2-7B backbone (gated -> needs auth).
    "openvla_7b": VLAConfig(
        name="openvla_7b",
        vision_model_id="dinov2_vit_large_p14_518",
        llm_model_id="meta-llama/Llama-2-7b-hf",
    ),
}
