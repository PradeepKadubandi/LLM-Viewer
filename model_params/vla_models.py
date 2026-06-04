"""VLA model presets: a vision encoder + an LLM backbone + an action head.

Referenced by analyze_vla_cli.py. The `tinyvla_*` presets use a non-gated
backbone (TinyLlama) so they run end-to-end without HF auth; `openvla_7b` /
`pi0_like` document real scale (Llama-2 / PaliGemma are gated). OpenVLA actually
uses a *dual* SigLIP+DINOv2 encoder; this single-encoder approximation captures
the dominant cost and will be extended later.

Action heads:
  ar       -> autoregressive discrete tokens (OpenVLA: 7-DoF -> 7 tokens/action)
  flow     -> diffusion / flow-matching chunk via a separate action expert (pi0)
  parallel -> single-pass action expert
"""

from vla import VLAConfig


VLA_MODELS = {
    # --- Autoregressive (OpenVLA-style) -----------------------------------
    # Runnable demo: CLIP-L vision + TinyLlama 1.1B backbone, AR action head.
    "tinyvla_demo": VLAConfig(
        name="tinyvla_demo",
        vision_model_id="clip_vit_large_p14_336",
        llm_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        action_head="ar", action_horizon=1, tokens_per_action=7,
    ),
    # OpenVLA scale: DINOv2-L vision + Llama-2-7B backbone. Uses the ungated
    # NousResearch mirror (identical config to meta-llama/Llama-2-7b-hf, which is
    # gated and requires per-account Meta approval) so it runs without HF access
    # grants. The roofline numbers are the same -- only the config dims matter.
    "openvla_7b": VLAConfig(
        name="openvla_7b",
        vision_model_id="dinov2_vit_large_p14_518",
        llm_model_id="NousResearch/Llama-2-7b-hf",
        action_head="ar", action_horizon=1, tokens_per_action=7,
    ),
    # --- Flow / diffusion (pi0 / Octo-style) ------------------------------
    # NOTE: action head (ar/flow/parallel) and expert attention (quadratic/linear
    # SARA-RT) are now UI/CLI config options, not separate presets -- a base
    # model can be analyzed with any action decoder. The fields below are just
    # the preset's default; switch the action head via vla_config instead of
    # picking a different model. (tinyvla_demo above defaults to AR; override it
    # with action_head="flow" for the flow/diffusion path on the same backbone.)
    # pi0 scale: SigLIP vision + PaliGemma backbone (gated) + ~300M flow expert.
    "pi0_like": VLAConfig(
        name="pi0_like",
        vision_model_id="siglip_vit_large_p16_384",
        llm_model_id="google/paligemma-3b-pt-224",
        action_head="flow", num_flow_steps=10, action_chunk=50,
    ),
}
