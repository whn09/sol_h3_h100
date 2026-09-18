#!/usr/bin/env python3
"""Sol-H3 on ONE H100 80GB: what fits, and what it costs.

WHY THIS IS NOT `infer.py`. Sol-H3's engine does `self.pipe.to(self.device)` -- correct on the
system it was validated on, where one B300 has 288 GB and the whole component set is 134.13 GiB
(measured: text_encoder 62.13, transformer 61.73, vae 9.70, audio_vae 0.56). On 79.18 GiB that
call cannot succeed, and no ordering of it can, because the two big components alone are 123.86.
So placement is the patch, and this harness is the placement:

  1. the conditioner runs FIRST, alone on the card, and is then *freed* -- not offloaded;
  2. `prompt_embeds` (1 x T x 5120 bf16, ~13 MB) is cached and the conditioner is replaced by a
     stub that returns it, so the pipeline's own text-encoder block still runs unmodified;
  3. the DiT and both VAEs are placed after that, and stay resident across requests.

That is the *server* shape, and it is the same conclusion the 8-card work reached for a different
reason: the conditioner's output is 13 MB and its forward is 135 ms, so co-locating 62 GiB of
Qwen3-VL with the denoiser buys nothing. On one 80 GiB card it is not even a trade -- 48.9 GiB of
trimmed conditioner next to a 37 GiB post-AdaLN denoiser is 85.9, over the card. Two consequences
are reported rather than hidden: the number below EXCLUDES text encoding, where upstream's B300
figures include it (~135 ms of forward on a resident conditioner), and a prompt-to-video service
on one card pays either a second process or a ~7.6 s pinned upload per new prompt.

WHY IT RUNS WITHOUT THE FASTH3 ADAPTER. `--lora-mode merged` -- Sol-H3's default -- folds the
adapter into the weights before the loop, so it changes no shape, no kernel and no step count.
The four-forward *latency* is therefore exactly measurable on base weights with
`--steps 5`; only the *video* needs the adapter, which is gated on Hugging Face. Runs without an
adapter are tagged `adapter: null` in the JSON and their MP4s are not quality artifacts.

ARMS. `--steps 5` is Sol-H3's four-forward profile (5 scheduler points). `--steps 50` is the
49-forward base profile that upstream's own SGLang and Diffusers rows use, on this same code and
this same card -- which is what separates "the schedule is shorter" from "the runtime is faster".

  HF_HOME=/opt/dlami/nvme/vdn/hf bench_h100.py --duration 5 --steps 5 --repeats 3
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

MODEL = os.environ.get("H3_MODEL", "MiniMaxAI/MiniMax-H3")
WIDTH, HEIGHT, FPS = 1344, 768, 24
DURATION_FRAMES = {5: 124, 10: 243, 15: 362}
# The conditioning layer the pipeline reads. Copied from diffusers rather than imported so a stub
# built here cannot silently disagree with the block that consumes it.
TEXT_ENCODER_LAYER = 50
GiB = 2**30

PROMPT = (
    "A slow cinematic push-in on a rain-washed Tokyo alley at night. Neon signage reflects in the "
    "puddles, steam drifts from a ramen stall, and a woman in a red coat turns to look back at "
    "the camera. Ambient rain, distant traffic, and the hiss of the stall's burner."
)


def mem() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        "alloc_gib": torch.cuda.memory_allocated() / GiB,
        "reserved_gib": torch.cuda.memory_reserved() / GiB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GiB,
        "driver_used_gib": (total - free) / GiB,
    }


def note(label: str) -> None:
    m = mem()
    print(f"  [{label:<40}] resident {m['alloc_gib']:6.2f}  reserved {m['reserved_gib']:6.2f}  "
          f"peak {m['peak_reserved_gib']:6.2f}  driver {m['driver_used_gib']:6.2f} GiB", flush=True)


class _StubEncoderModel(torch.nn.Module):
    """Returns the cached conditioning instead of re-running 62 GiB of Qwen3-VL.

    The pipeline calls `text_encoder.model(...)` and reads `outputs.hidden_states[50]`, so that is
    the whole contract. The bf16 parameter exists only so `text_encoder.dtype`, which the block
    reads, still answers bfloat16 once the real stack is gone.
    """

    def __init__(self, prompt_embeds: torch.Tensor) -> None:
        super().__init__()
        self.register_parameter("dtype_anchor", torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16)))
        self.cached = prompt_embeds

    def forward(self, **kwargs):
        hidden_states = [None] * (TEXT_ENCODER_LAYER + 1)
        hidden_states[TEXT_ENCODER_LAYER] = self.cached
        return SimpleNamespace(hidden_states=hidden_states)

    __call__ = forward


def build_pipeline():
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import before_encoder

    # Same two overrides the shipped engine applies: 362 frames is a native 17n+5 H3 sequence.
    before_encoder.MINIMAX_H3_MAX_DURATION = 364 / FPS
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(MODEL, components_manager=manager)
    started = time.perf_counter()
    pipe.load_components(dtype=torch.bfloat16)
    print(f"components loaded to host in {time.perf_counter() - started:.1f} s")
    return pipe


def encode_once(pipe, prompt: str, device: torch.device) -> torch.Tensor:
    """Run the conditioner alone on the card, cache its output, and free all 62 GiB of it."""
    block = pipe.blocks.sub_blocks["text_encoder"]
    encoder = pipe.text_encoder

    # The language-model head is a 5120 x vocab projection the block never calls.
    if getattr(encoder, "lm_head", None) is not None:
        head_gib = sum(p.numel() * p.element_size() for p in encoder.lm_head.parameters()) / GiB
        encoder.lm_head = torch.nn.Identity()
        print(f"  dropped the unused lm_head ({head_gib:.2f} GiB)")

    started = time.perf_counter()
    encoder.model.to(device)
    upload_s = time.perf_counter() - started
    note("conditioner resident")
    started = time.perf_counter()
    prompt_embeds, tags = block.encode_prompt(
        pipe, prompt, None, device=device, dtype=torch.bfloat16
    )
    torch.cuda.synchronize()
    forward_s = time.perf_counter() - started
    print(f"  conditioner: {upload_s:.1f} s host->device, {forward_s * 1e3:.0f} ms forward, "
          f"embeds {tuple(prompt_embeds.shape)} = {prompt_embeds.numel() * 2 / 2**20:.1f} MiB, "
          f"{tags.numel()} rows")

    prompt_embeds = prompt_embeds.detach().clone()
    encoder.model = _StubEncoderModel(prompt_embeds).to(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    note("conditioner freed, stub installed")
    return prompt_embeds, upload_s, forward_s


def install_runtime(pipe, device, *, adapter: str | None, lora_mode: str, steps: int,
                    vae_resident: bool, compile_vae: bool):
    """The shipped engine's install sequence, minus everything that needs a second GPU."""
    from h3_runtime import adaln, fusion_install, vae_parallel

    transformer = pipe.transformer
    started = time.perf_counter()
    transformer.to(device)
    print(f"  DiT host->device in {time.perf_counter() - started:.1f} s")
    note("DiT resident")

    pipe.scheduler.set_shift(12.0)
    pipe.audio_scheduler.set_shift(3.0)

    if adapter:
        from h3_runtime.lora import fuse_lora, load_lora_branches

        load_adapter = fuse_lora if lora_mode == "merged" else load_lora_branches
        report = load_adapter(transformer, adapter, alpha=64, scale=1.0)
        print(f"  adapter: {report}")
        note("adapter merged")

    fusion_install.install(transformer, lora=lora_mode == "fused")
    adaln.enable_adaln_precompute(transformer, verbose=True, component_name="transformer")
    note("fusions + adaln installed")

    if vae_resident:
        pipe.vae.to(device)
    pipe.audio_vae.to(device)
    vae_parallel.install(
        pipe.vae, batched=True, compile_mode="default" if compile_vae else None,
        encode_parallel=False,
    )
    note("VAEs placed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=5, choices=(5, 10, 15))
    # 1344x768 is a module constant in the shipped engine, but the pipeline itself takes the canvas
    # per request, so a smaller one costs nothing to reach here. Both edges must be multiples of 32
    # (16x VAE, then the 2x2 transformer patch); 864x480 is the 480p canvas the 8-card work used.
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--steps", type=int, default=5, help="scheduler points; 5 = four forwards")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--lora-mode", default="merged", choices=("merged", "separate", "fused"))
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--vae-after-denoise", action="store_true",
                        help="keep the video VAE on the host until the decode (memory fallback)")
    parser.add_argument("--no-compile-vae", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name(0)} sm{''.join(map(str, torch.cuda.get_device_capability()))}"
          f"  {total / GiB:.2f} GiB, torch {torch.__version__}")

    frames = DURATION_FRAMES[args.duration]
    record: dict = {
        "gpu": torch.cuda.get_device_name(0),
        "gpus": 1,
        "duration_s": args.duration,
        "frames": frames,
        "canvas": [args.width, args.height],
        "scheduler_points": args.steps,
        "transformer_forwards": args.steps - 1,
        "attention": "dense",
        "adapter": args.adapter or None,
        "lora_mode": args.lora_mode if args.adapter else None,
        "seed": args.seed,
    }

    pipe = build_pipeline()
    prompt_embeds, upload_s, forward_s = encode_once(pipe, args.prompt, device)
    record["conditioner"] = {
        "tokens": int(prompt_embeds.shape[1]),
        "upload_s": upload_s,
        "forward_s": forward_s,
        "policy": "encode once, free, stub -- excluded from the request timings below",
    }

    install_runtime(
        pipe, device, adapter=args.adapter or None, lora_mode=args.lora_mode, steps=args.steps,
        vae_resident=not args.vae_after_denoise, compile_vae=not args.no_compile_vae,
    )
    record["resident_weights_gib"] = mem()["alloc_gib"]

    request = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": frames,
        "num_inference_steps": args.steps,
        "output_type": "pt",
    }

    latencies: list[float] = []
    state = None
    for index in range(args.warmup + args.repeats):
        label = "warmup" if index < args.warmup else f"request {index - args.warmup + 1}"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            state = pipe(**request, generator=torch.Generator().manual_seed(args.seed))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = mem()["peak_reserved_gib"]
        print(f"  {label:<12} {elapsed:8.3f} s   {elapsed / args.duration:6.3f} s per video-second"
              f"   peak reserved {peak:6.2f} GiB", flush=True)
        if index >= args.warmup:
            latencies.append(elapsed)
        record.setdefault("runs", []).append(
            {"label": label, "seconds": elapsed, "peak_reserved_gib": peak}
        )

    if latencies:
        record["median_s"] = statistics.median(latencies)
        record["stdev_s"] = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        print(f"\nmedian of {len(latencies)}: {record['median_s']:.3f} s "
              f"(+/- {record['stdev_s']:.3f}) for {args.duration} s / {frames} f at "
              f"{args.width}x{args.height}, {args.steps - 1} DiT forwards")

    if args.output and state is not None:
        from h3_runtime.encoding import start_fast_video_encode

        videos = state.get("videos") if hasattr(state, "get") else getattr(state, "videos", None)
        audio = state.get("audio") if hasattr(state, "get") else getattr(state, "audio", None)
        rate = state.get("sampling_rate") if hasattr(state, "get") else None
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        job = start_fast_video_encode(
            videos[0], fps=FPS, output_path=args.output,
            audio=None if audio is None else audio[0], audio_sample_rate=rate,
            chunk_frames=8, fragmented=True, encoder_threads=16, pyav_zero_copy=True,
            overlap_audio=True,
        )
        job.wait()
        record["mux_s"] = time.perf_counter() - started
        record["output"] = str(Path(args.output).resolve())
        print(f"wrote {args.output} in {record['mux_s']:.2f} s (excluded from the medians above)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(record, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
