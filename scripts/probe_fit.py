#!/usr/bin/env python3
"""Where does one H100's 80 GB go, and what does Sol-H3 as shipped do with it?

Sol-H3's engine calls `self.pipe.to(self.device)` once, which is the right move on the system it
was validated on: 8x B300 is 8x288 GB, and the whole MiniMax-H3 component set is ~135 GB. On one
80 GB H100 that call cannot succeed, so before writing any placement patch this script measures
the four numbers a patch has to work with:

  1. the size of every component, as loaded (bf16, on the host);
  2. what the shipped `pipe.to(cuda)` does -- reported, not guessed;
  3. what the transformer costs on the card, and what `adaln.enable_adaln_precompute` does at
     install time -- which is NOTHING, and that is the finding. Upstream's docstring claims the
     denoiser goes from 61.7 GB to ~37 GB, and it does, but the release is LAZY: `enable_` only
     patches `MiniMaxH3LoopDenoiser.__call__`, and the 24 GB of `adaln_proj` weights are not
     dropped until `i == 0` of the denoising loop, because the table is sized by a schedule that
     does not exist until the pipeline has built `row_timestep_plan`. So the number to size a card
     against is the one BEFORE the release (all 61.9 GB of DiT plus the VAEs plus first-step
     activations, measured at 72.10 GiB peak in bench_h100.py's warmup), not the 37 GB after it;
  4. whether DiT + both VAEs then co-reside, and how much is left for activations.

It also prints the pipeline's block structure and how the text encoder is reached, because a
single-card placement policy has to know which stage owns which component.

No adapter and no render: the LoRA is merged into the weights, so it changes neither the shapes
nor the residency this script is about, and the FastH3 adapter is gated on Hugging Face.

  HF_HOME=/opt/dlami/nvme/vdn/hf python probe_fit.py [--to-cuda]

--to-cuda opts into step 2, the deliberate OOM. Off by default so the useful numbers land first.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch

MODEL = os.environ.get("H3_MODEL", "MiniMaxAI/MiniMax-H3")
GiB = 2**30


def smi() -> tuple[float, float, float]:
    """allocated, reserved, and what the driver says is used, all GiB."""
    free, total = torch.cuda.mem_get_info()
    return (
        torch.cuda.memory_allocated() / GiB,
        torch.cuda.memory_reserved() / GiB,
        (total - free) / GiB,
    )


def mark(label: str) -> None:
    a, r, d = smi()
    print(f"  [{label:<44}] alloc {a:7.2f}  reserved {r:7.2f}  driver {d:7.2f} GiB", flush=True)


def module_bytes(module) -> tuple[int, int, str]:
    """Parameter bytes, buffer bytes, and the device set they live on."""
    params = sum(p.numel() * p.element_size() for p in module.parameters())
    buffers = sum(b.numel() * b.element_size() for b in module.buffers())
    devices = {str(p.device) for p in module.parameters()} | {
        str(b.device) for b in module.buffers()
    }
    return params, buffers, ",".join(sorted(devices))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--to-cuda", action="store_true", help="run the shipped pipe.to(cuda)")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)} "
          f"sm{''.join(map(str, torch.cuda.get_device_capability()))}")
    free, total = torch.cuda.mem_get_info()
    print(f"card: {total / GiB:.2f} GiB total, {free / GiB:.2f} GiB free before we start")

    from diffusers import ComponentsManager, ModularPipeline

    started = time.perf_counter()
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(MODEL, components_manager=manager)
    print(f"pipeline built in {time.perf_counter() - started:.1f} s")

    # How the pipeline is assembled, and therefore which stage owns which component.
    blocks = getattr(pipe, "blocks", None)
    if blocks is not None:
        sub = getattr(blocks, "sub_blocks", None)
        names = list(sub.keys()) if hasattr(sub, "keys") else [type(b).__name__ for b in (sub or [])]
        print(f"pipeline blocks ({len(names)}): {names}")

    started = time.perf_counter()
    pipe.load_components(dtype=torch.bfloat16)
    print(f"components loaded to host in {time.perf_counter() - started:.1f} s")

    report: dict = {"components": {}}
    total_bytes = 0
    print("\ncomponent sizes as loaded (bf16):")
    for name in (
        "text_encoder", "transformer", "transformer_ref", "vae", "audio_vae",
    ):
        component = getattr(pipe, name, None)
        if component is None or not hasattr(component, "parameters"):
            print(f"  {name:<16} absent")
            continue
        params, buffers, devices = module_bytes(component)
        total_bytes += params + buffers
        print(f"  {name:<16} {(params + buffers) / GiB:7.2f} GiB "
              f"(params {params / GiB:6.2f} + buffers {buffers / GiB:5.2f})  on {devices}  "
              f"{type(component).__name__}")
        report["components"][name] = {
            "gib": (params + buffers) / GiB, "class": type(component).__name__
        }
    print(f"  {'TOTAL':<16} {total_bytes / GiB:7.2f} GiB against an 80 GB card")
    report["total_gib"] = total_bytes / GiB

    # The text encoder is the component a single card cannot co-locate, so record how it is
    # reached: a forward pre-hook on the top module only works if the pipeline calls it there.
    encoder = getattr(pipe, "text_encoder", None)
    if encoder is not None:
        print(f"\ntext encoder {type(encoder).__name__}: "
              f"children {[n for n, _ in encoder.named_children()]}")
        print(f"  forward defined on: {type(encoder).forward.__qualname__}")

    transformer = pipe.transformer

    if args.to_cuda:
        print("\n--- step 2: the shipped pipe.to(cuda) ---")
        try:
            pipe.to(torch.device("cuda", 0))
            print("  it fit (unexpected on 80 GB)")
        except torch.OutOfMemoryError as error:
            first = str(error).split("\n")[0]
            print(f"  torch.OutOfMemoryError: {first}")
            mark("after the OOM")
            report["shipped_to_cuda"] = first
        pipe.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    print("\n--- step 3: the transformer alone, then AdaLN precompute ---")
    mark("empty card")
    started = time.perf_counter()
    transformer.to("cuda")
    load_s = time.perf_counter() - started
    mark("transformer resident")
    before = torch.cuda.memory_allocated()
    print(f"  host->device {before / GiB:.2f} GiB in {load_s:.1f} s "
          f"({before / GiB / load_s:.2f} GiB/s)")

    from h3_runtime import adaln, fusion_install

    fusion_install.install(transformer, lora=False)
    mark("fusions installed")

    started = time.perf_counter()
    adaln.enable_adaln_precompute(transformer, verbose=True, component_name="transformer")
    gc.collect()
    torch.cuda.empty_cache()
    adaln_s = time.perf_counter() - started
    mark("adaln armed (nothing freed yet -- by design)")
    after = torch.cuda.memory_allocated()
    # Expected to be 0.00: `enable_` only patches the loop denoiser, and the projections are
    # dropped at i == 0 of the first denoise, once the schedule exists to size the table with.
    # bench_h100.py logs the real release ("freed 24.23 GB of weights") from inside the loop.
    print(f"  arming AdaLN precompute took {adaln_s:.2f} s and released "
          f"{(before - after) / GiB:.2f} GiB at install; the 24 GB release happens on the first "
          f"denoise step, so a card must hold {after / GiB:.2f} GiB of DiT until then")
    report["transformer_gib_resident"] = after / GiB
    report["adaln_release_is_lazy"] = True
    report["adaln_arm_seconds"] = adaln_s

    print("\n--- step 4: do the VAEs fit next to it? ---")
    for name in ("vae", "audio_vae"):
        component = getattr(pipe, name, None)
        if component is None:
            continue
        component.to("cuda")
        mark(f"{name} resident")
    resident = torch.cuda.memory_allocated()
    free, total = torch.cuda.mem_get_info()
    print(f"\n  resident weights {resident / GiB:.2f} GiB; "
          f"{free / GiB:.2f} GiB left on the card for activations")
    report["resident_weights_gib"] = resident / GiB
    report["headroom_gib"] = free / GiB

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
