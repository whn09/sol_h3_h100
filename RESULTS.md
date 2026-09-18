# Sol-H3 on one H100 80GB — measured

Box: `p5.4xlarge`, 1× **NVIDIA H100 80GB HBM3, sm90, 79.18 GiB**, torch 2.10.0+cu130, diffusers
`abc5e9bf`, triton 3.6.0. Model `MiniMaxAI/MiniMax-H3`, T2VA with stereo audio, 1344×768, 24 FPS,
seed 20260903, `--attention-backend dense`, `--compute-quant none`. Every row is the median of two
measured requests after one warmup, from `scripts/matrix.sh` and `scripts/extras.sh`; raw logs and
per-run JSON in `logs/`.

Upstream's published figures come from the [Sol-H3 project
page](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/), *Benchmark · 8 / 4 / 1× NVIDIA B300*.

> **Provenance correction.** An earlier note in this work attributed upstream's 1×B300 baseline
> numbers (129.898 / 376.942 / 746.885 s) to *SGLang*, and cited a second *Diffusers* row of
> 159.547 / 442.533 / 847.486 s. Both attributions were wrong. The page publishes exactly one
> baseline, labelled **"Base H3 · Dense · 50 steps"**, and it is the 129.898 / 376.942 / 746.885
> row; the 159.547 numbers do not appear anywhere in the page source and are not used here. The
> comparison below is therefore against *upstream's own dense 50-step baseline*, not against
> another serving framework.

## 1. The headline: same code, same profile, 2.1–3.2× the time

Four DiT forwards (5 scheduler points), which is Sol-H3's distilled profile.

| output | frames | **1× H100 (measured)** | 1× B300 (upstream) | ratio | peak reserved |
|---|---|---|---|---|---|
| 5 s | 124 | **29.292 s** ± 0.098 | 13.745 s | 2.13× | 56.82 GiB |
| 10 s | 243 | **84.094 s** ± 0.204 | 37.813 s | 2.22× | 64.44 GiB |
| 15 s | 362 | **166.221 s** ± 0.144 | 52.260 s | 3.18× | 72.88 GiB |

The comparison is clean in a way single-card comparisons usually are not. Upstream's *Measured
scope* states that Sol-H3 uses **dense attention on 1× B300** and reserves SOL sparse attention plus
INT8/FP8 transport for the 4- and 8-GPU rows, and `--compute-quant` defaults to `none`, so
**MXFP8 — the one genuinely Blackwell-only item in the stack — is not in their 1-GPU number
either.** Same repo, same commit, same schedule, same seed, same dense bf16 attention, same bf16
GEMMs. What is left is B300 against H100.

Two asymmetries, both against this repo's numbers:

- **Upstream's timing includes text encoding; this does not.** Measured here at 0.48–0.49 s of
  conditioner forward once warm (0.94 s on the first call). Adding it moves 5 s to ~29.8 s and the
  ratio to 2.17×. It does not change any conclusion.
- **Upstream's 15 s figure is the outlier in upstream's own table, not this card falling over.**
  Their Sol-H3 1×B300 row goes 13.745 → 37.813 (2.75× the time for 1.96× the frames, superlinear,
  as dense attention should be) and then 37.813 → 52.260 — **1.38× the time for 1.49× the frames,
  sublinear.** Nothing else in the table behaves that way: their own Base H3 1×B300 row is
  2.90× then 1.98×, and this H100 is 2.87× then **1.98×** — the same 1.98 exactly. So the 3.18×
  at 15 s is best read as ~2.2× card gap plus an unexplained soft spot in the published 15 s
  number, and the honest summary of the card gap is **2.1–2.2×**.

## 2. How much of Sol-H3 is the four-step schedule, and how much is the engine

This is the arm that cost the most GPU time and settles the most: **49 forwards of the same fused
kernels, on the same card, in the same process** — not a comparison against someone else's runtime.

| 5 s / 124 f | forwards | **1× H100** | 1× B300 (upstream) | ratio |
|---|---|---|---|---|
| Sol-H3 distilled | 4 | 29.292 s | 13.745 s | 2.13× |
| 50 scheduler points | 49 | **303.361 s** ± 0.376 | 129.898 s (Base H3) | 2.34× |
| **speedup from the schedule alone** | | **10.36×** | 9.45× | |

So on this H100 the shorter schedule is worth **10.36×**, against the 9.45× upstream reports between
the same two rows on B300 — the distilled schedule is where nearly all of Sol-H3's headline
"9.45× faster" comes from, and it is **not** a Blackwell effect. The residual (10.36 vs 9.45) is not
a claim that the H100 benefits *more*; upstream's baseline is Base H3 rather than this same code at
49 steps, so the two ratios are not measuring quite the same denominator.

Two points give the cost decomposition, which is what makes the number useful:

```
303.361 = C + 49f          f = (303.361 - 29.292) / 45 = 6.090 s per DiT forward
 29.292 = C +  4f          C =  29.292 - 4 x 6.090     = 4.93 s fixed
```

**Of a 5-second clip's 29.3 s, 24.4 s is four DiT forwards and 4.93 s is everything else** — latent
prep, the video VAE's parallel tile decode over 124 frames, and the audio VAE. That fixed 4.93 s is
17 % of the 5 s request and would be 3 % of the 50-step one, which is why VAE decode is worth
optimizing only on the distilled profile. It also puts a floor under this card: no further step
reduction can take a 5 s clip below ~5 s here.

## 3. Where the 80 GB goes — and why the published 37 GB figure will mislead you

`scripts/probe_fit.py`. Components as loaded, bf16, on the host:

| component | class | GiB |
|---|---|---|
| `text_encoder` | `Qwen3VLForConditionalGeneration` | 62.13 |
| `transformer` | `MiniMaxH3Transformer3DModel` | 61.73 |
| `vae` | `AutoencoderKLMiniMaxH3` | 9.70 |
| `audio_vae` | `AutoencoderKLMiniMaxH3Audio` | 0.56 |
| **total** | | **134.13** vs a 79.18 GiB card |

The shipped `MiniMaxH3Inference` does `self.pipe.to(self.device)`. On 79.18 GiB that cannot succeed
in any order — the two large components alone are 123.86 GiB — and it is not an offload-ordering
problem either: 62.13 of conditioner beside even a fully-released 37 GiB denoiser is 85.9.

**`enable_adaln_precompute()` frees nothing when you call it.** It patches
`MiniMaxH3LoopDenoiser.__call__`; the 24 GB of per-block `adaln_proj` weights are dropped at
`i == 0` of the denoising loop, because the table is sized by a schedule that does not exist until
the pipeline has built `row_timestep_plan`. Measured, in order, for the 5 s / 4-forward arm:

```
[conditioner resident              ] resident 60.68 GiB    (after dropping the unused 1.45 GiB lm_head)
  conditioner: 0.6 MiB of embeds out of 62 GiB of weights
[conditioner freed, stub installed ] resident  0.03 GiB
[DiT resident                      ] resident 61.76 GiB
[fusions + adaln armed             ] resident 61.76 GiB    <- unchanged, by design
[VAEs placed                       ] resident 72.02 GiB
  first denoise step: [h3opt.adaln] cached 50 blocks x 4 steps: table 0.22 GB, freed 24.23 GB
warmup  peak reserved 72.10 GiB     <- the number that sizes the card
request peak reserved 56.82 GiB     <- 15 GiB below the resident-weight figure
```

So the figure to size a deployment against is **72.1 GiB, leaving 7.1 GiB of headroom** — not the
~37 GB the docstring's "denoiser drops from 61.7 GB to roughly 37 GB" suggests. A reader taking the
37 GB at face value would budget ~40 GiB of headroom and has 7.

The AdaLN table scales with the schedule, and at 49 steps it is no longer free:

| schedule | table | net freed |
|---|---|---|
| 5 points (4 steps) | 0.22 GB | 24.01 GB |
| 50 points (49 steps) | 2.65 GB | 21.58 GB |

**15 s at 768p is this card's ceiling.** Its steady-state request peak, 72.88 GiB, is *higher* than
the warmup peak that holds the un-released DiT, and leaves 6.3 GiB. There is no room above it for a
longer clip, a larger canvas, or a batch.

## 4. Cold start, and the cost of the placement policy

| | measured | note |
|---|---|---|
| `load_components` to host | 131.8 s | 268 GiB tree on local NVMe |
| DiT 61.9 GiB host→device, **cold page cache** | 27.2 s (2.27 GiB/s) | first run after bringup |
| DiT 61.9 GiB host→device, **warm** | 5.4 s (11.4 GiB/s) | every later run |
| conditioner 60.7 GiB host→device | 5.9–6.4 s (9.5–10.3 GiB/s) | |
| conditioner forward | 0.94 s first, 0.48–0.49 s warm | 63 tokens |
| MP4 mux (excluded from all timings) | 1.37 s (5 s) / 2.80 s (10 s) | `start_fast_video_encode`, 16 threads |

The 5× spread on the DiT upload is worth naming: diffusers loads safetensors by mmap, so `.to(cuda)`
faults pages in from disk, and the first upload after a fresh boot is NVMe-bound rather than
PCIe-bound. (That is the explanation the two numbers fit, not something measured directly — the
measurement is the 27.2 s and the 5.4 s.)

**What the placement policy costs in service.** Because the conditioner cannot co-reside with the
denoiser, a single-card prompt-to-video service pays, per *new prompt*, either a second process
holding 60.7 GiB of Qwen3-VL or a ~6 s upload — to produce 0.6 MiB of conditioning for a 29 s
request. One process per role on two cards is the arrangement this measurement argues for; on one
card, batching prompts through the conditioner before touching the DiT is the only way to amortize
it.

## 5. The FastH3 adapter: latency-neutral, as designed — and it is what makes the video

`FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA` (public, `dense-datafree/`, 1416.8 MiB), merged
with `--lora-mode merged`. Every latency number in sections 1–2 was measured on **base weights**, on
the argument that a merged LoRA folds into the weights before the loop and therefore changes no
shape, no kernel and no step count. That argument is now measured rather than asserted:

| 5 s / 124 f / 4 forwards | median | peak reserved |
|---|---|---|
| base weights | 29.292 s ± 0.098 | 57.13 GiB |
| **+ FastH3 adapter, merged** | **29.311 s** ± 0.029 | 57.13 GiB |

**+0.019 s, or 0.07 %** — inside the run-to-run spread. The merge itself costs **0.41 s, once**, at
install:

```
FuseReport(format='fastvideo-lora-v2', pairs=362, diffs=85, replacements=0,
           rank=64, alpha=64, scale=1.0, effective_scale=1.0, fuse_s=0.413)
```

So sections 1–4 stand as measured. **What the adapter changes is the picture, and by a lot.** Frame
60 of the same prompt and seed, three ways (`samples/`, all four forwards unless noted):

| | render |
|---|---|
| 4 forwards, **base weights** | `samples/frame60_4step_base_weights.jpg` — recognizably the prompt, and clearly **under-denoised**: dim, low-contrast, the signage illegible, the figure mushy |
| 4 forwards, **+ FastH3** | `samples/frame60_4step_with_fasth3_adapter.jpg` — properly exposed, legible signage, coherent figure |
| **49 forwards**, base weights | `samples/frame60_50step_base_weights.jpg` — sharp and correct: steam off the ramen stall, the woman in the red coat turned to camera |

The 49-forward control matters for two reasons. It shows the 4-forward softness is the *schedule*
and not this harness — and because it is prompt-faithful in detail (rain-washed alley, neon
reflections, ramen steam, red coat), it is end-to-end evidence that the encode-once-and-stub
conditioning path in §3 feeds the denoiser the right thing. **`out/s5_step5_noadapter.mp4` and
`out/d5_step50.mp4` are latency artifacts; `out/d5_adapter.mp4` is the quality artifact.**

## 6. 480p, where this card is actually comfortable

1344×768 is a module constant in Sol-H3's engine but a per-request argument to the pipeline
underneath, so 864×480 — the canvas the 8-card work in [`../minimax_h3_h100`](../minimax_h3_h100)
used — costs nothing to reach. It is **2.489× fewer pixels.**

| output | 864×480 | 1344×768 | time ratio | peak reserved (480p) |
|---|---|---|---|---|
| 5 s / 124 f | **9.151 s** ± 0.010 | 29.292 s | 3.20× | 52.93 GiB |
| 15 s / 362 f | **40.060 s** ± 0.156 | 166.221 s | 4.15× | 58.62 GiB |

Both ratios exceed the 2.489× pixel ratio, and the gap widens with length — 480p is **superlinearly**
cheaper, which is the signature of the quadratic attention term rather than of the linear layers.
Read along the other axis: from 5 s to 15 s (2.92× the frames) 480p costs 4.38× and 768p costs
5.67×, i.e. an effective exponent of 1.38 against 1.62. **At world_size=1 there is no Ulysses to
shard the sequence with, so the whole 362-frame sequence sits in one card's attention, and that is
where the H100 loses ground.** It is also the one place a real optimization is left on the table
here: Sol-Attn's sparse attention targets exactly this term and has no 1-GPU hook point (§7).

The practical read: **a 15-second 480p clip with native audio in 40 s on a single H100, with 20 GiB
to spare** — where the same clip at 768p needs 166 s and leaves 6.3 GiB. 480p is the profile this
card should serve.

> **Not a matched comparison, but the obvious question:** the sibling 8×H100 repo serves 480p / 345
> frames in **8.02 s** with SGLang. That is 5.0× faster on 8× the cards — but it is a *different
> checkpoint* (`OpenVDN/vdn-minimax-h3`, hybrid linear/window attention) at a *different step count*
> (8 DMD2 steps), so the 5.0× is not attributable to either the runtime or the cards. It is quoted
> only to place the single-card number on the same axis, not as a speedup.

## 7. What could not be run, and why

| | status | reason |
|---|---|---|
| Sol-Attn dynamic sparse attention | **not reachable** | `engine.py` raises `ValueError("SOL attention requires 2, 4, or 8 GPU processes")`; it installs into `ulysses_custom`'s attention slot rather than a diffusers attention processor, so there is no world_size=1 hook point. An `h3_runtime/third_party/sol_attn/sm90/` tree ships but nothing on this path references it |
| Ulysses INT8-QKV / FP8-output transport | **not reachable** | needs ≥2 ranks by construction |
| MXFP8 compute | **not reachable** | `compute_quant.validate_platform()` gates on `capability[0] == 10` (SM100); `scaled_mm` MX scaling has no sm90 path. Not in upstream's 1-GPU row either |
| `nvidia-cudnn-frontend[cutedsl]` | build fails at cmake | only `sol_bsa` needs it, and that path needs ≥2 GPUs — deliberately non-fatal in `bringup.sh` |

Everything else transfers to sm90 unmodified: the three Triton fusions, AdaLN precompute, the
compiled parallel VAE tile decode, and merged-LoRA four-forward sampling.
