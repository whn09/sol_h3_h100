# Sol-H3 on ONE H100 80GB — what runs, what it costs, and what the card gives up

The question: **NVlabs publishes a 1×B300 row for [Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) — 13.745 / 37.813 / 52.260 s for 5 / 10 / 15 s of 1344×768 MiniMax-H3 with native audio. What does the same code, same profile, same seed do on one H100 80GB?**

That comparison turns out to be unusually clean, and one sentence buried in upstream's *Measured scope* is why:

> Sol-H3 uses **Dense attention on 1× B300** and SOL with INT8 QKV / FP8 output transport on 4× / 8× B300.

So the one-GPU row upstream publishes is *already* the configuration a single H100 is restricted to — dense attention, no Ulysses, no sparse attention, no transport quantization. **The only variable left between their number and the number in this repo is the silicon.** Every other single-card comparison in the sibling repos had to argue around a config difference; this one does not.

| 1344×768, four DiT forwards | 5 s / 124 f | 10 s / 243 f | 15 s / 362 f |
|---|---|---|---|
| **Sol-H3, 1× B300** (upstream, published) | 13.745 s | 37.813 s | 52.260 s |
| **Sol-H3, 1× H100 80GB** (this repo, measured) | **29.292 s** | **84.094 s** | **166.221 s** |
| ratio | 2.13× | 2.22× | 3.18× \* |
| Base H3, dense, 50 steps, 1× B300 (upstream) | 129.898 s | 376.942 s | 746.885 s |
| **50 scheduler points, 1× H100** (this repo) | **303.361 s** | — | — |

\* The 15 s ratio is inflated by upstream's own table, not by this card: their 1×B300 row goes
2.75× from 5 s to 10 s and then **1.38× for 1.49× the frames**, which is sublinear and unlike every
other row on either side — their Base H3 does 2.90× then 1.98×, and this H100 does 2.87× then
**1.98×**. **The card gap is 2.1–2.2×**; see [`RESULTS.md` §1](RESULTS.md).

Every row: 24 FPS, stereo audio, T2VA, seed 20260903, dense attention, `--compute-quant none`,
median of two measured requests after one warmup. **The one asymmetry that cannot be removed is
disclosed rather than smoothed: upstream's timing includes text encoding and this repo's does
not**, because on 80 GB the conditioner cannot be resident while the denoiser is. It is worth
0.48 s of forward once warm — adding it makes the 5 s row 29.8 s and the ratio 2.17× — and the
reason it is excluded is the next section.

And the profile this card should actually serve, at 864×480 — 2.489× fewer pixels, and
superlinearly cheaper because at `world_size=1` there is no Ulysses to shard the sequence with:

| 864×480, four DiT forwards | 5 s / 124 f | 15 s / 362 f |
|---|---|---|
| 1× H100 measured | **9.151 s** (52.9 GiB peak) | **40.060 s** (58.6 GiB peak) |
| vs the same clip at 1344×768 | 3.20× cheaper | 4.15× cheaper |

**A 15-second 480p clip with native synchronized audio in 40 s on one H100, with 20 GiB to spare.**


## Sol-H3 will not start on this card, and it is not a bug

Sol-H3's engine does `self.pipe.to(self.device)` — one call, every component. On the system it was validated on that is correct and cheap: one B300 is 288 GB. Measured on the box (`scripts/probe_fit.py`):

| component | class | GiB (bf16) |
|---|---|---|
| `text_encoder` | `Qwen3VLForConditionalGeneration` | 62.13 |
| `transformer` | `MiniMaxH3Transformer3DModel` | 61.73 |
| `vae` | `AutoencoderKLMiniMaxH3` | 9.70 |
| `audio_vae` | `AutoencoderKLMiniMaxH3Audio` | 0.56 |
| **total** | | **134.13** against a **79.18 GiB** card |

**No ordering of that call can work**, because the two big components alone are 123.86 GiB. And this is not a case for an offload ladder either: 62.13 GiB of conditioner next to even a fully-released 37 GiB denoiser is 85.9 — still over the card. The conditioner and the denoiser cannot co-reside on one H100 *at all*, at any granularity short of layerwise streaming.

So placement is the patch, and `scripts/bench_h100.py` is the placement:

1. the conditioner runs **first, alone on the card**, and is then **freed** — not offloaded;
2. its output (`1 × 63 × 5120` bf16, **0.6 MiB**) is cached, and `text_encoder.model` is replaced by a stub that returns it in `hidden_states[50]`, so the pipeline's own text-encoder block still runs **unmodified**;
3. the DiT and both VAEs are placed after that, and stay resident across requests.

This is the same conclusion the 8-card work in [`../minimax_h3_h100`](../minimax_h3_h100) reached from the opposite direction — 0.6 MiB of output for 62 GiB of resident weights is a bad trade even when you have the room. Here it is not a trade, it is the only arrangement that runs. The cost is honest and it is a *serving* cost, not a benchmark artifact: a prompt-to-video service on one card pays either a second process for the conditioner or a ~6.4 s upload per new prompt.

## The AdaLN release is lazy, and that is what actually sizes the card

Upstream's `h3_runtime/adaln.py` claims the denoiser drops from 61.7 GB to ~37 GB, which is "what lets it share one 80 GB H100 with the VAEs". Both halves are true, but the timing is not what the docstring implies and it matters on exactly this card:

`enable_adaln_precompute()` **frees nothing**. It patches `MiniMaxH3LoopDenoiser.__call__`, and the 24 GB of per-block `adaln_proj` weights (`Linear(2688 → 6·5376·3)` × 50 blocks) are not dropped until **`i == 0` of the denoising loop** — the table cannot be built earlier because it is sized by a schedule that does not exist until the pipeline has built `row_timestep_plan`. Measured, in order:

```
[DiT resident            ] resident 61.76 GiB
[fusions + adaln armed   ] resident 61.76 GiB    <- unchanged, by design
[VAEs placed             ] resident 72.02 GiB
  ... first denoise step:  [h3opt.adaln] cached 50 blocks x 4 steps: table 0.22 GB, freed 24.23 GB
warmup   peak reserved 72.10 GiB
request  peak reserved 56.82 GiB
```

**The number that sizes the card is 72.10, not 37.** The peak lives in the warmup, where all 61.76 GiB of un-released DiT, both VAEs and the first step's activations are on the card simultaneously, with 7.1 GiB of headroom. Steady-state requests then run 15 GiB *below* the resident weight figure. A reader sizing a deployment from the 37 GB claim would conclude an H100 has ~40 GB of headroom; it has 7.

## What of the six Sol optimizations reaches sm90

Sol-H3 is not one trick, and the single-card restriction does not fall equally on all of it.

| optimization | on 1× H100 (sm90) | why |
|---|---|---|
| Triton kernel fusions (residual+RMSNorm+modulation, QKNorm+partial RoPE, SwiGLU) | **works** | plain Triton, no arch gate |
| AdaLN precompute | **works** | frees 24.23 GB, as claimed |
| Parallel VAE tile decode (`batched=True`, compiled) | **works** | intra-GPU batching, not multi-GPU |
| Four-forward FastH3 adapter | **works** | merged into the weights; **+0.07 % latency**, 0.41 s to fuse. It is what makes the 4-step render usable — `samples/` |
| **Sol-Attn dynamic sparse attention** | **unreachable** | `engine.py` raises below 2 processes; it plugs into `ulysses_custom`'s attention slot, not a diffusers processor, so there is no 1-GPU hook point. An `sm90/` kernel tree ships but is unvalidated and unreferenced from this path |
| **Ulysses all-to-all with INT8 QKV / FP8 output** | **unreachable** | needs ≥2 ranks by construction |
| **MXFP8 compute** | **unreachable** | `compute_quant.validate_platform()` hard-gates on `capability[0] == 10` (SM100). `torch.nn.functional.scaled_mm` with MX scaling has no sm90 path |

Two of the three unreachable items are unreachable *on any single GPU*, so upstream's own 1×B300 row does not have them either — which is precisely why the ratio in the table above is a card comparison. **MXFP8 is the one real Blackwell-only advantage in the list, and upstream's 1×B300 row does not use it** (`--compute-quant` defaults to `none`), so it is not in the 2.1× either. The gap is bf16 dense attention and bf16 GEMMs, B300 against H100.

## Layout

```
Sol-H3/                  vendored upstream source (UPSTREAM_REV.txt, SOURCE_SNAPSHOT.json)
scripts/bringup.sh       venv + the pinned stack + the FastH3 adapter, idempotent, ~15 min
scripts/probe_fit.py     the component/residency measurements above
scripts/bench_h100.py    the placement harness and the timing loop
scripts/matrix.sh        the three length arms + the 50-point control
scripts/extras.sh        the real FastH3 adapter, and 480p
RESULTS.md               every number, with the arm that produced it
samples/                 frame 60 of one prompt at 4 steps, 4 steps + FastH3, and 49 steps
```

Reproduce, on a `p5.4xlarge` with the MiniMax-H3 weights already in `HF_HOME`:

```bash
bash scripts/bringup.sh                 # ~15 min
HF_HOME=/opt/dlami/nvme/vdn/hf PYTHONPATH=$PWD/Sol-H3 \
  scripts/../venv/bin/python scripts/probe_fit.py --json logs/probe_fit.json
bash scripts/matrix.sh                  # the four arms
bash scripts/extras.sh                  # adapter + 480p
```

The pinned stack is torch 2.10.0+cu130, diffusers at `abc5e9bf` (for `diffusers.modular_pipelines.minimax_h3`), transformers 5.8.1, peft 0.20.0, triton 3.6.0, on Python 3.12 — see `scripts/bringup.sh` for why none of those are negotiable and why `nvidia-cudnn-frontend[cutedsl]` failing to build is harmless here.
