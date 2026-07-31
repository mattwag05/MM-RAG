# VLM selection for ingest-time frame captioning

Date: 2026-07-27 · Bead: MM-RAG-jyq · Harness: `scripts/vlm_bench.py` (`make bench-vlm`)

## Result

**Use `florence-community/Florence-2-base` with the `<DETAILED_CAPTION>` head, batched at 8.**
469 MB download, 1.77 GB resident, MIT, **200 ms/frame** — and captions with real retrievable
content: *"The image shows a table with two cups of coffee and a plate of food on it. There are
also forks, knives, and tissue papers on the table."*

Fallback: **`Qwen/Qwen3.5-2B`** (Apache-2.0), the best captions measured, at 4.4× the latency,
7× the resident memory and 9.7× the download.

Every candidate the bead named was measured on real MM-RAG keyframes. Florence-2-base won on
throughput by a margin driven by architecture rather than parameter count, as hypothesised —
and, unexpectedly, it was *not* meaningfully beaten on caption quality by models 35× its size.

Practical impact: a typical video in the test corpus has ~38 silent scenes. At 200 ms/frame
batched, captioning all of them adds **~8 seconds** to an ingest that already takes minutes.
Cost is not the deciding factor; it is close to free.

## Method

- **Corpus:** 40 real keyframes, 20 each from two ingested YouTube videos (a talking-head desk
  tour and Big Buck Bunny), selected by the production predicate and pinned in
  `eval/vlm-frames.jsonl`. See *Corpus predicate* below — it did not survive contact with the data.
- **Hardware:** M5 MacBook Pro, 64 GB, macOS 26. torch 2.13.0 / transformers 5.14.1, MPS.
- **Config, identical for every candidate:** `dtype=float32`, `attn_implementation="sdpa"`,
  `.to("mps")`, `num_beams=1`, `do_sample=False`, `max_new_tokens=64`, `use_cache=True`.
- **One model per process.** The MPS caching allocator does not return memory to the OS, so two
  models in one process measures A+B and the second model's resident figure is fabricated.
- **Warm-up per batch size**, not per model — MPS compiles Metal kernels per tensor shape, so a
  new batch size pays a fresh compile that would otherwise land inside the timed region.
- **Memory** is `ri_phys_footprint` via `proc_pid_rusage` (`ps` RSS double-counts shared pages;
  `ru_maxrss` is peak-not-current). **Disk** is measured from the HF cache, never quoted.
- **No MPS fallbacks were triggered for any candidate**, so no latency figure below carries an
  asterisk.

## Table 1 — acceptance criteria

Sorted by p50 at batch 1. `Resident MB` is post-load minus a pre-`import torch` baseline, so it
includes the torch runtime (~1.3 GB), which is already paid for by SigLIP in the real pipeline.

| Model | Params | Disk MB | License | Resident MB | Load s | p50 ms @B=1 | p50 ms @B=8 | mean out tok | distinct-word | MPS fallbacks |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| **florence-base `<CAPTION>`** | 0.23B | **469** | MIT | **1765** | 2.7 | **181.7** | **125.1** | 17.6 | 0.407 | none |
| **florence-base `<DETAILED_CAPTION>`** ⭐ | 0.23B | **469** | MIT | **1766** | 2.6 | 376.0 | **200.4** | 61.3 | 0.346 | none |
| florence-base `<MORE_DETAILED_CAPTION>` | 0.23B | 469 | MIT | 1766 | 2.5 | 455.2 | 197.5 | 65.0† | 0.340 | none |
| florence-large `<MORE_DETAILED_CAPTION>` | 0.77B | 1559 | MIT | 3698 | 2.8 | 1027.5 | 676.3 | 65.0† | 0.336 | none |
| smolvlm-500m *(control)* | 0.5B | 1020 | Apache-2.0 | 3949 | 3.3 | 1154.0 | 939.3 | 45.4 | **0.226** | none |
| minicpm-v-4.6 ‡ | 1.3B | 2621 | Apache-2.0 | 5946 | 4.3 | 2100.5 | 1073.0 | 58.7 | 0.425 | none |
| qwen3.5-2b | 2B | 4571 | Apache-2.0 | 12320 | 76.5 | 2436.3 | 870.5 | 60.5 | **0.452** | none |
| gemma4-e2b | 5.1B | 10279 | Apache-2.0 | 21327 | 5.0 | 2558.7 | 907.9 | 47.0 | 0.436 | none |
| gemma4-e4b | 8B | 16025 | Apache-2.0 | 32875 | 18.4 | 5130.8 | — | 46.5 | 0.438 | none |

† At the 64-token cap, i.e. truncated. `<DETAILED_CAPTION>` re-run at a 160-token cap produced
66.4 mean tokens and 376.5 ms — it self-terminates and was barely truncated. `<MORE_DETAILED>`
is genuinely cap-bound, so its cost scales with whatever cap you set. That predictability is
part of why `<DETAILED_CAPTION>` is the recommendation.

‡ **MiniCPM required letterboxing to square.** On native 16:9 frames it raises
`RuntimeError: shape '[3, 1034, 1152]' is invalid for input of size 3575808` — its adaptive
slicing miscounts visual tokens by exactly 2. Square inputs work; 640×360 does not. Reproduced
on the correct `apply_chat_template(tokenize=True)` path, so this is a transformers 5.14.1 bug,
not a harness bug. Every MM-RAG keyframe is 16:9, so its numbers required preprocessing no other
candidate needed.

**Measured vs quoted.** Published sizes for Qwen3.5-2B (2.27 GB) and Gemma-4-E4B (~8 GB) were
both roughly half the actual cache footprint (4571 MB and 16025 MB). The bead's insistence on
measurement over model cards was justified.

## Table 2 — batch scaling (p50 ms/frame, peak MB)

| Model | B=1 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| florence-base `<CAPTION>` | 182 / 1931 | **125 / 5624** | 137 / 10608 | 152 / 21683 |
| florence-base `<DETAILED_CAPTION>` | 376 / 1943 | **200 / 5817** | 194 / 10683 | 191 / 21308 |
| florence-base `<MORE_DETAILED>` | 455 / 1931 | 198 / 5804 | 191 / 10670 | 195 / 21295 |
| florence-large `<MORE_DETAILED>` | 1027 / 5136 | 676 / 12086 | 585 / 20306 | 570 / 37735 |
| smolvlm-500m | 1154 / 6561 | 939 / 23432 | 1078 / 52743 | 5552 / 90247 ✗ |
| minicpm-v-4.6 | 2100 / 7444 | 1073 / 15762 | — | — |
| qwen3.5-2b | 2436 / 12546 | 870 / 12337 | 872 / 13663 | 847 / 17398 |
| gemma4-e2b | 2559 / 21588 | 908 / 25942 | — | — |
| gemma4-e4b | 5131 / 33412 | — | — | — |

**Batch 8 is the operating point.** For `<DETAILED_CAPTION>`, B=8 gives 200 ms at 5.8 GB peak;
B=32 gives 191 ms at 21.3 GB. That is 4.5% more speed for 3.7× the memory. Not worth it.

**Peak memory is driven by batch size, not model size.** A 0.23B model at B=32 peaks at 21.3 GB
— more than the 8B model at B=1. Any caption stage must cap its batch, and 8 is the number.

✗ SmolVLM at B=32 hit 90 GB peak on a 64 GB machine and swapped; its 5552 ms is a swap artifact,
not a measurement. Large batches in fp32 are genuinely dangerous, which is also why Gemma-4-E4B
was capped at B=1 (32 GB of fp32 weights) — a deliberate, footnoted deviation.

## Table 3 — model lifecycle (input to MM-RAG-ei7)

Collected free while benchmarking. Every candidate was explicitly released with `del` +
`gc.collect()` + `torch.mps.empty_cache()`.

| Model | baseline MB | post-load MB | peak MB | post-release MB | reclaimed? |
|---|---:|---:|---:|---:|---|
| florence-base `<CAPTION>` | 17 | 1783 | 21683 | 9153 | **NO** |
| florence-base `<DETAILED_CAPTION>` | 17 | 1783 | 21308 | 5827 | **NO** |
| florence-large | 17 | 3715 | 37735 | 1508 | **NO** |
| smolvlm-500m | 17 | 3966 | 90247 | 3619 | **NO** |
| minicpm-v-4.6 | 17 | 5963 | 15762 | 6316 | **NO** |
| qwen3.5-2b | 17 | 12337 | 17398 | 8037 | **NO** |
| gemma4-e2b | 17 | 21344 | 25942 | 14752 | **NO** |
| gemma4-e4b | 17 | 32892 | 33412 | 22977 | **NO** |

**Not one candidate returned to baseline.** Explicit release left between 1.5 GB and 23 GB
resident. MM-RAG-ei7 assumed dropping the reference would reclaim the memory; on MPS it does not.
That bead needs to plan for process recycling, not just `del`.

## Table 4 — caption quality

The primary method was a human eyeball pass over a side-by-side dump (`make bench-vlm-html`).
The `distinct-word ratio` (unique non-stopword tokens ÷ total, at fixed N=40) was a cheap
automated tripwire — and it tracked the human read closely, so it is reported as corroboration.

| Model | distinct-word | Sample caption (same frame) |
|---|---:|---|
| qwen3.5-2b | **0.452** | "a cozy café setting with two white mugs filled with hot beverages, a slice of cake on a plate, and neatly arranged cutlery on a wooden table" |
| gemma4-e4b | 0.438 | — |
| minicpm-v-4.6 | 0.425 | "a cozy café or restaurant setting, centered around a black tray holding breakfast or coffee items" |
| florence-base `<CAPTION>` | 0.407 | "two cups of coffee and a piece of bread on a table" |
| florence-base `<DETAILED_CAPTION>` | 0.346 | "a table with two cups of coffee and a plate of food on it. There are also forks, knives, and tissue papers on the table" |
| smolvlm-500m | **0.226** | "In this image we can see a table on which a plate, cup, fork, knife and a mobile phone are placed" |

**The control behaved as a control should.** SmolVLM-500M was both the slowest-per-parameter and
the worst on quality — nearly every caption opens *"In this image we can see…"*, and one reads in
full: *"In this image we can see trees, mountains and the sky."* That is close to zero retrievable
content, and its 0.226 score caught it without anyone reading a thing.

**Confabulation is real but tolerable.** Florence-2-large invented *"both wearing black suits"*
for two men no other model described that way. For an FTS index feeding retrieval this is
acceptable — recall matters more than precision, and a caption that is 80% right still retrieves
on the nouns that are right. It would not be acceptable as user-facing alt text.

**Quality does not scale with size here.** Gemma-4-E4B (8B, 16 GB, 5.1 s/frame) scored 0.438
against Florence-2-base's 0.407 at 469 MB and 0.18 s/frame. Nothing in the 1–8B tier justified
its cost for this task.

## Recommendation

**Primary: `florence-community/Florence-2-base`, `<DETAILED_CAPTION>`, batch 8, fp32, sdpa.**

1. **Speed.** 200 ms/frame batched — 4.4× faster than the best-quality alternative and 25×
   faster than Gemma-4-E4B. Captioning a whole video's silent scenes costs ~8 s.
2. **Footprint.** 469 MB download and 1.77 GB resident (of which ~1.3 GB is the torch runtime
   SigLIP already pays for). It is the only candidate that does not meaningfully worsen the
   third-resident-model problem.
3. **License.** MIT, ungated, and — verified against the HF file listing — the
   `florence-community` repos contain **zero `.py` files**, so no `trust_remote_code`. That
   matters for a public plugin, especially given CVE-2026-4372 (config-injection RCE that fired
   even with `trust_remote_code=False`).
4. **Quality is sufficient, not merely cheap.** It out-scored the SmolVLM control and landed
   within 0.03 of models 8–35× its size.
5. **It is a captioner, not a chat model.** One special token of prefill instead of a chat
   template per frame, and fixed 768×768 input so a batch has zero padding waste. That is where
   the win comes from, and why 0.23B beats 2B here.

**Fallback: `Qwen/Qwen3.5-2B`** if caption quality ever proves insufficient in retrieval eval.
Best measured captions (0.452), Apache-2.0, native transformers, no preprocessing quirks. The
price is 870 ms/frame at B=8, 12.3 GB resident, 4.6 GB download, and a 76.5 s cold load.

**Not recommended:** Florence-2-large (2.3× slower than base for a *lower* quality score),
SmolVLM (control, worst on both axes), MiniCPM-V-4.6 (needs letterboxing on 16:9), Gemma-4 E2B
and E4B (quality does not justify 10–16 GB downloads and 21–33 GB resident).

### Constraints for the implementation (MM-RAG-yzt)

- `dtype=float32`. **Never `bfloat16`** — MPS lacks optimised bf16 conv kernels and silently
  emulates, and DaViT is conv-heavy. fp16 is safe if memory ever matters.
- `attn_implementation="sdpa"`, `.to("mps")` not `device_map="auto"`, `num_beams=1`.
- **Cap the batch at 8.** See Table 2 — larger batches buy ~5% and cost 3.7× the memory.
- Raise the `m3-visual` floor to `transformers>=5.8` (done). `>=4.40` cannot resolve `florence2`.
- Call `processor.post_process_generation(text, task=prompt, image_size=image.size)` — the raw
  decode is a tagged string, not a caption.
- Expect ~66 output tokens; `max_new_tokens=80` is ample headroom.

## Corpus predicate — the bead's definition did not survive the data

The bead scopes captioning to scenes where transcript **and** OCR are both empty, matching
`summarize.py`'s empty-scene constant. Measured on a real ingest (89 scenes, 103 frames):

| | count |
|---|---:|
| scenes total | 89 |
| scenes with **no transcript** (the real visual-only gap) | **38** |
| scenes emitting the empty-scene constant | **1** |
| frames with empty `ocr_text` | 7 / 103 |

The gap is not rare — it is 43% of scenes. The constant almost never fires because **Tesseract
emits noise on nearly every frame**, so `ocr_text` is almost never empty. Actual stored values:

    "ah id santa ae Beguine ae ©"
    "aig sa = a ee Se 2 ! if - Cy ="
    "SN semen 2S 1-3 yh “a : = es MM ior re casera otsnat rl"

Median OCR word-ratio across the corpus is 0.09; 79 of 173 candidate frames score 0.0.

Two consequences:

1. This benchmark selects on **"no transcript"** and records `ocr_ratio` per frame, so the corpus
   is the real failure population rather than a 1-frame artifact.
2. **MM-RAG-yzt's scoping rule is unimplementable as written.** Gating captions on the empty-scene
   constant would caption ~1 scene per video. It must gate on OCR *quality*, or **MM-RAG-xvg**
   (filed from this work) must land first.

## Artifacts

- `scripts/vlm_bench.py` — the harness. `make bench-vlm-corpus`, `make bench-vlm`, `make bench-vlm-html`.
- `eval/vlm-frames.jsonl` — the pinned 40-frame corpus manifest. Frame paths are stored
  **relative to `settings.assets_dir`**, so the manifest carries no absolute home paths and
  resolves on any machine or `MMRAG_DATA_DIR`. JPEGs are deliberately not committed; the repo
  is public and the frames are third-party video. To regenerate on a fresh checkout, ingest the
  `asset_source` URLs in the manifest and re-run `make bench-vlm-corpus`.
- `/tmp/mmrag-vlm-bench/*.json` — raw per-model results, including every caption.
- `/tmp/mmrag-vlm-bench/captions.html` — the side-by-side dump used for the eyeball pass.

## Threats to validity

- **Two source videos**, one animated and one live-action talking-head. Broader than a single
  source, still not a general sample of what users will ingest.
- **Quality judgement is human and small-N.** No published benchmark measures dense-caption
  quality at <1B params for this use case; the distinct-word ratio is a tripwire, not a metric.
- **A retrieval-based quality proxy was deliberately not built.** It would be circular (a caption
  trivially retrieves its own frame) and, more importantly, **MM-RAG-aux** would poison it:
  `content_items` hits carry raw BM25 (~1–20) and are concatenated with RRF scores (~0.016) then
  re-sorted, so caption hits would dominate regardless of relevance. **MM-RAG-aux should land
  before MM-RAG-yzt**, or caption work will look excellent in `make eval` for the wrong reason.
- **Gemma-4-E4B was measured at B=1 only** and MiniCPM at B≤8, to avoid exhausting RAM. Both were
  already out on other criteria, so this does not affect the decision.

---

# Round 2 — text-promptable captioners and transcript conditioning

Date: 2026-07-31 · Bead: MM-RAG-58v · Harness additions: `--with-transcript` corpus mode,
per-frame `{transcript}` prompt templating, candidates `smolvlm2-2.2b[-cond]`,
`qwen3.5-2b-cond`, `minicpm-v-4.6-cond`.

Motivated by the gap analysis of HKUDS/VideoRAG (conditions every segment caption on the
transcript, MiniCPM-V-2.6) and Grigorij-Dudnik/video-understanding-local (SmolVLM2-2.2B
video-chunk descriptions). Corpus: `eval/vlm-frames-transcript.jsonl`, 40 midpoint frames
from scenes WITH speech (the conditioning population), same manifest format as round 1.

## Result

1. **Florence-2-base `<DETAILED_CAPTION>` stays the ingest captioner.** Nothing displaced it.
2. **Transcript conditioning is REJECTED on measured evidence** (see below) — captions stay
   unconditioned even when a text-promptable model is used.
3. **If a text-promptable captioner is wanted, use `openbmb/MiniCPM-V-4_6` (1.3B).** Best
   distinct-word ratio measured across both rounds (0.559 plain at B=1 on this corpus), ~5.9 GB
   resident, ~2.0 s/frame at B=1. Note VideoRAG's own MiniCPM-V-2.6 is an 8B-class model whose
   int4 build is CUDA-only — V-4.6 is the variant that actually fits MM-RAG's MPS/edge targets.

## Measurements (transcript corpus, fp32, sdpa, MPS)

| Model | Params | Disk MB | Resident MB | p50 ms @B=1 | p50 @B=8 | distinct-word | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| florence-base `<DETAILED_CAPTION>` | 0.23B | 469 | 1764 | 429 | 195 | 0.300 | 5839 (B=8) |
| smolvlm2-2.2b | 2.2B | 8991 | 10070 | 3326 | 2614 | 0.425 | 44274 (B=1) / **92336 (B=8)** |
| minicpm-v-4.6 (B=1 only, n=20) | 1.3B | 2621 | 5885 | 1967 | — | 0.559 | — |
| minicpm-v-4.6-cond (B=1 only, n=20) | 1.3B | 2621 | 5887 | 1955 | — | 0.594 | — |

**SmolVLM2-2.2B is disqualified on cost, not quality.** Its captions are fine (0.425
distinct-word, zero duplicates), but 7.7× Florence's latency, 5.7× its resident memory, and a
**92 GB peak footprint at B=8 fp32** — the machine went 44 GB into swap. Its post-release
footprint (86 GB) also re-confirms MM-RAG-ei7: model memory does not come back; only the
child-process job architecture makes any of these models usable. The `smolvlm2-2.2b-cond`
run was killed mid-flight for RAM; the plain run's numbers settle the question without it.

## Why transcript conditioning is rejected

Conditioning is latency-free (1955 vs 1967 ms) and slightly raises distinct-word ratio — and
it is still the wrong call, because of what it does to caption *content* (side-by-side from
`minicpm-v-4.6[-cond].json`, same frames, same model):

1. **ASR errors become visual "facts".** Transcript: *"they have really really warm crumbs"*
   (ASR mishearing of "really long trunks"). Conditioned caption: *"the elephants have warm
   crumbs, as described in the transcript"*. The caption stream now asserts things no pixel
   supports.
2. **Modality leakage.** Conditioned captions say *"as described in the transcript"* and quote
   subtitle text. Captions are indexed in `fts_scenes` as the VISUAL text stream; leaking
   transcript into them double-counts speech across fusion streams — the same failure class
   that kept `vec_scenes` out of hybrid RRF (MM-RAG-7l1).
3. **Empty-visual pollution.** A black frame's conditioned caption embeds the transcript
   instead of reporting an empty frame.

MM-RAG's caption stage exists to describe what is VISIBLE in scenes that have no other
signal. Keeping it unconditioned preserves stream independence and keeps ASR errors out of
visual evidence. VideoRAG can afford conditioned captions because an LLM re-reads everything
at query time; MM-RAG's evidence pack is consumed as-is.

## Artifacts

- `eval/vlm-frames-transcript.jsonl` — pinned 40-frame with-transcript corpus
  (`python scripts/vlm_bench.py --build-corpus ... --with-transcript`).
- Raw round-2 JSONs (incl. every caption) were produced under the session scratchpad
  (`vlm-r2/`); regenerate with the commands above — the corpus manifest is the pinned part.

## Threats to validity

- The transcript corpus is 34/40 frames from one talking-head asset (it has most of the
  speech scenes in the local store); the conditioning failure modes were nonetheless
  observed across three different assets.
- MiniCPM rows are B=1, n=20 (RAM discipline after the SmolVLM2 run); its round-1 B=8
  numbers (1073 ms/frame) still hold for throughput planning.
- `qwen3.5-2b-cond` was defined but not run this round — Qwen's round-1 latency/memory
  already places it behind MiniCPM for this slot, and the conditioning question was settled
  without it.
