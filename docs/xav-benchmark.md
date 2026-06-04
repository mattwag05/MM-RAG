# XAV vs MM-RAG Pipeline Benchmark Report

Date: 2026-06-04

## Result

Do not integrate `emrakyz/xav` into MM-RAG now.

The MM-RAG Docker baseline completed successfully across six controlled inputs. XAV could not be benchmarked at runtime because the narrowest attempted build failed before compilation: the repo's checked-in Cargo config requires nightly-only `-Z` flags, while the available local Rust toolchain is stable and no `rustup`/nightly toolchain was available. Source inspection also shows `build.rs` links static FFmpeg, dav1d, Vulkan, Opus, libopusenc, and SVT-AV1 artifacts from `~/.local/src` or system paths. The provided `build.sh` builds a broad encoder dependency tree, which is not a narrow or Pi-safe MM-RAG runtime dependency.

## MM-RAG Docker Baseline

| Input | Duration s | Total s | Throughput video-s/s | Peak RSS MB | Scenes | Frames | Segments | Content Items | OCR Nonempty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| animation_slides.mp4 | 180.02 | 51.71 | 3.48 | 2771.1 | 6 | 90 | 7 | 103 | 45 |
| fast_cuts.mp4 | 180.02 | 40.29 | 4.47 | 2702.9 | 16 | 36 | 1 | 53 | 21 |
| low_light_noise.mp4 | 180.00 | 114.42 | 1.57 | 2579.6 | 1 | 91 | 0 | 92 | 91 |
| screen_text.mp4 | 180.02 | 51.85 | 3.47 | 2689.0 | 6 | 90 | 7 | 103 | 45 |
| static_talking_head.mp4 | 180.00 | 54.57 | 3.30 | 2596.8 | 1 | 91 | 0 | 92 | 91 |
| stress_long_static.mp4 | 600.00 | 115.15 | 5.21 | 2701.4 | 1 | 301 | 0 | 302 | 269 |

Notes:

- Peak RSS is Python process `ru_maxrss`; child-process peak RSS for ffmpeg/Tesseract is not captured by this first-pass helper.
- Docker used the existing `mmrag:0.1.0` image with `/tmp/mmrag-xav-benchmark` bind-mounted.
- The image contacted Hugging Face anonymously during embedding, so model availability/cache behavior is part of the baseline.
- Long single-scene videos drive frame count sharply because MM-RAG samples every 2 seconds after the midpoint policy triggers.

## XAV Feasibility

Attempted:

- Cloned `https://github.com/emrakyz/xav.git` into `/tmp/mmrag-xav-benchmark/xav-runs/xav-src`.
- Resolved Cargo metadata.
- Ran `cargo build --release --no-default-features`.

Observed failure:

- Cargo failed before compilation because `.cargo/config.toml` passes nightly-only `-Z panic_abort_tests` and `-Z dylib-lto` to stable `rustc`.

Additional source-level findings:

- `--sc-only` exists and is the relevant command path for any future MM-RAG comparison.
- `build.rs` still links static media libraries even for the no-default-features build.
- `build.sh` expects a substantial native build environment and can clone/build FFmpeg, Vulkan, Opus, dav1d, SVT-AV1, and related tooling.
- That dependency shape is not suitable for MM-RAG core or the current homelab-host Docker image.

## Recommendation

Do not add XAV to MM-RAG core, Docker, or MCP paths.

Revisit only if XAV provides one of:

- a prebuilt Linux arm64 binary,
- a documented container image,
- or a narrow scene-detection-only build that does not require the broad static encoder dependency tree.

If revisited, benchmark only `xav --sc-only` against MM-RAG's `scene_detect` and downstream frame sampling impact. Keep it behind an optional preprocessing adapter, not a new MCP tool.

## Artifacts

- Corpus metadata: `/tmp/mmrag-xav-benchmark/reports/input_metadata.json`
- MM-RAG raw reports: `/tmp/mmrag-xav-benchmark/reports/mmrag-*.json`
- MM-RAG summary: `/tmp/mmrag-xav-benchmark/reports/mmrag_summary.json`
- XAV feasibility: `/tmp/mmrag-xav-benchmark/reports/xav_build_feasibility.json`
- Generated inputs and runs: `/tmp/mmrag-xav-benchmark/`
