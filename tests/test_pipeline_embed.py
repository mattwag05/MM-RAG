"""Stage 7 embed: SigLIP 768-d vectors, L2-normalized, sane cosines."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.pipeline.stages.embed import embed

pytestmark = pytest.mark.m3_visual


def _solid_jpeg(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    Image.new("RGB", (256, 256), color).save(path, "JPEG", quality=95)


async def test_embed_produces_768d_normalized_vectors(tmp_path):
    import numpy as np

    red_a = tmp_path / "red_a.jpg"
    red_b = tmp_path / "red_b.jpg"
    blue = tmp_path / "blue.jpg"
    _solid_jpeg(red_a, (255, 0, 0))
    _solid_jpeg(red_b, (250, 5, 5))
    _solid_jpeg(blue, (0, 0, 255))

    frames = [
        {"scene_idx": 0, "frame_idx": 0, "path": str(red_a)},
        {"scene_idx": 0, "frame_idx": 1, "path": str(red_b)},
        {"scene_idx": 1, "frame_idx": 0, "path": str(blue)},
    ]
    scenes = [
        {"scene_idx": 0, "start_s": 0.0, "end_s": 1.0},
        {"scene_idx": 1, "start_s": 1.0, "end_s": 2.0},
    ]
    segments = [
        {"seg_idx": 0, "start_s": 0.0, "end_s": 1.0, "text": "a red square", "scene_idx": 0},
    ]

    patch = await embed(frames=frames, scenes=scenes, segments=segments)

    fvs = patch["frame_vectors"]
    svs = patch["scene_vectors"]
    gvs = patch["segment_vectors"]

    assert len(fvs) == 3
    assert len(svs) == 2
    assert len(gvs) == 1

    for entry in fvs + svs + gvs:
        vec = np.asarray(entry["vector"], dtype=np.float32)
        assert vec.shape == (768,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3

    red_0 = np.asarray(fvs[0]["vector"])
    red_1 = np.asarray(fvs[1]["vector"])
    blue_v = np.asarray(fvs[2]["vector"])
    cos_red_red = float(red_0 @ red_1)
    cos_red_blue = float(red_0 @ blue_v)
    assert cos_red_red > 0.9
    assert cos_red_blue < cos_red_red
