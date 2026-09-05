"""Offline procedural asset generation for the V2 kitchen.

Because the target machine may be offline, we do **not** download meshes/textures.
Instead we synthesise a small set of recognisable, textured assets with numpy:

* Procedural PNG textures (brushed steel, enamel, wood, stone counter, dark
  glass, tile, and wrap-around soda-can labels in several colours).
* A UV-mapped soda-can mesh (cylinder + caps) so cans read as real cans and
  their colour is controllable for the attribute/colour module.

Run once::

    python -m vision2action.env.assets_gen

Outputs land in ``vision2action/assets/generated/`` and are committed so the
scene loads without regenerating.  Deterministic (fixed RNG) for reproducibility.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

GEN_DIR = Path(__file__).resolve().parents[1] / "assets" / "generated"


def _save(name: str, arr: np.ndarray) -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB").save(GEN_DIR / name)


def tex_steel(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(1)
    base = np.linspace(150, 205, size)[None, :].repeat(size, axis=0)  # horizontal gradient
    streak = rng.normal(0, 6, size=(1, size)).repeat(size, axis=0)  # vertical brushing
    v = np.clip(base + streak, 120, 230)
    return np.stack([v, v, np.clip(v + 6, 0, 255)], axis=-1)


def tex_enamel(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(2)
    v = np.clip(238 + rng.normal(0, 3, (size, size)), 225, 255)
    return np.stack([v, v, v], axis=-1)


def tex_wood(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(3)
    x = np.linspace(0, 6 * math.pi, size)
    grain = (np.sin(x)[None, :].repeat(size, axis=0) * 14)
    grain += rng.normal(0, 6, (size, size))
    r = np.clip(150 + grain, 90, 200)
    g = np.clip(96 + grain, 55, 150)
    b = np.clip(52 + grain * 0.5, 25, 100)
    return np.stack([r, g, b], axis=-1)


def tex_counter(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(4)
    base = np.full((size, size), 185.0)
    speckle = rng.normal(0, 18, (size, size))
    dark = (rng.random((size, size)) > 0.985) * -60
    v = np.clip(base + speckle + dark, 120, 230)
    return np.stack([v, v, np.clip(v - 4, 0, 255)], axis=-1)


def tex_darkglass(size: int = 256) -> np.ndarray:
    v = np.full((size, size), 28.0)
    return np.stack([v, v, np.clip(v + 12, 0, 255)], axis=-1)


def tex_tile(size: int = 256, n: int = 6) -> np.ndarray:
    img = np.full((size, size, 3), 236.0)
    step = size // n
    grout = 3
    for i in range(0, size, step):
        img[i : i + grout, :, :] = 170
        img[:, i : i + grout, :] = 170
    return img


def tex_can_label(color: tuple[int, int, int], size: int = 256) -> np.ndarray:
    """Wrap-around soda-can label: coloured field, white band, metallic cap strips.

    Rows near v<0.05 and v>0.95 are grey so the can's top/bottom caps look metal.
    """
    img = np.zeros((size, size, 3), dtype=float)
    img[:] = color
    # white band across the middle (label graphic)
    band = slice(int(size * 0.40), int(size * 0.62))
    img[band, :, :] = 240
    # a darker accent line
    img[int(size * 0.30) : int(size * 0.33), :, :] = np.array(color) * 0.6
    # metallic cap strips (top/bottom rows -> UV maps caps here)
    cap = int(size * 0.05)
    img[:cap, :, :] = 175
    img[-cap:, :, :] = 175
    return img


def write_soda_can_obj(
    path: Path, radius: float = 0.033, height: float = 0.115, segments: int = 48
) -> None:
    """Closed cylinder with wrap-around side UVs and metallic caps.

    Origin at base centre (z in [0, height]) so a body at z=0 rests on the floor.
    """
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    norms: list[tuple[float, float, float]] = []
    faces: list[tuple[tuple[int, int, int], ...]] = []

    def add_v(p):
        verts.append(p)
        return len(verts)

    def add_vt(uv):
        uvs.append(uv)
        return len(uvs)

    def add_vn(n):
        norms.append(n)
        return len(norms)

    bottom_c = add_v((0.0, 0.0, 0.0))
    top_c = add_v((0.0, 0.0, height))
    n_up = add_vn((0.0, 0.0, 1.0))
    n_dn = add_vn((0.0, 0.0, -1.0))
    cap_uv = add_vt((0.5, 0.995))  # grey strip
    cap_uv_b = add_vt((0.5, 0.005))

    ring_b, ring_t, uv_b, uv_t, sn = [], [], [], [], []
    for i in range(segments + 1):
        a = 2 * math.pi * i / segments
        cx, sy = math.cos(a), math.sin(a)
        ring_b.append(add_v((radius * cx, radius * sy, 0.0)))
        ring_t.append(add_v((radius * cx, radius * sy, height)))
        u = i / segments
        uv_b.append(add_vt((u, 0.07)))
        uv_t.append(add_vt((u, 0.93)))
        sn.append(add_vn((cx, sy, 0.0)))

    for i in range(segments):
        # side quad -> 2 tris  (v/vt/vn)
        faces.append((
            (ring_b[i], uv_b[i], sn[i]),
            (ring_b[i + 1], uv_b[i + 1], sn[i + 1]),
            (ring_t[i + 1], uv_t[i + 1], sn[i + 1]),
        ))
        faces.append((
            (ring_b[i], uv_b[i], sn[i]),
            (ring_t[i + 1], uv_t[i + 1], sn[i + 1]),
            (ring_t[i], uv_t[i], sn[i]),
        ))
        # top cap
        faces.append((
            (top_c, cap_uv, n_up),
            (ring_t[i], cap_uv, n_up),
            (ring_t[i + 1], cap_uv, n_up),
        ))
        # bottom cap
        faces.append((
            (bottom_c, cap_uv_b, n_dn),
            (ring_b[i + 1], cap_uv_b, n_dn),
            (ring_b[i], cap_uv_b, n_dn),
        ))

    lines = ["# generated soda can"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts]
    lines += [f"vt {u:.5f} {w:.5f}" for u, w in uvs]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in norms]
    for tri in faces:
        lines.append("f " + " ".join(f"{v}/{vt}/{vn}" for v, vt, vn in tri))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    _save("steel.png", tex_steel())
    _save("enamel.png", tex_enamel())
    _save("wood.png", tex_wood())
    _save("counter.png", tex_counter())
    _save("darkglass.png", tex_darkglass())
    _save("tile.png", tex_tile())
    _save("can_red.png", tex_can_label((190, 40, 38)))
    _save("can_blue.png", tex_can_label((36, 78, 170)))
    _save("can_green.png", tex_can_label((40, 150, 70)))
    write_soda_can_obj(GEN_DIR / "soda_can.obj")
    print("generated assets in", GEN_DIR)
    for p in sorted(GEN_DIR.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
