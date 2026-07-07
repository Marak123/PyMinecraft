# Phase 1: Rendering Pipeline Overhaul

**Goal:** Transform the single-pass forward renderer into a multi-pass pipeline with FBOs, enabling shadows, post-processing, and deferred effects.

---

## Current State Analysis

**File:** `engine/graphics/renderer.py` (~600 lines)
- Renders directly to the default framebuffer (screen) — no intermediate FBOs.
- Single forward pass: sky → opaque → cutout → transparent → entities → rain → HUD.
- No shadow mapping infrastructure.
- Frustum culling is vectorized (good), but no occlusion culling.
- Chunk buffer pooling exists and works well — keep it.

**File:** `engine/graphics/shaders.py` (~280 lines)
- All GLSL shaders are stored as Python string constants.
- `CHUNK_FRAG`: Basic texture sampling + AO + sky/block light mixing + distance fog.
- No depth pre-pass, no G-buffer, no shadow sampling.

---

## Implementation Steps

### 1.1 Create `engine/graphics/framebuffer.py` [NEW]

```python
class FramebufferManager:
    """Manages all FBOs for the multi-pass pipeline."""
```

Create a utility class wrapping ModernGL FBO creation:

- **Main HDR FBO**: Color attachment (RGBA16F) + Depth attachment (DEPTH24). Render all 3D geometry here instead of the default framebuffer. HDR (16-bit float) enables proper bloom extraction and tone mapping later.
- **Shadow FBO**: Depth-only attachment, resolution 2048×2048 (configurable). Used by shadow map pass.
- **Cascaded Shadow Maps**: Create a `TextureArray` with 3–4 depth layers, one per cascade.
- **Post-processing ping-pong FBOs**: Two half/quarter-resolution FBOs (RGBA16F) for blur passes.
- Handle window resize: all FBOs must be recreated on resolution change (hook into `Window.on_resize()`).
- Provide `bind()` / `unbind()` / `clear()` helpers.

### 1.2 Create `engine/graphics/shadow.py` [NEW]

```python
class ShadowRenderer:
    """Cascaded Shadow Map (CSM) rendering."""
```

Implement Cascaded Shadow Maps for the sun/moon directional light:

- **Calculate cascade splits**: Use PSSM (Practical Split Scheme) or logarithmic split. 3 cascades recommended for performance:
  - Cascade 0: 0–16 blocks (sharp close shadows)
  - Cascade 1: 16–64 blocks (medium detail)
  - Cascade 2: 64–256 blocks (distant terrain shadows)
- **Per-cascade light matrix**: Build orthographic projection from the sun direction (`Environment.sun_dir`) that tightly fits the camera frustum slice.
- **Shadow pass**: Render all opaque chunk meshes into each cascade's depth texture using a minimal vertex-only shader (`SHADOW_VERT`). No fragment shader output needed (depth-only).
- **Stabilization**: Snap the orthographic projection to texel boundaries to prevent shadow edge shimmer when the camera moves.
- **PCF Filtering**: In the main `CHUNK_FRAG`, sample the shadow map with a 3×3 PCF kernel for soft shadow edges.

Add to `shaders.py`:
```glsl
// SHADOW_VERT — minimal shader for depth-only shadow pass
// SHADOW_FRAG — empty or alpha-test only for cutout blocks
```

### 1.3 Create `engine/graphics/pipeline.py` [NEW]

```python
class RenderPipeline:
    """Orchestrates the full multi-pass rendering pipeline."""
```

Replace the flat render sequence in `Renderer._render()` with a structured pipeline:

```
Pass 1: Shadow Map Pass
  → For each CSM cascade:
    → Set shadow FBO + cascade layer
    → Set light-space VP matrix
    → Draw all opaque chunks (shadow shader, no textures needed)
    → Draw cutout chunks (shadow shader + alpha test)

Pass 2: Geometry Pass (Main Scene)
  → Set main HDR FBO
  → Clear color + depth
  → Draw sky dome
  → Draw opaque chunks (front-to-back, with shadow sampling)
  → Draw cutout blocks (alpha test, no back-face culling)
  → Draw entities
  → Draw transparent blocks (back-to-front, alpha blend)
  → Draw rain/weather particles

Pass 3: Post-Processing (Phase 2 — stubbed for now)
  → Placeholder: just blit HDR FBO to screen with basic tonemapping

Pass 4: UI Overlay
  → Bind default framebuffer (screen)
  → Draw HUD
  → Draw UI menus
```

### 1.4 Modify `engine/graphics/renderer.py` [MODIFY]

- Extract the pipeline orchestration to `pipeline.py`. The `Renderer` class becomes a helper that owns GPU resources (shader programs, chunk VAOs, entity meshes) and provides draw methods.
- Move chunk sorting (front-to-back / back-to-front) into the pipeline.
- Keep buffer pooling and frustum culling in `Renderer`.
- Add methods: `render_chunks_shadow(cascade_vp)`, `render_chunks_main(vp, shadow_maps)`.

### 1.5 Modify `engine/graphics/shaders.py` [MODIFY]

Add new shader sources:

- **`SHADOW_VERT`**: Takes chunk vertex data + light-space MVP. Outputs `gl_Position` only.
- **`SHADOW_CUTOUT_FRAG`**: For cutout blocks — samples alpha from texture, discards if < 0.5.
- **`TONEMAP_VERT` / `TONEMAP_FRAG`**: Fullscreen quad that reads HDR FBO and applies Reinhard or ACES tone mapping + gamma correction.
- Modify **`CHUNK_FRAG`**: Add `uniform sampler2DArray shadow_maps` and cascade selection logic. Sample shadow map and multiply lighting by shadow factor.

### 1.6 Modify `engine/world/environment.py` [MODIFY]

- Add `sun_direction` as a normalized `vec3` computed from `sun_angle`. Currently sky shader calculates this internally — it must be exposed so `ShadowRenderer` can build light matrices.
- Add `moon_direction` for nighttime shadows.

---

## Verification

1. Run `py launcher.py --frames 300 --screenshot test_shadows.png` — verify shadows appear under trees and mountains.
2. Toggle time of day rapidly (add debug key) — shadows should rotate smoothly.
3. Check FPS: shadow pass should add no more than 2–3ms at render distance 10.
4. Run `tools/smoke_test.py` — ensure offscreen rendering still works with FBOs.

---

## Performance Budget

| Pass | Target Time (RTX 3050) |
|---|---|
| Shadow (3 cascades) | ≤ 3 ms |
| Geometry (unchanged from current) | ~ 6 ms |
| Tonemapping blit | ≤ 0.3 ms |
| **Total frame** | ≤ 12 ms (83+ FPS) |
