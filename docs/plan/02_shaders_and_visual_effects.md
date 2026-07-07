# Phase 2: Shaders and Visual Effects

**Goal:** Implement PBR lighting, post-processing effects (SSAO, bloom, volumetric light), and a stunning water shader. This phase transforms the game from "programmer art" to "screenshot-worthy."

**Depends on:** Phase 1 (FBOs, shadow maps, pipeline architecture).

---

## Current State Analysis

**File:** `engine/graphics/shaders.py`
- `CHUNK_FRAG`: Flat diffuse lighting = `base_color * ao * mix(block_light_color, sky_light_color, sky_factor) * face_shade`. No normals, no specular, no reflections.
- Water: transparent tinted quad with sinusoidal vertex displacement. No reflections, no refraction, no caustics.
- Sky: procedural gradient + sun disc + stars + scrolling cloud noise. Looks decent but lacks atmosphere scattering.

---

## Implementation Steps

### 2.1 PBR Lighting in `CHUNK_FRAG` [MODIFY shaders.py]

Rewrite the fragment shader to use a simplified PBR model. Since voxel faces always align to axes, we can optimize:

**Vertex Shader changes:**
- Output `v_world_pos` (world position) and `v_normal` (face normal — one of 6 axis directions, can be hardcoded from the face index already packed in vertex data).
- Output `v_tangent` and `v_bitangent` for normal mapping (hardcoded per face: e.g., +Y face → tangent=(1,0,0), bitangent=(0,0,1)).

**Fragment Shader changes:**
```glsl
// Inputs from vertex shader
in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;

// PBR texture layers (from texture array — see Phase 3)
uniform sampler2DArray u_albedo_array;     // RGB color
uniform sampler2DArray u_normal_array;     // Tangent-space normal map
uniform sampler2DArray u_mrao_array;       // R=Metallic, G=Roughness, B=AO

// Lighting calculation:
// 1. Sample normal map, transform to world space via TBN matrix
// 2. Compute direct sunlight: Cook-Torrance BRDF (GGX distribution, Schlick fresnel, Smith geometry)
// 3. Multiply direct light by shadow factor from CSM (Phase 1)
// 4. Add ambient: hemisphere ambient from sky color (top) + ground color (bottom), multiplied by SSAO
// 5. Add block light (torches): treat as omnidirectional warm point light, use voxel flood-fill value as attenuation
// 6. Add emissive (glowstone, lava): direct emission from emissive texture
// 7. Output to HDR framebuffer (values can exceed 1.0)
```

Key simplification: Since all block faces are axis-aligned, the TBN matrix is trivial — no per-vertex tangent calculation needed (unlike general 3D meshes).

### 2.2 SSAO (Screen-Space Ambient Occlusion) [NEW]

Create SSAO shaders in `shaders.py` and processing logic in `engine/graphics/postprocess.py`:

**Algorithm (John Chapman's SSAO):**
1. Render a depth + normal pre-pass (or reconstruct normals from depth in the SSAO shader).
2. For each pixel, sample 16–32 random points in a hemisphere oriented along the surface normal.
3. Compare each sample's depth with the actual depth buffer. If the sample is occluded (behind geometry), it contributes darkness.
4. Blur the raw SSAO texture (4×4 bilateral blur to preserve edges).
5. Multiply the SSAO value into the ambient term of the main shader.

**Performance notes:**
- Run SSAO at half resolution (e.g., 640×360 for a 1280×720 window) and upscale.
- Use a 4×4 noise texture (tiled) to rotate the sample kernel — reduces banding.
- Keep sample count at 16 for good performance; 32 for quality mode (setting toggle).

### 2.3 Bloom [NEW]

Add to `engine/graphics/postprocess.py`:

**Algorithm (dual-filter Kawase bloom):**
1. **Brightness extraction**: From the HDR FBO, extract pixels with luminance > 1.0 (or a configurable threshold) into a separate FBO.
2. **Downsample chain**: Iteratively downsample (bilinear) through 5–6 mip levels (e.g., 1/2, 1/4, 1/8, 1/16, 1/32).
3. **Upsample chain**: Iteratively upsample back, blending each level with the previous. Use tent filter (3×3 weighted average).
4. **Composite**: Additively blend the bloom texture onto the main HDR image before tonemapping.

**What should bloom:**
- Torches, glowstone, lava (emission > 0 in block data)
- The sun disc
- Lightning flashes (Phase 3 weather)

### 2.4 Volumetric Light / God Rays [NEW]

Add to `engine/graphics/postprocess.py`:

**Algorithm (screen-space radial blur):**
1. Determine the sun's screen-space position (project `sun_world_pos` through VP matrix).
2. If the sun is on screen (or near it), perform a radial blur pass:
   - For each pixel, march in a line from the pixel towards the sun's screen position (16–32 steps).
   - At each step, sample the depth buffer. If the depth is at the far plane (sky), accumulate light. If it's geometry, occlude.
   - Weight the accumulated light by distance and sun brightness.
3. Additively blend the volumetric light result onto the main image.
4. Only active during day, intensity scales with sun angle.

**Alternative (higher quality):** Volumetric raymarching through shadow map — march a ray from the camera through the shadow cascade. At each step, check if the point is in shadow. Accumulate inscattering. More expensive but produces true volumetric fog in forests and caves.

### 2.5 Advanced Water Shader [MODIFY shaders.py]

Replace the basic transparent-tint water with a dedicated water rendering pass:

**Vertex Shader (`WATER_VERT`):**
- Apply Gerstner wave displacement (sum of 3–4 wave components with different frequencies, amplitudes, and directions).
- Output displaced position + per-vertex normal from wave derivatives.

**Fragment Shader (`WATER_FRAG`):**
```glsl
// 1. Animated normal map: scroll two normal map layers in different directions, blend them.
// 2. Fresnel effect: more reflective at glancing angles (Schlick approximation).
// 3. Reflection: sample sky color at reflected view direction (cheap) or SSR (expensive, optional).
// 4. Refraction: sample the scene behind the water with UV offset based on the normal map (requires a copy of the opaque scene texture before water rendering).
// 5. Depth-based tint: compare water surface depth with scene depth behind it. Shallow = clear/light blue, deep = dark blue/green. Controls visibility of underwater terrain.
// 6. Foam/shore line: where water depth is very shallow (< 0.5 blocks), add animated white foam texture.
// 7. Caustics: project an animated caustic texture onto the underwater terrain (scrolling voronoi or cellular noise).
// 8. Specular highlight: sun reflection on the water surface (GGX specular with very low roughness).
```

**Pipeline change:** Before rendering water, copy the opaque+cutout scene to a separate texture (for refraction sampling). Then render water with blending.

### 2.6 Underwater Post-Processing [NEW]

When the camera is below water level:
- Apply a blue-green color tint overlay.
- Add slight blur (depth-of-field effect — fog increases faster).
- Distort the screen edges with animated wave UV displacement.
- Reduce visibility distance (fog distance = 2 chunks instead of full render distance).
- Darken everything — underwater should feel deep and mysterious.

### 2.7 Tonemapping & Color Grading [MODIFY shaders.py]

Upgrade the `TONEMAP_FRAG` (created in Phase 1) with:
- **ACES Filmic tonemapping** (better than Reinhard — preserves color saturation in highlights).
- **Exposure control**: Auto-exposure based on average screen luminance (compute in a downsampled luminance texture). Dark caves → high exposure → you can see a bit. Bright outdoors → low exposure → no blowout. Smooth transition over 1–2 seconds.
- **Vignette**: Subtle darkening at screen edges (configurable intensity, can be disabled).
- **Color temperature shift**: Warmer colors at sunset/sunrise, cooler at night.

### 2.8 Create `engine/graphics/postprocess.py` [NEW]

```python
class PostProcessPipeline:
    """Manages and executes post-processing effects in order."""
    
    def __init__(self, ctx, width, height):
        self.effects = []  # Ordered list: SSAO → Volumetric → Bloom → Tonemap → Vignette
    
    def add_effect(self, effect: PostEffect): ...
    def execute(self, input_fbo, output_fbo): ...
    def resize(self, width, height): ...
```

Each effect is a `PostEffect` subclass with `process(input_tex, output_fbo)` and its own shader program. Effects can be toggled individually in the settings menu.

### 2.9 Settings Integration [MODIFY game/ui.py, engine/core/config.py]

Add graphics quality settings:
- **Shadow Quality**: Off / Low (1024) / Medium (2048) / High (4096) — shadow map resolution.
- **SSAO**: Off / Low (16 samples, half-res) / High (32 samples, full-res).
- **Bloom**: Off / On.
- **Volumetric Light**: Off / On.
- **Water Quality**: Simple (current) / Realistic (Gerstner + reflections).
- **Tonemapping**: Reinhard / ACES.

Store in `configs/settings.json` under a `"graphics"` key.

---

## Verification

1. Screenshot comparison: same scene, same time of day, before vs. after each effect.
2. Performance: each post-process effect should add ≤ 1 ms at 1080p on RTX 3050.
3. Toggle each effect on/off in settings menu — verify no visual glitches or FBO leaks.
4. Test underwater rendering: submerge camera, verify tint + fog + distortion.
5. Test sunrise/sunset: verify warm color grading, long shadows, god rays.
6. Test caves: verify SSAO adds depth, torches bloom warmly, no light leaks.

---

## Visual Impact Summary

| Effect | What It Adds |
|---|---|
| PBR + Normal Maps | Blocks look 3D and textured, not flat painted |
| CSM Shadows | Sun casts realistic shadows through trees, buildings |
| SSAO | Corners and crevices have natural darkening |
| Bloom | Torches and lava glow warmly, sun has halo |
| Volumetric Light | Light shafts through forest canopy, cave openings |
| Water Shader | Reflective, refractive, animated ocean with foam |
| Tonemapping | Cinematic color, auto-exposure in caves |
