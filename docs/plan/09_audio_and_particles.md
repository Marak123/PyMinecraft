# Phase 9: Audio System & Particle Effects

**Goal:** Add 3D positional audio with ambient sounds, music, and block/mob sound effects. Implement a GPU-accelerated particle system for block breaking, environmental effects, and combat feedback.

**Depends on:** Phase 7 (mob events to trigger sounds), Phase 8 (tool breaking, eating, combat sounds).

---

## Current State Analysis

- **No audio system exists.** `Config` has a `volume` field but nothing uses it.
- **No particle effects.** Blocks appear/disappear instantly. No visual feedback for damage, walking, or environmental effects.
- **Rain** (`renderer.py`): 340 billboard streaks rendered as geometry — this is the closest thing to a particle system.

---

## Implementation Steps

### 9.1 Audio Engine [NEW: `engine/audio/`]

```
engine/audio/
  __init__.py
  audio_engine.py    # OpenAL context, listener, source management
  sound_manager.py   # High-level sound playback API
  music_player.py    # Background music with crossfading
```

**Dependencies:** Add `PyOpenAL` or `sounddevice` + `soundfile` to `requirements.txt`. Alternatively, use `pygame.mixer` (lighter, doesn't require full pygame — just `pip install pygame`).

**Recommended approach:** Use `pygame.mixer` for simplicity:
- `pygame.mixer.init()` — no window needed.
- `pygame.mixer.Sound` for short SFX.
- `pygame.mixer.music` for background music streaming.
- Positional audio: manually calculate volume + stereo pan from 3D position relative to camera.

```python
class AudioEngine:
    """3D positional audio engine using pygame.mixer."""
    
    def __init__(self, num_channels=32):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        pygame.mixer.set_num_channels(num_channels)
    
    def play_3d(self, sound_name: str, world_pos: np.ndarray, 
                camera_pos: np.ndarray, camera_forward: np.ndarray,
                volume: float = 1.0, pitch_variation: float = 0.1):
        """Play a sound with distance attenuation and stereo panning."""
        distance = np.linalg.norm(world_pos - camera_pos)
        if distance > 32.0:  # Max audible distance
            return
        attenuation = 1.0 / (1.0 + distance * 0.3)
        
        # Stereo pan: project position onto camera's right vector
        to_sound = world_pos - camera_pos
        right = np.cross(camera_forward, [0, 1, 0])
        pan = np.dot(to_sound, right) / (distance + 0.01)  # -1 to +1
        
        channel = self._get_channel()
        channel.set_volume(
            volume * attenuation * max(0, 1 - pan),   # Left
            volume * attenuation * max(0, 1 + pan)    # Right
        )
        channel.play(self.sounds[sound_name])
```

### 9.2 Procedural Sound Generation [NEW: `engine/audio/sound_gen.py`]

Since the project has zero external assets, generate all sounds procedurally at startup:

```python
def generate_sounds() -> dict[str, pygame.mixer.Sound]:
    """Generate all game sounds procedurally using numpy."""
    sounds = {}
    
    # Block break: short noise burst with decay
    sounds['block_break_stone'] = _noise_burst(freq=800, duration=0.15, decay=0.1)
    sounds['block_break_dirt']  = _noise_burst(freq=400, duration=0.12, decay=0.08)
    sounds['block_break_wood']  = _noise_burst(freq=600, duration=0.18, decay=0.12)
    sounds['block_break_glass'] = _glass_break(duration=0.3)
    
    # Block place: dull thud
    sounds['block_place'] = _thud(freq=200, duration=0.1)
    
    # Footsteps (per material)
    sounds['step_stone'] = _step(freq=1200, duration=0.08)
    sounds['step_dirt']  = _step(freq=600, duration=0.1)
    sounds['step_wood']  = _step(freq=800, duration=0.09)
    sounds['step_sand']  = _step(freq=300, duration=0.12)
    sounds['step_grass'] = _step(freq=500, duration=0.1)
    
    # Combat
    sounds['hit']     = _impact(freq=500, duration=0.15)
    sounds['crit']    = _impact(freq=1000, duration=0.2, ring=True)
    sounds['sword']   = _sweep(duration=0.2)
    
    # Environment
    sounds['splash']  = _splash(duration=0.4)
    sounds['lava_pop'] = _bubble(freq=100, duration=0.3)
    sounds['rain_ambient'] = _rain_loop(duration=5.0)  # Looping
    sounds['thunder'] = _thunder(duration=2.0)
    
    # Eating
    sounds['eat'] = _crunch(duration=0.3)
    
    # Mob sounds
    sounds['pig_oink']   = _oink(duration=0.4)
    sounds['zombie_groan'] = _groan(duration=0.8)
    sounds['skeleton_rattle'] = _rattle(duration=0.3)
    sounds['creeper_hiss'] = _hiss(duration=1.5)
    sounds['explosion'] = _explosion(duration=0.8)
    
    return sounds

def _noise_burst(freq, duration, decay):
    """White noise modulated by frequency, with exponential decay."""
    t = np.linspace(0, duration, int(44100 * duration))
    signal = np.random.randn(len(t)) * np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)
    return _to_sound(signal)
```

### 9.3 Sound Events [MODIFY game/game.py, game/player.py, game/entities.py]

**Block sounds:**
- `block_break_{material}`: On block break. Material determined from block type (stone, dirt, wood, glass, sand).
- `block_place`: On block place. Position = block world pos.
- Footstep: Every 0.4s while walking (0.3s while sprinting). Material = block under feet.

**Player sounds:**
- `splash`: Entering/exiting water.
- `hurt`: On taking damage.
- `eat`: During eating animation (3 crunch sounds spaced over 1.6s).
- `level_up`: On gaining XP level (if implemented).

**Mob sounds:**
- Each mob plays idle sounds randomly (every 5–15 seconds).
- Hurt sound on taking damage.
- Death sound on dying.
- Hostile mob alert sound when targeting player.

**Environment:**
- Rain loop: plays when `environment.raining` is True, fades in/out over 2 seconds.
- Ambient cave sounds: when player is underground (no sky light), play rare eerie sounds (every 30–120 seconds).
- Lava pops: random bubbling near lava blocks.

### 9.4 Background Music [NEW: `engine/audio/music_player.py`]

Generate simple procedural ambient music using sine wave harmonics:

```python
class MusicPlayer:
    """Generates and plays ambient background music."""
    
    def _generate_track(self, mood: str, duration: float = 120.0):
        """Generate a 2-minute ambient music track.
        
        mood: 'calm' (overworld day), 'night' (nighttime), 
              'cave' (underground), 'combat' (hostile nearby)
        
        Uses: pentatonic scale, slow arpeggios, reverb (convolution), 
              gentle pad (layered sine waves with chorus effect).
        """
```

- Crossfade between tracks based on context (day/night, underground, combat).
- Volume controlled by `Config.volume`.
- Can be disabled in settings.

### 9.5 Particle System [NEW: `engine/graphics/particle.py`]

```python
class ParticleSystem:
    """GPU-accelerated particle system using instanced rendering."""
    
    def __init__(self, ctx, max_particles=4096):
        self.max_particles = max_particles
        self.particles = np.zeros(max_particles, dtype=[
            ('pos', 'f4', 3),        # World position
            ('vel', 'f4', 3),        # Velocity
            ('color', 'f4', 4),      # RGBA
            ('life', 'f4'),          # Remaining lifetime
            ('max_life', 'f4'),      # Initial lifetime (for fade)
            ('size', 'f4'),          # Billboard size
            ('gravity', 'f4'),       # Gravity multiplier
        ])
        self.count = 0
        
        # Instanced rendering: one quad + per-particle instance data
        self._vao = ...
        self._instance_vbo = ...
    
    def emit(self, position, velocity, color, count=10, 
             spread=1.0, lifetime=1.0, size=0.1, gravity=1.0):
        """Spawn particles at position with random spread."""
    
    def update(self, dt):
        """Update all particle positions, apply gravity, remove dead."""
        # Vectorized NumPy update — no per-particle Python loop
        alive = self.particles[:self.count]
        alive['pos'] += alive['vel'] * dt
        alive['vel'][:, 1] -= 9.81 * alive['gravity'] * dt
        alive['life'] -= dt
        # Compact: remove dead particles
    
    def render(self, view_proj, camera_pos):
        """Render all particles as camera-facing billboards."""
```

### 9.6 Particle Effects [USING particle.py]

**Block break particles:**
- When a block is broken, emit 20–30 small colored particles.
- Color = dominant color of the block's albedo texture (sample from atlas).
- Particles fly outward with slight gravity, fade over 0.5–1.0 seconds.

**Block crack overlay:**
- While digging a block in survival mode, overlay a crack texture on the targeted block face.
- 10 stages of cracking (more cracks as progress increases).
- Rendered as a transparent quad on the block face with additive blending.

**Walking dust:**
- Emit 2–3 small dust particles at the player's feet when walking on dirt/sand/gravel.
- Color matches ground block.

**Torch/flame particles:**
- Emit 1 small orange/yellow particle every 0.5 seconds from torch blocks.
- Particle rises slowly (low velocity, no gravity), fades over 1 second.
- Small smoke particle (dark gray) rises above the flame particle.

**Water splash:**
- Emit 10 blue particles when entering water.
- Emit drip particles below blocks with water above.

**Lava:**
- Emit orange sparks rising from lava surface.
- Emit dark smoke particles.

**Critical hit:**
- Emit star-shaped particles (or bright white dots) in a burst at the hit location.

**Explosion (creeper):**
- Large burst of 100+ particles: white flash → expanding gray/orange cloud.
- Combine with block break particles for all destroyed blocks.

### 9.7 Particle Shader [ADD to shaders.py]

```glsl
// PARTICLE_VERT
in vec3 in_vert;     // Quad corners
in vec3 in_pos;      // Instance: world position
in vec4 in_color;    // Instance: RGBA
in float in_size;    // Instance: billboard size
in float in_life;    // Instance: normalized remaining life

uniform mat4 u_view_proj;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;

void main() {
    vec3 world = in_pos 
        + u_camera_right * in_vert.x * in_size 
        + u_camera_up * in_vert.y * in_size;
    gl_Position = u_view_proj * vec4(world, 1.0);
    v_color = in_color;
    v_alpha = in_life;  // Fade with remaining life
}

// PARTICLE_FRAG
in vec4 v_color;
in float v_alpha;
out vec4 frag_color;

void main() {
    // Soft circle: discard pixels outside radius
    vec2 uv = gl_PointCoord * 2.0 - 1.0;  // Or use quad UV
    float dist = length(uv);
    if (dist > 1.0) discard;
    float soft = 1.0 - smoothstep(0.5, 1.0, dist);
    frag_color = vec4(v_color.rgb, v_color.a * v_alpha * soft);
}
```

---

## Verification

1. Break a stone block — hear "stone break" sound, see gray particles scatter.
2. Walk on grass — hear footstep sounds in rhythm with movement.
3. Stand near lava — hear bubbling pops at correct volume/pan based on position.
4. Toggle rain — hear rain ambient loop fade in/out. See rain streaks (existing) + hear rain.
5. Fight a zombie — hear hit sounds, see damage particles, hear zombie groan.
6. Eat food — hear 3 crunching sounds during eating animation.
7. Place a torch — see flame particles rising above it.
8. Performance: 2000 active particles should add ≤ 1ms render time (instanced rendering).
9. Volume slider in settings — all sounds should scale correctly. Mute at 0.
