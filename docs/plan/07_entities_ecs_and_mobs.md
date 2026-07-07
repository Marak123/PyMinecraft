# Phase 7: Entity Component System, Mobs & AI

**Goal:** Refactor the pre-ECS entity system into a proper ECS, add hostile mobs with behavior-tree AI and A* pathfinding, implement a spawn cycle, mob drops, and animal breeding.

**Depends on:** Phase 4 (new biomes for biome-specific spawns), Phase 5 (fluid awareness for pathfinding).

---

## Current State Analysis

**File:** `game/entities.py` (~240 lines)
- `Pig` class: wander/idle/flee states, sine-wave walk animation, 6 HP, box rendering.
- `MobManager`: spawns up to 10 pigs within 18–34 blocks of player. Despawns beyond 64 blocks.
- `PlayerModel`: blocky humanoid for third-person view.
- Comment at top: "Deliberately pre-ECS."
- `_ray_aabb()`: simple slab-method ray intersection for attack targeting.
- No entity-entity collision. No drops. No hostile AI.

**File:** `game/player.py` (~310 lines)
- Player is a standalone class, not an entity. Has its own physics, collision, movement.

---

## Implementation Steps

### 7.1 ECS Core [NEW: `engine/ecs/`]

```
engine/ecs/
  __init__.py
  world.py          # ECS World container
  components.py     # All component types
  systems/
    __init__.py
    physics_system.py
    ai_system.py
    render_system.py
    spawn_system.py
    damage_system.py
```

**ECS World (`engine/ecs/world.py`):**

```python
class ECSWorld:
    """Lightweight ECS: entities are integer IDs, components are stored in typed dicts."""
    
    def __init__(self):
        self._next_id = 0
        self._components: dict[type, dict[int, Any]] = {}
        self._systems: list[System] = []
    
    def create_entity(self) -> int: ...
    def destroy_entity(self, entity_id: int): ...
    def add_component(self, entity_id: int, component): ...
    def get_component(self, entity_id: int, comp_type: type): ...
    def get_entities_with(self, *comp_types) -> Iterator[int]: ...
    def update(self, dt: float): ...  # Runs all systems in order
```

Keep it simple — no archetype storage or bitset queries. Dict-based lookup is fast enough for < 500 entities.

### 7.2 Components [NEW: `engine/ecs/components.py`]

```python
@dataclass
class Transform:
    position: np.ndarray  # float64 [3]
    yaw: float = 0.0
    pitch: float = 0.0

@dataclass  
class Velocity:
    value: np.ndarray  # float64 [3]

@dataclass
class Collider:
    half_width: float
    height: float
    on_ground: bool = False

@dataclass
class Health:
    current: float
    maximum: float
    invulnerable_timer: float = 0.0
    death_timer: float = 0.0

@dataclass
class MobAI:
    behavior_tree: 'BehaviorNode'
    state: dict  # Blackboard for the behavior tree
    mob_type: str

@dataclass
class Renderable:
    render_fn: Callable  # Function that emits box draws
    walk_cycle: float = 0.0

@dataclass
class Drops:
    loot_table: str  # Reference to loot table config

@dataclass
class Spawner:
    mob_type: str
    cooldown: float
    max_nearby: int
    radius: float

@dataclass
class Knockback:
    direction: np.ndarray
    strength: float
    decay: float = 10.0

@dataclass
class DamageFlash:
    timer: float = 0.0
    color: tuple = (1.0, 0.3, 0.3)
```

### 7.3 Behavior Tree AI [NEW: `game/ai/behavior_tree.py`]

```python
class NodeStatus(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3

class BehaviorNode:
    def tick(self, entity_id: int, ecs: ECSWorld, world: World, dt: float) -> NodeStatus: ...

class Selector(BehaviorNode):
    """Tries children in order until one succeeds."""

class Sequence(BehaviorNode):
    """Runs children in order, fails if any fails."""

class Condition(BehaviorNode):
    """Checks a predicate (e.g., 'is player nearby?')."""

class Action(BehaviorNode):
    """Performs an action (e.g., 'walk towards target')."""
```

**Example: Zombie Behavior Tree**
```
Selector:
  Sequence [Attack]:
    Condition: player_within(16 blocks)
    Condition: has_line_of_sight(player)
    Action: path_to(player)
    Condition: player_within(2 blocks)
    Action: melee_attack(player, damage=3)
  Sequence [Wander]:
    Condition: wander_timer_expired
    Action: pick_random_destination(8 blocks)
    Action: path_to(destination)
  Action [Idle]:
    wait(1-3 seconds)
```

### 7.4 A* Pathfinding [NEW: `game/ai/pathfinding.py`]

```python
def find_path(world: World, start: tuple, goal: tuple, 
              max_steps: int = 200, 
              can_swim: bool = False,
              jump_height: int = 1) -> list[tuple] | None:
    """3D A* pathfinding on the voxel grid.
    
    Returns a list of (x, y, z) waypoints or None if no path found.
    
    Walkability rules:
    - Block at feet must be air (or water if can_swim)
    - Block at head must be air
    - Block below feet must be solid (or water if can_swim)
    - Can step up 1 block (jump_height)
    - Can drop down up to 3 blocks (avoids fall damage)
    """
```

**Performance:**
- Run pathfinding on a background thread. Store the result in `MobAI.state['path']`.
- Re-path every 2 seconds or when the target moves significantly.
- Limit `max_steps` to 200 to prevent long searches for unreachable goals.
- Use Manhattan distance heuristic (fast, admissible in grid worlds).

### 7.5 Mob Types [NEW: `game/mobs/`, `configs/mobs.json`]

**Config format:**
```json
// configs/mobs.json
{
  "pig": {
    "health": 10, "half_width": 0.45, "height": 0.9,
    "speed": 2.0, "hostile": false, "spawns_in": ["plains", "forest"],
    "drops": [{"item": "raw_porkchop", "count": [1, 3]}],
    "behavior": "passive_wander",
    "model": {"body": [0.9, 0.5, 0.6], "head": [0.5, 0.4, 0.5], ...}
  },
  "zombie": {
    "health": 20, "half_width": 0.3, "height": 1.9,
    "speed": 2.3, "hostile": true, "spawns_in": ["any_dark"],
    "drops": [{"item": "rotten_flesh", "count": [0, 2]}],
    "behavior": "hostile_melee", "burns_in_sun": true,
    "model": {"body": [...], "head": [...], "arms": [...], "legs": [...]}
  },
  "skeleton": {
    "health": 20, "half_width": 0.3, "height": 1.9,
    "speed": 2.5, "hostile": true, "spawns_in": ["any_dark"],
    "drops": [{"item": "bone", "count": [0, 2]}, {"item": "arrow", "count": [0, 2]}],
    "behavior": "hostile_ranged", "burns_in_sun": true
  },
  "creeper": {
    "health": 20, "half_width": 0.3, "height": 1.7,
    "speed": 2.5, "hostile": true, "spawns_in": ["any_dark"],
    "drops": [{"item": "gunpowder", "count": [0, 2]}],
    "behavior": "hostile_explode"
  },
  "spider": {
    "health": 16, "half_width": 0.7, "height": 0.9,
    "speed": 3.0, "hostile": true, "spawns_in": ["any_dark"],
    "drops": [{"item": "string", "count": [0, 2]}],
    "behavior": "hostile_melee", "can_climb": true
  },
  "cow": {
    "health": 10, "half_width": 0.45, "height": 1.4,
    "speed": 2.0, "hostile": false, "spawns_in": ["plains", "forest"],
    "drops": [{"item": "raw_beef", "count": [1, 3]}, {"item": "leather", "count": [0, 2]}],
    "behavior": "passive_wander"
  },
  "sheep": {
    "health": 8, "half_width": 0.45, "height": 1.3,
    "speed": 2.0, "hostile": false, "spawns_in": ["plains"],
    "drops": [{"item": "wool", "count": 1}],
    "behavior": "passive_wander"
  },
  "chicken": {
    "health": 4, "half_width": 0.2, "height": 0.7,
    "speed": 1.5, "hostile": false, "spawns_in": ["plains", "forest"],
    "drops": [{"item": "feather", "count": [0, 2]}, {"item": "raw_chicken", "count": 1}],
    "behavior": "passive_wander", "slow_fall": true
  }
}
```

### 7.6 Spawn System [NEW: `engine/ecs/systems/spawn_system.py`]

**Rules:**
- **Passive mobs** (pig, cow, sheep, chicken): Spawn in appropriate biomes during chunk generation (initial population). Don't despawn naturally. Cap: 10 passive mobs within 128 blocks.
- **Hostile mobs** (zombie, skeleton, creeper, spider): Spawn at night or in dark areas (light level < 7). Spawn 24–128 blocks from the player. Despawn beyond 128 blocks. Cap: 30 hostile mobs within 128 blocks.
- **Spawn rate:** Check every 5 seconds. Spawn 1–4 mobs per check.
- **No spawning:** In creative mode. Within 24 blocks of player. In water (unless aquatic mob). On transparent blocks.

### 7.7 Mob Rendering [MODIFY game/entities.py → game/mobs/]

Each mob type defines a `render()` method that uses `BoxRenderer` to draw colored boxes:
- **Body parts:** Head, body, legs (2 or 4), arms (for humanoids), snout (for pigs/cows).
- **Walk animation:** Leg swing using sinusoidal functions of `walk_cycle` (already done for Pig — generalize).
- **Attack animation:** Arm swing for zombies, bow draw for skeletons.
- **Damage flash:** Tint red for 0.3 seconds on hit.
- **Death animation:** Entity falls over (rotate 90° over 0.5 seconds), then despawn.
- **Burn animation:** For zombies/skeletons in sunlight — add orange particle overlay.

### 7.8 Dropped Items [NEW system]

When a mob dies or a block is broken in survival:
- Create a dropped item entity with: position, velocity (small upward + random horizontal), item_id, count, pickup_delay (0.5 seconds).
- Dropped items bob up and down (sinusoidal Y offset) and spin slowly (increasing yaw).
- Render as a small 3D block icon (use BoxRenderer or render a single face quad).
- When the player walks within 1.5 blocks and pickup_delay has passed, add to inventory. If inventory full, leave on ground.
- Dropped items despawn after 5 minutes.

### 7.9 Migrate Existing Entities [MODIFY game/entities.py, game/player.py]

- Convert `Pig` to an ECS entity with `Transform`, `Velocity`, `Collider`, `Health`, `MobAI`, `Renderable`, `Drops` components.
- **Player stays as a standalone class** for now — it has too much unique input handling. But add an ECS entity ID for the player so systems can target it (e.g., hostile mob AI targets `player_entity_id`).
- Remove `MobManager` class. Replace with `SpawnSystem` in ECS.

---

## Verification

1. Walk around at night — hostile mobs should spawn in darkness.
2. Zombies should chase the player and deal damage.
3. Skeletons should shoot arrows (projectile entities).
4. Creepers should approach and explode (destroys blocks in radius).
5. Kill a pig — it should drop raw_porkchop items on the ground.
6. Walk over dropped items — they should fly into inventory.
7. Spawn limits: no more than 30 hostiles should exist simultaneously.
8. Performance: 50 active entities should not drop FPS below 60.
9. Save/load: entity positions should persist (at least for passive mobs in loaded chunks).
