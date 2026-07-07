# Phase 8: Survival Depth — Tools, Hunger, Crafting & Combat

**Goal:** Transform survival mode from a basic health system into a deep, engaging experience with tool tiers, hunger mechanics, shaped crafting, furnace smelting, enchanting, and tactical combat.

**Depends on:** Phase 7 (hostile mobs to fight, dropped items), Phase 4 (new ore types).

---

## Current State Analysis

**File:** `game/inventory.py` (~140 lines)
- 36-slot inventory, stack size 64. No item metadata (no durability, no enchantments).
- Crafting: shapeless only, from `configs/recipes.json`. Single-input recipes.

**File:** `game/player.py` (~310 lines)
- Health: 20 HP, regen after 5s delay. Fall/lava/drowning damage.
- No hunger. No tool speed multipliers. Digging time = block hardness (constant).

**File:** `game/hud.py` (~280 lines)
- Hearts + air bubbles rendered. No hunger bar, no XP bar, no armor display.

---

## Implementation Steps

### 8.1 Item System Overhaul [NEW: `game/items.py`, `configs/items.json`]

Currently the game only has "block items" (placing blocks). Add a proper item system:

```json
// configs/items.json
{
  "items": [
    // Tools
    {"id": 256, "name": "wooden_pickaxe", "type": "tool", "tool_type": "pickaxe", "tier": "wood", "durability": 59, "mining_speed": 2.0, "attack_damage": 2, "stack_size": 1},
    {"id": 257, "name": "stone_pickaxe", "type": "tool", "tool_type": "pickaxe", "tier": "stone", "durability": 131, "mining_speed": 4.0, "attack_damage": 3, "stack_size": 1},
    {"id": 258, "name": "iron_pickaxe", "type": "tool", "tool_type": "pickaxe", "tier": "iron", "durability": 250, "mining_speed": 6.0, "attack_damage": 4, "stack_size": 1},
    {"id": 259, "name": "diamond_pickaxe", "type": "tool", "tool_type": "pickaxe", "tier": "diamond", "durability": 1561, "mining_speed": 8.0, "attack_damage": 5, "stack_size": 1},
    // Same for axes, shovels, swords, hoes
    
    // Weapons
    {"id": 280, "name": "wooden_sword", "type": "weapon", "attack_damage": 4, "attack_speed": 1.6, "durability": 59, "stack_size": 1},
    {"id": 281, "name": "stone_sword", "type": "weapon", "attack_damage": 5, "attack_speed": 1.6, "durability": 131, "stack_size": 1},
    {"id": 282, "name": "iron_sword", "type": "weapon", "attack_damage": 6, "attack_speed": 1.6, "durability": 250, "stack_size": 1},
    {"id": 283, "name": "diamond_sword", "type": "weapon", "attack_damage": 7, "attack_speed": 1.6, "durability": 1561, "stack_size": 1},
    
    // Resources
    {"id": 300, "name": "coal", "type": "resource", "stack_size": 64},
    {"id": 301, "name": "iron_ingot", "type": "resource", "stack_size": 64},
    {"id": 302, "name": "gold_ingot", "type": "resource", "stack_size": 64},
    {"id": 303, "name": "diamond", "type": "resource", "stack_size": 64},
    {"id": 304, "name": "stick", "type": "resource", "stack_size": 64},
    
    // Food
    {"id": 350, "name": "raw_porkchop", "type": "food", "hunger": 3, "saturation": 1.8, "stack_size": 64},
    {"id": 351, "name": "cooked_porkchop", "type": "food", "hunger": 8, "saturation": 12.8, "stack_size": 64},
    {"id": 352, "name": "raw_beef", "type": "food", "hunger": 3, "saturation": 1.8, "stack_size": 64},
    {"id": 353, "name": "steak", "type": "food", "hunger": 8, "saturation": 12.8, "stack_size": 64},
    {"id": 354, "name": "bread", "type": "food", "hunger": 5, "saturation": 6.0, "stack_size": 64},
    {"id": 355, "name": "apple", "type": "food", "hunger": 4, "saturation": 2.4, "stack_size": 64}
  ]
}
```

**Item ID scheme:** Block IDs use 0–255 (uint8). Item-only IDs start at 256. Inventory slots store a 16-bit item ID + 8-bit count + 16-bit durability.

### 8.2 Tool Mining Mechanics [MODIFY game/game.py, configs/blocks.json]

Add harvest level requirements to blocks:

```json
// In blocks.json, add to each block:
{
  "name": "iron_ore",
  "hardness": 3.0,
  "harvest_level": "stone",    // Requires stone pickaxe or better
  "harvest_tool": "pickaxe"    // Must use a pickaxe
}
```

**Mining speed formula:**
```
base_time = hardness * 1.5                         # Bare hand
if holding correct tool type:
    base_time = hardness / mining_speed              # Tool speed
if tool tier < harvest_level:
    base_time = hardness * 5.0                       # Wrong tier = very slow + no drop
    drop = None
```

**Tool durability:** Each block mined decreases durability by 1. Each mob hit decreases durability by 2. When durability reaches 0, tool breaks (remove from inventory, play break sound in Phase 9).

### 8.3 Hunger System [MODIFY game/player.py, game/hud.py]

```python
# In Player class:
self.hunger = 20.0        # 10 drumsticks (0–20)
self.saturation = 5.0     # Hidden buffer before hunger depletes (0–20)
self.exhaustion = 0.0     # Accumulated from actions
```

**Exhaustion sources:**
| Action | Exhaustion |
|---|---|
| Sprint (per block) | 0.1 |
| Jump | 0.05 |
| Sprint-jump | 0.2 |
| Attack | 0.1 |
| Mine block | 0.005 |
| Swimming (per block) | 0.01 |

When `exhaustion >= 4.0`: reset to 0, decrease saturation by 1. When saturation = 0 and exhaustion >= 4.0: decrease hunger by 1.

**Hunger effects:**
- Hunger > 17: natural regeneration (0.5 HP/s, costs 6 exhaustion per HP).
- Hunger ≤ 6: cannot sprint.
- Hunger = 0: take 1 damage every 4 seconds (starvation, won't kill below 1 HP on Easy equivalent).

**Eating:** RMB while holding food → 1.6-second eating animation → restore hunger + saturation from item config.

**HUD:** Render 10 drumstick icons (full/half/empty) next to hearts, right-aligned. Drumsticks shake when hunger ≤ 6.

### 8.4 Shaped Crafting [MODIFY game/inventory.py, game/ui.py, configs/recipes.json]

Replace shapeless-only crafting with a 3×3 crafting grid:

```json
// configs/recipes.json — add shaped recipes:
{
  "type": "shaped",
  "pattern": [
    "##",
    "##"
  ],
  "key": {"#": "oak_planks"},
  "output": "crafting_table",
  "count": 1
}
```

```json
{
  "type": "shaped",
  "pattern": [
    "###",
    " | ",
    " | "
  ],
  "key": {"#": "cobblestone", "|": "stick"},
  "output": "stone_pickaxe",
  "count": 1
}
```

**UI changes (`game/ui.py`):**
- Inventory screen: add a 2×2 crafting grid (portable crafting) next to the inventory.
- When interacting with a crafting table block (E or RMB): open a 3×3 crafting grid.
- Player drags items into the grid → result appears in the output slot → clicking output crafts it.
- For the first version (before drag-and-drop): click to place 1 item from held stack into a grid slot. Click output to craft.

### 8.5 Furnace / Smelting [NEW: block + UI]

**New block:** `furnace` (solid, interactable). Has a "lit" variant with emissive front texture.

**Furnace UI (`game/ui.py`):**
- Input slot (top): ore or raw food.
- Fuel slot (bottom): coal, wood, planks (each with a burn time).
- Output slot (right): smelted result.
- Progress arrow: fills as smelting progresses (10 seconds per item).

**Smelting recipes:**
| Input | Output |
|---|---|
| Iron Ore | Iron Ingot |
| Gold Ore | Gold Ingot |
| Raw Porkchop | Cooked Porkchop |
| Raw Beef | Steak |
| Sand | Glass |
| Cobblestone | Stone |
| Oak Log | Charcoal (= Coal) |

**Fuel burn times:**
| Fuel | Burn Time (items smelted) |
|---|---|
| Coal | 8 |
| Planks | 1.5 |
| Log | 1.5 |
| Stick | 0.5 |

Furnace continues smelting while the player is away (if the chunk is loaded). Uses the block tick system from Phase 5.

### 8.6 Armor System [NEW]

**Armor slots:** Helmet, Chestplate, Leggings, Boots. Displayed in inventory UI left side.

**Materials:** Leather (weakest), Iron, Gold (low durability, high enchantability), Diamond (strongest).

**Damage reduction:** Each armor piece has a defense value. Total defense = sum of worn pieces. Damage reduced by `defense * 4%` (max 80% at full diamond = 20 defense).

**HUD:** Render armor icons above health bar when wearing armor.

### 8.7 Combat Improvements [NEW: `game/combat.py`]

**Attack cooldown:** After swinging, 0.625-second cooldown. Attacking during cooldown deals reduced damage proportional to cooldown progress.

**Knockback:** Hitting a mob pushes it away from the player. Sprint-attacks deal extra knockback.

**Critical hits:** Hitting while falling (not on ground) deals 150% damage. Show particle effect (Phase 9).

**Shield/Blocking:** Holding RMB with a shield equipped reduces incoming damage by 100% from the front. Shield has durability.

**Sweep attack:** When attacking with a sword at full cooldown, nearby mobs within 1 block of the primary target take 1 damage + knockback.

---

## Verification

1. Craft a wooden pickaxe from planks + sticks → mine stone → craft stone pickaxe → mine iron ore.
2. Smelt iron ore in furnace → get iron ingot → craft iron sword.
3. Fight zombies — sword should deal correct damage, cooldown should be visible.
4. Sprint and jump repeatedly — hunger should decrease. Eat food to restore.
5. Mine diamond ore with iron pickaxe → get diamond. Mine with wooden pickaxe → get nothing.
6. Tool durability: mine 59 blocks with wooden pickaxe → it should break.
7. Wear full iron armor → take less damage from mob attacks.
8. Open inventory → 2×2 crafting grid works. Use crafting table → 3×3 grid works.
