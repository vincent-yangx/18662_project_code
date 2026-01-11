# Dev Log: Version Updates, Next Steps, and Open Questions

This document records **change history**, **next-step planning**, and **current design questions** for the project.

---

## Change History / Updates

- Resources are added to the shared repository immediately after collection.
- Each resource type can appear multiple times on the map.
- A resource at the same location can be collected **only once**.
- Added a shared repository / warehouse mechanism.
- Tools are currently configured as **globally shared** across agents. ✅ (Done)
- Updated mining prerequisites. ✅ (Done)
- There is a fixed warehouse location on the map; when an agent reaches the warehouse tile, it deposits all carried items into the warehouse.
- Added an “exit” icon/tile. ✅ (Done; currently uses `exit`)

---

## Next Steps / Roadmap

### What to pass into the LLM (State + Hints)
1. Goal / objective description
2. Tool crafting / construction information
3. Agent status
4. Resource locations
5. Warehouse status (inventory)
6. Warehouse and exit positions
7. Tool suggestions
8. Distance hints
9. Crafting/build suggestions
10. High-level navigation suggestions (instead of low-level moves like up/down/left/right)
    - Question: should we integrate explicit pathfinding?

### Have Finished: LLM integration details
- How to store memory (data structure), and how to use past memory to optimize action trajectories

### Haven't Started:
- Whether to switch to partial observability (local / limited knowledge)
- If the map has insufficient resources for an agent to complete a task:
  - Will it retrieve resources from the warehouse?
  - Or request other agents to deposit resources first, then retrieve them?
- How to define **communication cost**
  - The resource-sharing considerations above can be part of the cost model
- Whether task execution can be interrupted / preempted
- Cooperation strategies have not been fully considered yet

---

## Current Questions / Confusions

- If an agent is given a goal, can it **communicate (update the goal)** or be **interrupted** while moving?
- How should **memory** be stored and retrieved/queried during execution?
- Should each agent have different attributes to encourage **specialization**?
- Consider making tools **non-shareable / carry-only** instead of globally shared (Haven’t started)
- Agent backpack capacity constraints

---

## Future Ideas (Wish List)

- Add obstacles, zombies, lava, etc.
- Add agent health / HP system
- Add pathfinding (e.g., A*, Dijkstra)
