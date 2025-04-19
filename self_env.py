# final version with separate backpack and warehouse

import gym
from gym import spaces
import numpy as np
import pygame
import sys
from collections import deque
import heapq

from gym.envs.registration import register
register(
        id='CustomMultiAgentEnv-v0',
        entry_point='self_env:MultiAgentResourceEnv',
    )

class MultiAgentResourceEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(MultiAgentResourceEnv, self).__init__()
        self.current_agent = None
        self.agents = ["agent_1", "agent_2", "agent_3", "agent_4"]
        self.grid_size = 20
        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)
            for agent in self.agents
        })
        self.action_space = spaces.Discrete(4)

        png_size = 35
        self.assets = {
            agent: pygame.transform.scale(pygame.image.load("assets/player.png"), (png_size, png_size))
            for agent in self.agents
        }
        for item in ["wood", "stone", "iron", "diamond", "warehouse","exit", "coal"]:
            self.assets[item] = pygame.transform.scale(pygame.image.load(f"assets/{item}.png"), (png_size, png_size))

        self.resource_counts = {
            "wood": 8,
            "stone": 7,
            "iron": 1,
            "coal": 1,
            "diamond": 1
        }

        # 工具的建造前提（资源或其他工具）
        self.tool_prerequisite = {
            "table": {"wood": 2},
            "wood pickaxe": {"wood": 2, "table": 1},
            "stone pickaxe": {"wood": 1, "stone": 1, "table": 1},
            "furnace": {"stone": 4, "table": 1},
            "iron pickaxe": {"coal": 1, "wood": 1, "iron": 1, "furnace": 1}
        }

        # 每种工具是否已建造（全局共享）
        self.tools_built = {tool: False for tool in self.tool_prerequisite}

        self.reset()

    def get_grid_matrix(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        resource_map = {"wood": 1, "stone": 2, "iron": 3, "diamond": 4, "coal": 5, "warehouse": 6, "exit": 7}

        for res_type, positions in self.resources.items():
            for x, y in positions:
                grid[x, y] = resource_map[res_type]

        for pos in self.agent_positions.values():
            grid[pos[0], pos[1]] = -1  # agent覆盖

        return grid

    def reset(self):
        self.agent_positions = {agent: np.array([0, 0]) for agent in self.agents}
        self.resources = {}
        for res_name, count in self.resource_counts.items():
            self.resources[res_name] = []
            for _ in range(count):
                while True:
                    pos = np.random.randint(0, self.grid_size, size=(2,))
                    if not any(np.array_equal(pos, p) for p in self.agent_positions.values()) and not any(np.array_equal(pos, existing) for lst in self.resources.values() for existing in lst):
                        self.resources[res_name].append(pos)
                        break

        # self.warehouse_position, self.exit_position = self._generate_adjacent_positions()

        self.collection_log = {
            agent: {res: 0 for res in self.resource_counts}
            for agent in self.agents
        }

        self.collected_flags = {res: [False]*len(pos_list) for res, pos_list in self.resources.items()}
        self.shared_resource_pool = {res: 0 for res in self.resource_counts}

        self.tools_built = {tool: False for tool in self.tool_prerequisite}

        return self.agent_positions

    def can_build_tool(self, agent, tool_name):
        prereq = self.tool_prerequisite[tool_name]
        for req, count in prereq.items():
            if req in self.tools_built:
                if not self.tools_built[req]:
                    return False
            else:
                if self.shared_resource_pool[req] < count:
                    return False
        return True

    def build_tool(self, agent, tool_name):
        if self.can_build_tool(agent, tool_name) and not self.tools_built[tool_name]:
            for req, count in self.tool_prerequisite[tool_name].items():
                if req not in self.tools_built:
                    self.shared_resource_pool[req] -= count
            self.tools_built[tool_name] = True
            return True
        return False

    def get_tool_status(self):
        return {tool: self.tools_built[tool] for tool in self.tool_prerequisite}

    def step(self, action):
        if self.current_agent is None:
            raise ValueError("当前没有 agent 被选中！")
        agent = self.current_agent
        movement = {
            0: np.array([0, 1]),
            1: np.array([0, -1]),
            2: np.array([1, 0]),
            3: np.array([-1, 0])
        }
        self.agent_positions[agent] += movement[action]
        self.agent_positions[agent] = np.clip(self.agent_positions[agent], 0, self.grid_size - 1)

        reward = 0
        done = False
        message = ""

        for res_name, pos_list in self.resources.items():
            for i, pos in enumerate(pos_list):
                if np.array_equal(self.agent_positions[agent], pos) and not self.collected_flags[res_name][i]:
                    if self._can_collect(agent, res_name):
                        self.shared_resource_pool[res_name] += 1
                        self.collected_flags[res_name][i] = True
                        reward = 10
                        message = f"{agent} 成功收集了 {res_name}!"
                        self.collection_log[agent][res_name] += 1

                        if res_name == "diamond":
                            done = True
                            message += f"\n💎 diamond 已被采集，游戏结束！"

        return self.agent_positions, reward, done, message

    def print_collected_summary(self):
        return dict(self.shared_resource_pool)

    def collect_resource(self, agent, resource_name):
        current_pos = self.agent_positions[agent]
        for i, pos in enumerate(self.resources.get(resource_name, [])):
            if not self.collected_flags[resource_name][i] and np.array_equal(current_pos, pos):
                if self._can_collect(agent, resource_name):
                    self.shared_resource_pool[resource_name] += 1
                    self.collected_flags[resource_name][i] = True
                    self.collection_log[agent][resource_name] += 1
                    self.resource_counts[resource_name] -= 1
                    return True, f"{agent} 成功采集 {resource_name}"
                else:
                    return False, f"{agent} 缺少采集 {resource_name} 所需工具"
        return False, f"{agent} 当前格子没有 {resource_name}"

    def _can_collect(self, agent, resource):
        # 每个资源需要的“前置工具”
        required_tools = {
            "wood": set(),  # 无需工具
            "stone": {"wood pickaxe"},
            "coal": {"wood pickaxe"},
            "iron": {"stone pickaxe"},
            "diamond": {"iron pickaxe"}
        }

        required = required_tools.get(resource, set())
        for tool in required:
            if not self.tools_built.get(tool, False):
                return False
        return True

    def print_shared_resources(self):
        print("📦 当前全局资源池：")
        for k, v in self.shared_resource_pool.items():
            print(f"  {k}: {v}")

        print("\n📈 各 Agent 采集记录：")
        for agent, res_dict in self.collection_log.items():
            res_str = ", ".join([f"{res}: {count}" for res, count in res_dict.items()])
            print(f"  {agent}: {res_str}")

    def get_env_state_summary(self):
        summary = []
        for agent in self.agents:
            pos = self.agent_positions[agent]
            summary.append(f"{agent} 位置: {list(pos)}")
        for k, v in self.shared_resource_pool.items():
            summary.append(f"资源 {k}: {v}")
        for tool, built in self.tools_built.items():
            status = "✅" if built else "❌"
            summary.append(f"工具 {tool}: {status}")
        return "\n".join(summary)

    def render(self, screen=None):
        if screen is None:
            if not hasattr(self, "_screen"):
                pygame.init()
                self._screen = pygame.display.set_mode((self.grid_size * 30, self.grid_size * 30))
            screen = self._screen

        screen.fill((255, 255, 255))
        cell_size = 30

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pygame.draw.rect(screen, (200, 200, 200), (y * cell_size, x * cell_size, cell_size, cell_size), 1)

        for res, pos_list in self.resources.items():
            for i, pos in enumerate(pos_list):
                if not self.collected_flags[res][i]:
                    screen.blit(self.assets[res], (pos[1]*cell_size, pos[0]*cell_size))

        # screen.blit(self.assets["warehouse"], (self.warehouse_position[1]*cell_size, self.warehouse_position[0]*cell_size))
        # screen.blit(self.assets["exit"], (self.exit_position[1] * cell_size, self.exit_position[0] * cell_size))

        for agent in self.agents:
            pos = self.agent_positions[agent]
            screen.blit(self.assets[agent], (pos[1]*cell_size, pos[0]*cell_size))

        pygame.display.flip()
    def get_shortest_path(self, start, goal):
        """
        A* algorithm to find the shortest path from `start` to `goal`.
        Returns a list of action integers (0=right, 1=left, 2=down, 3=up).
        """

        direction_map = {
            (0, 1): 0,    # right
            (0, -1): 1,   # left
            (1, 0): 2,    # down
            (-1, 0): 3    # up
        }

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

        start = tuple(start)
        goal = tuple(goal)
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), 0, start, []))
        visited = set()

        while open_set:
            _, cost, current, path = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return path  # List of action integers

            for (dx, dy), action_id in direction_map.items():
                nx, ny = current[0] + dx, current[1] + dy
                next_pos = (nx, ny)

                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if next_pos in visited:
                        continue

                    new_cost = cost + 1
                    priority = new_cost + heuristic(next_pos, goal)
                    heapq.heappush(open_set, (priority, new_cost, next_pos, path + [action_id]))

        return []  # No path found

    def close(self):
        pygame.quit()

