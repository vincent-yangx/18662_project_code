
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
            "wood": 10,
            "stone": 8,
            "iron": 6,
            "coal": 4,
            "diamond": 2
        }

        self.warehouse_storage = {res: 0 for res in self.resource_counts}
        self.collection_log = {agent: {res: 0 for res in self.resource_counts} for agent in self.agents}
        self.agent_backpack = {agent: {res: 0 for res in self.resource_counts} for agent in self.agents}

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

    # warehouse 和 exit在一起
    def _generate_adjacent_positions(self):
        directions = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]

        while True:
            base = np.random.randint(0, self.grid_size, size=(2,))
            neighbors = [base + d for d in directions]
            neighbors = [n for n in neighbors if 0 <= n[0] < self.grid_size and 0 <= n[1] < self.grid_size]

            np.random.shuffle(neighbors)  # 随机挑邻居

            for adj in neighbors:
                # 检查 base 和 adj 是否与 agent 起始点、资源点冲突
                conflict = False

                occupied_positions = list(self.agent_positions.values())
                for lst in self.resources.values():
                    occupied_positions += lst

                if any(np.array_equal(base, pos) or np.array_equal(adj, pos) for pos in occupied_positions):
                    conflict = True

                if not conflict:
                    return base, adj  # ✅ 成功生成一对不冲突的邻接点

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
                    if not any(np.array_equal(pos, p) for p in self.agent_positions.values()) and                        not any(np.array_equal(pos, existing) for lst in self.resources.values() for existing in lst):
                        self.resources[res_name].append(pos)
                        break

        self.warehouse_position, self.exit_position = self._generate_adjacent_positions()


        self.collected_flags = {res: [False]*len(pos_list) for res, pos_list in self.resources.items()}
        self.agent_backpack = {agent: {res: 0 for res in self.resource_counts} for agent in self.agents}
        self.warehouse_storage = {res: 0 for res in self.resource_counts}
        self.collection_log = {agent: {res: 0 for res in self.resource_counts} for agent in self.agents}

        self.tools_built = {tool: False for tool in self.tool_prerequisite}

        return self.agent_positions

    def can_build_tool(self, agent, tool_name):
        prereq = self.tool_prerequisite[tool_name]
        for req, count in prereq.items():
            if req in self.tools_built:
                if not self.tools_built[req]:
                    return False
            else:
                if self.agent_backpack[agent][req] < count:
                    return False
        return True

    def build_tool(self, agent, tool_name):
        if self.can_build_tool(agent, tool_name) and not self.tools_built[tool_name]:
            for req, count in self.tool_prerequisite[tool_name].items():
                if req not in self.tools_built:
                    self.agent_backpack[agent][req] -= count  # ✅ 扣除资源
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
                        self.agent_backpack[agent][res_name] += 1
                        self.collected_flags[res_name][i] = True
                        reward = 10
                        message = f"{agent} 成功收集了 {res_name}!"

                        if res_name == "diamond":
                            done = True
                            message += f"\n💎 diamond 已被采集，游戏结束！"

        # 存入仓库
        if np.array_equal(self.agent_positions[agent], self.warehouse_position):
            stored_any = False
            for res in self.resource_counts:
                amount = self.agent_backpack[agent][res]
                if amount > 0:
                    self.warehouse_storage[res] += amount
                    self.collection_log[agent][res] += amount
                    self.agent_backpack[agent][res] = 0
                    stored_any = True
            if stored_any:
                message += f"\n🏠 {agent} 将资源存入仓库！"

        if np.array_equal(self.agent_positions[agent], self.exit_position):
            # 将 warehouse 所有资源转移到 agent 背包
            for res in self.warehouse_storage:
                amount = self.warehouse_storage[res]
                if amount > 0:
                    self.agent_backpack[agent][res] += amount
                    self.warehouse_storage[res] = 0

            # 打印资源转移信息
            message += f"\n🚪 {agent} 走入出口，领取所有仓库资源！"
            message += f"\n🎒 {agent} 当前资源: {[self.agent_backpack[agent][res] for res in self.warehouse_storage]}"
            message += f"\n📦 仓库当前资源: {[self.warehouse_storage[res] for res in self.warehouse_storage]}"

        return self.agent_positions, reward, done, message

    def print_collected_summary(self):
        # print("Collected Materials")
        # for res, count in self.collected_resources.items():
        #     print(f"- {res}: {count}")

        return dict(self.resource_counts)

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

        screen.blit(self.assets["warehouse"], (self.warehouse_position[1]*cell_size, self.warehouse_position[0]*cell_size))
        screen.blit(self.assets["exit"], (self.exit_position[1] * cell_size, self.exit_position[0] * cell_size))

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

