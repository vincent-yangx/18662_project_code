# final version with separate backpack and warehouse

import gym
from gym import spaces
import numpy as np
import pygame
import sys

from gym.envs.registration import register
register(
        id='CustomMultiAgentEnv-v0',
        entry_point='self_env:MultiAgentResourceEnv',
    )

class MultiAgentResourceEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(MultiAgentResourceEnv, self).__init__()
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
    # def _generate_adjacent_positions(self):
    #     directions = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]
    #
    #     while True:
    #         base = np.random.randint(0, self.grid_size, size=(2,))
    #         neighbors = [base + d for d in directions]
    #         neighbors = [n for n in neighbors if 0 <= n[0] < self.grid_size and 0 <= n[1] < self.grid_size]
    #
    #         np.random.shuffle(neighbors)  # 随机挑邻居
    #
    #         for adj in neighbors:
    #             # 检查 base 和 adj 是否与 agent 起始点、资源点冲突
    #             conflict = False
    #
    #             occupied_positions = list(self.agent_positions.values())
    #             for lst in self.resources.values():
    #                 occupied_positions += lst
    #
    #             if any(np.array_equal(base, pos) or np.array_equal(adj, pos) for pos in occupied_positions):
    #                 conflict = True
    #
    #             if not conflict:
    #                 return base, adj  # ✅ 成功生成一对不冲突的邻接点

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

        # self.warehouse_position, self.exit_position = self._generate_adjacent_positions()

        self.collection_log = {
            agent: {res: 0 for res in self.resource_counts}
            for agent in self.agents
        }

        self.collected_flags = {res: [False]*len(pos_list) for res, pos_list in self.resources.items()}
        self.shared_resource_pool = {res: 0 for res in self.resource_counts}

        # self.agent_backpack = {agent: {res: 0 for res in self.resource_counts} for agent in self.agents}
        # self.warehouse_storage = {res: 0 for res in self.resource_counts}
        # self.collection_log = {agent: {res: 0 for res in self.resource_counts} for agent in self.agents}

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

    def step(self, agent, action):
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

    def close(self):
        pygame.quit()
