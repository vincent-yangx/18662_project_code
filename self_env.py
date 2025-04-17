
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

        png_size = 30
        self.assets = {
            agent: pygame.transform.scale(pygame.image.load("assets/player.png"), (png_size, png_size))
            for agent in self.agents
        }
        for item in ["wood", "stone", "iron", "diamond", "warehouse"]:
            self.assets[item] = pygame.transform.scale(pygame.image.load(f"assets/{item}.png"), (png_size, png_size))

        self.warehouse_position = np.array([0, self.grid_size - 1])
        self.resource_counts = {"wood": 8, "stone": 6, "iron": 4, "diamond": 2}
        self.warehouse_storage = {res: 0 for res in self.resource_counts}
        self.collection_log = {agent: {res: 0 for res in self.resource_counts} for agent in self.agents}
        self.agent_backpack = {agent: {res: 0 for res in self.resource_counts} for agent in self.agents}

        self.reset()

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

        self.collected_flags = {res: [False]*len(pos_list) for res, pos_list in self.resources.items()}
        self.agent_backpack = {agent: {res: 0 for res in self.resource_counts} for agent in self.agents}
        self.warehouse_storage = {res: 0 for res in self.resource_counts}
        self.collection_log = {agent: {res: 0 for res in self.resource_counts} for agent in self.agents}
        return self.agent_positions

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

        # if all(self.warehouse_storage[res] > 0 for res in self.resource_counts):
        #     done = True
        #     message += f"\n🎯 所有资源至少存入了一次，游戏结束！"

        return self.agent_positions, reward, done, message

    def _can_collect(self, agent, resource):
        required = {
            "wood": set(),
            "stone": {"wood"},
            "iron": {"wood", "stone"},
            "diamond": {"wood", "stone", "iron"}
        }
        for req in required[resource]:
            if self.agent_backpack[agent][req] <= 0:
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

        for agent in self.agents:
            pos = self.agent_positions[agent]
            screen.blit(self.assets[agent], (pos[1]*cell_size, pos[0]*cell_size))

        pygame.display.flip()

    def close(self):
        pygame.quit()
