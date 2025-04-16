import gym
from gym import spaces
import numpy as np
import pygame
import sys

from gym.envs.registration import register
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
        self.grid_size = 30
        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)
            for agent in self.agents
        })

        self.action_space = spaces.Discrete(4)

        png_size = 30
        self.assets = {
            "agent_1": pygame.transform.scale(pygame.image.load("assets/player.png"), (png_size, png_size)),
            "agent_2": pygame.transform.scale(pygame.image.load("assets/player.png"), (png_size, png_size)),
            "agent_3": pygame.transform.scale(pygame.image.load("assets/player.png"), (png_size, png_size)),
            "agent_4": pygame.transform.scale(pygame.image.load("assets/player.png"), (png_size, png_size)),
            "wood": pygame.transform.scale(pygame.image.load("assets/wood.png"), (png_size, png_size)),
            "stone": pygame.transform.scale(pygame.image.load("assets/stone.png"), (png_size, png_size)),
            "iron": pygame.transform.scale(pygame.image.load("assets/iron.png"), (png_size, png_size)),
            "diamond": pygame.transform.scale(pygame.image.load("assets/diamond.png"), (png_size, png_size)),
        }

        self.resources = {
            "wood": np.random.randint(0, self.grid_size, size=(2,)),
            "stone": np.random.randint(0, self.grid_size, size=(2,)),
            "iron": np.random.randint(0, self.grid_size, size=(2,)),
            "diamond": np.random.randint(0, self.grid_size, size=(2,))
        }

        self.warehouse_position = np.random.randint(0, self.grid_size, size=(2,))  # 可以改成任意固定位置
        self.assets["warehouse"] = pygame.transform.scale(pygame.image.load("assets/warehouse.png"),
                                                          (png_size, png_size))

        # 仓库中的资源总量
        self.warehouse_storage = {"wood": 0, "stone": 0, "iron": 0, "diamond": 0}

        # 每个 agent 的背包资源（即个人收集但未存入仓库的）
        self.agent_backpack = {
            agent: {"wood": 0, "stone": 0, "iron": 0, "diamond": 0}
            for agent in self.agents
        }

        # self.collected_resources = {"agent_1": set(), "agent_2": set()}
        self.collected_resources = set()
        self.reset()

        self.collected_resources = {"wood": 0, "stone": 0, "iron": 0, "diamond": 0}

    def reset(self):
        self.agent_positions = {agent: np.array([0, 0]) for agent in self.agents}


        # 定义每种资源的数量
        self.resource_counts = {
            "wood": 20,
            "stone": 16,
            "iron": 12,
            "diamond": 4
        }

        # 初始化资源位置：资源名 -> list of positions
        self.resources = {}

        for res_name, count in self.resource_counts.items():
            self.resources[res_name] = []
            for _ in range(count):
                while True:
                    pos = np.random.randint(0, self.grid_size, size=(2,))
                    if not any(np.array_equal(pos, p) for p in self.agent_positions.values()) and \
                            not any(
                                np.array_equal(pos, existing) for lst in self.resources.values() for existing in lst):
                        self.resources[res_name].append(pos)
                        break

        # self.collected_resources = {"agent_1": set(), "agent_2": set()}
        self.collected_resources = set()

        self.collected_resources = {"wood": 0, "stone": 0, "iron": 0, "diamond": 0}
        self.collection_log = {
            agent: {res: 0 for res in ["wood", "stone", "iron", "diamond"]}
            for agent in self.agents
        }

        self.collected_flags = {}
        for res_name, pos_list in self.resources.items():
            self.collected_flags[res_name] = [False] * len(pos_list)

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
                        self.collection_log[agent][res_name] += 1
                        self.collected_flags[res_name][i] = True  # ✅ 标记此位置已收集
                        reward = 10
                        message = f"{agent} 成功收集了 {res_name}!"
                    else:
                        message = f"⚠️ {agent} 未满足收集 {res_name} 的条件!"

        if all(v > 0 for v in self.collected_resources.values()):
            done = True
            message += f"\n🎯 {agent} 触发胜利！每种资源至少收集了一次！"

        # 检查是否进入仓库位置
        if np.array_equal(self.agent_positions[agent], self.warehouse_position):
            for res in ["wood", "stone", "iron", "diamond"]:
                amount = self.agent_backpack[agent][res]
                if amount > 0:
                    self.warehouse_storage[res] += amount
                    self.agent_backpack[agent][res] = 0
            message += f"\n🏠 {agent} 将资源存入仓库！"

        return self.agent_positions, reward, done, message

    def _can_collect(self, agent, resource):
        required_resources = {
            "wood": set(),
            "stone": {"wood"},
            "iron": {"wood", "stone"},
            "diamond": {"wood", "stone", "iron"}
        }

        for req in required_resources[resource]:
            if self.agent_backpack[agent][req] <= 0:
                return False
        return True

    def render(self, screen=None):
        if screen is None:
            if not hasattr(self, "_screen"):
                pygame.init()
                self._screen = pygame.display.set_mode((self.grid_size * 30, self.grid_size * 30))
            screen = self._screen

        cell_size = 30
        screen.fill((255, 255, 255))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pygame.draw.rect(screen, (200, 200, 200), (y * cell_size, x * cell_size, cell_size, cell_size), 1)

        for res_name, pos_list in self.resources.items():
            for i, pos in enumerate(pos_list):
                if not self.collected_flags[res_name][i]:
                    screen.blit(self.assets[res_name], (pos[1] * cell_size, pos[0] * cell_size))

        for agent in self.agents:
            screen.blit(self.assets["agent_1"], (self.agent_positions[agent][1] * cell_size,
                                                 self.agent_positions[agent][0] * cell_size))
        # 渲染仓库图标
        screen.blit(self.assets["warehouse"], (self.warehouse_position[1] * cell_size,
                                               self.warehouse_position[0] * cell_size))

        pygame.display.flip()

    def close(self):
        pygame.quit()


def main():
    pygame.init()
    env = MultiAgentResourceEnv()
    screen = pygame.display.set_mode((env.grid_size * 30, env.grid_size * 30))
    pygame.display.set_caption("🌳 资源收集游戏")
    clock = pygame.time.Clock()

    env.render(screen)
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                print(f"Key detected: {event.key}")
                agent = "agent_1"
                action = None

                if event.key == pygame.K_w:
                    action = 3
                elif event.key == pygame.K_s:
                    action = 2
                elif event.key == pygame.K_a:
                    action = 1
                elif event.key == pygame.K_d:
                    action = 0
                elif event.key == pygame.K_UP:
                    agent = "agent_2"
                    action = 3
                elif event.key == pygame.K_DOWN:
                    agent = "agent_2"
                    action = 2
                elif event.key == pygame.K_LEFT:
                    agent = "agent_2"
                    action = 1
                elif event.key == pygame.K_RIGHT:
                    agent = "agent_2"
                    action = 0

                print(f"Final action: {action}, agent: {agent}")
                if action is not None:
                    _, reward, done, message = env.step(agent, action)
                    print(f"{agent} moved to {env.agent_positions[agent]}")
                    print(message)
                    env.render(screen)

        # ✅ 移到事件循环外，每帧执行一次 AI agent 动作
        for agent in ["agent_3", "agent_4"]:
            action = np.random.randint(0, 4)
            _, reward, done, message = env.step(agent, action)
            if reward > 0 or "未满足" in message:
                print(f"{agent} moved to {env.agent_positions[agent]}")
                print(message)
                env.render(screen)

        clock.tick(50)  # 建议稍微调慢，1000 太快了

if __name__ == "__main__":

    main()
