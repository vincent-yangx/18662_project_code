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

        self.grid_size = 20
        self.observation_space = spaces.Dict({
            "agent_1": spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32),
            "agent_2": spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)
        })
        self.action_space = spaces.Discrete(4)

        png_size = 30
        self.assets = {
            "agent_1": pygame.transform.scale(pygame.image.load("assets/player.png"), (png_size, png_size)),
            "agent_2": pygame.transform.scale(pygame.image.load("assets/zombie.png"), (png_size, png_size)),
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

        # self.collected_resources = {"agent_1": set(), "agent_2": set()}
        self.collected_resources = set()
        self.reset()

    def reset(self):
        self.agent_positions = {
            "agent_1": np.array([0, 0]),
            "agent_2": np.array([self.grid_size - 1, self.grid_size - 1])
        }

        for resource in self.resources:
            while True:
                position = np.random.randint(0, self.grid_size, size=(2,))
                if not any(np.array_equal(position, pos) for pos in self.agent_positions.values()):
                    self.resources[resource] = position
                    break

        # self.collected_resources = {"agent_1": set(), "agent_2": set()}
        self.collected_resources = set()
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

        for resource, position in self.resources.items():
            if np.array_equal(self.agent_positions[agent], position):
                if self._can_collect(resource) and resource not in self.collected_resources:
                    self.collected_resources.add(resource)
                    reward = 10
                    message = f"{agent} 成功收集了 {resource}!"
                else:
                    message = f"⚠️ {agent} 未满足收集 {resource} 的条件!"

        if {"wood", "stone", "iron", "diamond"}.issubset(self.collected_resources):
            done = True
            message = f"🎯 {agent} 触发胜利！所有资源已被收集！"

        return self.agent_positions, reward, done, message

    def _can_collect(self, resource):
        required_resources = {
            "wood": set(),
            "stone": {"wood"},
            "iron": {"wood", "stone"},
            "diamond": {"wood", "stone", "iron"}
        }
        return required_resources[resource].issubset(self.collected_resources)

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

        for resource, pos in self.resources.items():
            screen.blit(self.assets[resource], (pos[1] * cell_size, pos[0] * cell_size))

        screen.blit(self.assets["agent_1"], (self.agent_positions["agent_1"][1] * cell_size,
                                             self.agent_positions["agent_1"][0] * cell_size))
        screen.blit(self.assets["agent_2"], (self.agent_positions["agent_2"][1] * cell_size,
                                             self.agent_positions["agent_2"][0] * cell_size))

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

        clock.tick(1000)


if __name__ == "__main__":

    main()
