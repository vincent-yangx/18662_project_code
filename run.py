import pygame
import sys
import time
import self_env  # 注册环境
import gym
import numpy as np


def get_backpack_vec(agent):
    return [env.unwrapped.agent_backpack[agent][res] for res in ["wood", "stone", "iron", "diamond"]]

def print_warehouse_info():
    print("\n📦 仓库资源总量:")
    total_vec = [env.unwrapped.warehouse_storage[res] for res in ["wood", "stone", "iron", "diamond"]]
    print(f"  [wood, stone, iron, diamond]: {total_vec}")

    print("\n📦 仓库资源贡献来源:")
    for a in all_agents:
        contrib_vec = [env.unwrapped.collection_log[a][res] for res in ["wood", "stone", "iron", "diamond"]]
        print(f"  {a}: {contrib_vec}")

# 初始化 pygame 和环境
pygame.init()
env = gym.make('CustomMultiAgentEnv-v0')
screen = pygame.display.set_mode((env.unwrapped.grid_size * 30, env.unwrapped.grid_size * 30))
pygame.display.set_caption("🌳 资源收集游戏")
clock = pygame.time.Clock()
env.reset()
env.render(screen)

done = False
collected_by = {}
all_agents = env.unwrapped.agents

print("🎮 控制说明：WASD 控制 agent_1，方向键控制 agent_2")

while not done:
    # 键盘控制 agent_1 和 agent_2
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
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

            if action is not None:
                pos, reward, agent_done, message = env.unwrapped.step(agent, action)
                done = done or agent_done

                if "成功收集了" in message or "存入仓库" in message:
                    print(f"\n🧭 {agent} moved to {pos[agent]}")
                    print(f"📣 {message}")

                    # 打印 agent 背包资源
                    print("\n🎒 agent 背包资源:")
                    for a in all_agents:
                        backpack_vec = [env.unwrapped.agent_backpack[a][res] for res in
                                        ["wood", "stone", "iron", "diamond"]]
                        print(f"  {a}: {backpack_vec}")

                    # 打印仓库资源总量
                    print("\n📦 仓库资源总量:")
                    total_vec = [env.unwrapped.warehouse_storage[res] for res in ["wood", "stone", "iron", "diamond"]]
                    print(f"  [wood, stone, iron, diamond]: {total_vec}")

                    # 打印每个 agent 对仓库的贡献（来自 collection_log）
                    print("\n📦 仓库资源贡献来源:")
                    for a in all_agents:
                        contrib_vec = [env.unwrapped.collection_log[a][res] for res in
                                       ["wood", "stone", "iron", "diamond"]]
                        print(f"  {a}: {contrib_vec}")

                    if done:
                        print("\n🎉 游戏结束！")

                env.render(screen)

    # 自动控制 agent_3 和 agent_4
    for agent in ["agent_3", "agent_4"]:
        action = np.random.randint(0, 4)
        pos, reward, agent_done, message = env.unwrapped.step(agent, action)
        done = done or agent_done

        if "成功收集了" in message or "存入仓库" in message:
            print(f"\n🤖 {agent} moved to {pos[agent]}")
            print(f"📣 {message}")

            print("\n🎒 agent 背包资源:")
            for a in all_agents:
                backpack_vec = [env.unwrapped.agent_backpack[a][res] for res in ["wood", "stone", "iron", "diamond"]]
                print(f"  {a}: {backpack_vec}")

            print("\n📦 仓库资源总量:")
            total_vec = [env.unwrapped.warehouse_storage[res] for res in ["wood", "stone", "iron", "diamond"]]
            print(f"  [wood, stone, iron, diamond]: {total_vec}")

            print("\n📦 仓库资源贡献来源:")
            for a in all_agents:
                contrib_vec = [env.unwrapped.collection_log[a][res] for res in ["wood", "stone", "iron", "diamond"]]
                print(f"  {a}: {contrib_vec}")

            if done:
                print("\n🎉 游戏结束！")

        env.render(screen)

    clock.tick(10)  # 控制刷新速度
