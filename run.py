import pygame
import sys
import time
import self_env  # 注册环境
import gym

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

print("🎮 控制说明：WASD 控制 agent_1，方向键控制 agent_2")

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            agent = "agent_1"
            action = None

            # 键位绑定
            if event.key == pygame.K_w:
                action = 3  # 上
            elif event.key == pygame.K_s:
                action = 2  # 下
            elif event.key == pygame.K_a:
                action = 1  # 左
            elif event.key == pygame.K_d:
                action = 0  # 右
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

            # 执行动作
            if action is not None:
                pos, reward, agent_done, message = env.unwrapped.step(agent, action)
                done = done or agent_done

                if reward > 0 or ("未满足收集" in message):
                    print(f"\n🧭 {agent} moved to {pos[agent]}")
                    print(f"📣 {message}")

                    if reward > 0:
                        for res in ["wood", "stone", "iron", "diamond"]:
                            if res in message:
                                collected_by[res] = agent

                    print(f"🧺 当前已收集资源: {env.unwrapped.collected_resources}")
                    print("🔍 资源采集记录:")
                    for res in sorted(env.unwrapped.collected_resources):
                        who = collected_by.get(res, "未知")
                        print(f"  - {res}: {who}")

                    if done:
                        print("\n🎉 游戏结束！")

                env.render(screen)

    clock.tick(60)  # 控制刷新速度
