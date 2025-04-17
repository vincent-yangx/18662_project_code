import pygame
import sys
import time
import self_env  # 注册环境
import gym
import numpy as np


def get_backpack_vec(agent):
    return [env.unwrapped.agent_backpack[agent][res] for res in res_order]

def print_warehouse_info():
    print("\n📦 仓库资源总量:")
    total_vec = [env.unwrapped.warehouse_storage[res] for res in res_order]
    print(f"  [wood, stone, iron, diamond]: {total_vec}")

    print("\n📦 仓库资源贡献来源:")
    for a in all_agents:
        contrib_vec = [env.unwrapped.collection_log[a][res] for res in res_order]
        print(f"  {a}: {contrib_vec}")

def build_tool(agent, tool_name):
    if env.unwrapped.build_tool(agent, tool_name):
        print(f"✅ {agent} 成功制造了 {tool_name}！")
    else:
        print(f"❌ {agent} 无法制造 {tool_name}。")

    print(f"🎒 {agent} 当前背包资源: {get_backpack_vec(agent)}")

    print("\n🛠️ 工具状态：")
    for tool, built in env.unwrapped.tools_built.items():
        print(f"  {tool}: {'✅' if built else '❌'}")


# 初始化 pygame 和环境
pygame.init()
env = gym.make('CustomMultiAgentEnv-v0')

current_agent_index = 0
agent_list = ["agent_1", "agent_2", "agent_3", "agent_4"]
current_agent = agent_list[current_agent_index]

res_order = ["wood", "stone", "iron", "coal", "diamond"]
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
            if event.key == pygame.K_TAB:
                current_agent_index = (current_agent_index + 1) % len(agent_list)
                current_agent = agent_list[current_agent_index]
                print(f"🎮 当前控制的 agent: {current_agent}")

            action = None
            agent = None

            # agent_2 使用方向键控制
            if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                agent = "agent_2"
                if event.key == pygame.K_UP:
                    action = 3
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 1
                elif event.key == pygame.K_RIGHT:
                    action = 0

            # 其他 agent 使用 current_agent + WASD 控制
            elif event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]:
                agent = "agent_1"
                if event.key == pygame.K_w:
                    action = 3
                elif event.key == pygame.K_s:
                    action = 2
                elif event.key == pygame.K_a:
                    action = 1
                elif event.key == pygame.K_d:
                    action = 0

            # agent_3 使用 I K J L 控制
            elif event.key in [pygame.K_i, pygame.K_k, pygame.K_j, pygame.K_l]:
                agent = "agent_3"
                if event.key == pygame.K_i:
                    action = 3
                elif event.key == pygame.K_k:
                    action = 2
                elif event.key == pygame.K_j:
                    action = 1
                elif event.key == pygame.K_l:
                    action = 0

            # agent_4 使用 T G F H 控制
            elif event.key in [pygame.K_t, pygame.K_g, pygame.K_f, pygame.K_h]:
                agent = "agent_4"
                if event.key == pygame.K_t:
                    action = 3
                elif event.key == pygame.K_g:
                    action = 2
                elif event.key == pygame.K_f:
                    action = 1
                elif event.key == pygame.K_h:
                    action = 0

            if action is not None:
                pos, reward, agent_done, message = env.unwrapped.step(agent, action)
                done = done or agent_done

                if "成功收集了" in message:
                    print(f"\n🧭 {agent} moved to {pos[agent]}")
                    print(f"📣 {message}")
                    print(f"🎒 {agent} 背包资源: {get_backpack_vec(agent)}")
                elif "存入仓库" in message:
                    print(f"\n🧭 {agent} moved to {pos[agent]}")
                    print(f"📣 {message}")
                    print_warehouse_info()
                    print(f"🎒 {agent} 背包资源: {get_backpack_vec(agent)}")

                    if done:
                        print("\n🎉 游戏结束！")

                env.render(screen)

        # 快捷制造工具键绑定（当前控制 agent 执行）
            tool_key_map = {
                pygame.K_1: "table",
                pygame.K_2: "wood pickaxe",
                pygame.K_3: "stone pickaxe",
                pygame.K_4: "furnace",
                pygame.K_5: "iron pickaxe"
            }

            if event.key in tool_key_map:
                tool_to_build = tool_key_map[event.key]
                build_tool(current_agent, tool_to_build)


    # 放在 while not done: 循环底部，紧靠 clock.tick 前
    keys = pygame.key.get_pressed()
    if keys[pygame.K_BACKSLASH]:  # \键触发制造工具
        tool_to_build = input("请输入要制造的工具名: ").strip()
        build_tool(current_agent, tool_to_build)

    clock.tick(10)  # 控制刷新速度
