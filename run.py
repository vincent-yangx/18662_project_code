import pygame
import sys
import time
import self_env  # 注册环境
import gym
# import numpy as np


def get_backpack_vec(agent):
    return [env.unwrapped.agent_backpack[agent][res] for res in res_order]

def print_warehouse_info():
    print("\n📦 仓库资源总量:")
    total_vec = [env.unwrapped.warehouse_storage[res] for res in res_order]
    print(f"  [wood, stone, iron, diamond]: {total_vec}")

    # print("\n📦 仓库资源贡献来源:")
    # for a in all_agents:
    #     contrib_vec = [env.unwrapped.collection_log[a][res] for res in res_order]
    #     print(f"  {a}: {contrib_vec}")

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

                if any(keyword in message for keyword in ["成功收集了", "存入仓库", "走入出口"]):
                    print(f"\n🧭 {agent} moved to {pos[agent]}")
                    print(f"📣 {message}")
                    print(f"🎒 {agent} 背包资源: {get_backpack_vec(agent)}")

                    if "存入仓库" in message:
                        print_warehouse_info()

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




# def build_full_llm_prompt(env):
#     lines = []
#
#     lines.append("🎯 最终目标：采集到 diamond 资源。")
#     lines.append("💡 注意：agent 走到 diamond 并成功采集即视为完成任务。")
#     lines.append("🚪 exit 的作用是取出仓库资源，并不表示游戏胜利或结束。")
#     lines.append("📦 仓库可以由 agent 存入资源，然后任何 agent 从 exit 取出使用。")
#
#     lines.append("\n📦 仓库：agent 可以将资源存入仓库（位置固定）。")
#     lines.append("🚪 出口：agent 走到出口后可以一次性取出仓库中所有资源，并加入自己背包中。")
#
#     lines.append("\n=== 🔧 资源采集前提规则 ===")
#     lines.append("1. wood 可直接采集")
#     lines.append("2. stone 和 coal 需要 wood pickaxe")
#     lines.append("3. iron 需要 stone pickaxe")
#     lines.append("4. diamond 需要 iron pickaxe")
#
#     lines.append("\n=== 🛠️ 工具建造前提规则 ===")
#     lines.append("table: 需要 2 wood")
#     lines.append("wood pickaxe: 需要 2 wood + table")
#     lines.append("stone pickaxe: 需要 1 wood + 1 stone + table")
#     lines.append("furnace: 需要 4 stone + table")
#     lines.append("iron pickaxe: 需要 1 coal + 1 wood + 1 iron + furnace")
#
#     # === Agent 状态 ===
#     lines.append("\n=== 🤖 Agent 状态 ===")
#     for agent in env.agents:
#         pos = env.agent_positions[agent]
#         backpack = env.agent_backpack[agent]
#         backpack_str = ", ".join([f"{k}: {v}" for k, v in backpack.items()])
#         lines.append(f"{agent} 在位置 {list(pos)}，背包资源：{backpack_str}")
#
#     # === 地图上的资源位置 ===
#     lines.append("\n=== 🗺️ 地图上的资源位置（未采集） ===")
#     for res_name, pos_list in env.resources.items():
#         for i, pos in enumerate(pos_list):
#             if not env.collected_flags[res_name][i]:
#                 lines.append(f"{res_name} at {list(pos)}")
#
#     # === 仓库信息 ===
#     lines.append(f"\n=== 📦 仓库 ===")
#     lines.append(f"仓库位置: {list(env.warehouse_position)}")
#     storage_str = ", ".join([f"{k}: {v}" for k, v in env.warehouse_storage.items()])
#     lines.append(f"当前资源：{storage_str}")
#
#     # === 出口信息 ===
#     lines.append(f"\n=== 🚪 出口 ===")
#     lines.append(f"出口位置: {list(env.exit_position)}")
#
#     # === 工具建造状态 ===
#     lines.append("\n=== 🛠️ 工具状态 ===")
#     for tool, built in env.tools_built.items():
#         status = "✅ 已建造" if built else "❌ 未建造"
#         lines.append(f"{tool}: {status}")
#
#     return "\n".join(lines)
#
#
# prompt = build_full_llm_prompt(env.unwrapped)
# print(prompt)


