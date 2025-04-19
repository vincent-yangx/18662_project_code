# final version with separate backpack and warehouse

import pygame
import sys
import time
import self_env  # 注册环境
import gym
import numpy as np
import random
import sys
import json
import pygame

from self_env import MultiAgentResourceEnv
from pydantic import BaseModel
from openai import OpenAI
from typing import Literal, Optional, List

pygame.init()
env = MultiAgentResourceEnv()

current_agent_index = 0
agent_list = ["agent_1", "agent_2", "agent_3", "agent_4"]

res_order = ["wood", "stone", "iron", "coal", "diamond"]

def build_planner_input(env, warehouse_summary):
    grid = env.get_grid_matrix().tolist()
    agents = {}
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == -1:
                agents[f"agent_{len(agents) + 1}"] = [i, j]

    return {
        "map": grid,
        "agents": agents,
        "resources_collected": warehouse_summary,
        "rules": {
            "resource_order": ["wood", "stone", "iron", "diamond"],
            "movement": ["up", "down", "left", "right"],
            "shared_resources": True
        }
    }


def build_prompt():
    return """You are a centralized planner for a 2D multi-agent resource collection environment.

Given:
- A 2D map (list of lists) where:
    - 0 = empty space
    - -1 = agent position
    - 1 = wood
    - 2 = stone
    - 3 = iron
    - 4 = diamond
- Agent positions and current warehouse status (resources collected).
- Game rules:
    1. Agents can only move up/down/left/right.
    2. Agents must collect resources in order: wood → stone → iron → diamond.
    3. Resources are shared across agents.

Please assign one action per agent. Each action should be either:
- move: with a direction (up/down/left/right)
- collect: collect the resource at current position
- wait: do nothing this turn

Use this structured JSON format:
{
  "actions": [
    {
      "agent_id": "agent_1",
      "action": "move",
      "target": [2, 3],
      "reason": "moving toward nearest wood"
    }
  ]
}
Respond with only valid JSON.
"""

# ----------- Models -----------

class AgentAction(BaseModel):
    agent_id: Literal["agent_1", "agent_2", "agent_3", "agent_4"]
    action: Literal["move", "collect", "create", "wait"]
    target_pos: Optional[List[int]] = None
    target_resource: Optional[Literal["wood", "stone", "coal", "diamond"]] = None
    target_tool: Optional[Literal["table", "wood pickaxe", "stone pickaxe", "furnace", "iron pickaxe"]] = None
    reason: str
    # direction: Optional[Literal["up", "down", "left", "right"]] = None


class AgentPlan(BaseModel):
    actions: List[AgentAction]


# ----------- GPT Planning Function -----------

client = OpenAI(api_key="replace your api here")

def ask_gpt_to_plan(client, env, warehouse_summary):
    planner_input = build_planner_input(env, warehouse_summary)  # I think memory should be passed here
    system_prompt = build_prompt()

    response = client.beta.chat.completions.parse(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(planner_input)}
        ],
        response_format=AgentPlan,
    )

    return response.choices[0].message.parsed


def build_tool(agent, tool_name):
    if env.unwrapped.build_tool(agent, tool_name):
        print(f"✅ {agent} 成功制造了 {tool_name}！")
    else:
        print(f"❌ {agent} 无法制造 {tool_name}。")

    env.unwrapped.print_shared_resources()
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
                env.unwrapped.current_agent = agent
                pos, reward, agent_done, message = env.unwrapped.step(action)
                done = done or agent_done

                if "成功收集了" in message:
                    print(f"\n🧭 {agent} moved to {pos[agent]}")
                    print(f"📣 {message}")
                    env.unwrapped.print_shared_resources()

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




# LLM控制框架组件（适配 "move_to" 目标指令 + 工具建造）

# def build_full_llm_prompt(env):
#     def manhattan_dist(p1, p2):
#         return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
#
#     required_tools = {
#         "wood": set(),
#         "stone": {"wood pickaxe"},
#         "coal": {"wood pickaxe"},
#         "iron": {"stone pickaxe"},
#         "diamond": {"iron pickaxe"}
#     }
#
#     def is_resource_collectible(env, resource):
#         required = required_tools.get(resource, set())
#         return all(env.tools_built.get(tool, False) for tool in required)
#
#     lines = []
#
#     lines.append("🎯 最终目标：采集到 diamond 资源。")
#     lines.append("💡 注意：agent 走到 diamond 并成功采集即视为完成任务。")
#     lines.append("🚪 exit 的作用是取出仓库资源，并不表示游戏胜利或结束。")
#     lines.append("📦 仓库可以由 agent 存入资源，然后任何 agent 从 exit 取出使用。")
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
#     lines.append("\n=== 🤖 Agent 状态 ===")
#     for agent in env.agents:
#         pos = env.agent_positions[agent]
#         backpack = env.agent_backpack[agent]
#         backpack_str = ", ".join([f"{k}: {v}" for k, v in backpack.items()])
#         lines.append(f"{agent} 在位置 {list(pos)}，背包资源：{backpack_str}")
#
#     lines.append("\n=== 🗺️ 地图上的资源位置（未采集） ===")
#     for res_name, pos_list in env.resources.items():
#         for i, pos in enumerate(pos_list):
#             if not env.collected_flags[res_name][i]:
#                 lines.append(f"{res_name} at {list(pos)}")
#
#     lines.append(f"\n=== 📦 仓库 ===")
#     lines.append(f"仓库位置: {list(env.warehouse_position)}")
#     storage_str = ", ".join([f"{k}: {v}" for k, v in env.warehouse_storage.items()])
#     lines.append(f"当前资源：{storage_str}")
#
#     lines.append(f"\n=== 🚪 出口 ===")
#     lines.append(f"出口位置: {list(env.exit_position)}")
#
#     lines.append("\n=== 🛠️ 工具状态 ===")
#     for tool, built in env.tools_built.items():
#         status = "✅ 已建造" if built else "❌ 未建造"
#         lines.append(f"{tool}: {status}")
#
#     lines.append("\n=== 📏 Agent 到“可采资源”的最短距离 ===")
#     for agent in env.agents:
#         agent_pos = env.agent_positions[agent]
#         dist_info = []
#         for res_name, pos_list in env.resources.items():
#             if not is_resource_collectible(env, res_name):
#                 continue
#             uncollected_positions = [pos for i, pos in enumerate(pos_list) if not env.collected_flags[res_name][i]]
#             if uncollected_positions:
#                 min_dist = min([manhattan_dist(agent_pos, pos) for pos in uncollected_positions])
#                 dist_info.append(f"{res_name}: {min_dist}")
#         dist_summary = ", ".join(dist_info) if dist_info else "无可采资源"
#         lines.append(f"{agent} 到可采资源最近距离: {dist_summary}")
#
#     lines.append("\n=== 🧩 建造建议 ===")
#     tool_priority = ["table", "wood pickaxe", "stone pickaxe", "furnace", "iron pickaxe"]
#     missing_tools = [tool for tool in tool_priority if not env.tools_built[tool]]
#     if not missing_tools:
#         lines.append("所有关键工具已建造完毕，无需建造新工具。")
#     else:
#         next_tool = missing_tools[0]
#         lines.append(f"建议优先建造：🛠️ {next_tool}")
#         prereq = env.tool_prerequisite[next_tool]
#         lacking = []
#         for res, amount in prereq.items():
#             if res in env.tools_built:
#                 if not env.tools_built[res]:
#                     lacking.append(f"{res}（未建）")
#             else:
#                 total_available = sum(agent[res] for agent in env.agent_backpack.values()) + env.warehouse_storage[res]
#                 if total_available < amount:
#                     lacking.append(f"{res}（缺 {amount - total_available}）")
#         if lacking:
#             lines.append("➡️ 当前缺少的前置资源或工具: " + ", ".join(lacking))
#         else:
#             lines.append("✅ 所有建造材料都已具备，可立即建造！")
#
#     # 📤 输出格式说明（目标坐标为主）
#     lines.append("\n=== 📤 输出格式要求（目标为位置） ===")
#     lines.append("请为每个 agent 输出下一步目标，使用如下 JSON 格式：")
#     lines.append("""
# {
#   "agent_1": {"action": "move_to", "target": [3, 5]},
#   "agent_2": {"action": "build", "tool": "stone pickaxe"},
#   "agent_3": {"action": "move_to", "target": "exit"},
#   "agent_4": {"action": "noop"}
# }
#     """)
#     lines.append("说明：")
#     lines.append("- move_to 的目标可以是 [x, y] 位置，或 'warehouse' 或 'exit'")
#     lines.append("- build 表示尝试建造工具")
#     lines.append("- noop 表示不执行操作")
#     lines.append("⚠️ 请确保返回的是有效 JSON 对象，不能包含其他文字或解释。")
#
#     return "\n".join(lines)
#
#
# # 计算 agent 当前朝目标前进一步的动作（方向）
# def navigate_one_step(current_pos, target_pos):
#     dx = target_pos[0] - current_pos[0]
#     dy = target_pos[1] - current_pos[1]
#     if abs(dx) > abs(dy):
#         return 2 if dx > 0 else 3  # down / up
#     elif dy != 0:
#         return 0 if dy > 0 else 1  # right / left
#     return None  # already at target
#
#
# # 解释 LLM 输出并将其转化为 agent 控制行为（目标规划）
# def interpret_llm_json(json_obj, env):
#     move_plan = {}  # agent -> action_id
#     for agent, cmd in json_obj.items():
#         if cmd["action"] == "noop":
#             continue
#         elif cmd["action"] == "build":
#             tool = cmd["tool"]
#             success = env.build_tool(agent, tool)
#             print(f"{agent} 建造 {tool} -> {'成功' if success else '失败'}")
#         elif cmd["action"] == "move_to":
#             target = cmd["target"]
#             if target == "warehouse":
#                 target_pos = env.warehouse_position
#             elif target == "exit":
#                 target_pos = env.exit_position
#             else:
#                 target_pos = target
#             current_pos = env.agent_positions[agent]
#             action = navigate_one_step(current_pos, target_pos)
#             if action is not None:
#                 move_plan[agent] = action
#     return move_plan



