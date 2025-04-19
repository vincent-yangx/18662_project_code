import pygame
import sys
import time
import self_env
import gym
import numpy as np
import random
import json
from self_env import MultiAgentResourceEnv
from pydantic import BaseModel
from openai import OpenAI
from typing import Literal, Optional, List, Dict
import re
import datetime

# ----------- 环境初始化 -----------
pygame.init()
env = MultiAgentResourceEnv()

agent_list = ["agent_1", "agent_2", "agent_3", "agent_4"]
res_order = ["wood", "stone", "iron", "coal","diamond"]

# ----------- JSON 模型结构 -----------
class AgentAction(BaseModel):
    agent_id: Literal["agent_1", "agent_2", "agent_3", "agent_4"]
    action: Literal["move", "collect", "create", "wait"]
    target_pos: Optional[List[int]] = None
    target_resource: Optional[Literal["wood", "stone", "coal", "iron", "diamond"]] = None
    target_tool: Optional[Literal["table", "wood pickaxe", "stone pickaxe", "furnace", "iron pickaxe"]] = None
    reason: str

class AgentPlan(BaseModel):
    actions: List[AgentAction]

# ----------- 工具构造函数 -----------
def build_tool(env, agent, tool_name):
    if env.unwrapped.build_tool(agent, tool_name):
        print(f"✅ {agent} 成功制造了 {tool_name}！")
    else:
        print(f"❌ {agent} 无法制造 {tool_name}。")

# ----------- Prompt 构造函数 -----------
def build_full_llm_prompt(env):
    def manhattan_dist(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    required_tools = {
        "wood": set(),
        "stone": {"wood pickaxe"},
        "coal": {"wood pickaxe"},
        "iron": {"stone pickaxe"},
        "diamond": {"iron pickaxe"}
    }

    def is_resource_collectible(resource):
        return all(env.tools_built.get(t, False) for t in required_tools.get(resource, []))

    lines = []
    lines.append("🎯 最终目标：采集到 diamond 资源。")
    lines.append("💡 注意：agent 走到 diamond 并成功采集即视为完成任务。")

    lines.append("\n=== 🤖 Agent 状态 ===")
    for agent in env.agents:
        pos = env.agent_positions[agent]
        resource_here = "无"
        for res, pos_list in env.resources.items():
            for i, p in enumerate(pos_list):
                if not env.collected_flags[res][i] and tuple(p) == tuple(pos):
                    resource_here = res
                    break
            if resource_here != "无":
                break
        lines.append(f"{agent} 在位置 {list(pos)}，当前格子资源：{resource_here}")

    lines.append("\n=== 地图资源位置 ===")
    for res_name, pos_list in env.resources.items():
        for i, pos in enumerate(pos_list):
            if not env.collected_flags[res_name][i]:
                lines.append(f"{res_name} at {list(pos)}")

    lines.append("\n=== 工具状态 ===")
    for tool, built in env.tools_built.items():
        status = "✅ 已建造" if built else "❌ 未建造"
        lines.append(f"{tool}: {status}")

    lines.append("\n请根据当前状态生成以下 JSON 格式的动作建议：")
    lines.append("""{
  "actions": [
    {
      "agent_id": "agent_1",
      "action": "move",
      "target_pos": [2, 3],
      "reason": "去 wood"
    },
    {
      "agent_id": "agent_2",
      "action": "collect",
      "target_resource": "wood",
      "reason": "当前位置是 wood，执行采集"
    },
    {
      "agent_id": "agent_3",
      "action": "create",
      "target_tool": "wood pickaxe",
      "reason": "已收集足够资源，制造 wood pickaxe"
    }
  ]
}""")

    lines.append("\n=== 🧠 重要规划提示 ===")
    lines.append("⚠️ 请注意：若目标资源无法直接采集，应优先完成所需工具的建造任务。")
    lines.append("✅ 规划顺序建议如下：")
    lines.append("1. 判断采集目标（如 iron、diamond）是否需要工具")
    lines.append("2. 若需要工具，则先采集工具所需的资源（如 wood、stone、coal）")
    lines.append("3. 进行工具的建造")
    lines.append("4. 所有 agent 协作，分阶段完成工具链构建")
    lines.append("🎯 示例（采集 diamond 的规划）:")
    lines.append("- 若缺 iron pickaxe，则先造 iron pickaxe")
    lines.append("- iron pickaxe 需要 furnace → 造 furnace")
    lines.append("- stone pickaxe 需要 wood和stone → 采集wood 和 stone")
    lines.append("- furnace 需要 stone → 先采 stone")
    lines.append("- stone 需要 wood pickaxe → 先造 wood pickaxe")
    lines.append("- wood pickaxe 需要 wood + table → 先采 wood + 造 table")

    lines.append("\n=== ⚠️ 注意事项和行为约束 ===")
    lines.append("🔒 若工具尚未建造，请不要采集需要该工具的资源。")
    lines.append("✅ 可采资源 = tools_built 所支持的资源，请依据当前工具状态判断。")
    lines.append("❌ 禁止 agent 执行 move 到当前位置 的动作。")
    lines.append("❌ 禁止前往墙体、障碍物或其他 agent 当前位置。")
    lines.append("🤖 若目标资源不可采，应先制造所需工具，再采资源。")
    lines.append("👥 agent 应合理分工协作，避免重复走位或重复任务。")

    lines.append("🛠️ 注意：建造工具请使用 'create' 作为动作类型，而非 'build'")
    lines.append("例如：{\"agent_id\": \"agent_1\", \"action\": \"create\", \"target_tool\": \"table\"}")

    lines.append("\n=== 🧠 智能建造建议 ===")
    lines.append("🛠️ 请根据系统提供的 tool_recommendation，判断当前是否具备建造条件。")
    lines.append("✅ 如果 status 是 ready，则安排 agent 去建造 next_tool。")
    lines.append("⛔ 如果 status 是 not_ready，则安排 agent 去采集 missing 中列出的资源。")
    lines.append("🛑 请不要在资源已充足时继续采集。")

    lines.append("📦 工具建造状态由 tools_built 指定。")
    lines.append("✅ 若 tools_built[tool] = true，则说明该工具已经建造完成，无需再次建造。")
    lines.append("⛔ 请勿对已建造的工具再次执行 create 动作。")
    lines.append("🛑 例如：若 table 已建造完成，不应再次采集 wood 来建造 table，更不应再次执行 create table。")
    lines.append("⛔ 若某个工具已经建造完成，请不要再采集用于该工具的材料。")
    lines.append("✅ 所有工具的建造状态见 already_built_tool 字段。")

    lines.append("\n=== 🛠️ 工具建造配方规则 ===")
    for tool, req in env.tool_prerequisite.items():
        req_str = ", ".join(f"{k}: {v}" for k, v in req.items())
        lines.append(f"{tool} 需要：{req_str}")
    lines.append("✅ 如果所有材料都具备，请立即执行 create 动作建造该工具。")

    lines.append("\n=== 🔍 当前推荐建造工具状态 ===")
    lines.append("tool_recommendation 字段说明当前应建造哪个工具、是否缺材料：")
    lines.append("""格式如下：
    {
      "next_tool": "stone pickaxe",
      "missing": {},
      "status": "ready"
    }
""")

    lines.append("📌 若 status 为 'ready'，则说明所有资源已备齐，请立即执行 create 动作建造工具。")
    lines.append("⛔ 不要继续采集任何资源，也不要等待或移动，应优先建造 next_tool。")

    lines.append("❌ 不允许使用 'idle' 作为 action 类型，请使用 'wait' 表示等待。")

    return "\n".join(lines)


# ----------- GPT 调用 -----------
planner_history: list[Dict] = []


def get_next_tool_recommendation(env):
    tool_priority = ["table", "wood pickaxe", "stone pickaxe", "furnace", "iron pickaxe"]
    for tool in tool_priority:
        if not env.tools_built[tool]:
            needed = env.tool_prerequisite[tool]
            lacking = {}
            for res, amt in needed.items():
                total = env.shared_resource_pool.get(res, 0)
                if total < amt:
                    lacking[res] = amt - total
            return {
                "next_tool": tool,
                "missing": lacking,
                "status": "ready" if not lacking else "not_ready"
            }
    return {"next_tool": None, "missing": {}, "status": "done"}

def ask_gpt_to_plan(client, env, warehouse_summary):
    def to_python(obj):
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def extract_json(text):
        matches = re.findall(r'{.*}', text, flags=re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def get_collectible_resources(env):
        required_tools = {
            "wood": set(),
            "stone": {"wood pickaxe"},
            "coal": {"wood pickaxe"},
            "iron": {"stone pickaxe"},
            "diamond": {"iron pickaxe"}
        }
        return [res for res, tools in required_tools.items() if all(env.tools_built.get(t, False) for t in tools)]

    planner_input = {
        "map": env.get_grid_matrix().tolist(),
        "agents": {agent: to_python(env.agent_positions[agent]) for agent in env.agents},
        "resources_collected": to_python(warehouse_summary),
        "tools_built": to_python(env.tools_built),
        "shared_resource_pool": to_python(env.shared_resource_pool),
        "collectible_resources": get_collectible_resources(env),
        "tool_recommendation": get_next_tool_recommendation(env),
        "tool_prerequisite": to_python(env.tool_prerequisite),
        # "already_built_tools": [tool for tool, built in env.tools_built.items() if built],
    }

    system_prompt = build_full_llm_prompt(env)

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": json.dumps(planner_input)})

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages,
        temperature=0.2
    )

    content = response.choices[0].message.content
    print("\n🧠 GPT 回复原文:\n", content)

    # ✅ 保存到文件中
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"gpt_response_step_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write(content)
    parsed = extract_json(content)
    print("\n🧠 GPT 回复原文:\n", parsed)

    if parsed:
        try:
            planner_history.append({"role": "assistant", "content": json.dumps(parsed)})
            return AgentPlan.model_validate(parsed)
        except Exception as e:
            print(f"⚠️ JSON 验证失败: {e}")
            print("内容:", parsed)
            return AgentPlan(actions=[])
    else:
        print("⚠️ GPT 输出非合法 JSON：")
        print(content)
        return AgentPlan(actions=[])

# ----------- 主逻辑 -----------
def main():
    screen = pygame.display.set_mode((env.grid_size * 30, env.grid_size * 30))
    pygame.display.set_caption("资源收集游戏")
    clock = pygame.time.Clock()
    env.reset()
    env.render(screen)

    llm = OpenAI(api_key="your key here")  # 使用你已有的 key
    steps = 0
    done = False

    def get_collectible_resources(env):
        required_tools = {
            "wood": set(),
            "stone": {"wood pickaxe"},
            "coal": {"wood pickaxe"},
            "iron": {"stone pickaxe"},
            "diamond": {"iron pickaxe"}
        }
        return [res for res, tools in required_tools.items() if all(env.tools_built.get(t, False) for t in tools)]

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        warehouse_summary = env.print_collected_summary()
        tool_reco = get_next_tool_recommendation(env)

        # ✅ 如果资源已经齐全且未建造，立即安排建造（优先执行）
        if tool_reco["status"] == "ready" and not env.tools_built[tool_reco["next_tool"]]:
            builder = random.choice(env.agents)
            print(f"🛠️ 强制安排 {builder} 建造 {tool_reco['next_tool']}")
            plan = AgentPlan(actions=[
                AgentAction(
                    agent_id=builder,
                    action="create",
                    target_tool=tool_reco["next_tool"],
                    reason="资源已齐备，立即建造"
                )
            ])
        else:
            plan = ask_gpt_to_plan(llm, env, warehouse_summary)

        # ✅ 执行 GPT 返回的动作
        for action in plan.actions:
            agent_id = action.agent_id
            env.current_agent = agent_id

            if action.action == "move":
                current_pos = env.agent_positions[agent_id]
                if tuple(current_pos) == tuple(action.target_pos):
                    continue
                moves = env.get_shortest_path(current_pos, action.target_pos)
                if not moves:
                    continue
                for move in moves:
                    try:
                        state, reward, done, msg = env.step(move)
                        env.render(screen)
                    except Exception as e:
                        break

            elif action.action == "collect":
                if not action.target_resource:
                    continue
                if action.target_resource not in get_collectible_resources(env):
                    continue
                env.collect_resource(agent_id, action.target_resource)

            elif action.action == "create":
                if env.tools_built.get(action.target_tool, False):
                    print(f"⚠️ {agent_id} 要建造 {action.target_tool}，但它已经建造，跳过")
                    continue
                build_tool(env, agent_id, action.target_tool)

        # ✅ 每隔 3 步由 agent_1 尝试建造推荐工具
        if steps % 3 == 0:
            auto_tool = get_next_tool_recommendation(env)
            tool = auto_tool["next_tool"]
            if tool and not env.tools_built[tool] and auto_tool["status"] == "ready":
                env.current_agent = "agent_1"
                print(f"🔁 agent_1 自动尝试建造 {tool}（每 3 步触发）")
                build_tool(env, "agent_1", tool)

        print(f"[DEBUG] 当前工具建造状态: {env.tools_built}")
        steps += 1
        clock.tick(5)



if __name__ == "__main__":
    main()
