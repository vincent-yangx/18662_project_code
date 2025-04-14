import gym
import self_env  # 确保环境注册

env = gym.make('CustomMultiAgentEnv-v0')
env.reset()
done = False

while not done:
    actions = {
        "agent_1": env.action_space.sample(),
        "agent_2": env.action_space.sample()
    }

    rewards = {}
    for agent, action in actions.items():
        pos, reward, agent_done, message = env.unwrapped.step(agent, action)
        rewards[agent] = reward
        done = done or agent_done

        if reward > 0:
            print(f"\n🧭 {agent} moved to position: {pos[agent]}")
            print(f"🎁 Reward: {reward}")
            print(f"🧺 Collected resources: {env.unwrapped.collected_resources}")
            if message:
                print(f"📣 {message}")

    env.render()
