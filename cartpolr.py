import gymnasium as gym
import time

# 创建环境（人类可视化）
env = gym.make("CartPole-v1", render_mode="human")
	# •	env.reset()：重置环境，得到初始状态
	# •	env.step(action)：执行动作，让小车和杆子运动
	# •	env.render()：渲染画面，看到真实的动画
	# •	env.close()：关闭环境

# 初始化环境
obs, info = env.reset(return_info=True) if "return_info" in env.reset.__code__.co_varnames else (env.reset(), {})
#初始化环境并且将info赋值给info（return_info=Ture是用于兼容化才写的）
#将速度、角速度、角度、位置赋值给env
#初始化完成之后他会自动执行：
#self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
#return self.state

reward_sum = 0
step_hand = 1
for step in range(100):
    env.render()
	# •	env.render()：渲染画面，看到真实的动画

    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()
        #从动作空间随机采样动作，'.action_space'是指env中的合法状态空间（0，1）
        #'.sample'是在合法状态空间中随机取值

        obs, reward, terminated, truncated, info = env.step(action)
        # 核心交互：
        # obs：下一时刻的环境状态
        # reward：当前的奖励（支撑住了给1分，倒下了给0分，直到倾斜太大或者是跑出去之前的得分总和为本次的得分）
        # terminated： 是否自然结束（True为自然结束，False为还未自然结束）
        # truncated： 是否被打断（Unterbrechung？，True为被打断，同理false）
        # info：当前信息。
        reward_sum += 1
        print(f"  Step_in: {step}")
        print(f"  Action: {action}")
        print(f"  Reward      : {reward}")
        print(f"  Observation : {obs}")
        # 让打印不太快（可调节速度）
        time.sleep(0.5)

    # 实时打印信息
    print(f"  Step_hand : {step_hand}")
    print(f"  Observation : {obs}")
    print(f"  Sum_Reward  : {reward_sum}")
    print("-" * 50)


    # 如果回合结束，重置环境
    if terminated or truncated:
        step_hand += 1
        obs, info = env.reset(return_info=True) if "return_info" in env.reset.__code__.co_varnames else (env.reset(), {})

env.close()
