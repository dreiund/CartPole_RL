import gymnasium as gym
from stable_baselines3 import PPO

# 1. 创建环境
# render_mode="human" 告诉环境我们需要看到图形化界面
env = gym.make("CartPole-v1", render_mode="human")

# 2. 加载刚才训练好的模型
# 注意：文件名要和 train.py 里保存的一致
model = PPO.load("ppo_cartpole")

# 3. 初始化环境
obs, info = env.reset()

print("正在演示... 请看弹出的窗口。")
print("按 Ctrl+C 在终端结束程序")

try:
    while True:
        # 4. 模型预测
        # deterministic=True 表示让 AI 拿出最稳的策略，不要瞎蒙
        action, _states = model.predict(obs, deterministic=True)
        
        # 5. 环境执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 6. 如果杆子倒了或时间到了，重置环境
        if terminated or truncated:
            obs, info = env.reset()
            
except KeyboardInterrupt:
    print("演示结束")
    env.close()
