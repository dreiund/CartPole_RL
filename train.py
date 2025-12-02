import gymnasium as gym#世界库
from stable_baselines3 import PPO#引入ppo算法
import os

# 1. 创建环境
# CartPole-v1 是经典的倒立摆任务
env = gym.make("CartPole-v1")
#基于公式的仿真，输出环境ID为："CartPole-v1"的物理环境
#包括：[小车位置, 小车速度, 杆子角度, 杆顶速度]

# 2. 定义模型
# 我们使用 PPO 算法，MlpPolicy 表示使用全连接神经网络
# device="cpu"：对于这种极小的模型，M4 Max 的 CPU 处理速度极快，无需调用 GPU
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
#多层感知机，与环境交互（环境输出：小车位置、小车速度、杆子角度、杆顶速度），输出日志，使用cpu

print("---------- 🚀 开始训练 ----------")

# 3. 开始训练
# total_timesteps=10000：让 AI 尝试玩 20,000 步
# 在 M4 Max 上，这应该瞬间完成
model.learn(total_timesteps=20000)
# ---------------------------------------------------
# 这就是 model.learn() 内部偷偷做的事情 (简化版)
# ---------------------------------------------------

# def learn(self, total_timesteps):
#     # 1. 刚开始训练，先重置环境，拿到第一帧画面
#     obs = self.env.reset()  # <--- 这里！这里执行了第一次 reset
    
#     current_step = 0
    
#     while current_step < total_timesteps:
#         # 2. AI 决定动作
#         action = self.predict(obs)
#           
        
#         # 3. 环境执行动作
#         obs, reward, done, info = self.env.step(action)
        
#         # 4. 收集数据用于训练...
        
#         # 5. 【关键点】如果游戏结束了 (done is True)
#         if done:
#             # 自动帮你重置环境，开始下一局游戏！
#             # 你完全不用操心，它为了不中断训练，立刻开启新的一局
#             obs = self.env.reset()  # <--- 这里！这里执行了后续无数次 reset
            
#         current_step += 1

print("---------- ✅ 训练结束 ----------")

# 4. 保存模型
# 保存为 ppo_cartpole.zip
model.save("ppo_cartpole")
print("模型已保存！")
