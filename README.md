# 0. 离散动作的PPO算法原理

PPO算法是基于激进派和保守派两个RL算法发展出来的算法，他即保证每一步更新都会让神经网络有“放手一搏”的权利，又保留了自己对于神经网络的控制权（电子围栏clipping），防止他在更新过程中出现“经验主义”的问题。

# 角色：
PPO算法中存在两种角色：actor、critic、state（env）、collector；

# actor
actor：在离散动作中，在每一步（step）的每个动作都有一个发生概率，actor就是根据这些发生概率取出最高的那个进行动作的。actor的每一步动作都会产生一个新的状态state，比如：在CartPole模型中，如果你有60%的概率向左推小车，40%的概率向右推小车，那么actor在下一个step就会采用向左推的决策，于是小车就产生了新的state（位置，速度，角度，角速度）。
# critic
critic：在PPO算法中critic充当了一个“严格的老师”的角色，他会对每一步收集来的actor的动作进行评价（predict value）。

# state：
state（env）：表示env环境根据actor的动作或者决策输出的状态，对于CartPole这个项目来说，变化的状态有（位置，速度，角度，角速度），state是actor和critic之间的通讯方式。

# collector：
用于收集各个状态的参数。

# 1. PPO算法流程

当你激活PPO算法之后，开始进入PPO流程
```python
# just an example 😊
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=20000)
```
 1.1 内部代码：
```python
 def learn(self, total_timesteps):
     # 1. 刚开始训练，先重置环境，拿到第一帧画面
     obs = self.env.reset()  # <--- 这里！这里执行了第一次 reset
    
     current_step = 0
    
     while current_step < total_timesteps:
         # 2. AI 决定动作
         action = self.predict(obs)
           
        
         # 3. 环境执行动作
         obs, reward, done, info = self.env.step(action)
        
         # 4. 收集数据用于训练...
        
         # 5. 【关键点】如果游戏结束了 (done is True)
         if done:
             # 自动帮你重置环境，开始下一局游戏！
             # 你完全不用操心，它为了不中断训练，立刻开启新的一局
             obs = self.env.reset()  # <--- 这里！这里执行了后续无数次 reset
            
         current_step += 1
```
# 1.2 Step1 重置虚拟环境，定义循环步骤（从零开始）
# 1.3 Step2 进入巨大循环，开始逐步更新actor和critic
actor和critic是协同步进更新的
1-首先actor会自己进行2048步[GAE (Generalized Advantage Estimation)]
2-收集器会收集每一步的数据（obs，reward，action，current_critic，log_pro）
3-计算各项指标：
    Advantage = Taget - current_critic
    Target = reward_sum_gamma + next_critic
4-切片：一般会把2048个数据在经过上面的计算之后，将计算结果都给数据本身（作为标签），然后将整个2048个数据打乱顺序并且分成32份（每一份64个数据），然后每一份数据都进行10次训练（迭代完成）。
5-训练：训练的过程就是通过方向传播来更新神经网络的过程。actor和critic两部分分别有一个神经网络。

        actor更新方式：
            1、Avdantage的结果与0做比较；在这个state情况下，Advantage大于0需要提高概率，小于0需要减小这个动作概率
            2、概率是由actor的神经网络计算出来的（给定state输出两个动作的概率）
            3、根据Advantage与0的关系对actor神经网络进行反向传播得到新的概率
            4、然后再进行下一次更新。。。直到10次完成
        
        critic更新方式：
            1、根据当下的state给出一个当前步的critic分数（critic_current）
            2、根据下一步的state给出下一步的critic分数（只看这一步和下一步：critic_next_step）
            3、根据公式计算出Target = reward + critic_next_step
                          Advantage = Target - critic_current
            4、根据Target对critic神经网络进行反向传播，更新critic的值
            5、十次迭代
6-目的是得到reward总和最大！

# 2. 输出模型
