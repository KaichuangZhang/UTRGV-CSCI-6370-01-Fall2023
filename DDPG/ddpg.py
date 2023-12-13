import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 加载 MovieLens 数据集
# 替换为你的数据集路径
movie_data = pd.read_csv('../dataset/ml-latest/ratings.csv')

# 数据预处理（根据你的数据集结构进行调整）
# 假设 movie_data 包含用户ID、电影ID和评分
user_ids = movie_data['userId'].values
movie_ids = movie_data['movieId'].values
ratings = movie_data['rating'].values

# 将用户ID和电影ID转换为连续的索引
user_ids = pd.factorize(user_ids)[0]
movie_ids = pd.factorize(movie_ids)[0]

# 归一化评分
scaler = MinMaxScaler()
ratings = scaler.fit_transform(ratings.reshape(-1, 1)).flatten()

# 分割数据集
train_data, test_data, train_ratings, test_ratings = train_test_split(
    np.stack([user_ids, movie_ids], axis=1), ratings, test_size=0.2, random_state=42)

# 定义 Actor 和 Critic 模型
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # 使用 Sigmoid 以保证输出在 0 到 1 之间
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

# 初始化模型
actor = Actor(input_size=2, hidden_size=128, output_size=1)
critic = Critic(input_size=3, hidden_size=128)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# 训练函数
def train(actor, critic, actor_optimizer, critic_optimizer, train_data, train_ratings, epochs=10):
    for epoch in range(epochs):
        for i in range(len(train_data)):
            state = torch.FloatTensor(train_data[i])
            reward = torch.FloatTensor([train_ratings[i]])
            print (f'state {state}, reward {reward}')

            # Actor 选择动作
            action = actor(state)

            # Critic 评估动作
            q_value = critic(torch.cat([state, action], dim=0))
            #q_value = critic(state, action)

            # 计算损失
            actor_loss = -critic(torch.cat([state, actor(state)], dim=0)).mean()
            critic_loss = nn.MSELoss()(q_value, reward)

            # 更新 Actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # 更新 Critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
        

# 训练模型
train(actor, critic, actor_optimizer, critic_optimizer, train_data, train_ratings)
