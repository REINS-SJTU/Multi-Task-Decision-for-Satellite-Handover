import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import rl_utils

from env_new import CustomEnv


def make_env():
    num_agents = 150
    num_satellites = 29
    env = CustomEnv(num_agents, num_satellites)
    return env


# 令DDPG适用于离散动作
def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]], requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y


class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)

        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu()

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents = []
        for i in range(env.num_agents):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device)
            )
        self.gamma = gamma
        self.tau = tau
        # 均方误差
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        states = torch.chunk(states, env.num_agents, dim=0)
        actions = [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]
        combined_actions = torch.cat(actions, dim=0)
        return combined_actions

    def update(self, states, actions, rewards, next_states, dones, i_agent):
        # 转换为批量张量
        states = torch.stack(states, dim=0).to(self.device)  # shape: (batch_size, num_agents, obs_dim)
        actions = torch.stack(actions, dim=0).to(self.device)  # shape: (batch_size, num_agents, act_dim)
        rewards = torch.stack(rewards, dim=0).to(self.device)  # shape: (batch_size, num_agents)
        next_states = torch.stack(next_states, dim=0).to(self.device)  # shape: (batch_size, num_agents, obs_dim)
        dones = torch.stack(dones, dim=0).to(self.device)  # shape: (batch_size, num_agents)

        cur_agent = self.agents[i_agent]

        # 更新critic网络
        cur_agent.critic_optimizer.zero_grad()

        # 计算并拼接所有agent的target_actor
        all_target_act = [onehot_from_logits(pi(_next_obs)) for pi, _next_obs in
                          zip(self.target_policies, next_states.transpose(0, 1))]
        all_target_act = torch.stack(all_target_act, dim=1)  # shape: (batch_size, num_agents, act_dim)

        with torch.no_grad():
            # 计算critic输入
            flat_next_states = next_states.view(batch_size, -1)
            flat_all_target_act = all_target_act.view(batch_size, -1)
            critic_input = torch.cat((flat_next_states, flat_all_target_act), dim=1)  #状态和动作横着拼接
            critic_values = cur_agent.target_critic(critic_input).to(device)
            # print(critic_values) # shape: (batch, 1)

        compute_rewards = torch.tensor([
            rewards[i, i_agent] for i in range(rewards.shape[0])
        ]).view(-1, 1).to(self.device)
        compute_dones = torch.tensor([
            dones[i, i_agent] for i in range(dones.shape[0])
        ]).view(-1, 1).to(self.device)  # shape: (batch, 1)

        # 计算target_critic值
        target_critic_value = (
                compute_rewards +  # shape: (batch_size, 1)
                self.gamma * critic_values *  # shape: (batch_size, 1)
                (1 - compute_dones)  # shape: (batch_size, 1)
        )  # shape: (batch_size, 1)

        # 生成当前critic网络的输入
        flat_states = states.view(batch_size, -1)
        flat_actions = actions.view(batch_size, -1)
        critic_input_current = torch.cat((flat_states, flat_actions), dim=1)
        # shape: (batch_size, critic_input_dim)
        # 通过当前critic网络计算 Q 值
        critic_value = cur_agent.critic(critic_input_current)  # shape: (batch_size, 1)

        # 计算critic loss
        critic_loss = self.critic_criterion(critic_value, target_critic_value)
        critic_loss.backward()
        cur_agent.critic_optimizer.step()
        # 更新actor网络
        cur_agent.actor_optimizer.zero_grad()
        # 获取当前agent的动作
        cur_actor_out = cur_agent.actor(states[:, i_agent, :])  # shape: (batch_size, act_dim)
        cur_act = gumbel_softmax(cur_actor_out)  # shape: (batch_size, act_dim)

        # 获取其他agent的动作
        other_acts = []
        for i in range(env.num_agents):
            if i == i_agent:
                other_acts.append(cur_act)
            else:
                other_pi = self.policies[i]
                other_act_out = other_pi(states[:, i, :])  # shape: (batch_size, act_dim)
                other_act = onehot_from_logits(other_act_out)  # shape: (batch_size, act_dim)
                other_acts.append(other_act)

        # 拼接所有agent的动作
        all_actor_acs = torch.stack(other_acts, dim=1)  # shape: (batch_size, num_agents, act_dim)
        # 拼接当前states和所有agent的动作
        flat_states = states.view(batch_size, -1)  # shape: (batch_size, num_agents*obs_dim)
        flat_actions = all_actor_acs.view(batch_size, -1)  # shape: (batch_size, num_agents*act_dim)
        vf_in = torch.cat((flat_states, flat_actions), dim=1)  # shape: (batch_size, num_agents*(obs_dim+act_dim))

        q_values = cur_agent.critic(vf_in)  # shape: (batch_size, 1)
        # 计算actor的损失，更新actor网络
        actor_loss = -q_values.mean()
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

    # def save_models(self, save_dir, episode_number):
    #     save_dir = os.path.join(save_dir, f'episode_{episode_number}')
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     for i, agent in enumerate(self.agents):
    #         torch.save(agent.actor.state_dict(), os.path.join(save_dir, f'actor_{i}.pth'))
    #         torch.save(agent.critic.state_dict(), os.path.join(save_dir, f'critic_{i}.pth'))


if __name__ == "__main__":
    num_episodes = 500
    episode_length = 60  # 每条探索序列的最大长度
    buffer_size = 1000
    hidden_dim = 256
    actor_lr = 1e-2
    critic_lr = 1e-2
    gamma = 0.95
    tau = 1e-2
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_interval = 240
    minimal_size = 120

    env = make_env()
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    # 设置维度值
    state_dims = []
    action_dims = []
    for i in range(env.num_agents):
        state_dims.append(131)
        action_dims.append(3)  # 动作空间为3， 根据这三个值寻找每个agent可见的卫星进行连接
    critic_input_dim = sum(state_dims) + sum(action_dims)
    print(critic_input_dim)

    maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                    action_dims, critic_input_dim, gamma, tau)


    def evaluate(maddpg, n_episode=5, episode_length=60):
        # 对学习的策略进行评估,此时不会进行探索
        env = make_env()
        returns = np.zeros(env.num_agents)
        for _ in range(n_episode):
            obs = env.reset()
            for t_i in range(1, episode_length):
                actions = maddpg.take_action(obs, explore=False)
                obs, rew, done, truncated, info = env.step(actions.numpy(), time=t_i)
                rew = rew.cpu().numpy()
                returns += rew / episode_length
        return returns.tolist()


    return_list = []  # 记录每一轮的回报（return）
    total_step = 0
    for i_episode in range(0, num_episodes):
        print("i_episode:" + str(i_episode))
        state = env.reset()
        for e_i in range(episode_length):
            print("e_i:"+str(e_i))
            actions = maddpg.take_action(state, explore=True)
            next_state, reward, done, truncated, info = env.step(actions.numpy(), time=e_i)
            replay_buffer.add(state, actions, reward, next_state, done)
            state = next_state

            total_step += 1
            if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
                state, action, reward, next_state, done = replay_buffer.sample(batch_size)
                for a_i in range(env.num_agents):
                    maddpg.update(state, action, reward, next_state, done, a_i)
                maddpg.update_all_targets()
        # 貌似不需要eval环节
        # if (i_episode + 1) % 20 == 0:
        #     print("eval start:")
        #     with open('./rewards.txt', 'a') as f:
        #         f.write('Eval:\n')
        #     ep_returns = evaluate(maddpg)
        #     return_list.append(ep_returns)
        #     print(f"Episode: {i_episode + 1}, {ep_returns}")
            # maddpg.save_models('models', i_episode + 1)
