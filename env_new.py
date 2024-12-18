import sys

import gym
import numpy as np
import torch
from gym import spaces
from datetime import timedelta
from dateutil import parser
from collections import defaultdict

from reader import get_final_np, get_num_satellites, get_K_max, get_num_agents, add_task

sys.path.append("./")


class CustomEnv(gym.Env):
    def __init__(self, num_agents, num_satellites):
        super(CustomEnv, self).__init__()

        # N个用户， M个卫星
        self.num_agents = num_agents
        self.num_satellites = num_satellites

        # TODO：设置观察空间信息貌似没有什么用处，后续计算用不到。。。
        self.observation_space = spaces.Box(
            low=50, high=100, shape=(self.num_agents, self.num_satellites), dtype=np.float32
        )

        # 动作空间（N * 3）: 每个agent选择一个卫星进行连接，可见区间为3
        self.action_space = spaces.Box(
            low=0, high=0.5, shape=(self.num_agents, 3), dtype=np.float32
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_state = None  # 上一时刻的全局状态
        self.last_actions = None  # 上一时刻的连接动作
        self.state = None  # 全局状态
        self.json_read = None  # 读取的JSON数据
        self.actions = None  # 当前时刻的连接动作

        self.done_time = None  # 记录已完成的连接时间
        self.done = None  # 记录任务是否完成
        self.current_connections = None  # 当前时刻的连接状态,agent--->satellite
        self.connections = None  # 当前卫星的连接数量(k),satellite--->number
        self.K_max = None
        self.reward_file = None
        self.connection_file = None
        # new add
        self.task_id = -1  # 记录当前处理的最大task id(对应task.json中的id，从1开始)，下一个待获取的task是tasks[task_id]
        self.task_idx_np = None  # 记录每个agent正在处理的任务id,agent--->task，结合current_connections得到每个task对应的satellite
        self.already_finished = 0  # 已经完成的任务数量
        self.final_already_finished = 0
        self.agent_use_np = None  # 记录每个agent是否完成了全部task

    def reset(self):
        self.last_state = None
        self.last_actions = None
        self.current_connections = np.zeros(self.num_agents, dtype=float)
        self.connections = np.zeros(self.num_satellites, dtype=float)
        self.done_time = np.zeros(self.num_agents, dtype=float)
        self.done = np.zeros(self.num_agents, dtype=float)
        self.K_max = get_K_max(self.num_satellites)

        # 横向拼接两个np，初始化的连接动作为0
        self.actions = np.zeros((self.num_agents, self.num_satellites), dtype=float)
        # 初始调度150个任务
        self.task_id = 150
        self.task_idx_np = np.arange(1, 151)
        self.already_finished = 0
        self.final_already_finished = 0
        self.json_read = get_final_np("2024-07-19T00:00:00Z", self.num_satellites, self.task_idx_np)
        self.state = np.hstack((self.actions, self.json_read))

        self.reward_file = open('./rewards_3000.txt', 'a', buffering=1)
        self.connection_file = open('./connections_3000.txt', 'a', buffering=1)
        self.agent_use_np = np.zeros(self.num_agents)

        return self.get_state()

    def step(self, actions, **kwargs):
        time = kwargs.get("time", 0)
        # 处理并记录时间
        start_time_str = "2024-07-19T00:00:00Z"
        start_time = parser.isoparse(start_time_str)
        current_time = start_time + timedelta(seconds=time * 20)
        current_time_str = current_time.strftime("%Y-%m-%dT%H:%M:%S") + 'Z'
        next_time = start_time + timedelta(seconds=(time+1) * 20)
        next_time_str = next_time.strftime("%Y-%m-%dT%H:%M:%S") + 'Z'

        self.last_actions = self.actions
        self.last_state = self.state
        # 用于将3个维度的动作 转化为 对应卫星的独热编码
        temp_actions = np.zeros((self.num_agents, self.num_satellites), dtype=float)
        # 每次step时读取当前时隙的位置信息，需要得知当前的task数据，据此得到需要的对应的位置数据
        self.json_read = get_final_np(current_time_str, self.num_satellites, self.task_idx_np)

        # STEP1：输入动作处理
        # 修改json_read的内容，（改变done_time）
        # 新的值由当前输入的actions确定，如果该值不全为0，说明能够执行一个时隙，
        # 则new_done_time对应位置的值在原本的值上加1
        new_done_time = self.done_time
        for i in range(self.num_agents):
            if not np.all(actions[i] == 0) and new_done_time[i] < self.json_read[i, 1]:
                new_done_time[i] += 1
            if new_done_time[i] >= self.json_read[i, 1]:
                self.done[i] = 1  # 说明该任务已经完成
        # json_read的第三个元素位置为done_time
        self.json_read[:, 2] = new_done_time
        self.done_time = new_done_time

        # 处理输入的动作actions（改变k）
        new_connections = np.zeros(self.num_satellites, dtype=float)  # new_connections用于存储每个卫星的k
        for agent_id in range(self.num_agents):
            # 对于本次step不执行动作的agent，特殊处理
            if np.all(actions[agent_id] == 0) or self.agent_use_np[agent_id] == 1:
                temp_actions[agent_id] = 0
            else:
                # Actor返回的最佳的选择，未必是可见卫星，在可见卫星里面选值最大的
                satellite = self.select_satellite(agent_id, actions)
                if satellite == -1:
                    temp_actions[agent_id] = 0
                else:
                    temp_actions[agent_id][satellite] = 1
                    new_connections[satellite] += 1
                    self.current_connections[agent_id] = satellite
        self.actions = temp_actions

        # 根据new_connections 更新 json_read中的 k（3，13，23）(6个卫星的情况下)
        id_ranges = [(3, 3 + self.num_satellites), (3 + self.num_satellites + 4, 3 + 2 * self.num_satellites + 4),
                     (3 + 2 * self.num_satellites + 8, 3 + 3 * self.num_satellites + 8)]
        k_ranges = [3 + self.num_satellites + 2, 3 + 2 * self.num_satellites + 6, 3 + 3 * self.num_satellites + 10]
        for index, row in enumerate(self.json_read):
            for i, (start, end) in enumerate(id_ranges):
                one_hot = row[start:end]
                if np.all(one_hot == 0):
                    continue
                satellite_index = np.argmax(one_hot)
                connection = new_connections[satellite_index]
                self.json_read[index, k_ranges[i]] = connection
        self.state = np.hstack((self.last_actions, self.json_read))
        self.connections = new_connections

        # STEP2：reward等的计算
        alpha1 = 0.1
        alpha2 = 0.3
        alpha3 = 0.6
        r1 = 0.0
        r2 = 0.0
        r3 = 0.0
        # PART1: 延迟开销(by last_actions, actions ,state and satellites_attributes)
        for i, row in enumerate(self.state):
            # 当前无动作 或者 当前与上一次动作相同， 无需计算延迟
            if np.all(self.actions[i] == 0) or np.array_equal(self.actions[i], self.last_actions[i]):
                r1 += 0.0
            else:
                b = row[self.num_satellites]
                satellite_idx = np.argmax(self.actions[i])
                assert (np.sum(self.actions[i]) == 1)
                # self.state 的第16，26，36个元素为v，第10~15个元素类推为独热编码(在6个卫星的情况下)
                one_hot_1 = row[self.num_satellites + 3: 2 * self.num_satellites + 3]
                one_hot_2 = row[2 * self.num_satellites + 7: 3 * self.num_satellites + 7]
                one_hot_3 = row[3 * self.num_satellites + 11: 4 * self.num_satellites + 11]
                assert (np.sum(one_hot_1) == 0 or np.sum(one_hot_1) == 1)
                assert (np.sum(one_hot_2) == 0 or np.sum(one_hot_2) == 1)
                assert (np.sum(one_hot_3) == 0 or np.sum(one_hot_3) == 1)
                v = 1.0
                if np.argmax(one_hot_1) == satellite_idx:
                    v = row[2 * self.num_satellites + 3]
                elif np.argmax(one_hot_2) == satellite_idx:
                    v = row[3 * self.num_satellites + 7]
                elif np.argmax(one_hot_3) == satellite_idx:
                    v = row[4 * self.num_satellites + 11]
                r1 += b / v
        # PART2: 利用率
        for i, row in enumerate(self.state):
            # 只有采取了连接动作（非0），才需要计算
            if np.all(self.actions[i] == 0):
                r2 += 0.0
            else:
                u = row[self.num_satellites + 1]
                d = row[self.num_satellites + 2]
                satellite_idx = np.argmax(self.actions[i])
                one_hot_1 = row[self.num_satellites + 3:2 * self.num_satellites + 3]
                one_hot_2 = row[2 * self.num_satellites + 7:3 * self.num_satellites + 7]
                one_hot_3 = row[3 * self.num_satellites + 11:4 * self.num_satellites + 11]
                r = 1.0
                if np.argmax(one_hot_1) == satellite_idx:
                    r = row[2 * self.num_satellites + 4]
                elif np.argmax(one_hot_2) == satellite_idx:
                    r = row[3 * self.num_satellites + 8]
                elif np.argmax(one_hot_3) == satellite_idx:
                    r = row[4 * self.num_satellites + 12]
                r2 += abs((1 - (u - d) / r))
        # print(r2)
        # PART3: 负载均衡
        div = self.connections / self.K_max
        r3 = np.std(div)

        # print(self.connections)
        print("already: " + str(self.final_already_finished))
        # print(r3)

        rewards = alpha1 * r1 * 0.001 + alpha2 * r2 * 0.001 + alpha3 * r3
        rewards = 1 / (rewards + 1e-8)
        self.reward_file.write(f"{time}, {r1:.2f}, {r2:.2f}, {r3:.2f}, {rewards:.5f}\n")
        self.connection_file.write(f"{str(self.connections)}\n")

        # STEP3：替换已完成的任务，结合下一时隙拓扑信息，得到下一个State；
        # 根据done的情况，选择是否加入新的任务进行调度（先检查done,将这个done掉的任务替换为新的任务；
        # 修改done为0, done_time为0）
        for idx, item in enumerate(self.done):
            if item == 1.0:
                tmp = add_task(self.task_id, current_time_str, 1, self.num_satellites)
                if len(tmp) == 1:  # 能够增加新task
                    self.already_finished += 1
                    self.task_id += 1
                    self.task_idx_np[idx] = self.task_id
                    self.done[idx] = 0
                    self.done_time[idx] = 0
                    self.final_already_finished = self.already_finished
                else:
                    self.task_id = get_num_agents()
                    self.agent_use_np[idx] = 1  # 该agent已经闲置
                    self.final_already_finished = self.already_finished + np.sum(self.agent_use_np)

        self.last_actions = self.actions
        # 读取下一个时隙的拓扑信息
        self.json_read = get_final_np(next_time_str, self.num_satellites, self.task_idx_np)
        # 更新k
        new_connections = np.zeros(self.num_satellites, dtype=float)  # new_connections用于存储每个卫星的k
        for agent_id in range(self.num_agents):
            # 对于本次step不执行动作的agent，特殊处理
            if np.all(actions[agent_id] == 0):
                pass
            else:
                # Actor返回的最佳的选择，未必是可见卫星，在可见卫星里面选值最大的
                satellite = self.select_satellite(agent_id, actions)
                if satellite == -1:
                    pass
                else:
                    new_connections[satellite] += 1
                    self.current_connections[agent_id] = satellite
        # 根据new_connections 更新 json_read中的 k
        for index, row in enumerate(self.json_read):
            for i, (start, end) in enumerate(id_ranges):
                one_hot = row[start:end]
                if np.all(one_hot == 0):
                    continue
                satellite_index = np.argmax(one_hot)
                connection = new_connections[satellite_index]
                self.json_read[index, k_ranges[i]] = connection
        # 更新self.state,得到next_state
        self.state = np.hstack((self.last_actions, self.json_read))

        # 处理返回值
        truncated = 0.0
        info = {}
        # 将返回值转换为tensor格式，并设置元素数量(暂未确定reward函数的设计，先按此模拟)
        rewards = torch.full((self.num_agents,), rewards, dtype=torch.float32).to(self.device)
        terminated = torch.tensor(self.done, dtype=torch.float32).to(self.device)
        truncated = torch.full((self.num_agents,), truncated, dtype=torch.float32).to(self.device)
        info = {key: torch.full((self.num_agents,), value, dtype=torch.float32) for key, value in info.items()}
        return self.get_state(), rewards, terminated, truncated, info

    def render(self):
        print("Satellite connections:", self.connections)

    def close(self):
        pass

    # 返回从0开始的卫星索引
    def select_satellite(self, agent_id, actions):
        if np.all(actions[agent_id] == 0):
            raise ValueError(f"find all 0 action")
        visible_satellites = []
        for i in range(3):
            # 选择卫星根据的是当前time可连接的卫星
            start_idx = 3 + i * (self.num_satellites + 4)
            end_idx = start_idx + self.num_satellites
            one_hot = self.json_read[agent_id, start_idx:end_idx]
            # 检查独热编码是否为独热或者全0
            if not self.is_one_hot(one_hot):
                raise ValueError(f"One-hot encoding is not valid for agent {agent_id}, satellite group {i}: {one_hot}")
            if np.any(one_hot):
                visible_satellites.append(np.argmax(one_hot))
        if not visible_satellites:
            return -1
        # new add
        selected_satellites = np.argmax(actions[agent_id])
        if selected_satellites >= len(visible_satellites):
            return -1
        else:
            return visible_satellites[selected_satellites]

    def is_one_hot(self, one_hot):
        return np.sum(one_hot) == 1 or np.sum(one_hot) == 0

    def get_num_by_onehot(self, onehot):
        if np.all(onehot == 0):
            return -1
        return np.argmax(onehot)

    def generate_onehot_actions(self, actions):
        for i in range(actions.shape[0]):
            row = actions[i]
            if np.all(row == 0):
                continue
            if np.sum(row) == 1 and np.any(row == 1):
                continue
            max_index = np.argmax(row)
            row[:] = 0
            row[max_index] = 1
        return actions

    # 返回Env的整体状态（tensor格式）
    def get_state(self):
        return torch.tensor(self.state, dtype=torch.float32).to(self.device)


if __name__ == "__main__":
    num_satellites = get_num_satellites()
    num_agents = 150
    env = CustomEnv(num_agents=num_agents, num_satellites=num_satellites)
    observations = env.reset()
    # print("Initial Observations:", observations)
    for step in range(0, 55):
        print("time: " + str(step))
        actions = env.action_space.sample()
        actions = env.generate_onehot_actions(actions)
        observations, reward, terminated, truncated, info = env.step(actions, time=step)
        env.render()
        print(f"Step {step}, Reward: {reward[0]}")
    env.close()
