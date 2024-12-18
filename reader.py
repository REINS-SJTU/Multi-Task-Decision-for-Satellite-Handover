import json
import numpy as np
from datetime import timedelta
from dateutil import parser
from itertools import chain


# 根据卫星的标号生成独热编码
def get_onehot(satellite_number, max_id):
    onehot_vector = [0] * max_id
    if 0 <= satellite_number < max_id:
        onehot_vector[satellite_number] = 1
    return onehot_vector


def get_num_satellites():
    with open('./data/data.json', 'r') as f:
        data = json.load(f)
    return data['max_sa_id']


def get_num_agents():
    with open('./data/task_3000.json', 'r') as f:
        data = json.load(f)
    return len(data)


# 更新id_list 中对应id在 round时隙的位置信息
def get_final_np(current_time_str, max_id, id_list):
    result_array = []
    result = []

    with open('./data/data.json', 'r') as f:
        data = json.load(f)['data']
    with open('./data/task_3000.json', 'r') as f:
        tasks = json.load(f)

    for id in id_list:
        task = tasks[id - 1]
        task_pos_id = task['pos_id']
        # 在data.json中找到对应时间点的位置信息
        for data_entry in data:
            if data_entry['time'] == current_time_str:
                for position in data_entry['positions']:
                    if position['pos_id'] == task_pos_id:
                        # 获取前3个可见卫星的数据
                        satellites = position['satellites'][:3]
                        satellite_data = [
                            {
                                'id': get_onehot(sat['id'], max_id),
                                'altitude': sat['altitude'],
                                'transmission_rate': sat['transmission_rate'],
                                'actual_k': sat['actual_k'],
                                'max_K': sat['max_K'],
                                'remain_time': sat['remain_time']
                            }
                            for sat in satellites
                        ]
                        # 不足3个，填充0
                        while len(satellite_data) < 3:
                            satellite_data.append({
                                'id': [0] * max_id,
                                'altitude': 0,
                                'transmission_rate': 0,
                                'actual_k': 0,
                                'max_K': 0,
                                'remain_time': 0
                            })
                        task_info = {
                            'up_data_size': task['up_data_size'],  # 待上传的任务量
                            'duration_time': task['duration_time'],  # 需要的累积连接时间
                            'done_time': task['done_time']  # 已完成的连接时间
                        }
                        combined_data = {**task_info, **{'satellites': satellite_data}}
                        result.append(combined_data)
                        break
    for entry in result:
        task_info = entry['up_data_size'], entry['duration_time'], entry['done_time']
        satellite_data = [
            chain(sat['id'],
                  [sat['transmission_rate'], sat['remain_time'], sat['actual_k'], sat['max_K']])
            for sat in entry['satellites']
        ]
        combined_data = task_info + tuple(chain(*satellite_data))
        result_array.append(combined_data)
    result_array = np.array(result_array)
    return result_array
    # # 遍历每个时间点的任务
    # for task in tasks:
    #     task_pos_id = task['pos_id']
    #     # 在data.json中找到对应时间点的位置信息
    #     for data_entry in data:
    #         if data_entry['time'] == current_time_str:
    #             for position in data_entry['positions']:
    #                 if position['pos_id'] == task_pos_id:
    #                     # 获取前3个可见卫星的数据
    #                     satellites = position['satellites'][:3]
    #                     satellite_data = [
    #                         {
    #                             'id': get_onehot(sat['id'], max_id),
    #                             'altitude': sat['altitude'],
    #                             'transmission_rate': sat['transmission_rate'],
    #                             'actual_k': sat['actual_k'],
    #                             'max_K': sat['max_K'],
    #                             'remain_time': sat['remain_time']
    #                         }
    #                         for sat in satellites
    #                     ]
    #                     # 不足3个，填充0
    #                     while len(satellite_data) < 3:
    #                         satellite_data.append({
    #                             'id': [0] * max_id,
    #                             'altitude': 0,
    #                             'transmission_rate': 0,
    #                             'actual_k': 0,
    #                             'max_K': 0,
    #                             'remain_time': 0
    #                         })
    #                     task_info = {
    #                         'up_data_size': task['up_data_size'],  # 待上传的任务量
    #                         'duration_time': task['duration_time'],  # 需要的累积连接时间
    #                         'done_time': task['done_time']  # 已完成的连接时间
    #                     }
    #                     combined_data = {**task_info, **{'satellites': satellite_data}}
    #                     result.append(combined_data)
    #                     break
    # for entry in result:
    #     task_info = entry['up_data_size'], entry['duration_time'], entry['done_time']
    #     satellite_data = [
    #         chain(sat['id'],
    #               [sat['transmission_rate'], sat['remain_time'], sat['actual_k'], sat['max_K']])
    #         for sat in entry['satellites']
    #     ]
    #     combined_data = task_info + tuple(chain(*satellite_data))
    #     result_array.append(combined_data)
    # result_array = np.array(result_array)
    # return result_array


def get_K_max(max_id):
    with open('./data/data.json', 'r') as file:
        data = json.load(file)
    max_K_array = np.zeros(max_id, dtype=int)
    for i in range(max_id):
        max_K_array[i] = data['max_K']
    return max_K_array


# TODO: 根据当前id获取下一个用户的信息，返回一个np数组
# 参数：起始任务id， 当前时间， 需要添加的任务长度, 卫星数量
def add_task(task_id, cur_time_str, length, max_id):
    with open('./data/task_3000.json', 'r') as file:
        tasks = json.load(file)
    if task_id >= len(tasks):
        # print("已经无剩余任务待调度！")
        return np.array([])
    with open('./data/data.json', 'r') as file:
        data = json.load(file)['data']

    result = []
    start_idx = task_id
    end_idx = min(task_id + length, len(tasks))
    return np.zeros(1)
    # TODO: 这个add_task函数貌似只需要判断能否继续加task就行了，不需要具体的res
    pass

    # for i in range(start_idx, end_idx):
    #     task = tasks[i]
    #     task_pos_id = task['pos_id']
    #     for data_entry in data:
    #         if data_entry['time'] == cur_time_str:
    #             for position in data_entry['positions']:
    #                 if position['pos_id'] == task_pos_id:
    #                     # 获取前3个可见卫星的数据
    #                     satellites = position['satellites'][:3]
    #                     satellite_data = [
    #                         {
    #                             'id': get_onehot(sat['id'], max_id),
    #                             'altitude': sat['altitude'],
    #                             'transmission_rate': sat['transmission_rate'],
    #                             'actual_k': sat['actual_k'],
    #                             'max_K': sat['max_K'],
    #                             'remain_time': sat['remain_time']
    #                         }
    #                         for sat in satellites
    #                     ]
    #                     # 不足3个，填充0
    #                     while len(satellite_data) < 3:
    #                         satellite_data.append({
    #                             'id': [0] * max_id,
    #                             'altitude': 0,
    #                             'transmission_rate': 0,
    #                             'actual_k': 0,
    #                             'max_K': 0,
    #                             'remain_time': 0
    #                         })
    #                     task_info = {
    #                         'up_data_size': task['up_data_size'],  # 待上传的任务量
    #                         'duration_time': task['duration_time'],  # 需要的累积连接时间
    #                         'done_time': task['done_time']  # 已完成的连接时间
    #                     }
    #                     combined_data = {**task_info, **{'satellites': satellite_data}}
    #                     result.append(combined_data)
    #                     break
    # result_array = []
    # for entry in result:
    #     task_info = entry['up_data_size'], entry['duration_time'], entry['done_time']
    #     satellite_data = [
    #         chain(sat['id'],
    #               [sat['transmission_rate'], sat['remain_time'], sat['actual_k'], sat['max_K']])
    #         for sat in entry['satellites']
    #     ]
    #     combined_data = task_info + tuple(chain(*satellite_data))
    #     result_array.append(combined_data)
    # result_array = np.array(result_array)
    # return result_array


if __name__ == '__main__':
    # id_list = np.arange(1, 151)
    # res = get_final_np(0, 29, id_list)
    # print(res.shape)
    #
    # res = add_task(149, "2024-07-19T00:00:00Z", 4, 29)
    # print(res.shape)
    from collections import defaultdict
    dict = defaultdict(list)
    dict[0].append(1)
    dict[0].append(2)
    dict[1].append(3)
    print(dict)

