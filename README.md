### Data Simulation
1. Simulation code of satellite-ground data and task dataset is in **data_simulation**
2. **getSatelliteData.py** is the entry of satellite-ground data simulation.
3. **getTaskData.py** is the entry of task data simulation.
###  MADDPG-based Algorithm of Multi-Task Decision for Satellite Handover (MTDSH)
1. The program entry of is **main.py**.
2. **reader.py:** reads data.json and task.json according to time, generates and returns json_read to Env in the format (num_agents *(3+3*(num_satellites+4)); In addition, it also includes max_K,num_agents,num_satellites and other data.
3. **env_new.py:** initializes the environment, performs step according to time, calls reader function in step to read json, after some data processing (incoming actions may not connect,done tasks can not continue to increase done_time, incoming actions lead to each satellite k change).
After the processing is completed, the state is concatenated with the last action recorded by Env to obtain (num_agents *(num_satellites+ 3+3*(num_satellites+4)))),done, etc.). Finally, the reward is calculated.
4. **main.py:** Implement MADDPG, set episode, length, eval, etc., and update the network for eval each time.
 
