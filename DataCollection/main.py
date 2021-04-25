from gSolver import path_cost
import numpy as np
import phyre
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

tasks = ['00002:026', '00002:044', '00002:052', '00002:066', '00002:079', '00003:001']
task = tasks[0]
num_iters = 10
num_update_steps = 20

sim = phyre.initialize_simulator([task], 'ball')
actions = sim.build_discrete_action_space(max_actions=num_iters)

for iter in range(num_iters):
    print("new random action loaded")
    base_action = actions[iter]
    costs = []

    base_cost, _, status = path_cost(task, sim, base_action)
    print("Base cost: ", base_cost)

    cost = base_cost
    while cost == base_cost or cost == 255.:
        grads = gen_random_grads(0, 0.2)
        new_action = np.clip(base_action + grads, 0.1, 0.9)
        cost, _, status = path_cost(task, sim, new_action)

    print("Action found, new cost = ", cost)
    action = new_action
    costs.append(cost)

    step = 0
    while step < num_update_steps:
        grads = gen_random_grads(0, 0.05)
        new_action = np.clip(action + grads, 0.1, 0.9)
        new_cost, _, status = path_cost(task, sim, new_action)
        # print(action, new_action, new_cost)

        if new_cost < cost:
            cost = new_cost
            action = new_action

        costs.append(new_cost)
        step += 1

    plt.plot(range(1, num_update_steps + 2), costs)
    plt.savefig(f'./tmp_2/00002/044/costs{iter + 1}.png')
    plt.cla()

    if status == phyre.simulation_cache.SOLVED:
        print(f"Solved in {iter * num_update_steps + step + 1} iterations")
        break

step = 0
actions = sim.build_discrete_action_space(max_actions=200)
action = random.choice(actions)
try:
    simulation = sim.simulate_action(0, action, need_featurized_objects=True, stride=5)
except:
    pass
while simulation.status != phyre.simulation_cache.SOLVED:
    action = random.choice(actions)
    try:
        simulation = sim.simulate_action(0, action, need_featurized_objects=True, stride=5)
    except:
        pass
    step += 1

print(step)
