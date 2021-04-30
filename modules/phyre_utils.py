import phyre
import numpy as np


def simulate_action(task_id, x, y, r):
    sim = phyre.initialize_simulator(task_id, 'ball')
    action = np.array([x, y, r])

    res = sim.simulate_action(task_id, action, need_featurized_objects=True, stride=1)

    return res.status.is_solved()