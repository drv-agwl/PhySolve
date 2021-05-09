import phyre
import numpy as np
from DataCollection.data_collector import get_collision_timestep


def simulate_action(sim, task_idx, x, y, r):
    x = x * 256. / 255.
    y = y * 256. / 255.

    r = r.cpu().numpy()[0] * 256.
    r = (r - 2.) / 30.
    action = np.array([x, y, r])

    collided, solved = 0, 0

    try:
        res = sim.simulate_action(task_idx, action, need_featurized_objects=True, stride=1)
        if get_collision_timestep(res) != -1:
            collided = 1
        if res.status.is_solved():
            solved = 1
        return collided, solved
    except:
        return 0, 0
