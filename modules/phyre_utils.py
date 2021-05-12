import phyre
import numpy as np
from DataCollection.data_collector import get_collision_timestep


def simulate_action(sim, task_idx, x, y, r, num_attempts=10):
    x = x * 256. / 255.
    y = y * 256. / 255.

    r = r.cpu().numpy()[0] * 256.
    r = (r - 2.) / 30.
    action = np.array([x, y, r])

    collided, solved = 0, 0
    attempt = 0

    delta_generator = action_delta_generator(pure_noise=True)
    action_memory = [action]

    try:
        res = sim.simulate_action(task_idx, action, need_featurized_objects=True, stride=1)
        if get_collision_timestep(res) != -1:
            collided = 1
        if res.status.is_solved():
            solved = 1

    except:
        pass

    while not solved or attempt < num_attempts:
        delta = delta_generator.__next__()
        new_action = np.clip(action + delta, 0., 1.)

        if new_action in action_memory:
            continue

        try:
            res = sim.simulate_action(task_idx, new_action, need_featurized_objects=True, stride=1)
            attempt += 1
            action_memory.append(new_action)

            if get_collision_timestep(res) != -1:
                collided = 1
            if res.status.is_solved():
                solved = 1
            return collided, solved
        except:
            return 0, 0


def similar_action_tried(self, action, tries):
    for other in tries:
        if (np.linalg.norm(action - other) < 0.02):
            print("similiar action already tried", action, other, end="\r")
            return True
    return False


def action_delta_generator(pure_noise=False):
    temp = 1
    radfac = 0.025
    coordfac = 0.1

    # for x,y,r in zip([0.05,-0.05,0.1,-0.1],[0,0,0,0],[-0.1,-0.2,-0.3,0]):
    # yield x,y,r

    if not pure_noise:
        for fac in [0.5, 1, 2]:
            for rad in [0, 1, -1]:
                for xd, yd in [(1, 0), (-1, 0), (2, 0), (-2, 0), (-1, 2), (1, 2), (-1, -2), (-1, -2)]:
                    # print((fac*np.array((coordfac*xd, coordfac*yd, rad*radfac))))
                    yield (fac * np.array((coordfac * xd, coordfac * yd, rad * radfac)))
    count = 0
    while True:
        count += 1
        action = ((np.random.randn(3)) * np.array([0.2, 0.1, 0.2]) * temp) * 0.1
        # print(count,"th", "ACTION:", action)
        if np.linalg.norm(action) < 0.05:
            continue
        yield action
        temp = 1.04 * temp if temp < 5 else temp
