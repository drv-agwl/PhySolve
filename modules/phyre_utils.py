import os

import phyre
import numpy as np
from DataCollection.data_collector import get_collision_timestep
import os.path as osp
import imageio


def simulate_action(sim, task_idx, task_id, x, y, r, num_attempts=10, save_rollouts_dir=None):
    x = x * 256. / 255.
    y = y * 256. / 255.

    r = r.cpu().numpy()[0] * 256.
    r = (r - 2.) / 30.
    action = np.array([x, y, r])

    collided, solved = 0, 0
    attempt = 0

    delta_generator = action_delta_generator(pure_noise=True)
    action_memory = [action]

    res_first_guess = None

    try:
        res = sim.simulate_action(task_idx, action, need_featurized_objects=True, stride=1)
        res_first_guess = res
        # if get_collision_timestep(res) != -1:
        #     collided = 1
        if res.status.is_solved():
            solved = 1

    except:
        pass

    while not solved and attempt < num_attempts:
        delta = delta_generator.__next__()
        new_action = np.clip(action + delta, 0., 1.)

        if similar_action_tried(new_action, action_memory):
            continue

        try:
            attempt += 1
            action_memory.append(new_action)
            res = sim.simulate_action(task_idx, new_action, need_featurized_objects=True, stride=1)

            # if get_collision_timestep(res) != -1:
            #     collided = 1
            if res.status.is_solved():
                solved = 1

                if save_rollouts_dir is not None:
                    save_rollout_as_gif(res, get_collision_timestep(res), save_rollouts_dir, task_id)

                return collided, solved
        except:
            continue

    if save_rollouts_dir is not None and res_first_guess is not None:
        try:
            save_rollout_as_gif(res_first_guess, get_collision_timestep(res_first_guess), save_rollouts_dir, task_id)
        except:
            pass
    return collided, solved


def save_rollout_as_gif(res, collision_timestep, save_dir, task_id):
    template = f"Task-{str(int(task_id.split(':')[0]))}"
    os.makedirs(osp.join(save_dir, template), exist_ok=True)
    start_sleep = 50

    if collision_timestep != -1:
        rollout = np.repeat(res.images[collision_timestep][None], start_sleep, axis=0)
        rollout = np.concatenate([rollout, res.images], axis=0)
        rollout = np.concatenate([phyre.observations_to_uint8_rgb(x)[None] for x in rollout])

    else:
        rollout = np.concatenate([phyre.observations_to_uint8_rgb(x)[None] for x in res.images])

    imageio.mimsave(osp.join(save_dir, template, task_id + ".gif"), rollout, fps=25)


def similar_action_tried(action, action_memory):
    for other in action_memory:
        if np.linalg.norm(action - other) < 0.02:
            print("similar action already tried", action, other, end="\r")
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
