import os

import phyre
import numpy as np
from DataCollection.data_collector import get_collision_timestep, get_obj_channels
import os.path as osp
import imageio
import random
from PIL import Image, ImageDraw, ImageFont

cache = phyre.get_default_100k_cache('ball')
cache_actions = cache.action_array


def simulate_action(sim, task_idx, task_id, x, y, r, num_attempts=10, save_rollouts_dir=None, size=(64, 64)):
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
    imgs_lfm = np.zeros(size)

    try:
        res = sim.simulate_action(task_idx, action, need_featurized_objects=True, stride=1)
        res_first_guess = res
        imgs_lfm = np.max(get_obj_channels(res.images, size=size), axis=0)

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
            imgs_lfm = np.max(get_obj_channels(res.images, size=(64, 64)), axis=0)

            # if get_collision_timestep(res) != -1:
            #     collided = 1
            if res.status.is_solved():
                solved = 1

                if save_rollouts_dir is not None:
                    collision_scene = get_solving_collision_scene(sim, task_id, task_idx)
                    save_rollout_as_gif(res, collision_scene, save_rollouts_dir, "solved", task_id)

                return collided, solved, imgs_lfm
        except:
            continue

    if save_rollouts_dir is not None and res_first_guess is not None:
        try:
            collision_scene = get_solving_collision_scene(sim, task_id, task_idx)
            if solved:
                save_rollout_as_gif(res_first_guess, collision_scene, save_rollouts_dir, "solved", task_id)
            else:
                save_rollout_as_gif(res_first_guess, collision_scene, save_rollouts_dir, "unsolved", task_id)
        except:
            pass
    return collided, solved, imgs_lfm


def get_text_image(text, size=(256, 256, 3)):
    img = Image.new('RGB', (256, 256), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype('/home/dhruv/Desktop/PhySolve/arial.ttf', 25)
    d.text((30, 110), text, font=font, fill=(255, 0, 0), align="center")
    return np.array(img)


def get_solving_collision_scene(sim, task_id, task_idx):
    solved = 0
    cache_list = cache_actions[cache.load_simulation_states(task_id) == 1]
    res = None
    while not solved:
        actionlist = cache_list

        if len(actionlist) == 0:
            print("WARNING no solution action in cache at task", task_id)
            actionlist = [np.random.rand(3)]

        action = random.choice(actionlist)
        res = sim.simulate_action(task_idx, action,
                                  need_featurized_objects=True, stride=1)

        if (res.status.is_solved() == 1) and not res.status.is_invalid():
            solved = 1

    if solved and res:
        try:
            collision_timestep = get_collision_timestep(res)
            return res.images[collision_timestep]

        except:
            return np.zeros((256, 256, 3))

    return np.zeros((256, 256, 3))


def save_rollout_as_gif(res, collision_scene, save_dir, status, task_id):
    template = f"Task-{str(int(task_id.split(':')[0]))}"
    os.makedirs(osp.join(save_dir, template, status), exist_ok=True)
    start_sleep = 50

    text_solving = np.repeat(get_text_image("Simulator Solution")[None], start_sleep, axis=0)
    text_pred = np.repeat(get_text_image("Predicted Solution")[None], start_sleep, axis=0)

    sim_soln = np.concatenate(
        [phyre.observations_to_uint8_rgb(x)[None] for x in np.repeat(collision_scene[None], start_sleep,
                                                                     axis=0)])
    rollout = np.concatenate([phyre.observations_to_uint8_rgb(x)[None] for x in res.images])
    rollout = np.concatenate([text_solving, sim_soln, text_pred, rollout], axis=0)

    imageio.mimsave(osp.join(save_dir, template, status, task_id + ".gif"), rollout, fps=25)


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
