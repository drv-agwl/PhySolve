import cv2
import matplotlib.pyplot as plt
import phyre
import random
import os.path as osp
import numpy as np
from tqdm import tqdm
import itertools
from itertools import permutations


def get_obj_channels(imgs, size=(256, 256)):
    channels = range(1, 7)
    if len(imgs.shape) == 2:
        obj_channels = np.array([cv2.resize(((imgs == ch).astype(float)), size, cv2.INTER_MAX)
                                 for ch in channels])
        obj_channels = np.flip(obj_channels, axis=1)

    else:
        obj_channels = np.array(
            [np.array([cv2.resize((frame == ch).astype(float), size, cv2.INTER_MAX) for ch in channels]) for
             frame in imgs])
        obj_channels = np.flip(obj_channels, axis=2).astype(np.uint8)

    return obj_channels


def getSplitLines(x, y, width=1):
    # hline coordinates
    xs = list(range(0, 256, 1))
    ys = list(range(y - width, y + width + 1, 1))
    hline = [i for i in itertools.product(xs, ys)]

    # yline coordinates
    xs = list(range(x - width, x + width + 1, 1))
    ys = list(range(0, 256, 1))
    yline = [i for i in itertools.product(xs, ys)]

    return hline, yline


if __name__ == '__main__':
    eval_setup = 'ball_cross_template'
    fold_id = 0
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    tasks = train_tasks + dev_tasks + test_tasks

    templates = [2]
    templates = [str(i).zfill(5) for i in templates]

    tasks_ids = sorted([x for x in tasks if x.startswith(tuple(templates))])[:10]

    sim = phyre.initialize_simulator(tasks_ids, 'ball')

    solution_maps = [np.zeros((256, 256, 3)) for i in range(len(tasks_ids))]

    pbar = tqdm(len(tasks_ids))
    for task_idx, task in enumerate(tasks_ids):
        cache = phyre.get_default_100k_cache('ball')
        actions = cache.action_array

        cache_list = actions[cache.load_simulation_states(task) == 1]

        solved = 0
        tries = 0
        max_tries = 1000
        num_solving = 100
        while solved < num_solving:
            tries += 1
            actionlist = cache_list

            if len(actionlist) == 0:
                print("WARNING no solution action in cache at task", task)
                actionlist = [np.random.rand(3)]

            action = random.choice(actionlist)
            res = sim.simulate_action(task_idx, action,
                                      need_featurized_objects=True, stride=1)

            # make split lines
            body_list = res.body_list
            green_idx = body_list.index('GreenObject')
            black_idx = body_list.index('BlackObject')
            goal_idx = body_list.index('GoalObject')
            #
            # obj_centers = [(x, y) for x, y in zip(res.featurized_objects.features[0, :, 0],
            #                                       res.featurized_objects.features[0, :, 1])]
            # obj_centers = [(round(x * 255.), round((1. - y) * 255.)) for x, y in obj_centers]
            #
            # splitLines = [getSplitLines(c[0], c[1]) for c in obj_centers[:-1]]
            #
            # for obj_line in splitLines:
            #     for line in obj_line:
            #         for coord in line:
            #             solution_maps[0][coord[1], coord[0]] = 5

            try:
                features = res.featurized_objects.features
            except:
                if tries > max_tries:
                    break
                else:
                    continue

            if res.status.is_solved() and not res.status.is_invalid():
                tries = 0
                solved += 1

                # images = [phyre.observations_to_uint8_rgb(x) for x in res.images]
                x, y = res.featurized_objects.features[0][-1][0] * 255., \
                       (1 - res.featurized_objects.features[0][-1][1]) * 255.
                x, y = round(x), round(y)

                obj_channels = get_obj_channels(np.array([res.images[0]]))[0]
                # images[0][y, x, :2] += 1
                solution_maps[task_idx][:, :, 2] = (obj_channels[3] + obj_channels[5])
                solution_maps[task_idx][:, :, 1] = obj_channels[1]
                solution_maps[task_idx][y, x, 0] += 1.

        pbar.update(1)

        for i, img in enumerate(solution_maps):
            plt.imsave(f"{i+1}.png", solution_maps[i])

print("")
