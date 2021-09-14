import cv2
import phyre
import random
from DataCollection.gSolver import path_cost
from DataCollection.utils import *
import pickle
import gzip
import os
import os.path as osp


def get_collision_timestep(res):
    # return timestep where collision occurred
    collision_mat = res.bitmap_seq
    body_list = res.body_list
    green_ball_idx = body_list.index('GreenObject')
    red_ball_idx = body_list.index('RedObject')

    for timestep in range(collision_mat.shape[0]):
        if collision_mat[timestep][green_ball_idx][red_ball_idx] == 1:
            return timestep

    return -1


def check_collision(res):
    # find if balls collided with other objects
    collision_mat = res.bitmap_seq
    body_list = res.body_list
    green_ball_idx = body_list.index('GreenObject')
    red_ball_idx = body_list.index('RedObject')

    collision_red = 0
    collision_green = 0

    for timestep in range(collision_mat.shape[0]):
        for obj_idx in range(len(body_list)):
            if obj_idx == red_ball_idx or obj_idx == green_ball_idx:
                continue
            if collision_mat[timestep][green_ball_idx][obj_idx] == 1:
                collision_green = 1
            if collision_mat[timestep][red_ball_idx][obj_idx] == 1:
                collision_red = 1

    return collision_red, collision_green


def get_obj_channels(imgs, size):
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


def get_noised_action(action, step):
    noise_factor = step * 0.1
    while True:
        noise = np.random.randn(3) * np.array([0.2, 0.1, 0.2]) * noise_factor
        if np.linalg.norm(noise) >= 0.05 or sum(noise) == 0.:
            noised_action = np.clip(action + noise, 0., 1.)
            # keep radius same
            noised_action[-1] = action[-1]
            return noised_action
        elif np.linalg.norm(noise) < 0.05:
            noise_factor += 0.1


if __name__ == '__main__':

    eval_setup = 'ball_cross_template'
    fold_id = 0
    num_samples = 100
    sigma = 0.5
    size = (64, 64)

    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)

    tasks = train_tasks + dev_tasks + test_tasks

    no_grey_ids = [0, 1, 2, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
    # no_grey_ids = [14]
    task_ids = [str(i).zfill(5) for i in no_grey_ids]
    tasks_ids = sorted([x for x in tasks if x.startswith(tuple(task_ids))])

    sim = phyre.initialize_simulator(tasks_ids, 'ball')
    database = []

    for task_idx, task in enumerate(tasks_ids):
        if osp.exists(f'./Database_newPosModel/{task}.pkl'):
            continue
        database = []
        solving = True  # collect solving or non-solving task
        stride = 5  # num frames to skip
        number_to_solve = 10
        max_tries = 10000
        channels = range(1, 7)

        data = []
        print(task)

        cache = phyre.get_default_100k_cache('ball')
        actions = cache.action_array

        cache_list = actions[cache.load_simulation_states(task) == (1 if solving else -1)]

        solved = 0

        tries = 0
        while solved < number_to_solve:
            tries += 1
            actionlist = cache_list

            if len(actionlist) == 0:
                print("WARNING no solution action in cache at task", task)
                actionlist = [np.random.rand(3)]

            action = random.choice(actionlist)
            res = sim.simulate_action(task_idx, action,
                                      need_featurized_objects=True, stride=1)
            try:
                features = res.featurized_objects.features
            except:
                if tries > max_tries:
                    break
                else:
                    continue

            # IF SOLVED PROCESS ROLLOUT
            if (res.status.is_solved() == solving) and not res.status.is_invalid():
                tries = 0
                solved += 1

                collision_timestep = get_collision_timestep(res)
                path_images_till_collision = get_obj_channels(res.images[: collision_timestep + 1], size=(64, 64)) if \
                    collision_timestep != -1 else get_obj_channels(res.images, size=(64, 64))

                imgs_solved = res.images[
                    range(collision_timestep - 2 * stride, collision_timestep + 3 * stride, stride)]
                imgs_solved = get_obj_channels(imgs_solved, size=(64, 64))

                features = res.featurized_objects.features[
                    range(collision_timestep - 2 * stride, collision_timestep + 3 * stride, stride)]

                unsolving_action = np.array([0.1, 0.1, 0.0003])
                try:
                    res_unsolved = sim.simulate_action(task_idx, unsolving_action,
                                                       need_featurized_objects=True, stride=1)
                except:
                    continue
                imgs_unsolved = res_unsolved.images[
                    range(collision_timestep - 2 * stride, collision_timestep + 3 * stride, stride)]
                imgs_unsolved = get_obj_channels(imgs_unsolved, size=(64, 64))

                # LFM data
                # lfm_solving = 1
                # lfm_unsolving = 2
                #
                # step = 1
                # while lfm_solving:  # get noisy action with solved results
                #     noised_action = get_noised_action(action, step=step)
                #     try:
                #         res_lfm = sim.simulate_action(task_idx, noised_action,
                #                                       need_featurized_objects=True, stride=1)
                #     except:
                #         continue
                #     while res_lfm.images is None:
                #         step -= 0.2
                #         step = max(0, step)
                #
                #         noised_action = get_noised_action(action, step=step)
                #         try:
                #             res_lfm = sim.simulate_action(task_idx, noised_action,
                #                                           need_featurized_objects=True, stride=1)
                #         except:
                #             continue
                #
                #     if res_lfm.status.is_solved():
                #         lfm_solving -= 1
                #         step += 0.5
                #
                #         imgs_lfm = np.max(get_obj_channels(res_lfm.images, size=(64, 64)), axis=0)
                #         imgs_unsolved_complete = np.max(get_obj_channels(res_unsolved.images, size=(64, 64)), axis=0)

                database.append({'images_solved': np.array(imgs_solved),
                                 'path_till_collision': np.array(path_images_till_collision),
                                 'images_unsolved': np.array(imgs_unsolved),
                                 # 'path_lfm': np.asarray(imgs_lfm),
                                 # 'lfm_scene-0': get_obj_channels(np.array(res_lfm.images[0]), size=(64, 64)),
                                 # 'path_unsolved': np.array(imgs_unsolved_complete),
                                 'features': features,
                                 'collision_timestep': collision_timestep / stride,
                                 'scene-0': get_obj_channels(np.array(res.images[0]), size=(64, 64)),
                                 'scene-33': get_obj_channels(np.array(res.images[collision_timestep * 1 // 3]),
                                                              size=(64, 64)),
                                 'scene-66': get_obj_channels(np.array(res.images[collision_timestep * 2 // 3]),
                                                              size=(64, 64)),
                                 'task-id': task})

                # UNCOMMENT THE WHOLE CODE BELOW - THIS IS DONE TEMPORARILY

                # step = 1
                # while lfm_unsolving:  # get noisy action with unsolved results
                #     noised_action = get_noised_action(action, step=step)
                #     if (noised_action == action).all():
                #         step = random.randint(1, 10)
                #         continue
                #     try:
                #         res_lfm = sim.simulate_action(task_idx, noised_action,
                #                                       need_featurized_objects=True, stride=1)
                #     except:
                #         continue
                #     while res_lfm.images is None:
                #         step -= 0.2
                #         step = max(0, step)
                #
                #         noised_action = get_noised_action(action, step=step)
                #         try:
                #             res_lfm = sim.simulate_action(task_idx, noised_action,
                #                                           need_featurized_objects=True, stride=1)
                #         except:
                #             continue
                #
                #     if not res_lfm.status.is_solved():
                #         lfm_unsolving -= 1
                #         step += 1
                #
                #         imgs_lfm = np.max(get_obj_channels(res_lfm.images, size=(64, 64)), axis=0)
                #         imgs_unsolved_complete = np.max(get_obj_channels(res_unsolved.images, size=(64, 64)), axis=0)
                #
                #         database.append({'images_solved': np.array(imgs_solved),
                #                          'images_unsolved': np.array(imgs_unsolved),
                #                          'path_lfm': np.asarray(imgs_lfm),
                #                          'lfm_scene-0': get_obj_channels(np.array(res_lfm.images[0]), size=(64, 64)),
                #                          'path_unsolved': np.array(imgs_unsolved_complete),
                #                          'features': features,
                #                          'collision_timestep': collision_timestep / stride,
                #                          'scene-0': get_obj_channels(np.array(res.images[0]), size=(64, 64)),
                #                          'scene-33': get_obj_channels(np.array(res.images[collision_timestep * 1 // 3]),
                #                                                       size=(64, 64)),
                #                          'scene-66': get_obj_channels(np.array(res.images[collision_timestep * 2 // 3]),
                #                                                       size=(64, 64)),
                #                          'task-id': task})
                #
                #     elif res.status.is_solved():
                #         step = random.randint(1, 10)

        file = gzip.GzipFile(f'./Database_newPosModel/{task}.pkl', 'wb')
        pickle.dump(database, file)
        file.close()

        # with open() as handle:
        #     pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)
