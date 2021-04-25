import cv2
import phyre
import random
from gSolver import path_cost
from utils import *
import pickle

eval_setup = 'ball_cross_template'
fold_id = 0
num_samples = 100
sigma = 0.5
size = (128, 128)

train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)

# for task in [x for x in train_tasks if x.startswith('00002') == True]:
#     try:
#         os.mkdir('./tmp_2/00002/'+task[6:])
#     except:
#         continue

# body-list = [Black, Green, Goal, Red]
tasks_ids = [x for x in test_tasks if x.startswith('00015')]


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


sim = phyre.initialize_simulator(tasks_ids, 'ball')
database = []

for task_idx, task in enumerate(tasks_ids):
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
            imgs_solved = res.images[range(collision_timestep - 2 * stride, collision_timestep + 3 * stride, stride)]

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

            database.append({'images_solved': np.array(imgs_solved),
                             'images_unsolved': np.array(imgs_unsolved),
                             'features': features,
                             'collision_timestep': int(collision_timestep // stride),
                             'initial_scene': np.array(res.images[0])})

with open('database_task15.pkl', 'wb') as handle:
    pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)
