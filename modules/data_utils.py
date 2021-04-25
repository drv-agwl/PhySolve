import pickle
import numpy as np
import cv2
import random


def load_data_position(data_paths, seq_len, size):
    channels = range(1, 7)

    train_data = []

    for data_path in data_paths:
        with open(data_path, 'rb') as handle:
            task_data = pickle.load(handle)
        for data in task_data:
            frames = data['images_solved']
            collision_time = data["collision_timestep"]
            initial_scene = data["initial_scene"]
            collision_idx = seq_len // 2

            obj_channels = np.array(
                [np.array([cv2.resize((frame == ch).astype(float), size, cv2.INTER_MAX) for ch in channels]) for
                 frame in frames])
            obj_channels = np.flip(obj_channels, axis=2)

            initial_scene = np.array([cv2.resize(((initial_scene == ch).astype(float)), size, cv2.INTER_MAX)
                                      for ch in channels])
            initial_scene = np.flip(initial_scene, axis=1)

            if data_path.split('.')[0][-1] == '2':  # Task-2:
                green_ball_idx = 1
                red_ball_idx = 0
                static_obj_idxs = [3, 5]
            elif data_path.split('.')[0][-1] == '0':  # Task-20
                green_ball_idx = 1
                red_ball_idx = 0
                static_obj_idxs = [3, 5]
            elif data_path.split('.')[0][-1] == '5':  # Task-15
                green_ball_idx = 1
                red_ball_idx = 0
                static_obj_idxs = [3, 5]

            green_ball_collision = obj_channels[collision_idx, green_ball_idx].astype(np.uint8)
            red_ball_collision = obj_channels[collision_idx, red_ball_idx].astype(np.uint8)
            # red_ball_path = np.flip(obj_channels[:seq_len // 2 + 1, red_ball_idx], axis=0).astype(
            #     np.uint8)

            red_ball_gt = initial_scene[red_ball_idx].astype(np.uint8)

            static_objs = np.max(obj_channels[0, static_obj_idxs, :, :][None], axis=1).astype(np.uint8)

            initial_scene = np.concatenate(
                [initial_scene[i][None] for i in [red_ball_idx, green_ball_idx]] + [static_objs],
                axis=0)
            combined = np.concatenate([green_ball_collision[None], red_ball_collision[None], static_objs,
                                       initial_scene, red_ball_gt[None]], axis=0).astype(np.uint8)

            # combined = np.concatenate([red_ball_path, static_objs, red_ball_gt[None]],
            #                           axis=0).astype(np.uint8)

            train_data.append({"Images": combined,
                               "Collision_time": collision_time})

    random.seed(7)
    random.shuffle(train_data)

    train_data, test_data = train_data[:int(0.9 * len(train_data))], train_data[int(0.9 * len(train_data)):]

    return train_data, test_data