import pickle
import numpy as np
import cv2
import random
import gzip


def load_data_position(data_paths, seq_len, size, all_samples=False):
    channels = range(1, 7)

    train_data = []

    for data_path in data_paths:
        with gzip.open(data_path, 'rb') as fp:
            task_data = pickle.load(fp)
        for data in task_data:
            obj_channels = data['images_solved']
            collision_time = data["collision_timestep"]
            features = data["features"]

            scene_0 = data["scene-0"]
            scene_33 = data["scene-33"]
            scene_66 = data["scene-66"]

            collision_idx = seq_len // 2

            # obj_channels = np.array(
            #     [np.array([cv2.resize((frame == ch).astype(float), size, cv2.INTER_MAX) for ch in channels]) for
            #      frame in frames])
            # obj_channels = np.flip(obj_channels, axis=2)

            # scene_0 = np.array([cv2.resize(((scene_0 == ch).astype(float)), size, cv2.INTER_MAX)
            #                           for ch in channels])
            # scene_0 = np.flip(scene_0, axis=1)
            #
            # scene_33 = np.array([cv2.resize(((scene_33 == ch).astype(float)), size, cv2.INTER_MAX)
            #                     for ch in channels])
            # scene_33 = np.flip(scene_33, axis=1)
            #
            # scene_66 = np.array([cv2.resize(((scene_66 == ch).astype(float)), size, cv2.INTER_MAX)
            #                     for ch in channels])
            # scene_66 = np.flip(scene_66, axis=1)

            # if data_path.split('.')[0][-1] == '2':  # Task-2:
            green_ball_idx = 1
            red_ball_idx = 0
            static_obj_idxs = [3, 5]
            # elif data_path.split('.')[0][-1] == '0':  # Task-20
            #     green_ball_idx = 1
            #     red_ball_idx = 0
            #     static_obj_idxs = [3, 5]
            # elif data_path.split('.')[0][-1] == '5':  # Task-15
            #     green_ball_idx = 1
            #     red_ball_idx = 0
            #     static_obj_idxs = [3, 5]

            green_ball_collision = obj_channels[collision_idx, green_ball_idx].astype(np.uint8)
            red_ball_collision = obj_channels[collision_idx, red_ball_idx].astype(np.uint8)
            # red_ball_path = np.flip(obj_channels[:seq_len // 2 + 1, red_ball_idx], axis=0).astype(
            #     np.uint8)

            red_ball_gt = scene_0[red_ball_idx].astype(np.uint8)

            static_objs = np.max(obj_channels[0, static_obj_idxs, :, :][None], axis=1).astype(np.uint8)

            empty_channel = np.zeros_like(static_objs)
            scene_0 = np.concatenate(
                [scene_0[i][None] for i in [red_ball_idx, green_ball_idx]] + [empty_channel, static_objs],
                axis=0)

            scene_33 = np.concatenate(
                [scene_33[i][None] for i in [red_ball_idx, green_ball_idx]] + [empty_channel, static_objs],
                axis=0)

            scene_66 = np.concatenate(
                [scene_66[i][None] for i in [red_ball_idx, green_ball_idx]] + [empty_channel, static_objs],
                axis=0)

            combined = np.concatenate([green_ball_collision[None], red_ball_collision[None], static_objs,
                                       scene_0, scene_33, scene_66, red_ball_gt[None]], axis=0).astype(np.uint8)

            # combined = np.concatenate([red_ball_path, static_objs, red_ball_gt[None]],
            #                           axis=0).astype(np.uint8)

            red_diam = features[0][-1][3]
            train_data.append({"Images": combined,
                               "Collision_time": collision_time,
                               "Red_diam": red_diam})

    random.seed(7)
    random.shuffle(train_data)

    if all_samples:
        return train_data

    train_data, test_data = train_data[:int(0.9 * len(train_data))], train_data[int(0.9 * len(train_data)):]
    return train_data, test_data
