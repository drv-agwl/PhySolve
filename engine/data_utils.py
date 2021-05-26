import pickle
import numpy as np
import cv2
import random
import gzip
from PIL import Image, ImageDraw


def draw_ball(size, y, x, r):
    background = np.zeros(size)
    img = Image.fromarray(background)
    draw = ImageDraw.Draw(img)
    draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=1.)
    return np.asarray(img)


def load_data_collision(data_paths, seq_len, all_samples=False, shuffle=True):
    channels = range(1, 7)
    train_data = []
    for data_path in sorted(data_paths):
        with gzip.open(data_path, 'rb') as fp:
            task_data = pickle.load(fp)
        for data in task_data:
            obj_channels_solved = data['images_solved']
            obj_channels_unsolved = data['images_unsolved']
            features = data["features"]
            task_id = data["task-id"]

            # obj_channels_solved = np.array(
            #     [np.array([cv2.resize((frame == ch).astype(float), size, cv2.INTER_MAX) for ch in channels]) for
            #      frame in frames_solved])
            # obj_channels_solved = np.flip(obj_channels_solved, axis=2)
            #
            # obj_channels_unsolved = np.array(
            #     [np.array([cv2.resize((frame == ch).astype(float), size, cv2.INTER_MAX) for ch in channels]) for
            #      frame in frames_unsolved])
            # obj_channels_unsolved = np.flip(obj_channels_unsolved, axis=2)

            green_ball_idx = 1
            red_ball_idx = 0
            static_obj_idxs = [3, 5]

            green_ball_solved = obj_channels_solved[:, green_ball_idx].astype(np.uint8)
            green_ball_unsolved = obj_channels_unsolved[:, green_ball_idx].astype(np.uint8)
            red_ball_gt = np.flip(obj_channels_solved[:seq_len // 2 + 1, red_ball_idx], axis=0).astype(
                np.uint8)
            static_objs = np.max(obj_channels_solved[0, static_obj_idxs, :, :][None], axis=1).astype(np.uint8)
            red_ball_zeros = np.zeros_like(red_ball_gt).astype(np.uint8)

            combined = np.concatenate([green_ball_solved, green_ball_unsolved, static_objs, red_ball_zeros,
                                       red_ball_gt], axis=0).astype(np.uint8)

            red_diam = features[0][-1][3]
            train_data.append({"Images": combined,
                               "Red_diam": red_diam,
                               "task-id": task_id})

    if shuffle:
        np.random.seed(7)
        np.random.shuffle(train_data)

    if all_samples:
        return train_data

    train_data, test_data = train_data[:int(0.9 * len(train_data))], train_data[int(0.9 * len(train_data)):]
    return train_data, test_data


def load_data_position(data_paths, seq_len, all_samples=False, shuffle=True):
    channels = range(1, 7)

    train_data = []

    for data_path in sorted(data_paths):
        with gzip.open(data_path, 'rb') as fp:
            task_data = pickle.load(fp)
        for data in task_data:
            obj_channels = data['images_solved']
            collision_time = data["collision_timestep"]
            features = data["features"]
            task_id = data["task-id"]

            scene_0 = data["scene-0"]
            scene_33 = data["scene-33"]
            scene_66 = data["scene-66"]

            collision_idx = seq_len // 2

            green_ball_idx = 1
            red_ball_idx = 0
            static_obj_idxs = [3, 5]

            green_ball_collision = obj_channels[collision_idx, green_ball_idx].astype(np.uint8)
            red_ball_collision = obj_channels[collision_idx, red_ball_idx].astype(np.uint8)

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

            red_diam = features[0][-1][3]
            train_data.append({"Images": combined,
                               "Collision_time": collision_time,
                               "Red_diam": red_diam,
                               # "red_ball_pos": [features[int(collision_time)][red_ball_idx][0],
                               #                  features[int(collision_time)][red_ball_idx][1]],
                               # "green_ball_pos": [features[int(collision_time)][green_ball_idx][0],
                               #                    features[int(collision_time)][green_ball_idx][1]],
                               "task-id": task_id})

    if shuffle:
        random.seed(7)
        random.shuffle(train_data)

    if all_samples:
        return train_data

    train_data, test_data = train_data[:int(0.9 * len(train_data))], train_data[int(0.9 * len(train_data)):]
    return train_data, test_data


def load_lfm_data(data_paths, seq_len, all_samples=False, shuffle=True):
    train_data = []

    for data_path in sorted(data_paths):
        with gzip.open(data_path, 'rb') as fp:
            task_data = pickle.load(fp)
        for data in task_data:
            unsolved_path = data['path_unsolved']
            imgs_solved = data['images_solved']
            lfm_path = data['path_lfm']
            collision_time = data["collision_timestep"]
            features = data['features']
            task_id = data["task-id"]
            scene_0 = data["scene-0"]

            collision_idx = seq_len // 2

            green_ball_idx = 1
            red_ball_idx = 0
            static_obj_idxs = [3, 5]

            green_ball_collision = imgs_solved[collision_idx, green_ball_idx].astype(np.uint8)
            red_ball_collision = imgs_solved[collision_idx, red_ball_idx].astype(np.uint8)
            green_ball_unsolved_path = unsolved_path[green_ball_idx].astype(np.uint8)
            static_objs = np.max(unsolved_path[static_obj_idxs], axis=0).astype(np.uint8)
            wrong_red_start = data['lfm_scene-0'][red_ball_idx]
            green_ball_lfm_path = lfm_path[green_ball_idx].astype(np.uint8)

            red_ball_gt = scene_0[red_ball_idx].astype(np.uint8)

            combined = np.concatenate([green_ball_collision[None], red_ball_collision[None],
                                       green_ball_unsolved_path[None], green_ball_lfm_path[None], wrong_red_start[None],
                                       static_objs[None],
                                       red_ball_gt[None]], axis=0).astype(np.uint8)

            red_diam = features[0][-1][3]
            train_data.append({"Images": combined,
                               "Collision_time": collision_time,
                               "Red_diam": red_diam,
                               "task-id": task_id})

    if shuffle:
        random.seed(7)
        random.shuffle(train_data)

    if all_samples:
        return train_data

    train_data, test_data = train_data[:int(0.9 * len(train_data))], train_data[int(0.9 * len(train_data)):]
    return train_data, test_data
