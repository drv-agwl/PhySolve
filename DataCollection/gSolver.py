import phyre
import cv2
import pickle
import pathlib
import os, random
from PIL import Image, ImageFont, ImageDraw
import numpy as np
# from planner.planner_agent import solve, find_dir
from DataCollection import dijkstra  # in the local folder
# from ttictoc import tic,toc
import matplotlib
# matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

tries = 0
tasks = []

tasks_block = [
    #     '00000:001', '00000:002', '00000:003', '00000:004', '00000:005',
    #     '00001:001', '00001:002', '00001:003', '00001:004', '00001:005',
    #     '00002:007', '00002:011', '00002:015', '00002:017', '00002:023',
    '00002:026', '00002:044', '00002:052', '00002:066', '00002:079',
    #     '00003:000', '00003:001', '00003:002', '00003:003', '00003:004',
    #     '00004:063', '00004:071', '00004:092', '00004:094', '00004:095',
]
tasks.append(tasks_block)
tasks = sum(tasks, [])
# tasks = ["00013:011"]  # "00007:100", "00007:113", "00007:125"

base_path = "tmp_2"  # "fiddeling"
stride = 1
save_ind = 500
font = ImageFont.truetype("./arial.ttf", 10)  # Keyboard.ttf

with open("./DataCollection/095_collected_traj_256.pickle", 'rb') as handle:
    tr_xy256 = pickle.load(handle)  # ±256 pixel representation of 12x5 trajectories


def centerXY2YX(tr_xy256, c_xy256):
    xyl = np.array(tr_xy256, dtype=int) + np.array(c_xy256, dtype=int)
    xyl.T[[0, 1]] = xyl.T[[1, 0]]  # translate from plot xy to list -yx
    xyl[:, 0] = 255 - xyl[:, 0]  # translate from plot xy to list -yx
    return xyl


def trjBoxConstrain(tr_YX256, img_constrains):
    trj_return_box_constrain = []
    for i, yx in enumerate(tr_YX256):
        trg_center_constrained = False
        if yx[0] < 0 or yx[0] > 255 or yx[1] < 0 or yx[1] > 255:
            break
        if img_constrains[yx[0], yx[1]] == 1:
            trg_center_constrained = True
            if i > 2:
                break
        if not trg_center_constrained:
            trj_return_box_constrain.append(yx)
    return trj_return_box_constrain


def trj_shift_const(xy256, tr_xy256, img_constrains, distance_map, trg_saveImg=False):
    distance_list = []
    for k in tr_xy256.keys():
        tr_YX256 = trjBoxConstrain(
            centerXY2YX(tr_xy256[k], xy256),  # tr_xy256[k] - plotXY_256 trajectory; # xy256 - plotXY coordinates center
            img_constrains
        )
        for i, yx in enumerate(tr_YX256):
            v = distance_map[int(yx[0]), int(yx[1])]
            distance_list.append([v, (int(yx[0]), int(yx[1])), k])
    distance_list_sorted = sorted(distance_list, key=lambda x: x[0])
    # -------------------------------------------------
    # save image
    if trg_saveImg:
        distance_map_trj = 255 - distance_map
        # for k in [distance_list_sorted[0][2]]:
        for k in tr_xy256.keys():
            tr_YX256 = trjBoxConstrain(
                centerXY2YX(tr_xy256[k], xy256),
                # tr_xy256[k] - plotXY_256 trajectory; # xy256 - plotXY coordinates center
                img_constrains
            )
            for yx in tr_YX256:
                # print(yx)
                distance_map_trj[yx[0], yx[1]] = 0.0
        try:
            distance_map_trj = add_cross_to_img(distance_map_trj, distance_list_sorted[0][1][0],
                                                distance_list_sorted[0][1][1], color=255.)
        except:
            pass
        # cv2.imwrite(path_str + os.sep + '__dmap_trj_' + str(imgID).zfill(2) + '.png', distance_map_trj)
    return distance_list_sorted, distance_map_trj


def feature_space(feature_scene,
                  dot_obj_relative_yx):  # extend the black objects constrains by the relative displacement of ball pixels around its center
    feature_obj_inds_yx = np.transpose(np.where(feature_scene))
    for c in feature_obj_inds_yx:
        for g in dot_obj_relative_yx:
            p = c + g  # p = [255, 100]
            if p[0] >= 0 and p[0] <= 255 and p[1] >= 0 and p[1] <= 255:
                feature_scene[p[0]][p[1]] = 1
    # scene_rgb = phyre.observations_to_uint8_rgb(feature_scene[::-1])
    return feature_scene


def add_cross_to_img(img, target_pos_y, target_pos_x, color=0.):
    img[target_pos_y, target_pos_x] = color
    if target_pos_y > 0:
        img[target_pos_y - 1, target_pos_x] = color
    if target_pos_y < 255:
        img[target_pos_y + 1, target_pos_x] = color
    if target_pos_x > 0:
        img[target_pos_y, target_pos_x - 1] = color
    if target_pos_x < 255:
        img[target_pos_y, target_pos_x + 1] = color
    return img


def constr_pull(img_constr_fs, r=2):
    img_constr_fs_b = np.all(img_constr_fs == np.array([0, 0, 0], dtype=np.uint8), axis=2).astype(int)
    img_resized = np.zeros((int(img_constr_fs.shape[0] / r), int(img_constr_fs.shape[1] / r)), dtype=int)
    for i in range(int(img_constr_fs.shape[0] / r)):
        for j in range(int(img_constr_fs.shape[1] / r)):
            if np.any(img_constr_fs_b[i * r:i * r + r, j * r:j * r + r]):
                img_resized[i, j] = 1
            else:
                img_resized[i, j] = 0
    return img_resized


def path_cost(task, sim, action):
    path_str = f"{base_path}/{task[:5]}/{task[6:]}"
    # pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
    # print('path_str', path_str)
    # action = sim.sample()
    # action = np.array([0.1, 0.9, 0.01])
    img_seq_orig = []

    try:
        res = sim.simulate_action(0, action, need_featurized_objects=True, stride=stride)
    except:
        return 255., None, None, None, None

    if res.featurized_objects is not None:
        for i, scene in enumerate(res.images):
            if i * stride >= save_ind:
                break
            img = Image.fromarray(phyre.observations_to_uint8_rgb(scene))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), str(i * stride), (15, 15, 15), font=font)
            img_seq_orig.append(img)
        img_seq_orig[0].save(path_str + os.sep + '_1_orig.gif', save_all=True, append_images=img_seq_orig[1:],
                             optimize=True, duration=30, loop=0)

    if res.featurized_objects is None:
        return 255.  # Max cost, unsolvable

    # try:
    #     obj_target_idx = int(np.argwhere(np.asarray(res.featurized_objects.colors) == 'PURPLE'))
    # except:
    #     obj_target_idx = int(np.argwhere(np.asarray(res.featurized_objects.colors) == 'BLUE'))
    #
    obj_green_idx = int(np.argwhere(np.asarray(res.featurized_objects.colors) == 'GREEN'))
    im_size = res.images.shape[-1]

    img_phyre_YX = res.images[0][::-1].copy()
    img_constr = np.array(img_phyre_YX == 6, dtype=np.uint8)
    img_target = np.array(img_phyre_YX == 4, dtype=np.uint8)
    if not img_target.max():
        img_target = np.array(img_phyre_YX == 3, dtype=np.uint8)

    green_obj_inds_yx = np.transpose(np.where(img_phyre_YX == 2))  # 'GREEN' object
    obj_green_idx = int(np.argwhere(np.asarray(res.featurized_objects.colors) == 'GREEN'))
    obj_green_center_yx = np.array((255 - int(round(res.featurized_objects.ys[0][obj_green_idx] * 255.)),
                                    int(round(res.featurized_objects.xs[0][obj_green_idx] * 255.))), dtype=int)
    dot_obj_relative_yx = green_obj_inds_yx - obj_green_center_yx

    green_pos_x_list = np.array(np.round(res.featurized_objects.xs[:, obj_green_idx] * (im_size - 1)), dtype='uint8')
    green_pos_y_list = np.array(np.round((1 - res.featurized_objects.ys[:, obj_green_idx]) * (im_size - 1)),
                                dtype='uint8')
    green_xs = res.featurized_objects.xs[:, obj_green_idx]  # original plot positions
    green_ys = res.featurized_objects.ys[:, obj_green_idx]  # original plot positions

    img_constr_fs = feature_space(img_constr, dot_obj_relative_yx)

    img_target_fs = feature_space(img_target, dot_obj_relative_yx)

    distance_map_fs = dijkstra.find_distance_map_obj(phyre.observations_to_uint8_rgb(img_constr_fs[::-1] * 6),
                                                     img_target_fs & (1 - img_constr_fs))

    costs = distance_map_fs[green_pos_y_list, green_pos_x_list]

    # if np.min(costs) < 37.:  # plotting when solved 00003:001
    #     path_img = np.zeros((256, 256))
    #     for x, y in zip(green_pos_x_list, green_pos_y_list):
    #         path_img[y, x] = 1
    #     plt.imshow(path_img)
    #     plt.show()

    return np.min(costs), zip(green_pos_y_list, green_pos_x_list), res.status, res
