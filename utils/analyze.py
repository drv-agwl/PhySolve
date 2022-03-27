import os
import pandas as pd
import os.path as osp
import phyre
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from phyre_utils import invert_bg


def merge_csv(src_dir, task_ids):
    """
    merges all the task csv files
    """
    csv_files = [osp.join(src_dir, f"{task_id}.csv") for task_id in task_ids]
    combined_df = pd.read_csv(csv_files[0])

    for csv_file in csv_files[1:]:
        all_regions = list(combined_df.values[:, 1])
        df = pd.read_csv(csv_file)
        cur_regions = list(df.values[:, 1])

        for region in cur_regions:
            if region not in all_regions:
                row_df = df[df["Region Names"] == region]
                combined_df = pd.concat([combined_df, row_df], axis=0)

            else:
                row_combined_df = combined_df[combined_df["Region Names"] == region]
                row_df = df[df["Region Names"] == region]

                num_solved = row_combined_df["# Solving Points"].values[0] + row_df["# Solving Points"].values[0]
                total_points = row_combined_df["Total Points in Region"].values[0] + \
                               row_df["Total Points in Region"].values[0]
                density = 0. if total_points == 0 else num_solved / total_points

                row_combined_df["Density of Solution Points"] = density
                row_combined_df["# Solving Points"] = num_solved
                row_combined_df["Total Points in Region"] = -1.
                combined_df[combined_df["Region Names"] == region] = row_combined_df

    combined_df.to_csv(osp.join(src_dir, "combined.csv"), index=False)


def plot_combined_density(combined_csv_file, src_dir, task_ids):
    """
    plots an image (num_episodes x num_regions) of solution density
    """

    combined_data = pd.read_csv(osp.join(src_dir, combined_csv_file))
    num_regions = combined_data.shape[0]

    num_tasks = 100
    slab_size = 60

    height = num_tasks * slab_size
    width = num_regions * slab_size

    density_image = np.zeros((height, width, 3))

    all_regions = list(combined_data.values[:, 1])

    csv_files = [osp.join(src_dir, f"{task_id}.csv") for task_id in task_ids]

    for task_itr, csv_file in enumerate(csv_files):
        task_data = pd.read_csv(csv_file)

        for region_itr, region in enumerate(all_regions):
            try:
                density = task_data[task_data["Region Names"] == region]["Density of Solution Points"].values[0]
                density_image[task_itr * slab_size: (task_itr + 1) * slab_size,
                region_itr * slab_size: (region_itr + 1) * slab_size, 0] = density
            except:
                # region not present in current task(episode)
                continue

    density_image /= np.max(density_image)
    density_image = invert_bg(density_image)
    # density_image *= 255.

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(36, 30)
    # plt.tick_params(axis="x", colors="white")
    # plt.tick_params(axis="y", colors="white")

    x = list(range(0, width, slab_size))
    y = list(range(0, height, slab_size))
    labelx = all_regions
    labely = task_ids

    plt.xticks(x, labelx, rotation="vertical")
    plt.yticks(y, labely, rotation="horizontal")
    plt.imshow(density_image)
    plt.savefig('./density_image_rgb.png', dpi=200)


def plot_csv(src_dir, csv_file):
    data = pd.read_csv(csv_file)
    density = list(data["Density of Solution Points"].values)
    density = [100 * x for x in density]
    regions = list(data["Region Names"])

    figure(figsize=(30, 25), dpi=120)

    rev_sort_idx = np.argsort(density)
    regions = [regions[i] for i in reversed(rev_sort_idx)][:35]
    density = [density[i] for i in reversed(rev_sort_idx)][:35]

    plt.bar(regions, density, color='blue', width=0.4)
    plt.xticks(rotation=90)
    plt.xlabel("Region Names")
    plt.ylabel("% Solving Points")
    plt.savefig(osp.join(src_dir, "density_hist.png"))


def get_reciprocal_gain(density_roi, col1, col2):
    d1 = density_roi[:, col1]
    d2 = density_roi[:, col2]

    mean_d1 = np.mean(d1)
    mean_d2 = np.mean(d2)

    comb = np.maximum(d1, d2)
    mean_comb = np.mean(comb, axis=0)

    reciprocal_gain = min(mean_comb - mean_d1, mean_comb - mean_d2)
    return reciprocal_gain


def plot_reciprocal_gain(combined_csv_file, src_dir, task_ids):
    combined_data = pd.read_csv(osp.join(src_dir, combined_csv_file))
    num_regions = combined_data.shape[0]
    num_tasks = 100

    all_regions = list(combined_data.values[:, 1])

    csv_files = [osp.join(src_dir, f"{task_id}.csv") for task_id in task_ids]

    density_table = np.zeros((num_tasks, num_regions))

    for task_itr, csv_file in enumerate(csv_files):
        task_data = pd.read_csv(csv_file)

        for region_itr, region in enumerate(all_regions):
            try:
                density = task_data[task_data["Region Names"] == region]["Density of Solution Points"].values[0]
                density_table[task_itr, region_itr] = density
            except:
                # region not present in current task(episode)
                continue

    k = int(0.3 * num_regions)  # top 30%
    mean_region_density = np.mean(density_table, axis=0)
    idx_of_interest = np.argpartition(mean_region_density, -k)[-k:]  # Index of top k% values
    regions_of_interest = np.array(all_regions)[idx_of_interest]  # roi
    density_roi = density_table[:, idx_of_interest]

    reciprocal_gains = np.zeros((k, k))  # reciprocal gain is commutative: filling only lower triangle
    for i in range(k):
        for j in range(i, k):
            reciprocal_gains[j, i] = get_reciprocal_gain(density_roi, i, j)

    # plot results
    slab_size = 60
    height = slab_size * k
    width = slab_size * k

    image = np.zeros((height, width, 3))

    for i in range(0, k):
        for j in range(i, k):
            image[j * slab_size: (j + 1) * slab_size,
            i * slab_size: (i + 1) * slab_size, 1] = reciprocal_gains[j, i]

    image /= np.max(image)

    def k_largest_index_argpartition_v1(a, topk):
        idx = np.argpartition(-a.ravel(), topk)[:topk]
        return np.column_stack(np.unravel_index(idx, a.shape))

    top5 = k_largest_index_argpartition_v1(reciprocal_gains, 5)

    def annotate_image(image, idxs):
        for idx in idxs:
            row = idx[0] * slab_size
            col = idx[1] * slab_size

            image[row:, col:col+2, :] = np.array([1., 0., 0.])
            image[row:row+2, :col, :] = np.array([1., 0., 0.])
            image[row-10:row+10, col-10:col+10, :] = np.array([1., 0., 0.])

        return image

    image = annotate_image(image, top5)
    image = invert_bg(image)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(36, 30)
    # plt.tick_params(axis="x", colors="white")
    # plt.tick_params(axis="y", colors="white")

    x = list(range(0, width, slab_size))
    y = list(range(0, height, slab_size))
    labelx = regions_of_interest
    labely = regions_of_interest

    plt.xticks(x, labelx, rotation="vertical")
    plt.yticks(y, labely, rotation="horizontal")
    plt.imshow(image)
    plt.savefig('./reciprocal_gains_rgb.png', dpi=200)


if __name__ == '__main__':
    src_dir = "/home/dhruv/Desktop/PhySolve/engine/experience_data"

    eval_setup = 'ball_cross_template'
    fold_id = 0
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    tasks = train_tasks + dev_tasks + test_tasks

    templates = [2]
    templates = [str(i).zfill(5) for i in templates]

    tasks_ids = sorted([x.replace(':', '_') for x in tasks if x.startswith(tuple(templates))])

    # merge_csv(src_dir, tasks_ids)

    # plot_combined_density("combined.csv", src_dir, tasks_ids)
    plot_reciprocal_gain("combined.csv", src_dir, tasks_ids)

    # plot_csv(src_dir, osp.join(src_dir, "combined.csv"))
