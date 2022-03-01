import os
import pandas as pd
import os.path as osp
import phyre
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np


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
                non_solving_points = row_combined_df["# Non Solving Points"].values[0] + \
                                     row_df["# Non Solving Points"].values[0]
                total = num_solved + non_solving_points
                density = 0. if total == 0 else num_solved / total

                row_combined_df["Density of Solution Points"] = density
                row_combined_df["# Solving Points"] = num_solved
                row_combined_df["# Non Solving Points"] = non_solving_points
                row_combined_df["Total Points in Region"] = -1.
                combined_df[combined_df["Region Names"] == region] = row_combined_df

    combined_df.to_csv(osp.join(src_dir, "combined.csv"), index=False)

def plot_csv(src_dir, csv_file):
    data = pd.read_csv(csv_file)
    density = list(data["Density of Solution Points"].values)
    density = [100*x for x in density]
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


if __name__ == '__main__':
    src_dir = "/home/dhruv/Desktop/PhySolve/engine"

    eval_setup = 'ball_cross_template'
    fold_id = 0
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    tasks = train_tasks + dev_tasks + test_tasks

    templates = [2]
    templates = [str(i).zfill(5) for i in templates]

    tasks_ids = sorted([x for x in tasks if x.startswith(tuple(templates))])

    # merge_csv(src_dir, tasks_ids)

    plot_csv(src_dir, osp.join(src_dir, "combined.csv"))
