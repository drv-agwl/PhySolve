import cv2
import matplotlib.pyplot as plt
import phyre
import os.path as osp
import numpy as np
from tqdm import tqdm


class Line:
    def __init__(self, start=(0, 0), end=(0, 0), orientation="H"):
        """"
        Line: ax + by = c
        """
        self.start = start
        self.end = end
        self.orientation = orientation

        self.a = end[1] - start[1]
        self.b = start[0] - end[0]
        self.c = start[0] * (end[1] - start[1]) + start[1] * (start[0] - end[0])

    def setStart(self, start):
        self.start = start

    def setEnd(self, end):
        self.end = end

    def setOrientation(self, orientation):
        self.orientation = orientation

    def locatePoint(self, point):
        """
        locates a point wrt Line
        True means outside or on the line, False means inside
        """

        val = self.a * point[0] + self.b * point[1]
        return val >= self.c


class GridSolver:
    """
    Grid for an object looks like this

                            L1  L2  L3
                            |   |   |
                            |   |   |
 L4 ------------------------|---|---|-------------------------------
 L5 ------------------------|---|---|------------------------------
 L6 ------------------------|---|---|-------------------------------
                            |   |   |
                            |   |   |
                            |   |   |
    """

    def __init__(self, featurized_objects):
        self.featurized_objects = featurized_objects
        self.objectFeatures = self.featurized_objects.features[0]
        self.num_objects = self.objectFeatures.shape[0]
        self.num_lines = 6
        self.regions = [[Line() for j in range(self.num_lines)] for i in range(self.num_objects)]

    def getSplitPoints(self):
        """
        input: object features from phyre simulator (num_objects, 14)
        output: 5 split points per object, output shape (num_objects, num_split_points, 2). 2 for x and y coordinate
        values
        """
        num_objects = self.num_objects
        num_split_points = 5  # 1 center, 4 at the centers of each edge

        splitPoints = np.zeros((num_objects, num_split_points, 2))

        for i in range(num_objects):
            cx = self.objectFeatures[i][0] * 255.
            cy = (1. - self.objectFeatures[i][1]) * 255.

            objectType = "ball" if self.objectFeatures[i][4] == 1 else "other"
            latRadius = self.objectFeatures[i][3] * 127.5
            longRadius = 2.5 if objectType != "ball" else latRadius

            ux, uy = cx, cy - longRadius
            dx, dy = cx, cy + longRadius
            lx, ly = cx - latRadius, cy
            rx, ry = cx + latRadius, cy

            splitPoints[i][0][0] = cx
            splitPoints[i][0][1] = cy
            splitPoints[i][1][0] = lx
            splitPoints[i][1][1] = ly
            splitPoints[i][2][0] = ux
            splitPoints[i][2][1] = uy
            splitPoints[i][3][0] = rx
            splitPoints[i][3][1] = ry
            splitPoints[i][4][0] = dx
            splitPoints[i][4][1] = dy

            for j in range(num_split_points):
                splitPoints[i][j][0] = max(0, min(splitPoints[i][j][0], 255.))
                splitPoints[i][j][1] = max(0, min(splitPoints[i][j][1], 255.))

        return splitPoints

    def makeRegions(self):
        splitPoints = self.getSplitPoints()
        for i in range(self.num_objects):
            l1 = Line(start=(splitPoints[i][1][0], 0), end=(splitPoints[i][1][0], 255.), orientation="V")
            l2 = Line(start=(splitPoints[i][0][0], 0), end=(splitPoints[i][0][0], 255.), orientation="V")
            l3 = Line(start=(splitPoints[i][3][0], 0), end=(splitPoints[i][3][0], 255.), orientation="V")
            l4 = Line(start=(0, splitPoints[i][2][1]), end=(255., splitPoints[i][2][1]), orientation="H")
            l5 = Line(start=(0, splitPoints[i][0][1]), end=(255., splitPoints[i][0][1]), orientation="H")
            l6 = Line(start=(0, splitPoints[i][4][1]), end=(255., splitPoints[i][4][1]), orientation="H")

            self.regions[i] = [l1, l2, l3, l4, l5, l6]


if __name__ == '__main__':
    eval_setup = 'ball_cross_template'
    fold_id = 0
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    tasks = train_tasks + dev_tasks + test_tasks

    templates = [2]
    templates = [str(i).zfill(5) for i in templates]

    tasks_ids = sorted([x for x in tasks if x.startswith(tuple(templates))])[:10]

    sim = phyre.initialize_simulator(tasks_ids, 'ball')

    gridSolver = GridSolver(sim.initial_featurized_objects[0])
    print(gridSolver.getSplitPoints().shape)
    gridSolver.makeRegions()

#     pbar = tqdm(len(tasks_ids))
#     for task_idx, task in enumerate(tasks_ids):
#         cache = phyre.get_default_100k_cache('ball')
#         actions = cache.action_array
#
#         cache_list = actions[cache.load_simulation_states(task) == 1]
#
#         solved = 0
#         tries = 0
#         max_tries = 1000
#         num_solving = 100
#         while solved < num_solving:
#             tries += 1
#             actionlist = cache_list
#
#             if len(actionlist) == 0:
#                 print("WARNING no solution action in cache at task", task)
#                 actionlist = [np.random.rand(3)]
#
#             action = random.choice(actionlist)
#             res = sim.simulate_action(task_idx, action,
#                                       need_featurized_objects=True, stride=1)
#
#             # make split lines
#             body_list = res.body_list
#             green_idx = body_list.index('GreenObject')
#             black_idx = body_list.index('BlackObject')
#             goal_idx = body_list.index('GoalObject')
#             #
#             # obj_centers = [(x, y) for x, y in zip(res.featurized_objects.features[0, :, 0],
#             #                                       res.featurized_objects.features[0, :, 1])]
#             # obj_centers = [(round(x * 255.), round((1. - y) * 255.)) for x, y in obj_centers]
#             #
#             # splitLines = [getSplitLines(c[0], c[1]) for c in obj_centers[:-1]]
#             #
#             # for obj_line in splitLines:
#             #     for line in obj_line:
#             #         for coord in line:
#             #             solution_maps[0][coord[1], coord[0]] = 5
#
#             try:
#                 features = res.featurized_objects.features
#             except:
#                 if tries > max_tries:
#                     break
#                 else:
#                     continue
#
#             if res.status.is_solved() and not res.status.is_invalid():
#                 tries = 0
#                 solved += 1
#
#                 # images = [phyre.observations_to_uint8_rgb(x) for x in res.images]
#                 x, y = res.featurized_objects.features[0][-1][0] * 255., \
#                        (1 - res.featurized_objects.features[0][-1][1]) * 255.
#                 x, y = round(x), round(y)
#
#                 obj_channels = get_obj_channels(np.array([res.images[0]]))[0]
#                 # images[0][y, x, :2] += 1
#                 solution_maps[task_idx][:, :, 2] = (obj_channels[3] + obj_channels[5])
#                 solution_maps[task_idx][:, :, 1] = obj_channels[1]
#                 solution_maps[task_idx][y, x, 0] += 1.
#
#         pbar.update(1)
#
#         for i, img in enumerate(solution_maps):
#             plt.imsave(f"{i+1}.png", solution_maps[i])
#
# print("")
