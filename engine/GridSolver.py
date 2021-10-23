import cv2
import matplotlib.pyplot as plt
import phyre
import os.path as osp
import numpy as np
from tqdm import tqdm
from math import sqrt
import random


class Line:
    def __init__(self, start=(0, 0), end=(0, 0), orientation="H", obj_idx=None, line_no=0):
        """"
        Line: ax + by = c
        """
        self.start = (round(start[0]), round(start[1]))
        self.end = (round(end[0]), round(end[1]))
        self.orientation = orientation

        self.a = self.end[1] - self.start[1]
        self.b = self.start[0] - self.end[0]
        self.c = self.start[0] * (self.end[1] - self.start[1]) + self.start[1] * (self.start[0] - self.end[0])

        self.obj_idx = obj_idx  # obj_idx to which the line belongs
        self.line_no = line_no

        # make c positive
        if self.c < 0:
            self.c *= -1
            self.a *= -1
            self.b *= -1

        # check c=0
        if self.c == 0:
            if self.a != 0:
                self.a = 1
            elif self.b != 0:
                self.b = 1

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

    def getDistFromPoint(self, point):
        x = point[0]
        y = point[1]

        numerator = abs(self.a * x + self.b * y - self.c)
        denominator = sqrt(self.a ** 2 + self.b ** 2) + 1e-6

        return numerator / denominator


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
        self.width = 256
        self.height = 256

        self.lineDatabase = [[Line() for j in range(self.num_lines)] for i in range(self.num_objects)]
        self.allLines = [Line(start=(0, 0), end=(0, 255), orientation="V", obj_idx="L"),
                         Line(start=(255, 0), end=(255, 255), orientation="V", obj_idx="R"),
                         Line(start=(0, 0), end=(255, 0), orientation="H", obj_idx="U"),
                         Line(start=(0, 255), end=(255, 255), orientation="H", obj_idx="D")]
        self.regions = {}

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
            l1 = Line(start=(splitPoints[i][1][0], 0), end=(splitPoints[i][1][0], 255.), orientation="V", obj_idx=i,
                      line_no=0)
            l2 = Line(start=(splitPoints[i][0][0], 0), end=(splitPoints[i][0][0], 255.), orientation="V", obj_idx=i,
                      line_no=1)
            l3 = Line(start=(splitPoints[i][3][0], 0), end=(splitPoints[i][3][0], 255.), orientation="V", obj_idx=i,
                      line_no=2)
            l4 = Line(start=(0, splitPoints[i][2][1]), end=(255., splitPoints[i][2][1]), orientation="H", obj_idx=i,
                      line_no=3)
            l5 = Line(start=(0, splitPoints[i][0][1]), end=(255., splitPoints[i][0][1]), orientation="H", obj_idx=i,
                      line_no=4)
            l6 = Line(start=(0, splitPoints[i][4][1]), end=(255., splitPoints[i][4][1]), orientation="H", obj_idx=i,
                      line_no=5)

            self.lineDatabase[i] = [l1, l2, l3, l4, l5, l6]

        for i in range(self.num_objects):
            for line in self.lineDatabase[i]:
                self.allLines.append(line)

        pbar = tqdm(total=self.width * self.height)
        for x in range(self.width):
            for y in range(self.height):
                point = (x, y)
                # finding the bounding lines
                minDistLineL = self.allLines[0]
                minDistLineR = self.allLines[0]
                minDistLineU = self.allLines[0]
                minDistLineD = self.allLines[0]
                minDistL = 1e6
                minDistR = 1e6
                minDistU = 1e6
                minDistD = 1e6

                for line in self.allLines:
                    if line.orientation == "V" and line.locatePoint(point) and line.getDistFromPoint(point) < minDistL:
                        minDistLineL = line
                        minDistL = line.getDistFromPoint(point)

                    if line.orientation == "V" and not line.locatePoint(point) and line.getDistFromPoint(
                            point) < minDistR:
                        minDistLineR = line
                        minDistR = line.getDistFromPoint(point)

                    if line.orientation == "H" and line.locatePoint(point) and line.getDistFromPoint(
                            point) < minDistU:
                        minDistLineU = line
                        minDistU = line.getDistFromPoint(point)

                    if line.orientation == "H" and not line.locatePoint(point) and line.getDistFromPoint(
                            point) < minDistD:
                        minDistLineD = line
                        minDistD = line.getDistFromPoint(point)

                regionName = f"{minDistLineL.obj_idx}{minDistLineL.line_no}:{minDistLineR.obj_idx}{minDistLineR.line_no}:{minDistLineU.obj_idx}{minDistLineU.line_no}:{minDistLineD.obj_idx}{minDistLineD.line_no}"

                if regionName not in self.regions.keys():
                    self.regions[regionName] = []

                self.regions[regionName].append(point)

                pbar.update(1)

    def getRegionForPointWrtObject(self, point, obj_idx):
        region_marker = np.array([0 for i in range(16)])  # 1 if point belongs to region with region id = array idx.
        # 0 otherwise
        lines = self.lineDatabase[obj_idx]

        if lines[0].locatePoint(point) and not lines[1].locatePoint(point) \
                and lines[3].locatePoint(point) and not lines[4].locatePoint(point):
            region_marker[0] = 1

        if lines[1].locatePoint(point) and not lines[2].locatePoint(point) \
                and lines[3].locatePoint(point) and not lines[4].locatePoint(point):
            region_marker[1] = 1

        if lines[0].locatePoint(point) and not lines[1].locatePoint(point) \
                and lines[4].locatePoint(point) and not lines[5].locatePoint(point):
            region_marker[2] = 1

        if lines[1].locatePoint(point) and not lines[2].locatePoint(point) \
                and lines[4].locatePoint(point) and not lines[5].locatePoint(point):
            region_marker[3] = 1

        if not lines[0].locatePoint(point) \
                and lines[3].locatePoint(point) and not lines[4].locatePoint(point):
            region_marker[4] = 1

        if lines[2].locatePoint(point) \
                and lines[3].locatePoint(point) and not lines[4].locatePoint(point):
            region_marker[5] = 1

        if not lines[0].locatePoint(point) \
                and lines[4].locatePoint(point) and not lines[5].locatePoint(point):
            region_marker[6] = 1

        if lines[2].locatePoint(point) \
                and lines[4].locatePoint(point) and not lines[5].locatePoint(point):
            region_marker[7] = 1

        if lines[0].locatePoint(point) and not lines[1].locatePoint(point) \
                and not lines[3].locatePoint(point):
            region_marker[8] = 1

        if lines[1].locatePoint(point) and not lines[2].locatePoint(point) \
                and not lines[3].locatePoint(point):
            region_marker[9] = 1

        if lines[0].locatePoint(point) and not lines[1].locatePoint(point) \
                and lines[5].locatePoint(point):
            region_marker[10] = 1

        if lines[1].locatePoint(point) and not lines[2].locatePoint(point) \
                and lines[5].locatePoint(point):
            region_marker[11] = 1

        if not lines[0].locatePoint(point) and not lines[3].locatePoint(point):
            region_marker[12] = 1

        if lines[2].locatePoint(point) and not lines[3].locatePoint(point):
            region_marker[13] = 1

        if not lines[0].locatePoint(point) and lines[5].locatePoint(point):
            region_marker[14] = 1

        if lines[2].locatePoint(point) and lines[5].locatePoint(point):
            region_marker[15] = 1

        return region_marker

    @staticmethod
    def makeLineOnGrid(grid, line):
        x1, x2 = line.start[0], line.end[0]
        y1, y2 = line.start[1], line.end[1]

        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                grid[y, x, :] = np.array([255., 0., 0.])

        return grid

    def makeGridForObject(self, grid, obj_idxs, visualise=False):
        if len(self.allLines) == 0:
            self.makeRegions()

        for obj_idx in obj_idxs:
            for line in self.lineDatabase[obj_idx]:
                grid = self.makeLineOnGrid(grid, line)

        if visualise:
            plt.imshow(grid)
            plt.show()

        return grid

    def visualiseRegions(self):
        # check if regions have been made
        if len(self.allLines) == 0:
            print("Call makeRegions first")
            return

        grid = np.zeros((256, 256, 3))
        for region_name, region in self.regions.items():
            color = np.random.randint(0, 255, (3,)) / 255.
            for point in region:
                grid[point[1], point[0], :] = color

        plt.imshow(grid)
        plt.show()

    def randomSolve(self, sim, task_idx, task, visualise=False):
        solution_points = []
        initial_scene = phyre.observations_to_float_rgb(sim.initial_scenes[task_idx])
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
            res = sim.simulate_action(task_idx, action, need_featurized_objects=True, stride=1)

            initial_scene = self.makeGridForObject(initial_scene, obj_idxs=[0, 1, 2], visualise=False)

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

                x, y = res.featurized_objects.features[0][-1][0] * 255., \
                       (1 - res.featurized_objects.features[0][-1][1]) * 255.
                x, y = round(x), round(y)

                initial_scene[y, x, :] = [1., 0., 0.]

                solution_points.append((x, y))

        if visualise:
            plt.imshow(initial_scene)
            plt.show()

        return solution_points

    def getRegionDensities(self, solution_points):
        if len(self.allLines) == 0:
            self.makeRegions()

        region_density = {}

        most_dense_region = list(self.regions.keys())[0]
        max_density = 0
        for region_name, region in self.regions.items():
            num_solving_points = 0
            total_points = len(region)

            for point in solution_points:
                if point in region:
                    num_solving_points += 1

            density = num_solving_points / total_points
            region_density[region_name] = density

            if density > max_density:
                most_dense_region = region_name
                max_density = density

        return region_density, most_dense_region

    def gridSolve(self, sim, task_idx, task, proposed_region, visualise=False):
        if proposed_region not in self.regions.keys():
            print("Proposed region not found in the grid")
            return False

        region_space = self.regions[proposed_region]
        initial_scene = phyre.observations_to_float_rgb(sim.initial_scenes[task_idx])
        initial_scene = self.makeGridForObject(initial_scene, obj_idxs=[0, 1, 2], visualise=True)

        cache = phyre.get_default_100k_cache('ball')
        actions = cache.action_array
        actionList = actions[cache.load_simulation_states(task) == 1]
        if len(actionList) == 0:
            print("WARNING no solution action in cache at task", task)
            actionList = [np.random.rand(3)]

        solved = 0
        tries = 0
        max_tries = 1000
        num_solving = 100
        while solved < num_solving:
            tries += 1

            point = random.choice(region_space)  # choose random point from proposed region
            radius = random.choice(actionList)[-1]  # choose a random radius from cache
            action = np.array([point[0] / 255., 1. - point[1] / 255., radius])

            try:
                res = sim.simulate_action(task_idx, action, need_featurized_objects=True, stride=1)
            except:
                if tries > max_tries:
                    break
                else:
                    continue

            if res.status.is_solved() and not res.status.is_invalid():
                tries = 0
                solved += 1

        if visualise:
            plt.imshow(initial_scene)
            plt.show()

        return solved / num_solving


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
    gridSolver.makeRegions()
    gridSolver.makeGridForObject(phyre.observations_to_float_rgb(sim.initial_scenes[0]), obj_idxs=[0, 1, 2],
                                 visualise=True)
    gridSolver.visualiseRegions()
    solution_points = gridSolver.randomSolve(sim, task_idx=0, task=tasks_ids[0], visualise=True)
    _, solution_region = gridSolver.getRegionDensities(solution_points)

    gridSolver2 = GridSolver(sim.initial_featurized_objects[1])
    gridSolver2.makeRegions()
    solved = gridSolver2.gridSolve(sim, 1, tasks_ids[1], solution_region, visualise=False)
