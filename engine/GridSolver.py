import os
import shutil

import cv2
import matplotlib.pyplot as plt
import phyre
import os.path as osp
import numpy as np
from tqdm import tqdm
from math import sqrt
import random
import pandas as pd
from PIL import Image
import operator
import json
from RegionDiscriminator import RegionDiscriminator


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

                            L0  L1  L2
                            |   |   |
                            |   |   |   
 L3 ------------------------|---|---|-------------------------------
 L4 ------------------------|---|---|------------------------------
 L5 ------------------------|---|---|-------------------------------
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

        # pbar = tqdm(total=self.width * self.height)
        print("Building regions...")
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

                # pbar.update(1)

    def satisfy(self, point, constraint):
        keys = constraint.split("_")
        all_constraint_pass = True
        for key in keys:
            obj_idx = int(key[0])
            region_id = key[1:]

            constraint_pass = False

            if region_id == "0":
                line0 = self.lineDatabase[obj_idx][0]
                line3 = self.lineDatabase[obj_idx][3]
                if not line0.locatePoint(point) and not line3.locatePoint(point):
                    constraint_pass = True

            elif region_id == "1":
                line0 = self.lineDatabase[obj_idx][0]
                line1 = self.lineDatabase[obj_idx][1]
                line3 = self.lineDatabase[obj_idx][3]
                if line0.locatePoint(point) and not line1.locatePoint(point) and not line3.locatePoint(point):
                    constraint_pass = True

            elif region_id == "2":
                line1 = self.lineDatabase[obj_idx][1]
                line2 = self.lineDatabase[obj_idx][2]
                line3 = self.lineDatabase[obj_idx][3]
                if line1.locatePoint(point) and not line2.locatePoint(point) and not line3.locatePoint(point):
                    constraint_pass = True

            elif region_id == "3":
                line2 = self.lineDatabase[obj_idx][2]
                line3 = self.lineDatabase[obj_idx][3]
                if line2.locatePoint(point) and not line3.locatePoint(point):
                    constraint_pass = True

            elif region_id == "4":
                line0 = self.lineDatabase[obj_idx][0]
                line3 = self.lineDatabase[obj_idx][3]
                line4 = self.lineDatabase[obj_idx][4]
                if not line0.locatePoint(point) and line3.locatePoint(point) and not line4.locatePoint(point):
                    constraint_pass = True

            elif region_id == "5":
                line2 = self.lineDatabase[obj_idx][2]
                line3 = self.lineDatabase[obj_idx][3]
                line4 = self.lineDatabase[obj_idx][4]
                if line2.locatePoint(point) and line3.locatePoint(point) and not line4.locatePoint(point):
                    constraint_pass = True

            elif region_id == "6":
                line0 = self.lineDatabase[obj_idx][0]
                line4 = self.lineDatabase[obj_idx][4]
                line5 = self.lineDatabase[obj_idx][5]
                if not line0.locatePoint(point) and line4.locatePoint(point) and not line5.locatePoint(point):
                    constraint_pass = True

            elif region_id == "7":
                line2 = self.lineDatabase[obj_idx][2]
                line4 = self.lineDatabase[obj_idx][4]
                line5 = self.lineDatabase[obj_idx][5]
                if line2.locatePoint(point) and line4.locatePoint(point) and not line5.locatePoint(point):
                    constraint_pass = True

            elif region_id == "8":
                line0 = self.lineDatabase[obj_idx][0]
                line5 = self.lineDatabase[obj_idx][5]
                if not line0.locatePoint(point) and line5.locatePoint(point):
                    constraint_pass = True

            elif region_id == "9":
                line0 = self.lineDatabase[obj_idx][0]
                line1 = self.lineDatabase[obj_idx][1]
                line5 = self.lineDatabase[obj_idx][5]
                if line0.locatePoint(point) and not line1.locatePoint(point) and line5.locatePoint(point):
                    constraint_pass = True

            elif region_id == "10":
                line1 = self.lineDatabase[obj_idx][1]
                line2 = self.lineDatabase[obj_idx][2]
                line5 = self.lineDatabase[obj_idx][5]
                if line1.locatePoint(point) and not line2.locatePoint(point) and line5.locatePoint(point):
                    constraint_pass = True

            elif region_id == "11":
                line2 = self.lineDatabase[obj_idx][2]
                line5 = self.lineDatabase[obj_idx][5]
                if line2.locatePoint(point) and line5.locatePoint(point):
                    constraint_pass = True

            all_constraint_pass &= constraint_pass

        return all_constraint_pass

    def getBounds(self, obj_idx, region_id):
        obj_idx = int(obj_idx)
        min_x = 0
        max_x = 255
        min_y = 0
        max_y = 255

        if region_id == "0":
            line0 = self.lineDatabase[obj_idx][0]
            line3 = self.lineDatabase[obj_idx][3]
            x = line0.c / line0.a
            max_x = min(max_x, x)
            y = line3.c / line3.b
            max_y = min(max_y, y)

        elif region_id == "1":
            line0 = self.lineDatabase[obj_idx][0]
            line1 = self.lineDatabase[obj_idx][1]
            line3 = self.lineDatabase[obj_idx][3]
            y = line3.c / line3.b
            max_y = min(max_y, y)
            x = line1.c / line1.a
            max_x = min(max_x, x)
            x = line0.c / line0.a
            min_x = max(min_x, x)

        elif region_id == "2":
            line1 = self.lineDatabase[obj_idx][1]
            line2 = self.lineDatabase[obj_idx][2]
            line3 = self.lineDatabase[obj_idx][3]
            y = line3.c / line3.b
            max_y = min(max_y, y)
            x = line2.c / line2.a
            max_x = min(max_x, x)
            x = line1.c / line1.a
            min_x = max(min_x, x)

        elif region_id == "3":
            line2 = self.lineDatabase[obj_idx][2]
            line3 = self.lineDatabase[obj_idx][3]
            x = line2.c / line2.a
            min_x = max(min_x, x)
            y = line3.c / line3.b
            max_y = min(max_y, y)

        elif region_id == "4":
            line0 = self.lineDatabase[obj_idx][0]
            line3 = self.lineDatabase[obj_idx][3]
            line4 = self.lineDatabase[obj_idx][4]
            x = line0.c / line0.a
            max_x = min(max_x, x)
            y = line3.c / line3.b
            min_y = max(min_y, y)
            y = line4.c / line4.b
            max_y = min(max_y, y)

        elif region_id == "5":
            line2 = self.lineDatabase[obj_idx][2]
            line3 = self.lineDatabase[obj_idx][3]
            line4 = self.lineDatabase[obj_idx][4]
            x = line2.c / line2.a
            min_x = max(min_x, x)
            y = line3.c / line3.b
            min_y = max(min_y, y)
            y = line4.c / line4.b
            max_y = min(max_y, y)

        elif region_id == "6":
            line0 = self.lineDatabase[obj_idx][0]
            line4 = self.lineDatabase[obj_idx][4]
            line5 = self.lineDatabase[obj_idx][5]
            x = line0.c / line0.a
            max_x = min(max_x, x)
            y = line4.c / line4.b
            min_y = max(min_y, y)
            y = line5.c / line5.b
            max_y = min(max_y, y)

        elif region_id == "7":
            line2 = self.lineDatabase[obj_idx][2]
            line4 = self.lineDatabase[obj_idx][4]
            line5 = self.lineDatabase[obj_idx][5]
            x = line2.c / line2.a
            min_x = max(min_x, x)
            y = line4.c / line4.b
            min_y = max(min_y, y)
            y = line5.c / line5.b
            max_y = min(max_y, y)

        elif region_id == "8":
            line0 = self.lineDatabase[obj_idx][0]
            line5 = self.lineDatabase[obj_idx][5]
            x = line0.c / line0.a
            max_x = min(max_x, x)
            y = line5.c / line5.b
            min_y = max(min_y, y)

        elif region_id == "9":
            line0 = self.lineDatabase[obj_idx][0]
            line1 = self.lineDatabase[obj_idx][1]
            line5 = self.lineDatabase[obj_idx][5]
            y = line5.c / line5.b
            min_y = max(min_y, y)
            x = line1.c / line1.a
            max_x = min(max_x, x)
            x = line0.c / line0.a
            min_x = max(min_x, x)

        elif region_id == "10":
            line1 = self.lineDatabase[obj_idx][1]
            line2 = self.lineDatabase[obj_idx][2]
            line5 = self.lineDatabase[obj_idx][5]
            y = line5.c / line5.b
            min_y = max(min_y, y)
            x = line2.c / line2.a
            max_x = min(max_x, x)
            x = line1.c / line1.a
            min_x = max(min_x, x)

        elif region_id == "11":
            line2 = self.lineDatabase[obj_idx][2]
            line5 = self.lineDatabase[obj_idx][5]
            x = line2.c / line2.a
            min_x = max(min_x, x)
            y = line5.c / line5.b
            min_y = max(min_y, y)

        return int(min_x), int(max_x), int(min_y), int(max_y)

    def commonAreaExists(self, key1, key2):
        obj_idx1 = int(key1[0])
        reg_idx1 = key1[1:]
        obj_idx2 = int(key2[0])
        reg_idx2 = key2[1:]

        min_x1, max_x1, min_y1, max_y1 = self.getBounds(obj_idx1, reg_idx1)
        min_x2, max_x2, min_y2, max_y2 = self.getBounds(obj_idx2, reg_idx2)

        setx1, setx2 = set(range(min_x1, max_x1)), set(range(min_x2, max_x2))
        sety1, sety2 = set(range(min_y1, max_y1)), set(range(min_y2, max_y2))

        if len(setx1.intersection(setx2)) > 0 and len(sety1.intersection(sety2)) > 0:
            return True

        return False

    def makeRegions2(self, save=None, regionTreeDepth=3, relevantRegions=True):
        """
        returns region Tree formed using combination of regions made by different objects
        regionTree = {depth:0:
                        {
                            region_id0: [],
                            region_id1:[]
                        },
                      depth1:{...
                      }..
                    }
        """
        splitPoints = self.getSplitPoints()
        regionTree = {}

        for depth in range(regionTreeDepth):
            regionTree[depth + 1] = {}

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

            # adding region default keys (12 regions currently per object)
            regionTree[1][f"{i}0"] = []
            regionTree[1][f"{i}1"] = []
            regionTree[1][f"{i}2"] = []
            regionTree[1][f"{i}3"] = []
            regionTree[1][f"{i}4"] = []
            regionTree[1][f"{i}5"] = []
            regionTree[1][f"{i}6"] = []
            regionTree[1][f"{i}7"] = []
            regionTree[1][f"{i}8"] = []
            regionTree[1][f"{i}9"] = []
            regionTree[1][f"{i}10"] = []
            regionTree[1][f"{i}11"] = []

        unit_depth_keys = list(regionTree[1].keys())
        for depth in range(2, regionTreeDepth + 1):
            used_keys = []
            prev_depth_keys = list(regionTree[depth - 1].keys())
            for i in range(len(prev_depth_keys)):
                for j in range(len(unit_depth_keys)):
                    prev_depth_key = prev_depth_keys[i]
                    unit_depth_key = unit_depth_keys[j]

                    if relevantRegions and not self.commonAreaExists(prev_depth_key, unit_depth_key):
                        continue

                    # checking if constraint key is not already there
                    if unit_depth_key not in prev_depth_key:
                        new_key = min(prev_depth_key, unit_depth_key) + "_" + max(prev_depth_key, unit_depth_key)
                        new_key_item = sorted(new_key.split("_"))
                        if new_key_item not in used_keys:
                            regionTree[depth][new_key] = []
                            used_keys.append(new_key_item)

        ##################### Prefilling the region with all points is very time consuming ##############
        ########################### commenting below code to avoid prefilling #######################################
        # fill points in the regionTree
        # unique_constraints = set()
        # for depth in range(1, regionTreeDepth + 1):
        #     for constraint in list(regionTree[depth].keys()):
        #         keys = constraint.split("_")
        #         for key in keys:
        #             unique_constraints.add(key)
        #
        # point_constraint_table = {}  # stores which point satisfies which constraints
        # for x in range(self.width):
        #     for y in range(self.height):
        #         point = (x, y)
        #         point_constraint_table[point] = []
        #         for constraint in unique_constraints:
        #             if self.satisfy(point, constraint):
        #                 point_constraint_table[point].append(constraint)
        #
        # pbar = tqdm(total=self.width * self.height)
        # max_depth = max(regionTree.keys())
        # for x in range(self.width):
        #     for y in range(self.height):
        #         point = (x, y)
        #
        #         # for i in range(7140):
        #         #     pass
        #
        #         for depth in range(1, regionTreeDepth + 1):
        #             for constraint, region in regionTree[max_depth].items():
        #                 keys = constraint.split("_")
        #                 constraint_pass = True
        #                 for key in keys:
        #                     if key not in point_constraint_table[point]:
        #                         constraint_pass = False
        #                         break
        #
        #                 if constraint_pass:
        #                     region.append(point)
        #
        #         pbar.update(1)
        #
        # if save is not None:
        #     os.makedirs("./RegionTrees", exist_ok=True)
        #     with open(osp.join(f"./RegionTrees/{save}.json"), "w") as outfile:
        #         json.dump(regionTree, outfile, indent=2)

        # # flattening the regionTree structure
        for depth in range(1, regionTreeDepth + 1):
            for constraint, region in regionTree[depth].items():
                self.regions[f"depth-{depth}:{constraint}"] = region

        return regionTree

    def getRegionForPointWrtObject(self, point, obj_idx):
        region_marker = np.array([0 for i in range(12)])  # 1 if point belongs to region with region id = array idx.
        # 0 otherwise

        lines = self.lineDatabase[obj_idx]

        if not lines[0].locatePoint(point) and not lines[3].locatePoint(point):
            region_marker[0] = 1

        if lines[0].locatePoint(point) and not lines[1].locatePoint(point) \
                and not lines[3].locatePoint(point):
            region_marker[1] = 1

        if lines[1].locatePoint(point) and not lines[2].locatePoint(point) \
                and not lines[3].locatePoint(point):
            region_marker[2] = 1

        if lines[2].locatePoint(point) and not lines[3].locatePoint(point):
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

        if not lines[0].locatePoint(point) and lines[5].locatePoint(point):
            region_marker[8] = 1

        if lines[0].locatePoint(point) and not lines[1].locatePoint(point) \
                and lines[5].locatePoint(point):
            region_marker[9] = 1

        if lines[1].locatePoint(point) and not lines[2].locatePoint(point) \
                and lines[5].locatePoint(point):
            region_marker[10] = 1

        if lines[2].locatePoint(point) and lines[5].locatePoint(point):
            region_marker[11] = 1

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
            for i, line in enumerate(self.lineDatabase[obj_idx]):
                grid = self.makeLineOnGrid(grid, line)

        if visualise:
            plt.imshow(grid)
            plt.show()

        return grid

    def visualiseRegions(self, save=None):
        # check if regions have been made
        if len(self.allLines) == 0:
            print("Call makeRegions first")
            return

        grid = np.zeros((256, 256, 3))
        for region_name, region in self.regions.items():
            color = np.random.randint(0, 255, (3,)) / 255.
            for point in region:
                grid[point[1], point[0], :] = color

        if save is not None:
            plt.imsave(f"./experience_data/{save}_regions.png", grid)
            return

        plt.imshow(grid)
        plt.show()

    def randomSolve(self, sim, task_idx, task, save=None):
        solution_points = []
        non_solving_points = []
        initial_scene = phyre.observations_to_float_rgb(sim.initial_scenes[task_idx])
        cache = phyre.get_default_100k_cache('ball')
        actions = cache.action_array

        cache_list = actions[cache.load_simulation_states(task) == 1]

        solved = 0
        tries = 0
        max_tries = 65536
        print(f"Solving task-{task} randomly")
        pbar = tqdm(total=max_tries)
        actionlist = cache_list
        while tries < max_tries:
            tries += 1
            pbar.update(1)

            if len(actionlist) == 0:
                action = np.random.rand(3)
            else:
                action_idx = random.choice(range(len(actionlist)))
                action = actionlist[action_idx]
                actionlist = np.delete(actionlist, action_idx, axis=0)  # delete tried action

            try:
                res = sim.simulate_action(task_idx, action, need_featurized_objects=True, stride=1)
                features = res.featurized_objects.features
            except:
                continue

            initial_scene = self.makeGridForObject(initial_scene, obj_idxs=[0, 1, 2], visualise=False)

            x, y = res.featurized_objects.features[0][-1][0] * 255., \
                   (1 - res.featurized_objects.features[0][-1][1]) * 255.
            x, y = round(x), round(y)

            if res.status.is_solved() and not res.status.is_invalid():
                initial_scene[y, x, :] = np.array([1., 0., 0.])
                solution_points.append((x, y))

            else:
                non_solving_points.append((x, y))
                initial_scene[y, x, :] = np.array([0., 0., 1.])

        if save is not None:
            plt.imsave(f"./experience_data/{tasks_id}_solution_points.png", np.clip(initial_scene, 0., 1.).astype(np.float32))

        return solution_points, non_solving_points

    def getRegionArea(self, constraint):
        keys = constraint.split(":")[1].split("_")
        bounds = np.array([np.array(self.getBounds(key[0], key[1:])) for key in keys])
        min_x, max_x = np.max(bounds[:, 0]), np.min(bounds[:, 1])
        min_y, max_y = np.max(bounds[:, 2]), np.min(bounds[:, 3])

        if max_x < min_x or max_y < min_y:
            return 0

        area = (max_x - min_x) * (max_y - min_y)

        if area >= 0:
            return area

        return 0

    def getRegionProperties(self, solution_points, non_solving_points):
        if len(self.allLines) == 0:
            self.makeRegions()

        region_properties = {}

        most_dense_region = list(self.regions.keys())[0]
        max_density = 0
        pbar = tqdm(total=len(self.regions.items()))
        for region_name, region in self.regions.items():
            num_solving_points = 0
            total_points = self.getRegionArea(region_name)

            if total_points == 0:
                continue

            for point in set(solution_points):
                keys = region_name.split(":")[1].split("_")
                constraint_pass = True
                for key in keys:
                    if not self.satisfy(point, key):
                        constraint_pass = False
                        break

                if constraint_pass:
                    num_solving_points += 1

            density = 0. if total_points == 0 else num_solving_points / total_points
            region_properties[region_name] = [density, num_solving_points, total_points]

            if density > max_density:
                most_dense_region = region_name
                max_density = density

            pbar.update(1)

        region_properties = dict(sorted(region_properties.items(), key=operator.itemgetter(1), reverse=True))
        return region_properties, most_dense_region

    def gridSolve(self, sim, task_idx, task, proposed_region, visualise=False):
        if proposed_region not in self.regions.keys():
            print("Proposed region not found in the grid")
            return False

        region_space = self.regions[proposed_region]
        initial_scene = phyre.observations_to_float_rgb(sim.initial_scenes[task_idx])
        initial_scene = self.makeGridForObject(initial_scene, obj_idxs=[0, 1, 2], visualise=True)

        if visualise:
            for point in region_space:
                initial_scene[point[1], point[0], :] = np.array([255, 0., 0.])
            plt.imshow(initial_scene)
            plt.show()

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


def makeRegionEncodings(src_dir, task_ids, sim):
    data_dir = osp.join(src_dir, "experience_data")
    cluster_names = [i for i in os.listdir(osp.join(src_dir, 'clusters'))]
    cluster_dirs = [osp.join(src_dir, 'clusters', str(cluster_name)) for cluster_name in cluster_names]
    csv_files = [osp.join(data_dir, f"{task_id}.csv") for task_id in task_ids]

    regions = []

    gridSolver = GridSolver(sim.initial_featurized_objects[0])
    regionTree = gridSolver.makeRegions2(save=None, relevantRegions=False)
    regionTreeDepth = len(regionTree)

    for depth in range(1, regionTreeDepth + 1):
        for constraint, region in regionTree[depth].items():
            regions.append(f"depth-{depth}:{constraint}")

    regions = sorted(regions)

    taskEncodings = {key: np.zeros((len(regions),)) for key in task_ids}

    for task_id, csv_file in zip(task_ids, csv_files):
        df = pd.read_csv(csv_file)
        cur_regions = list(df.values[:, 1])

        encoding = np.zeros(len(regions), )

        for i, region in enumerate(regions):
            if region in cur_regions:
                encoding[i] = 1

        taskEncodings[task_id] = encoding

    cluster_data = {}

    for cluster_name, cluster_dir in zip(cluster_names, cluster_dirs):
        cluster_tasks = [i.split('_')[0] for i in os.listdir(cluster_dir) if i != "solution_regions.txt"]

        # remove the if part: only for debugging
        # if cluster_name == '3':
        #     cluster_tasks += ['00002:101', '00002:105', '00002:111', '00002:176']

        pos_encodings = np.array([np.array(taskEncodings[task_id]) for task_id in cluster_tasks])
        neg_encodings = np.array([np.array(taskEncodings[task_id]) for task_id in list(taskEncodings.keys())
                                  if task_id not in cluster_tasks])

        cluster_data[cluster_name] = {"pos": pos_encodings, "neg": neg_encodings}

    return cluster_data


def formTemplateClusters(src_dir, task_ids, sim, thresh=0.4):
    data_dir = osp.join(src_dir, "experience_data")
    csv_files = [osp.join(data_dir, f"{task_id}.csv") for task_id in task_ids]
    regions = []

    gridSolver = GridSolver(sim.initial_featurized_objects[0])
    regionTree = gridSolver.makeRegions2(save=None, relevantRegions=False)
    regionTreeDepth = len(regionTree)

    for depth in range(1, regionTreeDepth + 1):
        for constraint, region in regionTree[depth].items():
            regions.append(f"depth-{depth}:{constraint}")

    regions = sorted(regions)

    taskEncodings = {key: np.zeros((len(regions),)) for key in task_ids}

    for task_id, csv_file in zip(task_ids, csv_files):
        df = pd.read_csv(csv_file)
        cur_regions = list(df.values[:, 1])

        encoding = np.zeros(len(regions), )

        for i, region in enumerate(regions):
            if region in cur_regions:
                prob = df[df.values[:, 1] == region].values[:, 2][0]
                encoding[i] = 1 if prob >= thresh else 0

        taskEncodings[task_id] = encoding

    clusters = {}
    for task_id, encoding in taskEncodings.items():
        skey = ''.join([str(int(i)) for i in encoding])
        if skey in clusters.keys():
            clusters[skey].append(task_id)
        else:
            clusters[skey] = [task_id]

    # refilling clusters with tasks which have atleast one solution region common to encoding
    for task_id, encoding in taskEncodings.items():
        skey = ''.join([str(int(i)) for i in encoding])
        solution_idxs_task = set([i for i in range(len(skey)) if skey[i] == '1'])

        for cluster_key, task_ids in clusters.items():
            solution_idxs_cluster = set([i for i in range(len(cluster_key)) if cluster_key[i] == '1'])
            if len(solution_idxs_cluster.intersection(solution_idxs_task)) > 0 and task_id not in task_ids:
                task_ids.append(task_id)

    # move images to respective cluster folders
    clusters_dir = osp.join(src_dir, "clusters")
    if osp.exists(clusters_dir):
        shutil.rmtree(clusters_dir)
    os.makedirs(clusters_dir, exist_ok=True)

    cluster_id = 1
    for cluster_key, task_ids in clusters.items():
        os.makedirs(osp.join(clusters_dir, str(cluster_id)))
        solution_idxs = [i for i in range(len(cluster_key)) if cluster_key[i] == '1']
        with open(osp.join(clusters_dir, str(cluster_id), 'solution_regions.txt'), 'w') as f:
            if len(solution_idxs) == 0:
                print(f"[WARNING!] Found cluster with no solution region! - cluster {cluster_id}")
            for solution_idx in solution_idxs:
                f.write(regions[solution_idx])
                f.write('\n')

        f.close()
        for task_id in task_ids:
            shutil.copy(osp.join(data_dir, f"{task_id}_solution_points.png"),
                        osp.join(clusters_dir, str(cluster_id), f"{task_id}_solution_points.png"))

        cluster_id += 1


if __name__ == '__main__':
    eval_setup = 'ball_cross_template'
    fold_id = 0
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    tasks = train_tasks + dev_tasks + test_tasks

    templates = [2]
    templates = [str(i).zfill(5) for i in templates]

    tasks_ids = sorted([x for x in tasks if x.startswith(tuple(templates))])[:]

    sim = phyre.initialize_simulator(tasks_ids, 'ball')

    # for i, tasks_id in enumerate(tasks_ids):
    #     gridSolver = GridSolver(sim.initial_featurized_objects[i])
    #     regionTree = gridSolver.makeRegions2(save=tasks_id)
    #     grid = gridSolver.makeGridForObject(phyre.observations_to_float_rgb(sim.initial_scenes[i]), obj_idxs=[0, 1, 2],
    #                                         visualise=True)
    #     plt.imsave(f"./experience_data/{tasks_id}_grid.png", np.clip(grid, 0., 1.))
    #     # continue
    #     gridSolver.visualiseRegions(save=tasks_id)
    #     solution_points, non_solving_points = gridSolver.randomSolve(sim, task_idx=i, task=tasks_id, save=tasks_id)
    #     region_properties, solution_region = gridSolver.getRegionProperties(solution_points, non_solving_points)
    #
    #     regions_table = [[] for j in range(len(region_properties))]
    #     j = 0
    #     for region_name, region_property in region_properties.items():
    #         regions_table[j].append(region_name)
    #         regions_table[j].append(region_property[0])
    #         regions_table[j].append(region_property[1])
    #         regions_table[j].append(region_property[2])
    #         j += 1
    #
    #     regions_table = pd.DataFrame(data=regions_table, columns=['Region Names', 'Density of Solution Points',
    #                                                               '# Solving Points',
    #                                                               'Total Points in Region'])
    #     regions_table.to_csv(f"./experience_data/{tasks_id}.csv")

    dir = "/home/dhruv/Desktop/PhySolve/engine/"
    formTemplateClusters(dir, tasks_ids, sim)
    # cluster_data = makeRegionEncodings(dir, tasks_ids, sim)

    # regionDiscriminator = RegionDiscriminator('3', cluster_data)
    # regionDiscriminator.train()
    #
    # gridSolver2 = GridSolver(sim.initial_featurized_objects[1])
    # gridSolver2.makeRegions()
    # gridSolver2.visualiseRegions()
    # solved = gridSolver2.gridSolve(sim, 1, tasks_ids[1], solution_region, visualise=True)
