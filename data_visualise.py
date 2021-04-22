import pickle
import numpy as np
import phyre
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from math import sqrt

# Task: 00002  image channels ---> channel 0 = red object
#                                  channel 1 = green object
#                    features ---> channel 1 = green object
#                                  channel 3 = red object

# Task: 00015  image channels ---> channel 0 = red object
#                                  channel 1 = green object
#                    features ---> channel 3 = green object
#                                  channel 4 = red object

red_feature_database = []
green_feature_database = []
h_patch = w_patch = 64
h = w = 256

red_idx = 4
green_idx = 3

with open('DataCollection/database_task15.pkl', 'rb') as handle:
    database = pickle.load(handle)

channels = range(1, 7)
for task, data in database.items():
    frames = data['images']
    features = data['features']
    collision_timestep = data['collision_timestep']

    collision_frame = len(frames) // 2

    # img = phyre.observations_to_float_rgb(frames[0])

    obj_channels = np.array([np.array([(frame == ch).astype(float) for ch in channels]) for frame in frames])
    obj_channels = np.flip(obj_channels, axis=2)

    red_x = features[collision_frame][red_idx][0] - features[collision_frame][green_idx][0]
    red_y = -1 * (features[collision_frame][red_idx][1] - features[collision_frame][green_idx][1])
    red_r = features[collision_frame][red_idx][3] / 2
    red_dx = features[collision_frame][red_idx][0] - features[collision_frame - 1][red_idx][0]
    red_dy = -1 * (features[collision_frame][red_idx][1] - features[collision_frame - 1][red_idx][1])

    green_dx = features[collision_frame][green_idx][0] - features[collision_frame - 1][green_idx][0]
    green_dy = -1 * (features[collision_frame][green_idx][1] - features[collision_frame - 1][green_idx][1])
    green_x = features[collision_frame][green_idx][0]
    green_y = 1 - features[collision_frame][green_idx][1]
    green_r = features[collision_frame][green_idx][3] / 2

    patch = phyre.vis.observations_to_float_rgb(frames[collision_frame])
    patch = patch[round(max(0, green_y * h - h_patch // 2)): round(min(h, green_y * h + h_patch // 2)),
            round(max(0, green_x * w - w_patch // 2)): round(min(w, green_x * w + w_patch // 2))]

    red_feature_database.append([red_x, red_y, red_r, red_dx, red_dy])
    green_feature_database.append([green_dx, green_dy, collision_timestep, green_r, patch])

    patch = frames[collision_frame]

red_feature_database = np.array(red_feature_database)
green_feature_database = np.array(green_feature_database)

# pca = PCA(n_components=2)
# compressed_feature_database = pca.fit_transform(feature_database)
#
# kmeans = KMeans(n_clusters=10, random_state=0)
# predict the labels of clusters.
# label = kmeans.fit_predict(compressed_feature_database)
#
# Getting unique labels
# u_labels = np.unique(label)
#
# plotting the results:
#
# for i in u_labels:
#     plt.scatter(compressed_feature_database[label == i, 0], compressed_feature_database[label == i, 1], label=i)
# plt.legend()
# plt.show()

# Heatmap for position coords
# plt.hist2d(feature_database[:, 0] * 256, feature_database[:, 1] * 256,
#            bins=[np.arange(-50, 50, 1), np.arange(-50, 50, 1)],
#            cmap=plt.cm.BuPu)
# plt.colorbar()
# plt.show()
#
# Hist for radius
# plt.hist(feature_database[:, 2] * 256, bins=10)
# plt.show()
#
# Heatmap for velocities
# plt.hist2d(feature_database[:, 3] * 256, feature_database[:, 4] * 256,
#            bins=[np.arange(-2, 8, 0.2), np.arange(-2, 8, 0.2)],
#            cmap=plt.cm.BuPu)
# plt.colorbar()
# plt.show()

# Visualising data
samples = 100
num_rows = num_cols = int(sqrt(samples))
indices = np.random.randint(red_feature_database.shape[0], size=(samples))
h_gap = 80
v_gap = 50
grid = np.zeros((h_patch * num_rows + v_gap * num_rows, (w_patch * num_cols + h_gap * num_cols), 3))

h_start = w_start = 0

cnt = 0
for green_data in green_feature_database[indices]:
    patch = green_data[-1]
    grid[h_start: h_start + patch.shape[0], w_start: w_start + patch.shape[1], :] = patch

    w_start += (w_patch + h_gap)
    cnt += 1

    if cnt == 10:
        cnt = 0
        w_start = 0
        h_start += (h_patch + v_gap)

img = Image.fromarray(np.uint8(grid*255))
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", 10)

cnt = 0
h_start = w_start = 0
for red_data, green_data in zip(red_feature_database[indices], green_feature_database[indices]):

    green_dx, green_dy, t, green_r, _ = green_data
    green_dx, green_dy, green_r = round(green_dx*255, 2), round(green_dy*255, 2), round(green_r*255, 2)

    red_x, red_y, red_r, red_dx, red_dy = red_data
    red_x, red_y, red_r, red_dx, red_dy = round(red_x*255, 2), round(red_y*255, 2), round(red_r*255, 2), \
                                          round(red_dx*255, 2), round(red_dy*255, 2)

    draw.text((w_start, h_start+h_patch+2), f"dx={green_dx} dy={green_dy} r={green_r} t={t}", font=font, fill=(0, 255, 0, 0))

    draw.text((w_start, h_start + h_patch + 12), f"x={red_x} y={red_y} r={red_r}", font=font,
              fill=(255, 0, 0, 0))
    draw.text((w_start, h_start + h_patch + 22), f"dx = {red_dx} dy = {red_dy}", font=font,
              fill=(255, 0, 0, 0))

    w_start += (w_patch + h_gap)
    cnt += 1

    if cnt == 10:
        cnt = 0
        w_start = 0
        h_start += (h_patch + v_gap)

img.show()
img.save('./samples_task15.png')



