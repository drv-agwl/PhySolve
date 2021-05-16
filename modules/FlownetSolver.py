import phyre
import torch
from torch import nn
import numpy as np
import torch as T
import pickle
import torch.nn.functional as F
from utils.phyre_utils import vis_pred_path_task
import os
import cv2
from modules.data_utils import load_data_position, load_data_collision
from PIL import Image, ImageDraw
from modules.phyre_utils import simulate_action
from tqdm import tqdm
from modules.data_utils import draw_ball
import pandas as pd


class PosModelDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        images = self.data[idx]["Images"]
        collision_time = self.data[idx]["Collision_time"]
        red_diam = self.data[idx]["Red_diam"]
        task_id = self.data[idx]["task-id"]

        return images, collision_time, red_diam, task_id

    def __len__(self):
        return len(self.data)


class CollisionDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        images = self.data[idx]["Images"]
        red_diam = self.data[idx]["Red_diam"]
        task_id = self.data[idx]["task-id"]

        return images, red_diam, task_id

    def __len__(self):
        return len(self.data)


class CollisionDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        images = self.data[idx]["Images"]
        red_diam = self.data[idx]["Red_diam"]
        task_id = self.data[idx]["task-id"]

        return images, red_diam, task_id

    def __len__(self):
        return len(self.data)


class Pyramid2(nn.Module):
    def __init__(self, in_dim, chs):
        super().__init__()

        self.encoder = nn.Sequential(nn.Conv2d(in_dim, 8, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(8, 16, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(16, 32, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 4, 2, 1),
                                     nn.ReLU())

        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(nn.Linear(64 * 1 * 1 + 1, 64 * 1 * 1),
                                   nn.ReLU())

        self.dense_down = nn.Sequential(nn.Linear(1 * 1 * 128, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 8),
                                        nn.ReLU(),
                                        nn.Linear(8, 2),
                                        nn.ReLU(),
                                        nn.Linear(2, 1),
                                        nn.ReLU(),
                                        )

        self.dense_up = nn.Sequential(nn.Linear(2, 8),
                                      nn.ReLU(),
                                      nn.Linear(8, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 2 * 2 * 128),
                                      nn.ReLU())

        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, 16, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(16, 8, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(8, chs, 4, 2, 1),
                                     nn.Sigmoid())

    def forward(self, x, time):
        x = torch.cat([x, time], dim=1)
        x = self.encoder(x)
        x = self.flatten(x)

        b = x.size(0)
        x = x.view(b, 64, 1, 1)
        x = self.decoder(x)

        return x


class Pyramid(nn.Module):
    def __init__(self, in_dim, chs):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_dim, 8, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(8, 16, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(16, 32, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 4, 2, 1),
                                     nn.ReLU())

        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, 16, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(16, 8, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(8, chs, 4, 2, 1),
                                     nn.Sigmoid())
        self.flatten = nn.Flatten()
        self.radius_head = nn.Sequential(nn.Linear(64, 32),
                                         nn.ReLU(),
                                         nn.Linear(32, 1),
                                         nn.ReLU())

    def forward(self, X):
        x = self.encoder(x)
        r = self.radius_head(self.flatten(x))
        x = self.decoder(x)

        return x, r


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


class FlownetSolver:
    def __init__(self, seq_len, device):
        super().__init__()
        self.device = ("cuda" if T.cuda.is_available() else "cpu") if device == "cuda" else "cpu"
        print("device:", self.device)
        self.seq_len = seq_len
        self.logger = dict()

        self.collision_model = Pyramid(seq_len * 2 + seq_len // 2 + 2, 1)
        self.position_model = Pyramid2(4, 1)

        print("succesfully initialized models")

    def train_collision_model(self, data_paths, epochs=100, width=128, batch_size=32, smooth_loss=False):
        if self.device == "cuda":
            self.collision_model.cuda()

        size = (width, width)

        train_loss_log = []
        val_loss_log = []

        opti = T.optim.Adam(self.collision_model.parameters(recurse=True), lr=3e-4)
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(opti, 'min', patience=5, verbose=True)

        train_data, test_data = load_data_collision(data_paths, self.seq_len, size)

        train_data_loader = T.utils.data.DataLoader(CollisionDataset(train_data),
                                                    batch_size, shuffle=True)
        test_data_loader = T.utils.data.DataLoader(CollisionDataset(test_data),
                                                   batch_size, shuffle=True)

        # Load model from ckpt
        # self.collision_model.load_state_dict(T.load("./checkpoints/CollisionModel/100.pt"))

        rows = []
        pic_no = 1
        for epoch in range(epochs):
            print("Training")
            losses = []
            for i, batch in enumerate(train_data_loader):
                X_image = batch[0].float().to(self.device)
                radius = batch[1].float().to(self.device) / 2.

                num_steps = self.seq_len // 2 + 1
                model_input = X_image[:, :2 * self.seq_len + 1 + num_steps]
                red_ball_gt = X_image[:, 2 * self.seq_len + 1 + num_steps:]

                red_ball_preds = []
                for timestep in range(1):
                    red_ball_pred, pred_radius = self.collision_model(model_input)
                    red_ball_preds.append(red_ball_pred)

                    if smooth_loss:
                        red_ball_gt += 0.005
                        red_ball_gt = torch.clamp(red_ball_gt, 0., 1.)

                    loss_ball = F.binary_cross_entropy(red_ball_pred[:, 0], red_ball_gt[:, timestep])
                    loss_rad = F.mse_loss(radius, pred_radius)
                    loss = loss_ball + loss_rad
                    losses.append(loss.item())

                    opti.zero_grad()
                    loss.backward()
                    opti.step()

                    model_input[:, 2 * self.seq_len + 1 + timestep] = red_ball_gt[:, timestep]

                # Visualisation
                row = []

                green_ball_collision = X_image[-1, :self.seq_len][self.seq_len // 2][:, :, None].cpu()
                red_ball_collision = X_image[-1, 2 * self.seq_len + 1][:, :, None].cpu()
                static_objs = X_image[-1, 2 * self.seq_len][:, :, None].cpu()

                collision_scene = np.concatenate([red_ball_collision, green_ball_collision, static_objs], axis=-1)

                row.append(collision_scene)

                empty_channel = np.zeros_like(static_objs)
                for timestep in range(1):
                    red_ball_pred_channel = red_ball_preds[timestep][-1, 0][:, :, None].detach().cpu()
                    pred_scene = np.concatenate([red_ball_pred_channel, empty_channel, empty_channel], axis=-1)

                    row.append(pred_scene)

                rows.append(row)

                if i % 5 == 0:
                    print(f"Epoch-{epoch}, iteration-{i}: Loss = {loss.item()}")

                if len(rows) == 5:
                    os.makedirs(f"./results/train/CollisionModel/{epoch + 1}", exist_ok=True)
                    save_img_dir = f"./results/train/CollisionModel/{epoch + 1}/"
                    vis_pred_path_task(rows, save_img_dir, pic_no)
                    pic_no += 1
                    rows = []

            pic_no = 1

            train_loss_log.append(sum(losses) / len(losses))

            losses = []
            rows = []
            print("Validation")
            os.makedirs("./checkpoints/CollisionModel", exist_ok=True)
            T.save(self.collision_model.state_dict(), f"./checkpoints/CollisionModel/{epoch + 1}.pt")
            for i, batch in enumerate(test_data_loader):
                X_image = batch[0].float().to(self.device)

                num_steps = self.seq_len // 2 + 1
                model_input = X_image[:, :2 * self.seq_len + 1 + num_steps]
                red_ball_gt = X_image[:, 2 * self.seq_len + 1 + num_steps:]

                red_ball_preds = []
                for timestep in range(num_steps):
                    red_ball_pred = self.collision_model(model_input)
                    red_ball_preds.append(red_ball_pred)

                    loss = F.binary_cross_entropy(red_ball_pred[:, 0], red_ball_gt[:, timestep])
                    losses.append(loss.item())

                    model_input[:, 2 * self.seq_len + 1 + timestep] = red_ball_gt[:, timestep]

                # Visualisation
                row = []

                green_ball_collision = X_image[-1, :self.seq_len][self.seq_len // 2][:, :, None].cpu()
                red_ball_collision = X_image[-1, 2 * self.seq_len + 1][:, :, None].cpu()
                static_objs = X_image[-1, 2 * self.seq_len][:, :, None].cpu()

                collision_scene = np.concatenate([red_ball_collision, green_ball_collision, static_objs], axis=-1)

                row.append(collision_scene)

                empty_channel = np.zeros((64, 64, 1))
                red_ball_movements = []
                for timestep in range(1, num_steps):
                    red_ball_movement = rescale(red_ball_preds[timestep][-1, 0][:, :, None].detach().cpu() - \
                                                red_ball_preds[0][-1, 0][:, :, None].detach().cpu())
                    red_ball_movements.append(
                        np.concatenate([red_ball_movement, empty_channel, empty_channel], axis=-1))

                for timestep in range(num_steps):
                    red_ball_pred_channel = red_ball_preds[timestep][-1, 0][:, :, None].detach().cpu()
                    pred_scene = np.concatenate([red_ball_pred_channel, green_ball_collision, static_objs], axis=-1)
                    row.append(pred_scene)
                    if timestep > 0:
                        row.append(red_ball_movements[timestep - 1])

                green_ball_solved = X_image[-1, :self.seq_len].cpu()
                green_ball_unsolved = X_image[-1, self.seq_len: 2 * self.seq_len].cpu()
                green_ball_paths = np.max(np.concatenate([green_ball_solved, green_ball_unsolved], axis=0), axis=0)[:,
                                   :, None]

                row.insert(1, np.concatenate([empty_channel, green_ball_paths, empty_channel], axis=-1))

                rows.append(row)

                if i % 5 == 0:
                    print(f"Epoch-{epoch}, iteration-{i}: Loss = {loss.item()}")

                if len(rows) == 5 or len(rows) == 7:
                    os.makedirs(f"./results/test/CollisionModel/{epoch + 1}", exist_ok=True)
                    save_img_dir = f"./results/test/CollisionModel/{epoch + 1}/"
                    vis_pred_path_task(rows, save_img_dir, pic_no)
                    pic_no += 1
                    rows = []

            val_loss = sum(losses) / len(losses)
            val_loss_log.append(val_loss)
            scheduler.step(val_loss)

            pic_no = 1

        os.makedirs("./logs/CollisionModel", exist_ok=True)
        with open(f"./logs/CollisionModel/{epochs}.pkl", "wb") as f:
            pickle.dump({"Train losses": train_loss_log,
                         "Test losses": val_loss_log}, f)

    def make_visualisations(self, data_loader):
        rows = []
        pic_no = 1

        for i, batch in enumerate(data_loader):
            X_image = batch[0].float().to(self.device)
            X_time = batch[1].float().to(self.device) / 33.  # divide by max value of time

            scenes = X_image[:, 3:-1].detach().cpu().numpy()

            num_times = 3
            X_times = [X_time * i / num_times for i in range(1, num_times + 1)]
            red_ball_preds = []

            for X_time in X_times:
                model_input = X_image[:, :3], X_time
                red_ball_pred = self.position_model(model_input[0], model_input[1])
                red_ball_preds.append(red_ball_pred)

            # Visualisation
            for idx in range(X_image.shape[0]):
                row = []

                green_ball_collision = X_image[idx, 0][:, :, None].cpu()
                red_ball_collision = X_image[idx, 1][:, :, None].cpu()
                static_objs = X_image[idx, 2][:, :, None].cpu()

                collision_scene = np.concatenate([red_ball_collision, green_ball_collision, static_objs], axis=-1)

                row.append(collision_scene)

                # pred_channel = collision_scene.copy()
                # pred_channel[..., 0] = np.max(np.concatenate([pred_channel[..., 0][..., None],
                #                                               red_ball_gt.detach().cpu()[idx, ..., None]], axis=-1),
                #                               axis=-1)
                # # pred_channel[..., 1] = np.max(np.concatenate([pred_channel[..., 1][..., None],
                # #                                               red_ball_gt.detach().cpu()[idx, ..., None]], axis=-1),
                # #                               axis=-1)
                # pred_channel[..., 2] = np.max(np.concatenate([pred_channel[..., 2][..., None],
                #                                               red_ball_preds[-1].detach().cpu()[idx, 0, ..., None]],
                #                                              axis=-1),
                #                               axis=-1)
                # row.append(pred_channel)

                for j, red_ball_pred in enumerate(reversed(red_ball_preds)):
                    pred_channel = scenes[idx, j * 4: (j + 1) * 4, ...].copy()
                    pred_channel = np.moveaxis(pred_channel, 0, -1)
                    pred_channel[..., 2] = np.max(np.concatenate([pred_channel[..., 2][..., None],
                                                                  red_ball_pred.detach().cpu()[idx, 0, ..., None]],
                                                                 axis=-1),
                                                  axis=-1)
                    row.append(pred_channel)

                rows.append([row[0]] + row[::-1][:-1])

                if len(rows) == 20:
                    os.makedirs(f"./results/test/PositionModel/visualisation", exist_ok=True)
                    save_img_dir = f"./results/test/PositionModel/visualisation"
                    vis_pred_path_task(rows, save_img_dir, pic_no)
                    pic_no += 1
                    rows = []

    def get_position_pred(self, pred_channel, diam):
        diam = round(diam[0] * self.width)
        kernel = Image.new('1', (diam, diam))
        draw = ImageDraw.Draw(kernel)
        draw.ellipse((0, 0, diam - 1, diam - 1), fill=1)
        kernel = np.array(kernel).astype(np.float)

        pred_channel = pred_channel.detach().cpu().numpy()[0][0]

        filtered = cv2.filter2D(src=pred_channel, kernel=kernel, ddepth=-1)

        pred_y, pred_x = np.unravel_index(np.argmax(filtered, axis=None), filtered.shape)

        return pred_x, pred_y

    def get_collision_model_preds(self, checkpoint, data_paths, batch_size=32):
        if self.device == "cuda":
            self.collision_model.cuda()

        size = (self.width, self.width)

        self.collision_model.load_state_dict(T.load(checkpoint))

        data = load_data_collision(data_paths, self.seq_len, size, all_samples=True)

        data_loader = T.utils.data.DataLoader(CollisionDataset(data),
                                              batch_size, shuffle=False)

        for i, batch in enumerate(data_loader):
            X_image = batch[0].float().to(self.device)
            X_red_diam = batch[1].float().to(self.device)

            num_steps = self.seq_len // 2 + 1
            model_input = X_image[:, :2 * self.seq_len + 1 + num_steps]

            red_ball_preds = []
            for timestep in range(1):
                red_ball_pred = self.collision_model(model_input)
                red_ball_preds.append(red_ball_pred)

            pred_y, pred_x = self.get_position_pred(red_ball_preds[0], X_red_diam.cpu().numpy())

    def simulate_combined(self, collision_ckpt, position_ckpt, data_paths, batch_size=32,
                          save_rollouts_dir="/home/dhruv/Desktop/PhySolve/results/saved_rollouts"):
        if self.device == "cuda":
            self.collision_model.cuda().eval()
            self.position_model.cuda().eval()

        size = (self.width, self.width)

        self.collision_model.load_state_dict(T.load(collision_ckpt))
        self.position_model.load_state_dict(T.load(position_ckpt))

        data_collision = load_data_collision(data_paths, self.seq_len, size, all_samples=True, shuffle=False)
        data_position = load_data_position(data_paths, self.seq_len, size, all_samples=True, shuffle=False)

        collision_data_loader = T.utils.data.DataLoader(CollisionDataset(data_collision),
                                                        batch_size, shuffle=False)
        position_data_loader = T.utils.data.DataLoader(PosModelDataset(data_position),
                                                       batch_size, shuffle=False)

        task_idxs, tasks = [], []
        id = 0
        metrics_table = {}

        for batch in position_data_loader:
            task_id = batch[3][0]
            if task_id not in tasks:
                task_idxs.append(id)
                tasks.append(task_id)
                id += 1

                metrics_table[task_id.split(':')[0]] = [[0, 0], [0, 0]]

        sim = phyre.initialize_simulator(tasks, 'ball')

        num_solved = num_collided = 0
        pbar = tqdm(total=len(tasks))

        tasks = []
        doubt_ids = []

        id = 0
        for batch_collision, batch_position in zip(collision_data_loader, position_data_loader):
            assert batch_position[-1] == batch_collision[-1]
            task_id = batch_position[3][0]

            template = task_id.split(":")[0]

            if task_id in tasks:
                continue

            tasks.append(task_id)

            # Collision model prediction
            X_image = batch_collision[0].float().to(self.device)
            X_red_diam = batch_collision[1].float().to(self.device)

            num_steps = self.seq_len // 2 + 1
            model_input = X_image[:, :2 * self.seq_len + 1 + num_steps]

            red_ball_preds = []
            for timestep in range(1):
                red_ball_pred = self.collision_model(model_input)
                red_ball_preds.append(red_ball_pred)

            pred_x, pred_y = self.get_position_pred(red_ball_preds[0], X_red_diam.cpu().numpy())
            red_channel_collision = draw_ball(size, pred_y, pred_x,
                                              X_red_diam.cpu().numpy() * size[0] / 2.)  # Output of collision model

            # Position Model
            X_image = batch_position[0].float().to(self.device)
            X_time = batch_position[1].float().to(self.device) / 109.6  # divide by max value of time

            X_time = X_time[:, None, None].repeat(1, self.width, self.width)

            model_input = X_image[:, :3], X_time[:, None]
            model_input[0][:, 1] = torch.tensor(red_channel_collision).to(self.device)  # Replace ground truth with
            # collision model prediction

            red_ball_pred = self.position_model(model_input[0], model_input[1])

            pred_y, pred_x = self.get_position_pred(red_ball_pred, X_red_diam.cpu().numpy())
            collided, solved = simulate_action(sim, id, tasks[id],
                                               pred_y / (self.width - 1.), 1. - pred_x / (self.width - 1.),
                                               X_red_diam / 2., num_attempts=10, save_rollouts_dir=save_rollouts_dir)

            if collided:
                num_collided += 1
                metrics_table[template][0][0] += 1
            if solved:
                num_solved += 1
                metrics_table[template][1][0] += 1
            if not collided and solved:
                doubt_ids.append(task_id)

            metrics_table[template][0][1] += 1
            metrics_table[template][1][1] += 1
            id += 1
            pbar.update(1)

        pbar.close()
        print("Overall: ", num_collided * 100. / len(tasks), " ", num_solved * 100. / len(tasks))
        print()

        success = []
        for template, val in metrics_table.items():
            row = [template, round(val[0][0] * 100. / val[0][1], 2), round(val[1][0] * 100. / val[1][1], 2)]
            success.append(row)

        df = pd.DataFrame(success, columns=["Template", "Collision success", "Solved success"])
        df.to_csv("./success_combined.csv", index=False)

    def simulate_position_model(self, checkpoint, data_paths, batch_size=32):
        if self.device == "cuda":
            self.position_model.cuda()

        size = (self.width, self.width)

        self.position_model.load_state_dict(T.load(checkpoint))

        data = load_data_position(data_paths, self.seq_len, size, all_samples=True)

        data_loader = T.utils.data.DataLoader(PosModelDataset(data),
                                              batch_size, shuffle=False)

        task_idxs, tasks = [], []
        id = 0
        metrics_table = {}

        for batch in data_loader:
            task_id = batch[3][0]
            if task_id not in tasks:
                task_idxs.append(id)
                tasks.append(task_id)
                id += 1

                metrics_table[task_id.split(':')[0]] = [0, 0]

        sim = phyre.initialize_simulator(tasks, 'ball')

        num_solved = 0
        pbar = tqdm(total=len(tasks))

        tasks = []
        id = 0
        for batch in data_loader:
            task_id = batch[3][0]
            template = task_id.split(":")[0]

            if task_id in tasks:
                continue

            tasks.append(task_id)

            X_image = batch[0].float().to(self.device)
            X_time = batch[1].float().to(self.device) / 109.6
            X_red_diam = batch[2].float().to(self.device)  # divide by max value of time

            X_time = X_time[:, None, None].repeat(1, self.width, self.width)

            model_input = X_image[:, :3], X_time[:, None]

            red_ball_pred = self.position_model(model_input[0], model_input[1])

            pred_y, pred_x = self.get_position_pred(red_ball_pred, X_red_diam.cpu().numpy())

            solved = simulate_action(sim, id,
                                     pred_y / (self.width - 1.), 1. - pred_x / (self.width - 1.),
                                     X_red_diam / 2.)

            if solved:
                num_solved += 1
                metrics_table[template][0] += 1

            metrics_table[template][1] += 1
            id += 1
            pbar.update(1)

        pbar.close()
        print("Overall: ", num_solved * 100. / len(tasks))
        print()

        for template, val in metrics_table.items():
            print(template, ": ", val[0] * 100. / val[1])

    def train_position_model(self, data_paths, epochs=100, width=64, batch_size=32, smooth_loss=False):
        if self.device == "cuda":
            self.position_model.cuda()

        size = (width, width)

        train_loss_log = []
        val_loss_log = []

        opti = T.optim.Adam(self.position_model.parameters(recurse=True), lr=3e-4)
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(opti, 'min', patience=5, verbose=True)

        train_data, test_data = load_data_position(data_paths, self.seq_len, size)

        train_data_loader = T.utils.data.DataLoader(PosModelDataset(train_data),
                                                    batch_size, shuffle=True)
        test_data_loader = T.utils.data.DataLoader(PosModelDataset(test_data),
                                                   batch_size, shuffle=True)

        # Load model from ckpt
        # self.position_model.load_state_dict(T.load("./checkpoints/PositionModel/50.pt"))

        # self.make_visualisations(test_data_loader)

        rows = []
        pic_no = 1
        for epoch in range(epochs):
            print("Training")
            losses = []
            for i, batch in enumerate(train_data_loader):
                X_image = batch[0].float().to(self.device)
                X_time = batch[1].float().to(self.device) / 109.6
                X_red_diam = batch[2].float().to(self.device)  # divide by max value of time

                X_time = X_time[:, None, None].repeat(1, self.width, self.width)

                model_input = X_image[:, :3], X_time[:, None]
                red_ball_gt = X_image[:, -1]

                red_ball_pred = self.position_model(model_input[0], model_input[1])

                if smooth_loss:
                    red_ball_gt += 0.005
                    red_ball_gt = torch.clamp(red_ball_gt, 0., 1.)

                loss = F.binary_cross_entropy(red_ball_pred.squeeze(1), red_ball_gt)
                losses.append(loss.item())

                opti.zero_grad()
                loss.backward()
                opti.step()

                # Visualisation
                row = []

                green_ball_collision = X_image[-1, 0][:, :, None].cpu()
                red_ball_collision = X_image[-1, 1][:, :, None].cpu()
                static_objs = X_image[-1, 2][:, :, None].cpu()

                collision_scene = np.concatenate([red_ball_collision, green_ball_collision, static_objs], axis=-1)

                row.append(collision_scene)

                empty_channel = np.zeros_like(static_objs)

                row.append(
                    np.concatenate([red_ball_gt.detach().cpu()[-1][:, :, None], empty_channel, empty_channel], axis=-1))

                row.append(
                    np.concatenate([red_ball_pred.detach().cpu()[-1][0][:, :, None], empty_channel, empty_channel],
                                   axis=-1))

                rows.append(row)

                if i % 5 == 0:
                    print(f"Epoch-{epoch}, iteration-{i}: Loss = {loss.item()}")

                if len(rows) == 5:
                    os.makedirs(f"./results/train/PositionModel/{epoch + 1}", exist_ok=True)
                    save_img_dir = f"./results/train/PositionModel/{epoch + 1}/"
                    vis_pred_path_task(rows, save_img_dir, pic_no)
                    pic_no += 1
                    rows = []

            pic_no = 1

            train_loss = sum(losses) / len(losses)
            train_loss_log.append(train_loss)

            losses = []
            rows = []
            print("Validation")
            os.makedirs("./checkpoints/PositionModel", exist_ok=True)
            T.save(self.position_model.state_dict(), f"./checkpoints/PositionModel/{epoch + 1}.pt")
            for i, batch in enumerate(test_data_loader):
                X_image = batch[0].float().to(self.device)
                X_time = batch[1].float().to(self.device) / 109.6

                X_time = X_time[:, None, None].repeat(1, self.width, self.width)

                model_input = X_image[:, :3], X_time[:, None]
                red_ball_gt = X_image[:, -1]

                red_ball_pred = self.position_model(model_input[0], model_input[1])

                loss = F.binary_cross_entropy(red_ball_pred.squeeze(1), red_ball_gt)
                losses.append(loss.item())

                # Visualisation
                row = []

                green_ball_collision = X_image[-1, 0][:, :, None].cpu()
                red_ball_collision = X_image[-1, 1][:, :, None].cpu()
                static_objs = X_image[-1, 2][:, :, None].cpu()

                collision_scene = np.concatenate([red_ball_collision, green_ball_collision, static_objs], axis=-1)

                row.append(collision_scene)

                empty_channel = np.zeros_like(static_objs)

                row.append(
                    np.concatenate([red_ball_gt.detach().cpu()[-1][:, :, None], empty_channel, empty_channel], axis=-1))

                row.append(
                    np.concatenate([red_ball_pred.detach().cpu()[-1][0][:, :, None], empty_channel, empty_channel],
                                   axis=-1))

                rows.append(row)

                if i % 5 == 0:
                    print(f"Epoch-{epoch}, iteration-{i}: Loss = {loss.item()}")

                if len(rows) == 5:
                    os.makedirs(f"./results/test/PositionModel/{epoch + 1}", exist_ok=True)
                    save_img_dir = f"./results/test/PositionModel/{epoch + 1}/"
                    vis_pred_path_task(rows, save_img_dir, pic_no)
                    pic_no += 1
                    rows = []

            val_loss = sum(losses) / len(losses)
            val_loss_log.append(val_loss)
            print(f"Epoch-{epoch + 1}, Training loss = {train_loss}, Validation loss = {val_loss}")

            scheduler.step(val_loss)  # lr scheduler step

            pic_no = 1

        os.makedirs("./logs/PositionModel", exist_ok=True)
        with open(f"./logs/PositionModel/{epochs}.pkl", "wb") as f:
            pickle.dump({"Train losses": train_loss_log,
                         "Test losses": val_loss_log}, f)
