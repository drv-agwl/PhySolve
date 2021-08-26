import phyre
import torch
from torch import nn
import numpy as np
import torch as T
import pickle
import torch.nn.functional as F
from utils.phyre_utils import vis_pred_path_task, get_cross_image, get_text_image
import os
import cv2
from engine.data_utils import load_data_position, load_data_collision, load_lfm_data
from PIL import Image, ImageDraw
from engine.phyre_utils import simulate_action
from tqdm import tqdm
from engine.data_utils import draw_ball
from engine.SoftGatedHG import SoftGatedHG
import pandas as pd
from engine.LfM import LfM


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
        target_obj = self.data[idx]["Target_object"]

        return images, red_diam, target_obj, task_id

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

    def forward(self, x, time=None):
        if time is not None:
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
        x = self.encoder(X)
        r = self.radius_head(self.flatten(x))
        x = self.decoder(x)

        return x, r


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


class FlownetSolver:
    def __init__(self, args, seq_len, width, device):
        super().__init__()
        self.device = ("cuda" if T.cuda.is_available() else "cpu") if device == "cuda" else "cpu"
        print("device:", self.device)
        self.seq_len = seq_len
        self.width = width
        self.logger = dict()
        self.args = args

        self.collision_model = Pyramid(seq_len * 2 + seq_len // 2 + 2, 1)
        self.position_model = Pyramid2(4, 1)
        self.lfm = LfM(7, 1)
        self.position_model_unsupervised = Pyramid2(3, 1)

        # self.collision_model = SoftGatedHG(in_channels=seq_len * 2 + seq_len // 2 + 2, out_channels=1,
        #                                    time_channel=False, pred_radius=True, device=device)
        # self.position_model = SoftGatedHG(in_channels=4, out_channels=1,
        #                                   time_channel=True, pred_radius=False, device=device)
        # self.lfm = SoftGatedHG(in_channels=7, out_channels=1,
        #                        time_channel=True, pred_radius=False, device=device)

        print("succesfully initialized models")

    def train_collision_model(self, data_paths, epochs=100, width=128, batch_size=32, smooth_loss=False):
        if self.device == "cuda":
            self.collision_model.cuda()

        train_loss_log = []
        val_loss_log = []

        opti = T.optim.Adam(self.collision_model.parameters(recurse=True), lr=3e-4)
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(opti, 'min', patience=5, verbose=True)

        train_data, test_data = load_data_collision(data_paths, self.seq_len)

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
                    loss_rad = F.mse_loss(radius, pred_radius.squeeze(-1))
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
                    os.makedirs(f"./results/train/CollisionModel/SoftGatedHG/{epoch + 1}", exist_ok=True)
                    save_img_dir = f"./results/train/CollisionModel/SoftGatedHG/{epoch + 1}/"
                    vis_pred_path_task(rows, save_img_dir, pic_no)
                    pic_no += 1
                    rows = []

            pic_no = 1

            train_loss_log.append(sum(losses) / len(losses))

            losses = []
            rows = []
            print("Validation")
            os.makedirs("./checkpoints/CollisionModel/SoftGatedHG/", exist_ok=True)
            T.save(self.collision_model.state_dict(), f"./checkpoints/CollisionModel/SoftGatedHG/{epoch + 1}.pt")
            for i, batch in enumerate(test_data_loader):
                X_image = batch[0].float().to(self.device)
                radius = batch[1].float().to(self.device) / 2.

                num_steps = self.seq_len // 2 + 1
                model_input = X_image[:, :2 * self.seq_len + 1 + num_steps]
                red_ball_gt = X_image[:, 2 * self.seq_len + 1 + num_steps:]

                red_ball_preds = []
                for timestep in range(num_steps):
                    red_ball_pred, pred_radius = self.collision_model(model_input)
                    red_ball_preds.append(red_ball_pred)

                    loss_ball = F.binary_cross_entropy(red_ball_pred[:, 0], red_ball_gt[:, timestep])
                    loss_rad = F.mse_loss(radius, pred_radius.squeeze(-1))
                    loss = loss_ball + loss_rad
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
                    os.makedirs(f"./results/test/CollisionModel/SoftGatedHG/{epoch + 1}", exist_ok=True)
                    save_img_dir = f"./results/test/CollisionModel/SoftGatedHG/{epoch + 1}/"
                    vis_pred_path_task(rows, save_img_dir, pic_no)
                    pic_no += 1
                    rows = []

            val_loss = sum(losses) / len(losses)
            val_loss_log.append(val_loss)
            scheduler.step(val_loss)

            pic_no = 1

        os.makedirs("./logs/CollisionModel/SoftGatedHG/", exist_ok=True)
        with open(f"./logs/CollisionModel/SoftGatedHG/{epochs}.pkl", "wb") as f:
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

    def get_position_pred(self, pred_channel, diam, num_pos=1):
        diam = round(diam[0] * self.width)
        diam = max(1, diam)
        kernel = Image.new('1', (diam, diam))
        draw = ImageDraw.Draw(kernel)
        draw.ellipse((0, 0, diam - 1, diam - 1), fill=1)
        kernel = np.array(kernel).astype(np.float)

        pred_channel = pred_channel.detach().cpu().numpy()[0][0]

        filtered = cv2.filter2D(src=pred_channel, kernel=kernel, ddepth=-1)

        if num_pos == 1:
            pred_y, pred_x = np.unravel_index(np.argmax(filtered, axis=None), filtered.shape)
            return pred_x, pred_y
        elif num_pos > 1:
            # get the top num_pos for center of the ball according to the prbability map
            topk_pos = [np.unravel_index(i, filtered.shape) for i in (-filtered).argsort(axis=None)[:num_pos]]
            return topk_pos

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

    def simulate_combined(self, collision_ckpt, position_ckpt, lfm_ckpt, data_paths, batch_size=32,
                          save_rollouts_dir="./results/saved_rollouts", visualise_dir="./results/visualisations",
                          device='cuda',
                          num_lfm_attempts=10, num_random_attempts=0, visualize=False):

        self.collision_model.to(device).eval()
        self.position_model.to(device).eval()
        self.lfm.to(device).eval()

        size = (self.width, self.width)

        self.collision_model.load_state_dict(T.load(collision_ckpt, map_location=device))
        self.position_model.load_state_dict(T.load(position_ckpt, map_location=device))
        self.lfm.load_state_dict(T.load(lfm_ckpt, map_location=device))

        data_collision = load_data_collision(data_paths, self.seq_len, all_samples=True, shuffle=False)
        data_position = load_data_position(data_paths, self.seq_len, all_samples=True, shuffle=False)
        data_lfm = load_lfm_data(data_paths, self.seq_len, all_samples=True, shuffle=False)

        collision_data_loader = T.utils.data.DataLoader(CollisionDataset(data_collision),
                                                        batch_size, shuffle=False)
        position_data_loader = T.utils.data.DataLoader(PosModelDataset(data_position),
                                                       batch_size, shuffle=False)
        lfm_data_loader = T.utils.data.DataLoader(PosModelDataset(data_lfm), batch_size,
                                                  shuffle=False)

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
        lfm_works = 0
        for batch_collision, batch_position, batch_lfm in zip(collision_data_loader, position_data_loader,
                                                              lfm_data_loader):
            lfm_attempt = 0
            if visualize:
                row_titles = [get_text_image("Model", font_size=10),
                              get_text_image("Input-1", font_size=10),
                              get_text_image("Input-2", font_size=10),
                              get_text_image("Static objs", font_size=12, pos=(0, 25)),
                              get_text_image("Time", font_size=12),
                              get_text_image("G path - unsolved", font_size=10,pos=(0, 25)),
                              get_text_image("G path - prev attempt", font_size=10, pos=(0, 25)),
                              get_text_image("R - last pred", font_size=10, pos=(0, 25)),
                              get_text_image("R - pred", font_size=10),
                              get_text_image("Target", font_size=10)]

                grid = [row_titles]
                row_collision = [get_text_image("Collision Model", font_size=10, pos=(0, 25))]
                row_position = [get_text_image("Position Model", font_size=10, pos=(0, 25))]

            assert batch_position[-1] == batch_collision[-1] == batch_lfm[-1]
            task_id = batch_position[3][0]

            template = task_id.split(":")[0]

            if task_id in tasks:
                continue

            tasks.append(task_id)

            # Collision model prediction
            X_image = batch_collision[0].float().to(self.device)
            target_obj = batch_collision[-2].cpu().numpy()[0]

            num_steps = self.seq_len // 2 + 1
            model_input = X_image[:, :2 * self.seq_len + 1 + num_steps]

            red_ball_preds = []
            for timestep in range(1):
                red_ball_pred, radius = self.collision_model(model_input)
                red_ball_preds.append(red_ball_pred)

            if visualize:
                image = model_input.permute(0, 2, 3, 1).cpu().numpy()[0]
                green_ball_solved_paths = np.max(image[:, :, :5], axis=-1)
                green_ball_unsolved_paths = np.max(image[:, :, 5:10], axis=-1)
                static_objs = image[:, :, 10]
                collision_model_pred = red_ball_pred.permute(0, 2, 3, 1).detach().cpu().numpy()[0, :, :, 0]

                row_collision.append(green_ball_solved_paths)
                row_collision.append(green_ball_unsolved_paths)
                row_collision.append(static_objs)
                cross_images = [get_cross_image()] * 4
                for cross_image in cross_images:
                    row_collision.append(cross_image)
                row_collision.append(collision_model_pred)
                row_collision.append(target_obj)
                grid.append(row_collision)

            pred_x, pred_y = self.get_position_pred(red_ball_preds[0], radius.squeeze(1).detach().cpu().numpy() * 2)
            red_channel_collision = draw_ball(size, pred_y, pred_x,
                                              radius.squeeze(-1).detach().cpu().numpy() * size[0])  # Output of

            # Position Model
            X_image = batch_position[0].float().to(self.device)
            X_time = batch_position[1].float().to(self.device) / 109.6  # divide by max value of time
            X_time = X_time[:, None, None].repeat(1, self.width, self.width)

            model_input = X_image[:, :3], X_time[:, None]
            model_input[0][:, 1] = torch.tensor(red_channel_collision).to(self.device)  # Replace ground truth with
            # collision model prediction

            red_ball_pred = self.position_model(model_input[0], model_input[1])

            if visualize:
                images = model_input[0].permute(0, 2, 3, 1).cpu().numpy()[0]
                time = model_input[1].permute(0, 2, 3, 1).cpu().numpy()[0, :, :, 0]
                green_ball_collision = images[:, :, 0]
                red_ball_collision = images[:, :, 1]
                static_objs = images[:, :, 2]
                position_model_pred = red_ball_pred.detach().permute(0, 2, 3, 1).cpu().numpy()[0, :, :, 0]

                row_position.append(green_ball_collision)
                row_position.append(red_ball_collision)
                row_position.append(static_objs)
                row_position.append(time)
                cross_images = [get_cross_image()] * 3
                for cross_image in cross_images:
                    row_position.append(cross_image)
                row_position.append(position_model_pred)
                row_position.append(target_obj)
                grid.append(row_position)

            pred_y, pred_x = self.get_position_pred(red_ball_pred, radius.squeeze(1).detach().cpu().numpy() * 2)

            collided, solved, lfm_paths = simulate_action(self.args, sim, id, tasks[id],
                                                          pred_y / (self.width - 1.), 1. - pred_x / (self.width - 1.),
                                                          radius.squeeze(-1).detach(), num_attempts=num_random_attempts,
                                                          save_rollouts_dir=save_rollouts_dir,
                                                          red_ball_collision_scene=red_channel_collision)

            last_red_ball_pred = draw_ball(size, pred_x, pred_y, radius.squeeze(1).detach().cpu().numpy() * size[0])

            while not solved and lfm_attempt < num_lfm_attempts:
                row_lfm = [get_text_image("LfM Model", font_size=10, pos=(0, 25))]
                lfm_attempt += 1

                # LfM Model
                X_image = batch_lfm[0].float().to(self.device)
                X_time = batch_lfm[1].float().to(self.device) / 109.6  # divide by max value of time
                X_time = X_time[:, None, None].repeat(1, self.width, self.width)

                model_input = X_image[:, :-1], X_time[:, None]
                model_input[0][:, 4] = torch.Tensor(last_red_ball_pred).to(device)  # last prediction of red ball as
                # input
                if lfm_paths is not None:
                    model_input[0][:, 3] = torch.Tensor(lfm_paths[1]).to(device)  # 1 is green ball idx
                else:
                    model_input[0][:, 3] = model_input[0][:, 2]

                red_ball_pred = self.lfm(model_input[0], model_input[1])

                if visualize:
                    image = model_input[0].permute(0, 2, 3, 1).detach().cpu().numpy()[0]
                    green_ball_collision = image[:, :, 0]
                    red_ball_collision = image[:, :, 1]
                    green_ball_unsolved_path = image[:, :, 2]
                    green_ball_lfm_path = image[:, :, 3]
                    initial_red_start = image[:, :, 4]
                    static_objs = image[:, :, 5]
                    time = model_input[1].permute(0, 2, 3, 1).cpu().numpy()[0, :, :, 0]
                    lfm_pred = red_ball_pred.permute(0, 2, 3, 1).detach().cpu().numpy()[0, :, :, 0]

                    row_lfm.append(green_ball_collision)
                    row_lfm.append(red_ball_collision)
                    row_lfm.append(static_objs)
                    row_lfm.append(time)
                    row_lfm.append(green_ball_unsolved_path)
                    row_lfm.append(green_ball_lfm_path)
                    row_lfm.append(initial_red_start)
                    row_lfm.append(lfm_pred)
                    row_lfm.append(target_obj)

                    grid.append(row_lfm)

                pred_y, pred_x = self.get_position_pred(red_ball_pred, radius.squeeze(1).detach().cpu().numpy() * 2)

                collided, solved, lfm_paths = simulate_action(self.args, sim, id, tasks[id],
                                                              pred_y / (self.width - 1.),
                                                              1. - pred_x / (self.width - 1.),
                                                              radius.squeeze(-1).detach(),
                                                              num_attempts=num_random_attempts,
                                                              save_rollouts_dir=save_rollouts_dir,
                                                              red_ball_collision_scene=red_channel_collision)

                last_red_ball_pred = draw_ball(size, pred_x, pred_y, radius.squeeze(1).detach().cpu().numpy() * size[0])

            if collided:
                num_collided += 1
                metrics_table[template][0][0] += 1
            if solved:
                num_solved += 1
                metrics_table[template][1][0] += 1
            if not collided and solved:
                doubt_ids.append(task_id)
            if 0 < lfm_attempt < 10:
                lfm_works += 1

            metrics_table[template][0][1] += 1
            metrics_table[template][1][1] += 1
            id += 1

            if visualize:
                vis_pred_path_task(grid, visualise_dir, task_id, format="gray")

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

        data = load_data_position(data_paths, self.seq_len, all_samples=True)

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
            X_time = batch[1].float().to(self.device) / 109.6 # divide by max value of time
            X_red_diam = batch[2].float().to(self.device)
            X_red_pos = batch[3].float().to(self.device)
            X_green_pos = batch[4].float().to(self.device)

            X_time = X_time[:, None, None].repeat(1, self.width, self.width)

            model_input = X_image[:, :3], X_time[:, None]

            red_ball_pred = self.position_model(model_input[0], model_input[1])

            pred_y, pred_x = self.get_position_pred(red_ball_pred, X_red_diam.cpu().numpy())

            solved = simulate_action(self.args, sim, id,
                                     pred_y / (self.width - 1.), 1. - pred_x / (self.width - 1.),
                                     X_red_diam / 2.,
                                     save_rollouts=True)

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

    def train_lfm(self, data_paths, epochs=100, batch_size=32, smooth_loss=False):
        if self.device == "cuda":
            self.lfm.cuda()

        train_loss_log = []
        val_loss_log = []

        opti = T.optim.Adam(self.lfm.parameters(recurse=True), lr=3e-4)
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(opti, 'min', patience=5, verbose=True)

        train_data, test_data = load_lfm_data(data_paths, self.seq_len)

        train_data_loader = T.utils.data.DataLoader(PosModelDataset(train_data),
                                                    batch_size, shuffle=True)
        test_data_loader = T.utils.data.DataLoader(PosModelDataset(test_data),
                                                   batch_size, shuffle=True)

        rows = []
        pic_no = 1
        for epoch in range(epochs):
            print("Training", end='\r')
            losses = []
            for i, batch in enumerate(train_data_loader):
                X_image = batch[0].float().to(self.device)
                X_time = batch[1].float().to(self.device) / 109.6  # divide by max value of time

                X_time = X_time[:, None, None].repeat(1, self.width, self.width)

                model_input = X_image[:, :-1], X_time[:, None]
                red_ball_gt = X_image[:, -1]

                red_ball_pred = self.lfm(model_input[0], model_input[1])

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

                # Collision scene
                green_ball_collision = X_image[-1, 0][:, :, None].cpu()
                red_ball_collision = X_image[-1, 1][:, :, None].cpu()
                static_objs = X_image[-1, -2][:, :, None].cpu()
                collision_scene = np.concatenate([red_ball_collision, green_ball_collision, static_objs], axis=-1)
                row.append(collision_scene)

                # Trial attempt
                red_ball_wrong_start = X_image[-1, 4][:, :, None].cpu()
                green_ball_unsolved_path = X_image[-1, 2][:, :, None].cpu()
                green_ball_path_lfm = X_image[-1, 3][:, :, None].cpu()
                trial_scene = np.concatenate([red_ball_wrong_start, green_ball_unsolved_path,
                                              green_ball_path_lfm, static_objs], axis=-1)
                row.append(trial_scene)

                # Ground truth
                empty_channel = np.zeros_like(static_objs)
                row.append(
                    np.concatenate([red_ball_gt.detach().cpu()[-1][:, :, None], empty_channel, empty_channel], axis=-1))

                # Model prediction
                row.append(
                    np.concatenate([red_ball_pred.detach().cpu()[-1][0][:, :, None], empty_channel, empty_channel],
                                   axis=-1))

                rows.append(row)

                if i % 5 == 0:
                    print(f"Epoch-{epoch}, iteration-{i}: Loss = {loss.item()}", end='\r')

                if len(rows) == 5:
                    os.makedirs(f"./results/train/LfM/{epoch + 1}", exist_ok=True)
                    save_img_dir = f"./results/train/LfM/{epoch + 1}/"
                    vis_pred_path_task(rows, save_img_dir, pic_no)
                    pic_no += 1
                    rows = []

            pic_no = 1

            train_loss = sum(losses) / len(losses)
            train_loss_log.append(train_loss)

            losses = []
            rows = []
            print("Validation", end='\r')
            os.makedirs("./checkpoints/LfM", exist_ok=True)
            T.save(self.lfm.state_dict(), f"./checkpoints/LfM/{epoch + 1}.pt")
            for i, batch in enumerate(test_data_loader):
                X_image = batch[0].float().to(self.device)
                X_time = batch[1].float().to(self.device) / 109.6  # divide by max value of time
                X_red_diam = batch[2].float().to(self.device)

                X_time = X_time[:, None, None].repeat(1, self.width, self.width)

                model_input = X_image[:, :-1], X_time[:, None]
                red_ball_gt = X_image[:, -1]

                red_ball_pred = self.lfm(model_input[0], model_input[1])

                loss = F.binary_cross_entropy(red_ball_pred.squeeze(1), red_ball_gt)
                losses.append(loss.item())

                # Visualisation
                row = []

                # Collision scene
                green_ball_collision = X_image[-1, 0][:, :, None].cpu()
                red_ball_collision = X_image[-1, 1][:, :, None].cpu()
                static_objs = X_image[-1, -2][:, :, None].cpu()
                collision_scene = np.concatenate([red_ball_collision, green_ball_collision, static_objs], axis=-1)
                row.append(collision_scene)

                # Trial attempt
                red_ball_wrong_start = X_image[-1, 4][:, :, None].cpu()
                green_ball_unsolved_path = X_image[-1, 2][:, :, None].cpu()
                green_ball_path_lfm = X_image[-1, 3][:, :, None].cpu()
                trial_scene = np.concatenate([red_ball_wrong_start, green_ball_unsolved_path,
                                              green_ball_path_lfm, static_objs], axis=-1)
                row.append(trial_scene)

                # Ground truth
                empty_channel = np.zeros_like(static_objs)
                row.append(
                    np.concatenate([red_ball_gt.detach().cpu()[-1][:, :, None], empty_channel, empty_channel], axis=-1))

                # Model prediction
                row.append(
                    np.concatenate([red_ball_pred.detach().cpu()[-1][0][:, :, None], empty_channel, empty_channel],
                                   axis=-1))

                rows.append(row)

                if i % 5 == 0:
                    print(f"Epoch-{epoch}, iteration-{i}: Loss = {loss.item()}", end='\r')

                if len(rows) == 5:
                    os.makedirs(f"./results/test/LfM/{epoch + 1}", exist_ok=True)
                    save_img_dir = f"./results/test/LfM/{epoch + 1}/"
                    vis_pred_path_task(rows, save_img_dir, pic_no)
                    pic_no += 1
                    rows = []

            val_loss = sum(losses) / len(losses)
            val_loss_log.append(val_loss)
            print(f"Epoch-{epoch + 1}, Training loss = {train_loss}, Validation loss = {val_loss}")

            scheduler.step(val_loss)  # lr scheduler step

            pic_no = 1

        os.makedirs("./logs/LfM", exist_ok=True)
        with open(f"./logs/LfM/{epochs}.pkl", "wb") as f:
            pickle.dump({"Train losses": train_loss_log,
                         "Test losses": val_loss_log}, f)


    def build_target_map(self, target_map, pred_pos):
        """
        returns a target probability map with high values presenting areas which solves the environment
        and low values present areas which have low probability of solving the environment

        The target map is input to the function, whose values are updated based on the results of simulator at locations
        present in the pred_pos

        pred_pos: top k coordinates predicted by the model
        prob_map: current target map
        """

        for coord in pred_pos:


    def train_unsupervised(self, data_paths, epochs=100, batch_size=32, width=64, recur_len=10):
        """
        train a recurrent unsupervised model for predicting starting position of the red ball
        given the collision state.
        """
        if self.device == "cuda":
            self.position_model_unsupervised.cuda()

        size = (width, width)

        train_loss_log = []
        val_loss_log = []

        opti = T.optim.Adam(self.position_model_unsupervised.parameters(recurse=True), lr=3e-4)
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(opti, 'min', patience=5, verbose=True)

        train_data, test_data = load_data_position(data_paths, self.seq_len)

        train_data_loader = T.utils.data.DataLoader(PosModelDataset(train_data),
                                                    batch_size, shuffle=True)
        test_data_loader = T.utils.data.DataLoader(PosModelDataset(test_data),
                                                   batch_size, shuffle=True)

        for epoch in range(epochs):
            print("Training")
            losses = []

            for i, batch in enumerate(train_data_loader):
                X_image = batch[0].float().to(self.device)
                X_red_diam = batch[2].float().to(self.device)

                model_input = X_image[:, :3]

                red_ball_pred = self.position_model_unsupervised(model_input)
                pred_pos = self.get_position_pred(red_ball_pred, X_red_diam.cpu().numpy(), num_pos=recur_len)
                pred_pos = [(y / (self.width - 1.), 1. - x / (self.width - 1.)) for y, x in pred_pos]
                target = T.zeros(size).to(device)

                for step in recur_len:
                    target = self.build_target_map(target, pred_pos)  # map which is used to calculate loss