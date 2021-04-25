import torch
from torch import nn
import numpy as np
import torch as T
import pickle
import torch.nn.functional as F
from utils.phyre_utils import vis_pred_path_task
import os
import cv2
from modules.data_utils import load_data_position


class PosModelDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        images = self.data[idx]["Images"]
        collision_time = self.data[idx]["Collision_time"]

        return images, collision_time

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
                                     nn.Conv2d(64, 128, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 128, 4, 2, 1),
                                     nn.ReLU())

        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(nn.Linear(128 * 2 * 2 + 1, 128 * 2 * 2),
                                   nn.ReLU())

        self.dense_down = nn.Sequential(nn.Linear(2 * 2 * 128, 128),
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

        self.decoder = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(128, 64, 4, 2, 1),
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
        x = self.encoder(x)
        x = self.flatten(x)
        # x = self.dense_down(x)

        x = torch.cat((x, time[:, None]), dim=-1)
        x = self.dense(x)

        # x = self.dense_up(x)
        b = x.size(0)
        x = x.view(b, 128, 2, 2)
        x = self.decoder(x)

        return x


class Pyramid(nn.Module):
    def __init__(self, in_dim, chs, wid, hidfac):
        super().__init__()
        """
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, 8, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, chs, 4, 2, 1),
            nn.Sigmoid()
        )
        """

        folds = range(1, int(np.math.log2(wid)))
        acti = nn.ReLU
        convs = [nn.Conv2d(int(2 ** (2 + i) * hidfac), int(2 ** (3 + i) * hidfac), 4, 2, 1) for i in folds]
        encoder = [nn.Conv2d(in_dim, int(8 * hidfac), 4, 2, 1), acti()] + [acti() if i % 2 else convs[i // 2] for i in
                                                                           range(2 * len(folds))]
        trans_convs = [nn.ConvTranspose2d(int(2 ** (3 + i) * hidfac), int(2 ** (2 + i) * hidfac), 4, 2, 1) for i in
                       reversed(folds)]
        decoder = [acti() if i % 2 else trans_convs[i // 2] for i in range(2 * len(folds))] + [
            nn.ConvTranspose2d(int(8 * hidfac), chs, 4, 2, 1), nn.Sigmoid()]
        modules = encoder + decoder
        self.model = nn.Sequential(*modules)
        # print(self.model.state_dict().keys())

        """
        convs = [(2**(2+i), 2**(3+i)) for i in folds]
        trans_convs = [(2**(3+i), 2**(2+i)) for i in reversed(folds)]
        print(convs)
        print(trans_convs)
        encoder = [(in_dim,8), 'acti'] + [f"acti" if i%2 else convs[i//2] for i in range(2*len(folds))]
        print(encoder)
        decoder = [f"acti" if i%2 else trans_convs[i//2] for i in range(2*len(folds))] + [(8,chs), 'Sigmoid']
        print(decoder)
        print(*(encoder+decoder), sep='\n')
        """

    def forward(self, X):
        return self.model(X)


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


class FlownetSolver():
    def __init__(self, seq_len, width, device, hidfac=1, viz=100):
        super().__init__()
        self.device = ("cuda" if T.cuda.is_available() else "cpu") if device == "cuda" else "cpu"
        print("device:", self.device)
        self.width = width
        self.discr = None
        self.r_fac = 1
        # self.cache = phyre.get_default_100k_cache('ball')
        self.cache = None
        self.hidfac = hidfac
        self.viz = viz
        self.seq_len = seq_len
        self.logger = dict()

        self.collision_model = Pyramid(seq_len * 2 + seq_len // 2 + 2, 1, width, hidfac)
        self.position_model = Pyramid2(3, 1)

        print("succesfully initialized models")

    def train_collision_model(self, data_paths, epochs=100, width=128, batch_size=32, num_pred_steps=4):
        if self.device == "cuda":
            self.collision_model.cuda()

        channels = range(1, 7)
        size = (width, width)

        train_loss_log = []
        val_loss_log = []

        opti = T.optim.Adam(self.collision_model.parameters(recurse=True), lr=3e-4)

        train_data = []
        for data_path in data_paths:
            with open(data_path, 'rb') as handle:
                task_data = pickle.load(handle)
            for data in task_data:
                frames_solved = data['images_solved']
                frames_unsolved = data['images_unsolved']

                obj_channels_solved = np.array(
                    [np.array([cv2.resize((frame == ch).astype(float), size, cv2.INTER_MAX) for ch in channels]) for
                     frame in frames_solved])
                obj_channels_solved = np.flip(obj_channels_solved, axis=2)

                obj_channels_unsolved = np.array(
                    [np.array([cv2.resize((frame == ch).astype(float), size, cv2.INTER_MAX) for ch in channels]) for
                     frame in frames_unsolved])
                obj_channels_unsolved = np.flip(obj_channels_unsolved, axis=2)

                if data_path.split('.')[0][-1] == '2':  # Task-2:
                    green_ball_idx = 1
                    red_ball_idx = 0
                    static_obj_idxs = [3, 5]
                elif data_path.split('.')[0][-1] == '0':  # Task-20
                    green_ball_idx = 1
                    red_ball_idx = 0
                    static_obj_idxs = [3, 5]
                elif data_path.split('.')[0][-1] == '5':  # Task-15
                    green_ball_idx = 1
                    red_ball_idx = 0
                    static_obj_idxs = [3, 5]

                green_ball_solved = obj_channels_solved[:, green_ball_idx].astype(np.uint8)
                green_ball_unsolved = obj_channels_unsolved[:, green_ball_idx].astype(np.uint8)
                red_ball_gt = np.flip(obj_channels_solved[:self.seq_len // 2 + 1, red_ball_idx], axis=0).astype(
                    np.uint8)
                static_objs = np.max(obj_channels_solved[0, static_obj_idxs, :, :][None], axis=1).astype(np.uint8)
                red_ball_zeros = np.zeros_like(red_ball_gt).astype(np.uint8)

                combined = np.concatenate([green_ball_solved, green_ball_unsolved, static_objs, red_ball_zeros,
                                           red_ball_gt], axis=0).astype(np.uint8)
                train_data.append(combined)

        np.random.seed(7)
        np.random.shuffle(train_data)

        train_data, test_data = train_data[:int(0.9 * len(train_data))], train_data[int(0.9 * len(train_data)):]

        train_data = T.tensor(np.array(train_data))
        test_data = T.tensor(np.array(test_data))
        train_data_loader = T.utils.data.DataLoader(T.utils.data.TensorDataset(train_data),
                                                    batch_size, shuffle=True)
        test_data_loader = T.utils.data.DataLoader(T.utils.data.TensorDataset(test_data),
                                                   batch_size, shuffle=True)

        # Load model from ckpt
        # self.collision_model.load_state_dict(T.load("./checkpoints/CollisionModel/100.pt"))

        rows = []
        pic_no = 1
        for epoch in range(epochs):
            print("Training")
            losses = []
            for i, (X,) in enumerate(train_data_loader):
                X = X.float().to(self.device)

                num_steps = self.seq_len // 2 + 1
                model_input = X[:, :2 * self.seq_len + 1 + num_steps]
                red_ball_gt = X[:, 2 * self.seq_len + 1 + num_steps:]

                red_ball_preds = []
                for timestep in range(num_steps):
                    red_ball_pred = self.collision_model(model_input)
                    red_ball_preds.append(red_ball_pred)

                    loss = F.binary_cross_entropy(red_ball_pred[:, 0], red_ball_gt[:, timestep])
                    losses.append(loss.item())

                    opti.zero_grad()
                    loss.backward()
                    opti.step()

                    model_input[:, 2 * self.seq_len + 1 + timestep] = red_ball_gt[:, timestep]

                # Visualisation
                row = []

                green_ball_collision = X[-1, :self.seq_len][self.seq_len // 2][:, :, None].cpu()
                red_ball_collision = X[-1, 2 * self.seq_len + 1][:, :, None].cpu()
                static_objs = X[-1, 2 * self.seq_len][:, :, None].cpu()

                collision_scene = np.concatenate([red_ball_collision, green_ball_collision, static_objs], axis=-1)

                row.append(collision_scene)

                empty_channel = np.zeros_like(static_objs)
                for timestep in range(num_steps):
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
            for i, (X,) in enumerate(test_data_loader):
                X = X.float().to(self.device)

                num_steps = self.seq_len // 2 + 1
                model_input = X[:, :2 * self.seq_len + 1 + num_steps]
                red_ball_gt = X[:, 2 * self.seq_len + 1 + num_steps:]

                red_ball_preds = []
                for timestep in range(num_steps):
                    red_ball_pred = self.collision_model(model_input)
                    red_ball_preds.append(red_ball_pred)

                    loss = F.binary_cross_entropy(red_ball_pred[:, 0], red_ball_gt[:, timestep])
                    losses.append(loss.item())

                    model_input[:, 2 * self.seq_len + 1 + timestep] = red_ball_gt[:, timestep]

                # Visualisation
                row = []

                green_ball_collision = X[-1, :self.seq_len][self.seq_len // 2][:, :, None].cpu()
                red_ball_collision = X[-1, 2 * self.seq_len + 1][:, :, None].cpu()
                static_objs = X[-1, 2 * self.seq_len][:, :, None].cpu()

                collision_scene = np.concatenate([red_ball_collision, green_ball_collision, static_objs], axis=-1)

                row.append(collision_scene)

                empty_channel = np.zeros((width, width, 1))
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

                green_ball_solved = X[-1, :self.seq_len].cpu()
                green_ball_unsolved = X[-1, self.seq_len: 2 * self.seq_len].cpu()
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

            val_loss_log.append(sum(losses) / len(losses))

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

            initial_scenes = X_image[:, 3:-1].detach().cpu().numpy()
            red_ball_gt = X_image[:, -1]

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

                row.append(np.moveaxis(initial_scenes[idx], 0, 2))

                empty_channel = np.zeros_like(static_objs)

                collision_scene = np.concatenate([red_ball_collision, green_ball_collision, empty_channel, static_objs],
                                                 axis=-1)

                pred_channel = collision_scene.copy()
                pred_channel[..., 0] = np.max(np.concatenate([pred_channel[..., 0][..., None],
                                                              red_ball_gt.detach().cpu()[idx, ..., None]], axis=-1),
                                              axis=-1)
                # pred_channel[..., 1] = np.max(np.concatenate([pred_channel[..., 1][..., None],
                #                                               red_ball_gt.detach().cpu()[idx, ..., None]], axis=-1),
                #                               axis=-1)
                pred_channel[..., 2] = np.max(np.concatenate([pred_channel[..., 2][..., None],
                                                              red_ball_preds[-1].detach().cpu()[idx, 0, ..., None]],
                                                             axis=-1),
                                              axis=-1)
                row.append(pred_channel)

                for red_ball_pred in reversed(red_ball_preds[:-1]):
                    pred_channel = collision_scene.copy()
                    pred_channel[..., 2] = np.max(np.concatenate([pred_channel[..., 2][..., None],
                                                                  red_ball_pred.detach().cpu()[idx, 0, ..., None]],
                                                                 axis=-1),
                                                  axis=-1)
                    row.append(pred_channel)

                rows.append(row)

                if len(rows) == 20:
                    os.makedirs(f"./results/test/PositionModel/visualisation", exist_ok=True)
                    save_img_dir = f"./results/test/PositionModel/visualisation"
                    vis_pred_path_task(rows, save_img_dir, pic_no)
                    pic_no += 1
                    rows = []

    def train_position_model(self, data_paths, epochs=100, width=128, batch_size=32):
        if self.device == "cuda":
            self.position_model.cuda()

        channels = range(1, 7)
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
        self.position_model.load_state_dict(T.load("./checkpoints/PositionModel/50.pt"))

        self.make_visualisations(test_data_loader)

        rows = []
        pic_no = 1
        for epoch in range(epochs):
            print("Training")
            losses = []
            for i, batch in enumerate(train_data_loader):
                X_image = batch[0].float().to(self.device)
                X_time = batch[1].float().to(self.device) / 33.  # divide by max value of time

                model_input = X_image[:, :3], X_time
                red_ball_gt = X_image[:, -1]

                red_ball_pred = self.position_model(model_input[0], model_input[1])

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
                X_time = batch[1].float().to(self.device) / 33.

                model_input = X_image[:, :3], X_time
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
