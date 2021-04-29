from modules.FlownetSolver import FlownetSolver
import os
import os.path as osp

data_dir = "./DataCollection/Database"

paths = [osp.join(data_dir, i) for i in os.listdir(data_dir)]

solver = FlownetSolver(5, 128, "cuda")
solver.train_position_model(data_paths=paths,
                            epochs=100)
