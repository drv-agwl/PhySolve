from modules.FlownetSolver import FlownetSolver
import os
import os.path as osp
import sys

if __name__ == '__main__':
    smooth_loss = sys.argv[sys.argv.index("--smooth_loss") + 1] if "--smooth_loss" in sys.argv else False

    data_dir = "./DataCollection/Database"

    paths = [osp.join(data_dir, i) for i in os.listdir(data_dir)]

    solver = FlownetSolver(5, 64, "cuda")
    # solver.train_position_model(data_paths=paths,
    #                             epochs=100,
    #                             smooth_loss=smooth_loss)

    solver.simulate_position_model(checkpoint='/home/dhruv/Desktop/PhySolve/checkpoints/PositionModel/40.pt',
                                   data_paths=paths,
                                   batch_size=1)
