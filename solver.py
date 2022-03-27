from engine.FlownetSolver import FlownetSolver
from engine.GridSolver import GridSolver
import os
import os.path as osp
import sys
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('PhySolve', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)

    parser.add_argument('--train_collision_model', default=False, type=bool)
    parser.add_argument('--train_position_model', default=False, type=bool)
    parser.add_argument('--train_unsupervised', default=False, type=bool)
    parser.add_argument('--train_lfm', default=False, type=bool)
    parser.add_argument('--simulate_collision_model', default=False, type=bool)
    parser.add_argument('--simulate_position_model', default=False, type=bool)
    parser.add_argument('--simulate_model', default=False, type=bool)
    parser.add_argument('--smooth_loss', default=False, type=bool)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_rollouts_dir', default=None, type=str)
    parser.add_argument('--visualise_dir', default=None, type=str)
    parser.add_argument('--root_dir', default='./', type=str)

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PhySolve", parents=[get_args_parser()])
    args = parser.parse_args()

    data_dir = osp.join(args.root_dir, "DataCollection/Database_newPosModel")
    paths = [osp.join(data_dir, i) for i in os.listdir(data_dir)]

    solver = FlownetSolver(args, 5, 64, args.device)

    if args.train_position_model:
        solver.train_position_model(data_paths=paths,
                                    epochs=100,
                                    smooth_loss=args.smooth_loss)

    if args.train_collision_model:
        solver.train_collision_model(data_paths=paths,
                                     epochs=100,
                                     smooth_loss=args.smooth_loss)

    if args.train_lfm:
        solver.train_lfm(data_paths=paths,
                         epochs=100,
                         smooth_loss=args.smooth_loss)

    if args.simulate_collision_model:
        solver.simulate_collision_model(checkpoint=osp.join(args.root_dir, 'checkpoints/CollisionModel/26.pt'),
                                        data_paths=paths,
                                        batch_size=10)

    if args.simulate_position_model:
        solver.simulate_position_model(checkpoint=osp.join(args.root_dir, 'checkpoints/PositionModel/31.pt'),
                                       data_paths=paths,
                                       batch_size=32,
                                       visualize=True,
                                       visualize_dir='results/visualisations/Collision_Model')

    if args.simulate_model:
        solver.simulate_combined(collision_ckpt=osp.join(args.root_dir, 'checkpoints/CollisionModel/26.pt'),
                                 position_ckpt=osp.join(args.root_dir, 'checkpoints/PositionModel/31.pt'),
                                 lfm_ckpt=osp.join('checkpoints/LfM/31.pt'),
                                 data_paths=paths,
                                 batch_size=1,
                                 save_rollouts_dir=args.save_rollouts_dir,
                                 visualise_dir=args.visualise_dir,
                                 device=args.device,
                                 num_lfm_attempts=0,
                                 num_random_attempts=0,
                                 visualize=True,
                                 debug=False)

    if args.train_unsupervised:
        solver.train_unsupervised(data_paths=paths,
                                  epochs=100)
