from modules.FlownetSolver import FlownetSolver

solver = FlownetSolver(5, 128, "cuda")
solver.train_position_model(["/home/dhruv/Desktop/PhySolve/DataCollection/database_task2.pkl",
                             "/home/dhruv/Desktop/PhySolve/DataCollection/database_task20.pkl",
                             "/home/dhruv/Desktop/PhySolve/DataCollection/database_task15.pkl"],
                            epochs=100)
