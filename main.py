import random
import numpy as np
import toml
import torch
import os

from DQNAgent.DQNAgent import DQNAgent
from DQNAgent.MyEnvironment import MyEnviroment
from utils.dataset_lrw1000 import LRW1000_Dataset as Dataset
from utils.function import dataset2dataloader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run(filename):
    # parameters
    num_frames = 30000
    plotting_interval = 1000
    memory_size = 1000
    target_update = 100

    # Environment
    print("Loading options...")
    with open(filename, 'r') as optionsFile:
        args = toml.loads(optionsFile.read())
    TrainDataset = Dataset('train', args)
    TrainLoader = dataset2dataloader(TrainDataset, args["input"]["batchsize"], args["input"]["numworkers"])
    for (i_iter, input) in enumerate(TrainLoader):
        video = input.get('video')
        label = input.get('label')
        # if i_iter == 0:
        X_train = video
        y_train = label
        # else:
        #     X_train = torch.cat((video, X_train), dim=0)
        #     y_train = torch.cat([label, y_train], dim=0)

        if 'env' in locals().keys():
            env.update_data(X_train, y_train)
        else:
            env = MyEnviroment(X_train, y_train)

        seed = 777
        np.random.seed(seed=seed)
        random.seed(seed)
        seed_torch(seed=seed)
        env.seed(seed=seed)

        batch_size = 64

        checkpoint = torch.load("/home/czg/weight.pt")

        # train
        if 'agent' in locals().keys():
            agent.dqn.load_state_dict(torch.load("model_parameter.pkl"))
            agent.train(num_frames, plotting_interval)
            torch.save(agent.dqn.state_dict(), "model_parameter.pkl")
        else:
            agent = DQNAgent(env, memory_size, batch_size, target_update, args["general"]["gpuid"])
            # pre_train weight
            agent.dqn.load_state_dict(checkpoint['video_model'])
            agent.train(num_frames, plotting_interval)
            torch.save(agent.dqn.state_dict(), "model_parameter.pkl")
    # test
    # TestDataset = Dataset('test', args)
    # TestLoader = dataset2dataloader(TrainDataset, args["input"]["batchsize"], args["input"]["numworkers"])
    # video = input.get('video').numpy()
    # label = input.get('label').numpy()
    # X_test = video
    # y_test = label
    # env = MyEnviroment(X_test, y_test)
    # agent = DQNAgent(env, memory_size, args["input"]["batchsize"], target_update, args["general"]["gpuid"])
    # video_folder = "videos/rainbow"
    # agent.test(video_folder=video_folder)


# Press the green button in the gutter to run the script.
if (__name__ == '__main__'):
    run('options_lip.toml')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
