import os

from tensorboardX import SummaryWriter


class Tensorboard:

    def __init__(self, experiment, log_dir="tensorboard_logs"):
        experiment_name = experiment.split("/")[-1]
        save_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(save_dir)
        self.index_dict = dict()

    def log_scalar(self, name, value, index=-1):
        if index == -1:
            if name in self.index_dict:
                self.index_dict[name] += 1
                index = self.index_dict[name]
            else:
                self.index_dict[name] = 1
                index = 1
        self.writer.add_scalar(name, value, index)

    def log_histogram(self, name, value, bins, index=-1):
        if index == -1:
            if name in self.index_dict:
                self.index_dict[name] += 1
                index = self.index_dict[name]
            else:
                self.index_dict[name] = 1
                index = 1
        self.writer.add_histogram(name, value, index, bins)
