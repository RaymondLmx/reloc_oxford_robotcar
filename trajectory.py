import os
import numpy as np
import matplotlib.pyplot as plt
from tool.interpolate_poses import interpolate_vo_poses
from config.template import CONFIG

data_path = CONFIG.DATA_PATH


class Trajectory:
    """
        trajectory
    """
    def __init__(self, dataset):

        self.dataset = dataset
        self.is_sampled = False

        # 1. timestamps
        time_stamps_path = os.path.join(data_path, dataset, 'radar.timestamps')
        with open(time_stamps_path) as f:
            time_stamps = [int(l.rstrip().split(' ')[0]) for l in f]

        # 2. ground truth
        ground_truth_path = os.path.join(data_path, dataset, 'gt/radar_odometry.csv')
        poses = np.array(interpolate_vo_poses(ground_truth_path, time_stamps, time_stamps[0]))
        poses = np.reshape(poses[:, :3, :], (len(poses), -1))
        self.ground_truth = poses[:, [3, 7]]

        time_stamps.pop(0)
        self.time_stamps = time_stamps
        self.num = len(self.time_stamps)

    def downsample(self, interval):
        """
            downsample the trajectory with fixed interval
        """
        ts = []
        gt = []
        for i in range(0, self.num, interval):
            ts.append(self.time_stamps[i])
            gt.append(self.ground_truth[i])

        self.time_stamps = ts
        self.ground_truth = np.array(gt)
        self.num = len(self.time_stamps)
        self.is_sampled = True

    def intercept(self, start, end):
        """
            intercept a part of the trajectory as the index given
        """
        self.time_stamps = self.time_stamps[start: end]
        self.ground_truth = self.ground_truth[start: end, :]
        self.num = len(self.time_stamps)

    def plot(self):
        """
            plot the trajectory
        """
        plt.figure(figsize=(10, 10))

        plt.title('trajectory_%s%s' % (self.dataset, '_sampled' if self.is_sampled else ''), fontsize=15)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.xlim(-200, 1200)
        plt.ylim(-500, 900)
        plt.scatter(self.ground_truth[:, 0], self.ground_truth[:, 1], s=0.5)
        # plt.savefig('trajectory.png')
        plt.show()



