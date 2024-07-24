import numpy as np
from torch.utils.data import Dataset
import math
import torch

EMBEDDING_DIM = 768


class DataGenerator(Dataset):
    def __init__(self, x, y, window_size):
        'Initialization'
        self.x = x
        self.y = y
        self.window_size = window_size
        # self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches'
        return math.ceil(len(self.x))

    def __getitem__(self, index):
        'Generate one batch of data'
        # x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        # y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        x = self.x[index]
        y = self.y[index]
        # x = pad_sequences(x, dtype='object', padding='post',
        #                   value=np.zeros(EMBEDDING_DIM)).astype(np.float32)

        # 最大设置为40暂定不需要最大number的
        num_tokens = x.shape[0]
        # print('x.shape[0]:', num_tokens)

        mix_num_boxes = min(int(num_tokens), self.window_size)
        # # mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self.window_size, 768))#(window_size, 768)
        mix_features_pad[:mix_num_boxes] = x[:mix_num_boxes]
        x = mix_features_pad
        # print(x.shape)
        # x = pad_sequence([torch.from_numpy(np.array(x)) for x in input_x], batch_first=True).float()
        # print('x 类型：',type(x))
        return x, y

class DataGenerator_new(Dataset):
    def __init__(self, x, y, z, window_size):
        'Initialization'
        self.x = x
        self.y = y
        self.z = z
        self.window_size = window_size
        # self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches'
        return math.ceil(len(self.x))

    def __getitem__(self, index):
        'Generate one batch of data'
        # x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        # y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        x = self.x[index]
        y = self.y[index]
        z = self.z[index]
        # x = pad_sequences(x, dtype='object', padding='post',
        #                   value=np.zeros(EMBEDDING_DIM)).astype(np.float32)

        # 最大设置为40暂定不需要最大number的
        num_tokens = x.shape[0]
        # print('x.shape[0]:', num_tokens)

        mix_num_boxes = min(int(num_tokens), self.window_size)
        # # mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self.window_size, 768))#(window_size, 768)
        mix_features_pad[:mix_num_boxes] = x[:mix_num_boxes]
        x = mix_features_pad

        mix_features_pad_z = np.zeros((self.window_size, 768))  # (window_size, 768)
        mix_features_pad_z[:mix_num_boxes] = z[:mix_num_boxes]
        z = mix_features_pad_z
        # print(x.shape)
        # x = pad_sequence([torch.from_numpy(np.array(x)) for x in input_x], batch_first=True).float()
        # print('x 类型：',type(x))
        return x, y, z

class DomainAdaptationDataGenerator(Dataset):
    def __init__(self, training_data, label):
        'Initialization'
        self.training_data = training_data
        self.label = label

        # self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches'
        return len(self.training_data)

    def __getitem__(self, index):
        'Generate one batch of data'
        # x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        # y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        return torch.tensor(self.training_data[index]), torch.tensor(self.label[index])

class DomainAdaptationDataGenerator_2(Dataset):
    def __init__(self, source_data, target_data,source_label, target_label):
        'Initialization'
        self.source = source_data
        self.targrt = target_data
        self.source_label = source_label
        self.target_label = target_label

        # self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches'
        # print(len(self.source),len(self.targrt))
        # print(len(self.source_label),len(self.target_label))
        return len(self.source)

    def __getitem__(self, index):
        'Generate one batch of data'
        # x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        # y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        return torch.tensor(self.source[index]), torch.tensor(self.targrt[index]), torch.tensor(self.source_label[index]), torch.tensor(self.target_label[index])