import argparse
import random
import warnings
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, f1_score
from tqdm import tqdm

from dataloader import DataGenerator, DataGenerator_new
from model_new import Model
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import os
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets




def extract_penultimate_layer_output(model, x, z):
    with torch.no_grad():
        B, _, _ = x.size()
        output1, _ = model.attention1(x.transpose(0, 1), z.transpose(0, 1), z.transpose(0, 1))
        output2, _ = model.attention2(z.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        output1 = output1.transpose(0, 1)
        output2 = output2.transpose(0, 1)
        output1 = F.layer_norm(x + output1, normalized_shape=[output1.size(-1)])
        output2 = F.layer_norm(z + output2, normalized_shape=[output2.size(-1)])
        fused_embedding = torch.cat((output1, output2), dim=-1)
        fused_embedding = F.relu(model.fc(fused_embedding))
        #fused_embedding = model.trans_encder(fused_embedding)
        #fused_embedding = fused_embedding.contiguous().view(B, -1)
        return fused_embedding  # This is the output from the penultimate layer

parser = argparse.ArgumentParser()
# fine-tuning setting
parser.add_argument('--pretrained_log_name', type=str,
                    default='HDFS', help='log file name')
parser.add_argument("--load_path", type=str,
                    default='checkpoints/train_HDFS_classifier_1_64_5e-05-best.pt', help="latest model path")
parser.add_argument('--log_name', type=str,
                    default='BGL', help='log file name')
# parser.add_argument('--tune_mode', type=str, default='adapter',
#                     help='tune adapter or classifier only')
# model setting
parser.add_argument('--num_layers', type=int, default=1,
                    help='num of encoder layer')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--window_size', type=int,
                    default='20', help='log sequence length')
# parser.add_argument('--adapter_size', type=int, default=64,
#                     help='adapter size')
parser.add_argument('--epoch', type=int, default=20,
                    help='epoch')
args = parser.parse_args()
# args.tune_mode = "adapter"
# args.pretrained_log_name = 'BGL'
# args.load_path = 'checkpoints/train_BGL_classifier_1_64_1e-05-best.pt'
args.log_name = 'HDFS'
args.num_layers = 1

# suffix = f'{args.log_name}_from_{args.pretrained_log_name}_{args.num_layers}_{args.lr}_{args.epoch}'
# with open(f'result/tune_{suffix}.txt', 'w', encoding='utf-8') as f:
#     f.write(str(args)+'\n')

# hyper-parameters
EMBEDDING_DIM = 768
batch_size = 1
epochs = args.epoch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

# fix all random seeds
warnings.filterwarnings('ignore')
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True

# training_data = np.load(
#     f'./preprocessed_data/{args.log_name}_training_prompt4_new_data.npz', allow_pickle=True)
# load test data Hdfs
# testing_data = np.load(
#     f'./preprocessed_data/{args.log_name}_testing_prompt4_new_data.npz', allow_pickle=True)

testing_data = np.load(
    f'./preprocessed_data/{args.log_name}_testing_prompt4_new_data.npz', allow_pickle=True)


x_test, y_test, z_test = testing_data['x'], testing_data['y'], testing_data['z']

test_generator = DataGenerator_new(x_test, y_test, z_test, args.window_size)

test_loader = torch.utils.data.DataLoader(
    test_generator, batch_size=batch_size, shuffle=False)

# model = torch.load('model2plot/HDFS.pth')
model = Model(num_layers=args.num_layers, dim=EMBEDDING_DIM, window_size=args.window_size, nhead=8, dim_feedforward=4 *
              EMBEDDING_DIM, dropout=0.1)
model.load_state_dict(torch.load('model2plot/HDFS_model.pth'))
# model = torch.nn.DataParallel(model, device_ids=device_ids)

model = model.to(device)

model.eval()
n = 0.0


embeddings1 = []
embeddings2 = []

# count1 = 0
# count2 = 0
# max_samples = 500

with (torch.no_grad()):
    for batch_idx, data in enumerate(tqdm(test_loader)):
        x, y, z = data[0].to(device), data[1].to(device), data[2].to(device)

        # print(y)

        x = x.to(torch.float32)
        y = y.to(torch.float32)
        z = z.to(torch.float32)
        #out = model(x, z).cpu()
        penultimate_output = extract_penultimate_layer_output(model, x, z).cpu().numpy()
        # print(y)

        if torch.equal(y, torch.tensor([[1, 0]], device=device, dtype=torch.int32)
                        ):
            #print(type(penultimate_output))
            embeddings1.append(penultimate_output.flatten())
            #count1 += 1
        elif torch.equal(y, torch.tensor([[0, 1]], device=device, dtype=torch.int32)):
            embeddings2.append(penultimate_output.flatten())
            #count2 += 1
        # else:
        #     print("ERROR")

        # if count1 >= max_samples and count2 >= max_samples:
        #     break




embeddings1 = np.array(embeddings1)  # 转换为 numpy 数组
embeddings2 = np.array(embeddings2)  # 转换为 numpy 数组

tsne = TSNE(n_components=3)  # 修改为3维
reduced_embeddings1 = tsne.fit_transform(embeddings1)
reduced_embeddings2 = tsne.fit_transform(embeddings2)

# 使用 pyqtgraph 绘制3D散点图
# app = pg.QtGui.QApplication([])
app = QtWidgets.QApplication([])

w = gl.GLViewWidget()
w.show()
w.setWindowTitle('T-SNE of Event Templates (3D)')
w.setCameraPosition(distance=40)

g = gl.GLGridItem()
w.addItem(g)

# 创建 EventTemplate1 的散点图
sp1 = gl.GLScatterPlotItem(pos=reduced_embeddings1, color=(0, 0, 1, 1), size=5)
w.addItem(sp1)

# 创建 EventTemplate2 的散点图
sp2 = gl.GLScatterPlotItem(pos=reduced_embeddings2, color=(1, 0, 0, 1), size=5)
w.addItem(sp2)

# 开始 Qt 事件循环
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()








