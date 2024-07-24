# from torch import nn
#
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# from math import sqrt
# #
# class MultiHeadSelfAttention(nn.Module):
#     dim_in: int  # input dimension
#     dim_k: int   # key and query dimension
#     dim_v: int   # value dimension
#     num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
#
#     def __init__(self, dim_emd, num_heads=8):
#         super(MultiHeadSelfAttention, self).__init__()
#         assert dim_emd % num_heads == 0 and dim_emd % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
#         self.num_heads = num_heads
#         self.linear_q = nn.Linear(dim_emd, dim_emd, bias=False)
#         self.linear_k = nn.Linear(dim_emd, dim_emd, bias=False)
#         self.linear_v = nn.Linear(dim_emd, dim_emd, bias=False)
#         self._norm_fact = 1 / sqrt(dim_emd // num_heads)
#         self.linear_for_params = nn.Linear(dim_emd, dim_emd)  # 参数编码层
#         self.dim_emd = dim_emd
#
#
#     def forward(self, x, parsed_params):
#         # x: tensor of shape (batch, n, dim_in)
#         batch, n, dim_in = x.shape
#         assert dim_in == self.dim_emd
#
#         # 获取参数字符级嵌入
#         # chair_embeddings = self.feature_extractor(parsed_params)
#           # 参数编码
#
#         nh = self.num_heads
#         dk = self.dim_emd // nh  # dim_k of each head
#         dv = self.dim_emd // nh  # dim_v of each head
#
#         q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
#         k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
#         v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)
#         encoded_params = self.linear_for_params(parsed_params).reshape(batch, n, nh, dv).transpose(1, 2)
#
#
#
#
#         dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
#
#         encoded_params = torch.matmul(encoded_params, encoded_params.transpose(2, 3))
#         dist = torch.softmax(dist+encoded_params, dim=-1)  # batch, nh, n, n
#
#         att = torch.matmul(dist, v)  # batch, nh, n, dv
#         att = att.transpose(1, 2).reshape(batch, n, self.dim_emd)  # batch, n, dim_v
#         return att
#
# # 示例使用
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = MultiHeadSelfAttention(768, num_heads=8).to(device)
#
# # 假设我们已经得到了解析后参数的特征向量
# parsed_params_emb = torch.randn(64, 20, 768).to(device)  # (batch_size, sequence_length, embed_dim)
#
# # 以及相应的query, key, value张量
# query = torch.randn(64, 20, 768).to(device)
#
#
# output = model(query,parsed_params_emb)



# import numpy as np
# import random
# source_log_name = 'BGL'
# target_log_name = 'HDFS'
#
# source_training_data = np.load(
#     f'./preprocessed_data/{source_log_name}_training_data.npz', allow_pickle=True)
# # load test data Hdfs
# target_training_data = np.load(f'./preprocessed_data/{target_log_name}_training_data.npz', allow_pickle=True)
# source_x, source_y = source_training_data['x'], source_training_data['y']
# target_x, target_y = target_training_data['x'], target_training_data['y']
#
#
# source_data = []
# target_data = []
#
#
# for i in range(188536):
#     index1 = random.randint(0, len(source_x)-1)
#     source_data.append(source_x[index1])
#
#
# for i in range(188536):
#     index1 = random.randint(0, len(target_x)-1)
#     target_data.append(target_x[index1])
#
# source_mean = np.mean(source_data, axis=0)
# target_mean = np.mean(target_data, axis=0)
# distance = np.linalg.norm(source_mean - target_mean)
# simi12 = np.sum(source_mean * target_mean) / (np.linalg.norm(source_mean) * np.linalg.norm(target_mean))
#
# print("Source Mean Vector:", source_mean)
# print("Target Mean Vector:", target_mean)
# print("Distance between the mean vectors:", distance)
# print("simi between the mean vectors:", simi12)











#
# import matplotlib.pyplot as plt
# precision_list = [1.0000, 0.9985, 0.9979, 0.9986, 0.9987, 0.9987, 0.9987, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000 ]
# recall_list = [0.6152, 0.7892, 0.8475, 0.8719, 0.8851, 0.8862, 0.8868, 0.8862, 0.8862, 0.9065, 0.9035, 0.9333, 0.9333, 0.9339, 0.9416, 0.9363, 0.9398, 0.9398, 0.9363,  0.9398]
# f1_score_list = [0.7618,  0.8816, 0.9166, 0.9310, 0.9384, 0.9391, 0.9394, 0.9397, 0.9397, 0.9510, 0.9493, 0.9655, 0.9655, 0.9658, 0.9699, 0.9671, 0.9690, 0.9690, 0.9671, 0.9690 ]
# loss_list = [0.115809, 0.055543, 0.017619, 0.012936, 0.010550, 0.008734, 0.007312, 0.005980, 0.005053, 0.004697, 0.004245, 0.003854, 0.003518, 0.003418, 0.003141, 0.003001, 0.002992, 0.002865, 0.002803,0.002822]
#
#
# precision_list1 = [1.0000, 0.9985, 0.9986, 0.9986, 0.9980, 0.9973, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9993, 1.0000, 0.9994, 0.9994, 0.9994]
# recall_list1 = [0.6158, 0.7939, 0.8350, 0.8672, 0.8845, 0.8851, 0.8856, 0.8856, 0.8856, 0.8868, 0.8868, 0.8922, 0.8868, 0.9041, 0.9273, 0.9041, 0.9202, 0.9208, 0.9208, 0.9250]
# f1_score_list1 = [0.7623, 0.8845, 0.9095, 0.9283, 0.9378, 0.9378, 0.9388, 0.9388, 0.9388, 0.9394, 0.9394, 0.9424, 0.9394, 0.9490, 0.9617, 0.9493, 0.9584, 0.9585, 0.9585, 0.9607]
# loss_list1 = [0.150931, 0.053268, 0.020440, 0.013509, 0.010190, 0.008257, 0.006989, 0.006230, 0.005719, 0.005343, 0.004926, 0.004593, 0.004179, 0.004131, 0.003657, 0.003602, 0.003466, 0.003439, 0.003329, 0.003214]
#
# loss_list = [0.18433967313254399, 0.03747520854400101, 0.016796392999585175, 0.011524441122465338, 0.007309046191796039, 0.0050079997588215455, 0.003258957449085027, 0.002798171183334577, 0.002817541266595941, 0.002446137981465583, 0.0025273103463866035, 0.0024109108658329987, 0.0022427685596219443, 0.002182956742441706, 0.002122343853878843, 0.001929009780767023, 0.0022300890923519826, 0.0021014896712201266, 0.0020256461050281982, 0.0018544960685290294]
# loss_list1 = [0.15414022644052316, 0.03719147683585344, 0.011312060073992627, 0.006267600939090902, 0.004798434590194186, 0.0035387847507295715, 0.0029141611680857864, 0.0027196183203970705, 0.002780887654143715, 0.0025990837127115416, 0.0024023220235516083, 0.0024627727772787563, 0.0024682511262792265, 0.0023697973192458054, 0.0022458528788895375, 0.0020455795105896843, 0.0022615164050194436, 0.002331821669029919, 0.0022339297973261655, 0.0020005251466054183]
#
# F1_list = [0.0, 0.8430121250797703, 0.8836930455635491, 0.883623275344931, 0.9442154438032426, 0.9413744740532959, 0.9298974785259074, 0.9341849486253818, 0.9375696767001115, 0.9467755561813573, 0.9401565995525727, 0.9318785019168387, 0.9289489136817382, 0.9186214885606718, 0.924198250728863, 0.9207086842869591, 0.9188405797101449, 0.9249635036496351, 0.9228080396154966, 0.9225393127548049]
# F1_list1 = [0.0, 0.40240480961923847, 0.8251046025104602, 0.8878224355128974, 0.947043989913141, 0.9553420011305822, 0.9548022598870056, 0.9550720542526138, 0.9365079365079365, 0.9370588235294117, 0.9296761015465421, 0.94540059347181, 0.9437203791469194, 0.9406554472984943, 0.940100324579522, 0.9398230088495575, 0.9392688679245284, 0.9423247559893523, 0.9409332545776727, 0.9414893617021277]
#
#
# loss_list_pretrain = [0.2634944967942003, 0.12996129093605946, 0.07209444706781502, 0.03848231034443735, 0.024711529069983938, 0.01711745689140199, 0.012944442336710682, 0.008807887574566872, 0.006399534942342561, 0.005497882979362705, 0.005021412689690748, 0.0047774941149025425, 0.00464160853171029, 0.004530244566469533, 0.00420181376556923, 0.0043150011706260295, 0.004236904422265503, 0.004238416574761129, 0.00421680677167045, 0.004164783462644337]
# loss_list_random = [0.25438207530375484, 0.12562690695209072, 0.08777867308766632, 0.0562110916267169, 0.03305571933080297, 0.01965420243325377, 0.0136722781229327, 0.010045674346017332, 0.0075914995014139735, 0.006364022415282222, 0.005819727551695875, 0.0053678553206517226, 0.004983191591394025, 0.00480886613677839, 0.004604363160617548, 0.004602920523314759, 0.0043668576984726435, 0.004408903794803217, 0.004425060511369704, 0.004347023080210106]
#
# F1_list_pretrain = [0.0, 0.0011890606420927466, 0.796713111825652, 0.9009480222294868, 0.9016661221822934, 0.8922570016474465, 0.9137318255250404, 0.9296402419611589, 0.9370850458425546, 0.9377567140600316, 0.9384275339437954, 0.9384275339437954, 0.9384275339437954, 0.9384275339437954, 0.9384275339437954, 0.9384275339437954, 0.9384275339437954, 0.9384275339437954, 0.9384275339437954, 0.9384275339437954]
# F1_list_random = [0.0, 0.0, 0.7617994100294986, 0.8470507544581619, 0.8804528804528805, 0.8889623265036352, 0.9333756345177665, 0.9357391579613802, 0.9360759493670886, 0.9390975071000316, 0.9394321766561514, 0.9397666351308736, 0.9397666351308736, 0.9397666351308736, 0.9397666351308736, 0.9401008827238335, 0.9404349196344154, 0.9401008827238335, 0.9401008827238335, 0.9401008827238335]
#
# epochs = range(1, len(loss_list_pretrain) + 1)
#
# # plt.plot(epochs, loss_list_pretrain, 'b', label='loss_BGL2HDFS')
# # plt.plot(epochs, loss_list_random, 'r', label='loss_Random')
# plt.plot(epochs, F1_list_pretrain, 'b', label='F1_BGL2HDFS')
# plt.plot(epochs, F1_list_random, 'r', label='F1_Random')
# plt.title('Loss vs Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import torch
# from model_old import discriminator, Domaintransformer, Model
# from model_old import DAtrans
#
# # 加载数据
# hdfs_data = np.load('./preprocessed_data/BGL_training_data.npz', allow_pickle=False)#blue
# bgl_data = np.load('./preprocessed_data/BGL_testing_data.npz', allow_pickle=False)#red
#
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EMBEDDING_DIM = 768
# domain_transformer = Domaintransformer(emb_dim=EMBEDDING_DIM, output_dim=EMBEDDING_DIM, dropout=0.1).to(device)
# checkpoint = torch.load("domainadaptation/domain_transformer.pth")
# domain_transformer.load_state_dict(checkpoint)
#
# model = DAtrans().to(device)
# checkpoint = torch.load("domainadaptation/model.pth")
# model.load_state_dict(checkpoint)
#
# # 获取特征和标签
# hdfs_features = hdfs_data['x']
# bgl_features = bgl_data['x']
#
# unique_indices1 = np.unique(hdfs_features, axis=0, return_index=True)[1]
# unique_indices2 = np.unique(bgl_features, axis=0, return_index=True)[1]
# hdfs_features = hdfs_features[unique_indices1]
# bgl_features = bgl_features[unique_indices2]
#
#
# # 随机采样2000个样本
# hdfs_features = hdfs_features[np.random.choice(hdfs_features.shape[0], 1000, replace=False)]
# bgl_features = bgl_features[np.random.choice(bgl_features.shape[0], 1000, replace=False)]
#
#
# hdfs_features_2d = hdfs_features.reshape((hdfs_features.shape[0], -1))
# bgl_features_2d = bgl_features.reshape((bgl_features.shape[0], -1))
#
# # 合并特征
# all_features = np.concatenate((hdfs_features_2d, bgl_features_2d), axis=0)
# #print(hdfs_features.shape, bgl_features.shape)
#
# # 创建标签
# hdfs_labels = np.zeros(len(hdfs_features))
# bgl_labels = np.ones(len(bgl_features))
# all_labels = np.concatenate((hdfs_labels, bgl_labels), axis=0)
#
#
# # t-SNE降维
# tsne = TSNE(n_components=2, random_state=42)
# tsne_result = tsne.fit_transform(all_features)
#
# # PCA降维
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(all_features)
#
# # 可视化
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.title('t-SNE')
# plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=all_labels, cmap='coolwarm', alpha=0.7)
# plt.colorbar(label='Dataset')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
#
# plt.subplot(1, 2, 2)
# plt.title('PCA')
# plt.scatter(pca_result[:, 0], pca_result[:, 1], c=all_labels, cmap='coolwarm', alpha=0.7)
# plt.colorbar(label='Dataset')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
#
# plt.tight_layout()
# plt.show()
# # #
# #
# #
# for i in range(len(hdfs_features)):
#    # print(hdfs_features[i].shape)
#     hdfs_features[i] = model(torch.tensor(hdfs_features[i].reshape(1,20,768)).to(device),1)[0].detach().to('cpu').numpy().reshape(20,768)
# for i in range(len(bgl_features)):
#    # print(hdfs_features[i].shape)
#     bgl_features[i] = model(torch.tensor(bgl_features[i].reshape(1,20,768)).to(device),1)[0].detach().to('cpu').numpy().reshape(20,768)
# source_mean = np.mean(bgl_features, axis=0)
# target_mean = np.mean(hdfs_features, axis=0)
# distance = np.linalg.norm(source_mean - target_mean)
# print('distance: ', distance)
#
#
# hdfs_features_2d = hdfs_features.reshape((hdfs_features.shape[0], -1))
# bgl_features_2d = bgl_features.reshape((bgl_features.shape[0], -1))
# # 合并特征
# all_features = np.concatenate((hdfs_features_2d, bgl_features_2d), axis=0)
# #print(hdfs_features.shape, bgl_features.shape)
# # 创建标签
# hdfs_labels = np.zeros(len(hdfs_features))
# bgl_labels = np.ones(len(bgl_features))
# all_labels = np.concatenate((hdfs_labels, bgl_labels), axis=0)
# # t-SNE降维
# tsne = TSNE(n_components=2, random_state=42)
# tsne_result = tsne.fit_transform(all_features)
# # PCA降维
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(all_features)
# # 可视化
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('t-SNE')
# plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=all_labels, cmap='coolwarm', alpha=0.7)
# plt.colorbar(label='Dataset')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
#
# plt.subplot(1, 2, 2)
# plt.title('PCA')
# plt.scatter(pca_result[:, 0], pca_result[:, 1], c=all_labels, cmap='coolwarm', alpha=0.7)
# plt.colorbar(label='Dataset')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
#
# plt.tight_layout()
# plt.show()


# import pandas as pd
# df = pd.read_csv("parse_result/BGL.log_structured.csv")
# event_templates = df["EventTemplate"].values
# total_count = len(event_templates)
# top_20_percent_count = int(total_count * 0.2)
# top_20_percent_data = event_templates[:top_20_percent_count]
# bottom_80_percent_data = event_templates[top_20_percent_count:]
# top_20_percent_set = set(top_20_percent_data)
# bottom_80_percent_set = set(bottom_80_percent_data)
# repeated_categories = top_20_percent_set.intersection(bottom_80_percent_set)
# num_repeated_categories = len(repeated_categories)
#
# print(len(top_20_percent_set))
# print(len(bottom_80_percent_set))
#
# print("前20%数据中出现在后80%数据中的类别数量:", num_repeated_categories)

# import ast
# import os
# import re
#
# import numpy as np
# import pandas as pd
# import torch
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm
#
# import Drain
#
# log_name = 'HDFS'
# input_dir = '../log_data/'  # The input directory of log file
# output_dir = 'parse_result/'  # The output directory of parsing results
# max_length = 20
#
#
# if not os.path.exists(output_dir + log_name + '.log_structured.csv'):
#     log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
#     # Regular expression list for optional preprocessing (default: [])
#     regex = [
#         r'blk_(|-)[0-9]+',  # block id
#         r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
#         r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
#     ]
#     st = 0.5  # Similarity threshold
#     depth = 4  # Depth of all leaf nodes
#
#     parser = Drain.LogParser(log_format, indir=input_dir,
#                              outdir=output_dir, depth=depth, st=st, rex=regex)
#     parser.parse(log_name + '.log')
#
# num_workers = 6
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = SentenceTransformer(
#     'distilbert-base-nli-mean-tokens', device=device)
#
# structured_file_name = log_name + '.log_structured.csv'
# template_file_name = log_name + '_new.log_templates.csv'
#
# # load data
# df_template = pd.read_csv(output_dir + template_file_name)
# df_structured = pd.read_csv(output_dir + structured_file_name)
# df_label = pd.read_csv(input_dir + 'anomaly_label.csv')
#
# print('vector embedding...')
# embeddings = model.encode(df_template['EventTemplate'].tolist())  # num_workers=num_workers)
# df_template['Vector'] = list(embeddings)
# template_dict = df_template.set_index('EventTemplate')['Vector'].to_dict()
#
# embeddings1 = model.encode(df_template['Event Explanation'].tolist())
# df_template['VectorChat'] = list(embeddings1)
# template_dict1 = df_template.set_index('Event Explanation')['VectorChat'].to_dict()
#
#     # convert templates to vectors for all logs
# vectors = []
# vectors_chat = []
# set0 = set()
# for idx, template in enumerate(df_structured['EventTemplate']):
#     try:
#             # vectors.append(template_dict[template])
#             # vectors_chat.append(template_dict1[template])
#         template_dict[template]
#         template_dict1[template]
#
#     except KeyError:
#             # new template
#         print(template)
#         set0.add(template)
#             # vectors.append(model.encode(template))
#             # vectors_chat.append(model.encode(template))
# print(len(set0))





# import chardet
# # 以二进制的方式读取文件
# f = open('log_data/Thunderbird.log','rb')
# data = f.read()
# # 去掉['encoding']可以看完整输出，这里我做了筛选，只显示encoding
# print(chardet.detect(data)['encoding'])

# # from random import randint
# input_file_path = "log_data/Thunderbird.log"
# output_file_path = "log_data/Thunderbird_mini.log"
# # # #
# # # # # 读取前10000000行
# # #
# with open(input_file_path, "r", errors='ignore') as input_file:
#     #lines = input_file.readlines()[5*10000000:6*10000000]
#     lines = input_file.readlines()[0:10000000]
# count_sum = 0
# count_normal = 0
# count_unnormal = 0
# for line in lines:
#     count_sum += 1
#     if line[0]=='-':
#         count_normal += 1
#     else:
#         count_unnormal += 1
# print(count_sum,count_unnormal,count_normal)
#
#
# # # 将行写入新文件
# with open(output_file_path, "w") as output_file:
#     output_file.writelines(lines)


# count_sum = 0
# count_normal = 0
# count_unnormal = 0
# file_path = "log_data/Spirit.log"
# with open(file_path, "r") as file:
#
#     lines = file.readlines()
#     for line in lines:
#         count_sum += 1
#         if line[0]=='-':
#             count_normal+=1
#         else:
#             count_unnormal+=1
# print(count_sum,count_unnormal,count_normal)

# import sys
# print('Python: {}'.format(sys.version))
#
# import torch
# print(torch.__version__) #
#
# import transformers
# print(transformers.__version__)

# import pandas as pd
# import tqdm
# df = pd.read_csv('parse_result/BGL.log_structured.csv')
# print(df['Label'].value_counts())
# count_sum = 0
# count_normal = 0
# count_unnormal = 0
# print(df[0:20]["Label"].tolist())
# num_windows = int(len(df) / 20)
# #
# for i in range(num_windows):
#     count_sum +=1
#     df_blk = df[i * 20:i * 20 + 20]
#     labels = df_blk["Label"].tolist()
#     if labels == ['-'] * 20:
#         count_normal +=1
#     else:
#         count_unnormal +=1
# print(count_sum,count_normal,count_unnormal)

# import numpy as np
# log_name = 'BGL'
# training_data = np.load(
#     f'./preprocessed_data/{log_name}_training_prompt4_new_data.npz', allow_pickle=True)
# # load test data Hdfs
# # testing_data = np.load(
# #     f'./preprocessed_data/{args.log_name}_testing_data.npz', allow_pickle=True)
# testing_data = np.load(
#     f'./preprocessed_data/{log_name}_testing_prompt4_new_data.npz', allow_pickle=True)
#
#
# x_train = training_data['x']
# x_test = testing_data['x']
# print(x_train.shape)
# print(x_test.shape)
# print(len(x_train),len(x_test))
# print("read over")
#
# # 找到 x_test 中未在 x_train 中出现的日志
# unique_in_test = np.setdiff1d(x_test, x_train)
# print("unique over")
#
# # 计算未在 x_train 中出现的日志数量
# num_unique_in_test = len(unique_in_test)
# print(len(x_test))
#
# print(f'Number of unique logs in x_test not present in x_train: {num_unique_in_test}')
# print(num_unique_in_test/len(x_test))

# set_train = set(x_train.flatten())
# set_test = set(x_test.flatten())
# print("set over")
# unseen_elements = set_test - set_train
# print("-")
# num_unseen_elements = len(unseen_elements)
#
# print(f"Number of elements in x_test not present in x_train: {num_unseen_elements}")
# print(num_unseen_elements/len(set_test))

# y_train = training_data['y']
# y_test = testing_data['y']
# count_sum = 0
# count_normal = 0
# count_unnormal = 0
#
#
# for x in y_train:
#     count_sum+=1
#     if np.array_equal(x, y_train[0]):
#         count_normal+=1
#     else:
#         count_unnormal+=1
# for x in y_test:
#     count_sum+=1
#     if np.array_equal(x, y_train[0]):
#         count_normal+=1
#     else:
#         count_unnormal+=1
# print(count_sum,count_normal,count_unnormal)

# #nomal 9646206     max:482310
# #unnormal 353794
# #123000
# file_path = "log_data/Thunderbird10M.log"
# import chardet
# def get_encoding(file):
#     with open(file,'rb') as f:
#         tmp = chardet.detect(f.read(200000))
#         return tmp['encoding']
#
# print(get_encoding(file_path))

# import os
# count = 0
# with open('log_data/OpenStack.log', mode='w') as outfile:
#     for i in os.listdir('log_data/OpenStack'):
#         if i != 'abnormal_labels.txt':
#             with open(f'log_data/OpenStack/{i}', mode='r') as infile:
#                 for line in infile:
#                     outfile.write(f'{i} {line}')
#                     count += 1



# import ast
# import os
# import re
#
# import numpy as np
# import pandas as pd
# import torch
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm
#
# import Drain
# import csv

# log_name = 'HDFS'
# input_dir = './log_data/'  # The input directory of log file
# output_dir = './parse_result/'  # The output directory of parsing results
# max_length = 20
#
# def preprocess_data(df):
#     x_data, y_data = [], []
#     z_data = []
#     pbar = tqdm(total=df['BlockId'].nunique(),
#                 desc=f' data collection')
#
#     while len(df) > 0:
#         blk_id = df.iloc[0]['BlockId']
#         last_index = 0
#         for i in range(len(df)):
#             if df.iloc[i]['BlockId'] != blk_id:
#                 break
#             last_index += 1
#
#         df_blk = df[:last_index]
#
#         x_data.append(np.array(df_blk['Content'].tolist()))
#         y_index = int(df_blk.iloc[0]['Label'] == 'Anomaly')
#         y = [0, 0]
#         y[y_index] = 1
#         y_data.append(y)
#
#         df = df.iloc[last_index:]
#         pbar.update()
#     pbar.close()
#     print(len(x_data))
#
#     count_sum = 0
#     count_normal = 0
#     count_unnormal = 0
#
#     for i in range(len(x_data)):
#         #print(x_data[i].shape)
#         print(i,len(x_data))
#         count_sum+=len(x_data[i])
#         #print(y_data[i])
#         if y_data[i]==y_data[0]:
#             count_normal+=len(x_data[i])
#         else:
#             count_unnormal+=len(x_data[i])
#
#     print(count_sum,count_normal,count_unnormal)
#
#
#
#
# if __name__ == '__main__':
#
#     num_workers = 6
#
#     structured_file_name = log_name+'.log_structured.csv'
#     #template_file_name = log_name+'_new.log_templates.csv'
#
#     # load data
#     df_structured = pd.read_csv(output_dir + structured_file_name)
#     df_label = pd.read_csv(input_dir+'anomaly_label.csv')
#
#     # remove unused column
#     df_structured.drop(columns=['Date', 'Time', 'Pid', 'Level', 'Component', 'EventId', 'EventTemplate'], axis=1, inplace=True)
#
#     # extract BlockId
#     r1 = re.compile('^blk_-?[0-9]')
#     r2 = re.compile('.*blk_-?[0-9]')
#
#     paramlists = df_structured['ParameterList'].tolist()
#     blk_id_list = []
#     for paramlist in tqdm(paramlists, desc='extract BlockId'):
#         paramlist = ast.literal_eval(paramlist)
#         blk_id = list(filter(r1.match, paramlist))
#
#         if len(blk_id) == 0:
#             filter_str_list = list(filter(r2.match, paramlist))
#             # ex: '/mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906'
#             blk_id = filter_str_list[0].split(' ')[-1]
#         else:
#             # ex: ['blk_-1608999687919862906'], ['blk_-1608999687919862906', 'blk_-1608999687919862906'],
#             # ['blk_-1608999687919862906 terminating']
#             blk_id = blk_id[0].split(' ')[0]
#
#         blk_id_list.append(blk_id)
#
#     df_structured['BlockId'] = blk_id_list
#     df_structured.drop(columns=['ParameterList'], axis=1, inplace=True)
#
#
#     df_structured = pd.merge(df_structured, df_label, on='BlockId')
#
#
#     # group data by BlockId
#     df_structured.sort_values(by=['BlockId', 'LineId'], inplace=True)
#     df_structured.drop(columns=['LineId'], axis=1, inplace=True)
#
#     preprocess_data(df_structured)


#
# import numpy as np
#
# # 加载训练和测试数据
# log_name = 'Thunderbird_mini'
# training_data = np.load(f'./preprocessed_data/{log_name}_training_data.npz', allow_pickle=True)
# testing_data = np.load(f'./preprocessed_data/{log_name}_testing_data.npz', allow_pickle=True)
#
# x_train = training_data['x']
# x_test = testing_data['x']
#
# # 将训练数据转换为元组的集合以便快速查找
# train_set = set(map(tuple, x_train))
#
# # 初始化唯一测试序列的计数器
# unique_test_sequences = 0
#
# # 遍历每个测试序列
# for test_seq in x_test:
#     if tuple(test_seq) not in train_set:
#         unique_test_sequences += 1
#
# print(f"不在训练数据中的唯一测试序列数量: {unique_test_sequences}")
#
# print(unique_test_sequences / len(x_test))

# import pandas as pd
#
# def read_csv_file(filepath):
#     # 读取CSV文件
#     df = pd.read_csv(filepath)
#     return df['EventTemplate'].tolist()
#
# def split_sequences(log_lines, sequence_length):
#     # 将日志行按给定长度分割成序列
#     return [tuple(log_lines[i:i+sequence_length]) for i in range(0, len(log_lines), sequence_length)]
#
#
# def find_common_sequences(train_sequences, test_sequences):
#     # 查找测试集中出现在训练集中的序列
#     train_set = set(train_sequences)
#     common_sequences = [seq for seq in test_sequences if seq not in train_set]
#     return len(common_sequences), common_sequences
#
# def main():
#     filepath = 'parse_result/BGL.log_structured.csv'
#     sequence_length = 20
#
#     # Step 1: Read the CSV file
#     event_templates = read_csv_file(filepath)
#
#     # Step 2: Split into training and testing sets
#     train_size = int(len(event_templates) * 0.8)
#     train_lines = event_templates[:train_size]
#     test_lines = event_templates[train_size:]
#
#     temp = []
#     count = 0
#     train_set = set(train_lines)
#     print("set over")
#     for line in test_lines:
#         if line not in train_set:
#             count+=1
#             if line not in temp:
#                 temp.append(line)
#     print(count, len(test_lines))
#     print(count/len(test_lines))
#
#     for i in temp:
#         print(i)
#
# #     # Step 3: Split lines into sequences
# #     train_sequences = split_sequences(train_lines, sequence_length)
# #     test_sequences = split_sequences(test_lines, sequence_length)
# #
# #     # Step 4: Find common sequences
# #     common_count, common_sequences = find_common_sequences(train_sequences, test_sequences)
# #
# #     print(f"Number of common sequences: {common_count}")
# #     print(len(train_sequences), len(test_sequences))
# #     print(common_count/len(test_sequences))
# #     # print("Common sequences:")
# #     # for seq in common_sequences:
# #     #     print(seq)
# #
# if __name__ == "__main__":
#     main()
#
# #



# from transformers import AutoTokenizer
#
# def count_tokens_in_file(file_path, model_name='model/bert'):
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         total_tokens = 0
#
#         with open(file_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 tokens = tokenizer.tokenize(line)
#                 total_tokens += len(tokens)
#
#         return total_tokens
#     except FileNotFoundError:
#         print(f"The file {file_path} was not found.")
#         return 0
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return 0
#
#
# # Example usage:
# file_path = 'log_data/Spirit.log'
# token_count = count_tokens_in_file(file_path)
# print(f"The number of tokens in the file is: {token_count}")
# print(f"It will cost:{token_count/1000000*0.5}")


#
# from transformers import AutoTokenizer
# import pandas as pd
# def count_tokens_in_file(data, model_name='model/bert'):
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         total_tokens = 0
#
#         # with open(file_path, 'r', encoding='utf-8') as file:
#         #     for line in file:
#         #         tokens = tokenizer.tokenize(line)
#         #         total_tokens += len(tokens)
#         for index,row in data.iterrows():
#             tokens = tokenizer.tokenize(row['EventTemplate'])
#             total_tokens += len(tokens)
#         return total_tokens
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return 0
#
#
# # Example usage:
# file_path = 'parse_result/Spirit_new.log_templates.csv'
# data = pd.read_csv(file_path)
# token_count = count_tokens_in_file(data)
# print(f"The number of tokens in the file is: {token_count}")
# print(f"It will cost:{token_count/1000000*0.5}")


#

# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('parse_result/Spirit.log_structured.csv')
#
# # 获取Content列中不重复的元素
# unique_elements = df['Content'].unique()
#
# # 打印不重复元素的数量
# print("Content列中不重复的元素数量:", len(unique_elements))

# import numpy as np
# log_name = 'Spirit'
# training_data = np.load(
#     f'./preprocessed_data/{log_name}_training_prompt4_new_data.npz', allow_pickle=True)
# # load test data Hdfs
# # testing_data = np.load(
# #     f'./preprocessed_data/{args.log_name}_testing_data.npz', allow_pickle=True)
# testing_data = np.load(
#     f'./preprocessed_data/{log_name}_testing_prompt4_new_data.npz', allow_pickle=True)
#
# x_train, y_train, z_train = training_data['x'], training_data['y'], training_data['z']
# x_test, y_test, z_test = testing_data['x'], testing_data['y'], testing_data['z']
#
# print(len(y_train),len(y_test))



import math
from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=3072, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,**factory_kwargs)#创建了一个多头自注意力层。


        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.dropout1(src2)
        src = self.norm1(src + src2)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(src2)
        src = self.norm2(src + src2)

        return src

    # def activate_adapter(self):
    #     tune_layers = [self.adapter1, self.adapter2, self.norm1, self.norm2]
    #     for layer in tune_layers:
    #         for param in layer.parameters():
    #             param.requires_grad = True
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, num_layers=4, dim=768, window_size=100, nhead=8, dim_feedforward=3072, dropout=0.1):
        super(Model, self).__init__()
        #encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward, dropout, batch_first=True)

        #------------------------------------------new------------------------------------------
        self.attention1 = nn.MultiheadAttention(dim,nhead)
        self.attention2 = nn.MultiheadAttention(dim,nhead)
        self.fc = nn.Linear(dim*2, dim)
        # ------------------------------------------new------------------------------------------
        encoder_layer = TransformerEncoderLayer(
                dim, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

        self.trans_encder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers)
        self.pos_encoder1 = PositionalEncoding(d_model=768)
        # self.pos_encoder2 = LearnedPositionEncoding(
        #    d_model=768, max_len=window_size)
        self.fc1 = nn.Linear(dim * window_size, 2)

    def forward(self, x, z):
        B, _, _ = x.size()
        # ------------------------------------------new------------------------------------------
        # print(x.shape)
        # print(z.shape)
        output1, _ = self.attention1(x.transpose(0, 1), z.transpose(0, 1), z.transpose(0, 1))
        output2, _ = self.attention2(z.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        #print(output1.shape)
        output1 = output1.transpose(0, 1)
        output2 = output2.transpose(0, 1)
        output1 = F.layer_norm(x + output1, normalized_shape=[output1.size(-1)])
        output2 = F.layer_norm(z + output2, normalized_shape=[output2.size(-1)])
        fused_embedding = torch.cat((output1, output2), dim=-1)
        fused_embedding = F.relu(self.fc(fused_embedding))
        # ------------------------------------------new------------------------------------------

        fused_embedding = self.trans_encder(fused_embedding)
        fused_embedding = fused_embedding.contiguous().view(B, -1)
        fused_embedding = self.fc1(fused_embedding)
        return fused_embedding
num_layers = 1
EMBEDDING_DIM = 768
window_size = 20
model = Model(num_layers=num_layers, dim=EMBEDDING_DIM, window_size=window_size, nhead=8, dim_feedforward=4 *
              EMBEDDING_DIM, dropout=0.1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# 打印模型的参数量
print(f"Model has {count_parameters(model):,} parameters")





