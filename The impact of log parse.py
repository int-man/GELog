# #
# from openai import OpenAI
# import pandas as pd
#
# # df = pd.read_csv("parse_result/Spirit.log_templates.csv")
# # df1 = pd.read_csv("parse_result/Spirit_new.log_templates.csv")
# #
# # flag = 0
# # for i in range(0,len(df)):
# #
# #     if df.iloc[i]["EventTemplate"] == "Stopping <*> succeeded":
# #         flag =1
# #     if flag == 1:
# #         print(i,len(df))
# #         print(df.iloc[i]["EventTemplate"])
# #         content = "I will give you a template for a log from Spirit supercomputer and ask you to explain what the template means. Your answer should be less than 400 tokens. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\""
# #         print(content)
# #
# #         client = OpenAI(
# #             base_url="https://api.ai.cs.ac.cn/v1",
# #             # 填写 ETOChat 密钥
# #             api_key="sk-5JsKgj2mHon9KhOlXz2XSTpCOFo0wTARMBxFoIc7teUjzGnu",
# #         )
# #
# #         chat_completion = client.chat.completions.create(
# #             messages=[
# #                 {
# #                     "role": "user",
# #                     "content": content}
# #             ],
# #             model="gpt-3.5-turbo",
# #         )
# #
# #         # print(chat_completion['choices'][0]['text'])
# #         reply_text = chat_completion.choices[0].message.content
# #         print(reply_text)
# #
 # #         df1.at[i, 'Event Explanation'] = reply_text
# #         df1.to_csv('parse_result/Spirit_new.log_templates.csv', index=False)
# #
#
# # df = pd.read_csv("parse_result/HDFSwithlog.log_templates.csv")
# # df = pd.read_csv("parse_result/BGLwithlog.log_templates.csv")
# #df = pd.read_csv("parse_result/Thunderbird_m.
# # 0iniwithlog.log_templates.csv")
# df = pd.read_csv("parse_result/HDFSwithlog.log_templates1.csv")#Spiritwithlog.log_templates.csv
#
#
# for i in range(0,len(df)):
#
#     print(i,len(df))
#     print(df.iloc[i]["EventTemplate"])
#
#     #Hadoop Distributed File System/ BlueGene/L supercomputer / XXX supercomputer
#     content = "I will give you a template for a log from Hadoop Distributed File System and ask you to explain what the template means. Your answer should be less than 400 tokens. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\""
#     #content = "I will give you a template for a log and ask you to explain what the template means. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\" and the corresponding log example for this template is \""+df.iloc[i]["log"]+"\".Your answer should be less than 400 tokens."
#     #content = "Assuming you are a software developer for Hadoop Distributed File System. I will provide you with a log of the Hadoop Distributed File System. Please simulate the log evolution process of software updates and modify this log. During the update process, some words can be replaced, deleted, or added, but do not change the semantics of the entire log. The log you need to modify is:\""+df.iloc[i]["EventTemplate"]+"\". You just need to tell me the modified log template and don't give any other answers."
#     #content = "I will give you a template for a log and ask you to explain what the template means. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\". Your answer should be less than 400 tokens."
#
#     print(content)
#
#
#     client = OpenAI(
#         base_url="https://api.ai.cs.ac.cn/v1",
#         # 填写 ETOChat 密钥
#         api_key="sk-5JsKgj2mHon9KhOlXz2XSTpCOFo0wTARMBxFoIc7teUjzGnu",
#     )
#
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": content}
#         ],
#         #model="gpt-3.5-turbo",
#         model = "gpt-3.5-turbo",
#     )
#
#     # print(chat_completion['choices'][0]['text'])
#     reply_text = chat_completion.choices[0].message.content
#     print(reply_text)
#
#     df.at[i, 'Event Explanation'] = reply_text
#
#     df.to_csv('parse_result/HDFS_prompt_evolution1_withhuman_new.log_templates.csv', index=False)
#     # df.to_csv('parse_result/BGL_prompt_withhuman_new.log_templates.csv', index=False)
#     #df.to_csv('parse_result/Thunderbird_mini_prompt_withhuman_new.log_templates.csv', index=False)
#     #df.to_csv('parse_result/HDFS_prompt_evolution1_new.log_templates.csv', index=False)
#




import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import json
import random

def replace_words_with_placeholder(sentence, num_words_to_replace=1):
    words = sentence.split()
    non_placeholder_indices = [i for i, word in enumerate(words) if word != '<*>']
    if len(non_placeholder_indices) <= num_words_to_replace:
        # 如果非 '<*>' 单词的数量少于或等于要替换的数量，则全替换
        num_words_to_replace = len(non_placeholder_indices)
    indices_to_replace = random.sample(non_placeholder_indices, num_words_to_replace)
    for idx in indices_to_replace:
        words[idx] = '<*>'
    return ' '.join(words)


# 加载数据
df = pd.read_csv('parse_result/HDFSwithlog.log_templates.csv')
df['EventTemplate_new'] = df['EventTemplate'].apply(lambda x: replace_words_with_placeholder(x, num_words_to_replace=2))


#初始化SentenceTransformer模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(
    'distilbert-base-nli-mean-tokens', device=device)

#提取EventTemplate1和EventTemplate2的向量
embeddings1 = model.encode(df['EventTemplate'].tolist())
embeddings2 = model.encode(df['EventTemplate_new'].tolist())
embeddings3 = model.encode(df['log'].tolist())



# 使用PCA将向量降维到二维
pca = PCA(n_components=2)
reduced_embeddings1 = pca.fit_transform(embeddings1)
reduced_embeddings2 = pca.fit_transform(embeddings2)
reduced_embeddings3 = pca.fit_transform(embeddings3)

# 创建一个散点图
plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings1[:, 0], reduced_embeddings1[:, 1], c='blue', label='Original log template')
plt.scatter(reduced_embeddings3[:, 0], reduced_embeddings3[:, 1], c='green', label='Under-parsing log template')
plt.scatter(reduced_embeddings2[:, 0], reduced_embeddings2[:, 1], c='red', label='Misparsing log template')


# 为每个点添加序号
for i, (x, y) in enumerate(reduced_embeddings1):
    plt.annotate(i, (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=15, color='blue')

for i, (x, y) in enumerate(reduced_embeddings3):
    plt.annotate(i, (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=15, color='green')

for i, (x, y) in enumerate(reduced_embeddings2):
    plt.annotate(i, (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=15, color='red')




plt.legend(fontsize=15)
plt.title('Impact of log parsing noise',fontsize=20)
plt.xlabel('Principal Component 1',fontsize=20)
plt.ylabel('Principal Component 2', fontsize=20)
plt.savefig('picture/EMPIRICAL STUDY/ImpactOfLogParsing.pdf')

plt.show()



