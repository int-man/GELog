# import pandas as pd
# import re
# import ast
# def extract_parameters(content, template):
#     # 将模板中的<*>替换为正则表达式的捕获组
#     placeholder = '__PLACEHOLDER__'
#     template_temp = template.replace('<*>', placeholder)
#
#     pattern = re.escape(template_temp)
#     pattern = pattern.replace(re.escape(placeholder), '(.*?)')
#     # 使用正则表达式匹配日志内容
#     match = re.match(pattern, content)
#     if match:
#         # 返回所有匹配的捕获组
#         # print(match.groups())
#         return match.groups()
#     else:
#         return ()
#
# def fill_template(template, parameters):
#     # 使用参数填充模板
#     parameters1 = ast.literal_eval(parameters)
#     for param in parameters1:
#         # print(param)
#         # print(parameters1)
#         template = template.replace('<*>', param, 1)
#     return template
#
#
# def replace_event_template(df, output_path, mapping_dict):
#     # 读取CSV文件
#     # df = pd.read_csv(file_path)
#
#     # 使用字典映射替换EventTemplate列中的值
#     df['EventTemplate'] = df['EventTemplate'].map(mapping_dict).fillna(df['EventTemplate'])
#
#     # 将修改后的DataFrame保存回CSV文件
#     df.to_csv(output_path, index=False)
#
# # 加载数据
# df = pd.read_csv('parse_result/HDFS_prompt_evolution_withhuman_new.log_templates.csv')
# df1 = pd.read_csv('parse_result/HDFS_prompt_evolution1_withhuman_new.log_templates.csv')
#
# # 检查两个DataFrame的长度是否一致
# if len(df) != len(df1):
#     raise ValueError("DataFrames df and df1 do not have the same length")
#
# # 创建一个字典，将df的EventTemplate映射到df1的EventTemplate
# mapping_dict = {df['EventTemplate'].iloc[i]: df1['EventTemplate'].iloc[i] for i in range(len(df))}
#
# # 打印映射字典
# # print(mapping_dict)
#
# HDFS_struct = pd.read_csv('parse_result/HDFS.log_structured.csv')
# # start_index = int(len(df) * 0.8)
# # HDFS_struct_train = HDFS_struct[:start_index]
# # HDFS_struct_testing = HDFS_struct.iloc[start_index:]
#
# # del HDFS_struct
#
# # # 定义一个函数，根据EventTemplate从content中提取参数
# #
# # HDFS_struct_testing['Parameters'] = HDFS_struct_testing.apply(lambda row: extract_parameters(row['Content'], row['EventTemplate']), axis=1)
# #
# # print(mapping_dict)
#
#
# # print(HDFS_struct_testing['ParameterList'])
#
#
# #
# def generate_new_content(row):
#     original_template = row['EventTemplate']
#     #print(original_template)
#     parameters = row['ParameterList']
#     #print(parameters)
#     if original_template in mapping_dict:
#         new_template = mapping_dict[original_template]
#         return fill_template(new_template, parameters)
#     else:
#         print(original_template)
#         print("ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
#         return row['Content']  # 如果找不到对应的模板，则保持原内容不变
# #
# HDFS_struct['NewContent'] = HDFS_struct.apply(generate_new_content, axis=1)
#
# #
# #
# # # 拼接 HDFS_struct_train 和 HDFS_struct_testing
# # HDFS_struct_combined = pd.concat([HDFS_struct_train, HDFS_struct_testing])
#
# # for index, row in HDFS_struct_testing[:15]:
# #     print(row['NewContent'])
#
# replace_event_template(HDFS_struct,'parse_result/HDFS_struct_combined_evolution.csv',mapping_dict)
# # 将拼接后的数据保存到新的 CSV 文件
# # HDFS_struct.to_csv('parse_result/HDFS_struct_combined_evolution.csv', index=False)



# import csv
# csv_file_path = 'parse_result/HDFS_struct_combined_evolution.csv'
# txt_file_path = 'log_data/HDFS_evoluation.log'
#
# # start_index = int(len(df) * 0.8)
# # columns_to_select_first_80  = ['Date', 'Time', 'Pid', 'Level', 'Component', 'Content']
# columns_to_select  = ['Date', 'Time', 'Pid', 'Level', 'Component', 'NewContent']
#
# with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file, open(txt_file_path, 'w',
#                                                                               encoding='utf-8') as txt_file:
#     # 创建CSV字典读取器
#     csv_reader = csv.DictReader(csv_file)
#
#     # 将CSV文件内容读入列表
#     rows = list(csv_reader)
#     # # 计算分割点
#     # split_point = int(0.8 * len(rows))
#
#     # 处理前80%的数据
#     # for row in rows[:split_point]:
#     #     selected_columns = [row[column] + (':' if column == 'Component' else '') for column in
#     #                         columns_to_select_first_80 if column in row]
#     #     txt_file.write(' '.join(selected_columns) + '\n')
#
#         # 处理后20%的数据
#     for row in rows:
#         selected_columns = [row[column] + (':' if column == 'Component' else '') for column in columns_to_select
#                             if column in row]
#         txt_file.write(' '.join(selected_columns) + '\n')



#
# 打开原始日志文件进行读取
# 打开原始日志文件进行读取
# with open('log_data/HDFS_evoluation.log', 'r') as file:
#     lines = file.readlines()
#
# # 打开一个新的文件进行写入
# with open('log_data/HDFS_evolution_updated.log', 'w') as file:
#     for line in lines:
#         stripped_line = line.strip()
#
#         # 跳过空行
#         if not stripped_line:
#             continue
#
#         # 在每行的开头添加一个0
#         modified_line = '0' + stripped_line
#
#         # 按空格分割行
#         parts = modified_line.split()
#
#         # 检查是否有第二个元素，且其长度小于6
#         if len(parts) > 1 and len(parts[1]) < 6:
#             # 补充0至长度为6
#             parts[1] = parts[1].zfill(6)
#
#         # 将修改后的行重新拼接并写入新的文件
#         file.write(' '.join(parts) + '\n')

# import pandas as pd
#
#
# def replace_and_remove_newcontent(file_path, output_path):
#     # 读取CSV文件
#     df = pd.read_csv(file_path)
#
#     # 用NewContent列的值替换Content列的值
#     df['Content'] = df['NewContent']
#
#     # 删除NewContent列
#     df.drop(columns=['NewContent'], inplace=True)
#
#     # 将修改后的DataFrame保存回CSV文件
#     df.to_csv(output_path, index=False)
#
#
# # 示例：使用函数读取、修改并保存CSV文件
# input_file_path = 'parse_result/HDFS_struct_combined_evolution.csv'
# output_file_path = 'parse_result/HDFS_struct_combined_evolution_baseline.csv'
# replace_and_remove_newcontent(input_file_path, output_file_path)
#
# # 显示修改后的前几行数据以检查修改
# modified_df = pd.read_csv(output_file_path)
# print(modified_df['EventTemplate'].head(5))



# import pandas as pd
#
# df = pd.read_csv('parse_result/HDFS_struct_combined_evolution.csv')
# print(df['EventTemplate'].head(5))
#
# df1= pd.read_csv('parse_result/HDFS.log_structured.csv')
# print(df1['EventTemplate'].head(5))