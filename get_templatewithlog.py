import pandas as pd

#HDFS:Index(['LineId', 'Date', 'Time', 'Pid', 'Level', 'Component', 'Content','EventId', 'EventTemplate', 'ParameterList']
#BGL:['LineId', 'Label', 'Timestamp', 'Date', 'Node', 'Time', 'NodeRepeat','Type', 'Component', 'Level', 'Content', 'EventId', 'EventTemplate','ParameterList']
#Spirit:['LineId', 'Label', 'TimeStamp', 'Date', 'User', 'Month', 'Day', 'Time','UserGroup', 'Component', 'PID', 'Content', 'EventId', 'EventTemplate','ParameterList']
#Thunderbird:['LineId', 'Label', 'Id', 'Date', 'Admin', 'Month', 'Day', 'Time','AdminAddr', 'Content', 'EventId', 'EventTemplate']
df = pd.read_csv("parse_result/Thunderbird_mini.log_structured.csv")
df_sub = df[['Content', 'EventTemplate']]
dic = {}

for index,row in df_sub.iterrows():
    if row['EventTemplate'] not in dic.keys():
        dic[row['EventTemplate']] = row['Content']

print(len(dic))
df_new = pd.DataFrame(list(dic.items()), columns=['EventTemplate', 'log'])
df_new.to_csv("parse_result/Thunderbird_miniwithlog.log_templates.csv")