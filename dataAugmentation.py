
from openai import OpenAI
import pandas as pd

# df = pd.read_csv("parse_result/Spirit.log_templates.csv")
# df1 = pd.read_csv("parse_result/Spirit_new.log_templates.csv")
#
# flag = 0
# for i in range(0,len(df)):
#
#     if df.iloc[i]["EventTemplate"] == "Stopping <*> succeeded":
#         flag =1
#     if flag == 1:
#         print(i,len(df))
#         print(df.iloc[i]["EventTemplate"])
#         content = "I will give you a template for a log from Spirit supercomputer and ask you to explain what the template means. Your answer should be less than 400 tokens. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\""
#         print(content)
#
#         client = OpenAI(
#             base_url="https://api.ai.cs.ac.cn/v1",
#             # 填写 ETOChat 密钥
#             api_key="sk-5JsKgj2mHon9KhOlXz2XSTpCOFo0wTARMBxFoIc7teUjzGnu",
#         )
#
#         chat_completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": content}
#             ],
#             model="gpt-3.5-turbo",
#         )
#
#         # print(chat_completion['choices'][0]['text'])
#         reply_text = chat_completion.choices[0].message.content
#         print(reply_text)
#
#         df1.at[i, 'Event Explanation'] = reply_text
#         df1.to_csv('parse_result/Spirit_new.log_templates.csv', index=False)
#

# df = pd.read_csv("parse_result/HDFSwithlog.log_templates.csv")
# df = pd.read_csv("parse_result/BGLwithlog.log_templates.csv")
#df = pd.read_csv("parse_result/Thunderbird_m.
# 0iniwithlog.log_templates.csv")
df = pd.read_csv("parse_result/Spiritwithlog.log_templates.csv")#Spiritwithlog.log_templates.csv


for i in range(0,len(df)):

    print(i,len(df))
    print(df.iloc[i]["EventTemplate"])

    #Hadoop Distributed File System/ BlueGene/L supercomputer / XXX supercomputer
    #content = "I will give you a template for a log from Spirit supercomputer and ask you to explain what the template means. Your answer should be less than 400 tokens. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\""
    #content = "I will give you a template for a log and ask you to explain what the template means. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\" and the corresponding log example for this template is \""+df.iloc[i]["log"]+"\".Your answer should be less than 400 tokens."
    content = "Assuming you are a log analysis expert in the field of operations and maintenance. I will give you a template for a log from BlueGene/L supercomputer and ask you to explain what the template means. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\" and the corresponding log example for this template is \""+df.iloc[i]["log"]+"\". The '<*>' in the template is a placeholder, and you need to explain the meaning of the entire log template and the meaning of each placeholder."
    #content = "I will give you a template for a log and ask you to explain what the template means. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\". Your answer should be less than 400 tokens."

    #content = "Assuming you are a log analysis expert in the field of operations and maintenance. I will give you a template for a log from Hadoop Distributed File System and ask you to explain what the template means. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\" and the corresponding log example for this template is \""+df.iloc[i]["log"]+"\". The<*>in the template is a placeholder, and you need to explain the meaning of the entire log template and the meaning of each placeholder. Your answer should be less than 400 tokens. Here is a example：For the log template \"Receiving block<*>src:<*>dest:<*>\", the first<*>means the identifier of the accepted block, the second<*>means the source address identifier of the accepted block, and the third<*>means the destination address identifier of the accepted block. The meaning of this log template is to move a block from the source address to the destination address."
    #content = "Assuming you are a log analysis expert in the field of operations and maintenance. I will give you a template for a log from BlueGene/L supercomputer and ask you to explain what the template means. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\" and the corresponding log example for this template is \""+df.iloc[i]["log"]+"\". The<*>in the template is a placeholder, and you need to explain the meaning of the entire log template and the meaning of each placeholder. Your answer should be less than 400 tokens. Here is a example: For the log template \"MidplaneSwitchController performing bit sparring on<*>bit<*>\", the first<*>refers to the specific location of the device in the data center or server rack, the second<*>refers to the number of bits that perform the specific operation, and the entire log means that the MidplaneSwitchController has detected a bit corruption and is performing bit sparring"
    #content = "Assuming you are a log analysis expert in the field of operations and maintenance. I will give you a template for a log from Thunderbird supercomputer and ask you to explain what the template means. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\" and the corresponding log example for this template is \""+df.iloc[i]["log"]+"\". The<*>in the template is a placeholder, and you need to explain the meaning of the entire log template and the meaning of each placeholder. Your answer should be less than 400 tokens. Here is a example: For the log template \"in.tftpd[<*>]: tftp: client does not accept options\", the first<*>refers to a specific process ID (PID) used to identify the process number of the current running TFTP server instance. The meaning of the entire log is to indicate that the server attempted to negotiate certain transmission options with the client, but the client did not accept these options."
    #content = "Assuming you are a log analysis expert in the field of operations and maintenance. I will give you a template for a log from Spirit supercomputer and ask you to explain what the template means. The template you need to explain is \""+df.iloc[i]["EventTemplate"]+"\" and the corresponding log example for this template is \""+df.iloc[i]["log"]+"\". The <*> in the template is a placeholder, and you need to explain the meaning of the entire log template and the meaning of each placeholder. Your answer should be less than 400 tokens. Here is a example: For the log template \"DHCPDISCVER from <*> via <*> network <*> no free releases\", the first <*> refers to the MAC address of a specific client device sending a DHCP discovery request, the second <*> refers to the network interface through which the DHCPDISCVER request was received, and the third <*> refers to the subnet or network range where the request is located. The entire log represents a client device sending a DHCP discovery request through a specific network interface, but the DHCP server does not have an available IP address lease in the specified network that can be assigned to the client."

    print(content)


    client = OpenAI(
        base_url="https://api.ai.cs.ac.cn/v1",
        # 填写 ETOChat 密钥
        api_key="sk-5JsKgj2mHon9KhOlXz2XSTpCOFo0wTARMBxFoIc7teUjzGnu",
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content}
        ],
        model="gpt-3.5-turbo",
    )

    # print(chat_completion['choices'][0]['text'])
    reply_text = chat_completion.choices[0].message.content
    print(reply_text)

    df.at[i, 'Event Explanation'] = reply_text

    # df.to_csv('parse_result/HDFS_prompt_withhuman_new.log_templates.csv', index=False)
    # df.to_csv('parse_result/BGL_prompt_withhuman_new.log_templates.csv', index=False)
    #df.to_csv('parse_result/Thunderbird_mini_prompt_withhuman_new.log_templates.csv', index=False)
    df.to_csv('parse_result/BGL_prompt4(0619)_new.log_templates.csv', index=False)


