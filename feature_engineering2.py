#!/usr/bin/env python
# coding: utf-8

# @Copyright IQIYI 2021
# http://challenge.ai.iqiyi.com/

# In[1]:


import pandas as pd
import numpy as np
from itertools import groupby


# In[2]:


input_dir = "./"
output_dir = "./wsdm_model_data_v6/"


# In[3]:


launch = pd.read_csv(input_dir + "app_launch_logs.csv")
launch.date.min(), launch.date.max()


# In[4]:


launch_grp = launch.groupby("user_id").agg(
    launch_date=("date", list),
    launch_type=("launch_type", list)
).reset_index()
launch_grp


# In[5]:


# generating a sample for each user, a sample should has an anchor date for label
# best solution
def choose_end_date(launch_date):
    # 尽量让数据落在131到222这个区间里面,最好能向(160,222)这个区间靠近
    min_value, max_value = 131, 222
    
    n1, n2 = min(launch_date), max(launch_date)
    
    if n1 < n2 - 7:
        # 说明用户登录时间丰度是比较多的
        if n1<min_value and n2 < max_value:
            if n2 - min_value > 7:
                end_date = np.random.randint(min_value, n2 - 7)
            else:
                end_date = np.random.randint(n1, max_value - 7)
        else:
            end_date = np.random.randint(n1, n2 - 7)
    else:
        if n1 > min_value:
            end_date = np.random.randint(n1, max_value - 7)
        elif n1 < min_value and n2 > min_value:
            end_date = np.random.randint(min_value, max_value - 7)
        else:
            end_date = np.random.randint(160, 222-7)
    return int(end_date)

# def choose_end_date(launch_date):
#     # 尽量让数据落在131到222这个区间里面
#     min_value, max_value = 131, 222
#     # 尽量给(160,222)这个区间里面多分一些数据，但也要考虑是否会变得过拟合
#     n1, n2 = min(launch_date), max(launch_date)
#
#     if n1 < n2 - 7:
#         # 说明用户登录时间丰度是比较多的
#         if n1<min_value and n2 < max_value:
#             if n2 - min_value > 7 and n2-7 < 160:
#                 end_date = np.random.randint(min_value, n2 - 7)
#             elif n2 - min_value <= 7 and n2-7 > 160:
#                 end_date = np.random.randint(160, n2-7)
#             else:
#                 end_date = np.random.randint(n1, n2-7)
#         else:
#             if n2 - 7 < 160:
#                 end_date = np.random.randint(n1, max_value - 7)
#             else:
#                 end_date = np.random.randint(n1, n2-7)
#     else:
#         if n1 > min_value:
#             if n2 >= 160:
#                 end_date = np.random.randint(160, max_value - 7)
#             else:
#                 end_date = np.random.randint(n1, max_value - 7)
#         elif n1 < min_value and n2 > min_value:
#             end_date = np.random.randint(min_value, max_value - 7)
#         else:
#             end_date = np.random.randint(160, 222-7)
#     return int(end_date)

# def choose_end_date(launch_date):
#     # 尽量让数据落在160到222这个区间里面
#     min_value, max_value = 131, 222
    
#     n1, n2 = min(launch_date), max(launch_date)
    
#     if n1 < n2 - 7:
#         # 说明用户登录时间丰度是比较多的
#         if n1<min_value and n2 < max_value:
#             if n2 - min_value > 7 and n2 - 7 < 160:
#                 end_date = np.random.randint(min_value, n2 - 7)
#             elif n2 - min_value > 7 and n2 - 7 > 160:
#                 # 就在(160, max_value)中生成
#                 end_date = np.random.randint(160, max_value-7)
#             else:
#                 end_date = np.random.randint(n1, max_value - 7)
#         else:
#             # 如果在(160, max_value-7)之间
#             if n2 - 7 < 160:
#                end_date = np.random.randint(min_value, max_value-7)
#             else:     
#                 end_date = np.random.randint(n1, n2 - 7)
#     else:
#         if n1 > min_value:
#             end_date = np.random.randint(n1, max_value - 7)
#         elif n1 < min_value and n2 > min_value:
#             end_date = np.random.randint(min_value, max_value - 7)
#         else:
#             end_date = np.random.randint(160, max_value-7)
#     return int(end_date)
launch_grp["end_date"] = launch_grp.launch_date.apply(choose_end_date)
launch_grp
# In[6]:


def get_label(row):
    launch_list = row.launch_date
    end = row.end_date
    label = sum([1 for x in set(launch_list) if end < x < end+8])
    return label
launch_grp["label"] = launch_grp.apply(get_label, axis=1)
launch_grp


# In[7]:


launch_grp.label.value_counts()


# In[9]:


train = launch_grp[["user_id", "end_date", "label"]]
train


# In[10]:


# reading test data
test = pd.read_csv(input_dir + "test-b.csv")
test["label"] = -1
test


# In[11]:


# concat train and test data
data = pd.concat([train, test], ignore_index=True)
data


# # launch data process

# In[12]:


# append test data to launch_grp
launch_grp = launch_grp.append(
    test.merge(launch_grp[["user_id", "launch_type", "launch_date"]], how="left", on="user_id")
)
launch_grp


# In[13]:


# get latest 32 days([end_date-31, end_date]) launch type sequence
# 0 for not launch, 1 for launch_type=0, and 2 for launch_type=1
def gen_launch_seq(row):
    seq_sort = sorted(zip(row.launch_type, row.launch_date), key=lambda x: x[1])
    seq_map = {k: max(g)[0] + 1 for k, g in groupby(seq_sort, lambda x: x[1])}
    end = row.end_date
    seq = [seq_map.get(x, 0) for x in range(end-31, end+1)]
    return seq
launch_grp["launch_seq"] = launch_grp.apply(gen_launch_seq, axis=1)
launch_grp


# In[15]:


data = data.merge(
    launch_grp[["user_id", "end_date", "label", "launch_seq"]],
    on=["user_id", "end_date", "label"],
    how="left"
)
data


# # playback data and video data process

# In[16]:


# choose playback data in [end_date-31, end_date]
playback = pd.read_csv(input_dir + "user_playback_data.csv", dtype={"item_id": str})
playback = playback.merge(data, how="inner", on="user_id")
playback = playback.loc[(playback.date >= playback.end_date-31) & (playback.date <= playback.end_date)]
playback


# In[17]:


# add video info to playback data
video_data = pd.read_csv(input_dir + "video_related_data.csv", dtype=str)
playback = playback.merge(video_data[video_data.item_id.notna()], how="left", on="item_id")
playback


# In[18]:


# using target encoding
# Tutorial: https://www.kaggle.com/ryanholbrook/target-encoding
def target_encoding(name, df, m=1):
    df[name] = df[name].str.split(";")
    df = df.explode(name)
    overall = df["label"].mean()
    df = df.groupby(name).agg(
        freq=("label", "count"), 
        in_category=("label", np.mean)
    ).reset_index()
    df["weight"] = df["freq"] / (df["freq"] + m)
    df["score"] = df["weight"] * df["in_category"] + (1 - df["weight"]) * overall
    return df


# In[19]:


# father_id target encoding
df = playback.loc[(playback.label >= 0) & (playback.father_id.notna()), ["father_id", "label"]]
father_id_score = target_encoding("father_id", df)
father_id_score


# In[20]:


# tag_id target encoding
df = playback.loc[(playback.label >= 0) & (playback.tag_list.notna()), ["tag_list", "label"]]
tag_id_score = target_encoding("tag_list", df)
tag_id_score.rename({"tag_list": "tag_id"}, axis=1, inplace=True)
tag_id_score


# In[21]:


# cast_id target encoding
df = playback.loc[(playback.label >= 0) & (playback.cast.notna()), ["cast", "label"]]
cast_id_score = target_encoding("cast", df)
cast_id_score.rename({"cast": "cast_id"}, axis=1, inplace=True)
cast_id_score


# In[22]:


# group playback data for feature engineering
playback_grp = playback.groupby(["user_id", "end_date", "label"]).agg(
    playtime_list=("playtime", list),
    date_list=("date", list),
    duration_list=("duration", lambda x: ";".join(map(str, x))),
    father_id_list=("father_id", lambda x: ";".join(map(str, x))),
    tag_list=("tag_list", lambda x: ";".join(map(str, x))),
    cast_list=("cast", lambda x: ";".join(map(str, x)))
).reset_index()
playback_grp


# In[23]:


# generate latest 32 days([end_date-31, end_date]) playtime sequence
# playtime_norm = 1/(1 + exp(3 - playtime/450)). when playtime=3600s it's preference score is almost equal to 1
def get_playtime_seq(row):
    seq_sort = sorted(zip(row.playtime_list, row.date_list), key=lambda x: x[1])
    seq_map = {k: sum(x[0] for x in g) for k, g in groupby(seq_sort, key=lambda x: x[1])}
    seq_norm = {k: 1/(1+np.exp(3-v/450)) for k, v in seq_map.items()}
    seq = [round(seq_norm.get(i, 0), 4) for i in range(row.end_date-31, row.end_date+1)]
    return seq
playback_grp["playtime_seq"] = playback_grp.apply(get_playtime_seq, axis=1)
playback_grp


# In[24]:


drn_desc = video_data.loc[video_data.duration.notna(), "duration"].astype(int)
drn_desc.min(), drn_desc.max()


# In[25]:


# duration preference is a 16-dimentional prefer vector
# for a user, count the frequency of each duration
# prefer_score = freq / max(freq)
# if the user's duration_list is all null, then return null
# null duration_prefer will later be filled with zero vector
def get_duration_prefer(duration_list):
    drn_list = sorted(duration_list.split(";"))
    drn_map = {k: sum(1 for _ in g) for k, g in groupby(drn_list) if k != "nan"}
    if drn_map:
        max_ = max(drn_map.values())
        res = [round(drn_map.get(str(i), 0)/max_, 4) for i in range(1, 17)]
        return res
    else:
        return np.nan
playback_grp["duration_prefer"] = playback_grp.duration_list.apply(get_duration_prefer)


# In[26]:


# add all target encoding scores into a dict
id_score = dict()
id_score.update({x[1]: x[5] for x in father_id_score.itertuples()})
id_score.update({x[1]: x[5] for x in tag_id_score.itertuples()})
id_score.update({x[1]: x[5] for x in cast_id_score.itertuples()})

# check if features ids are duplicated
father_id_score.shape[0]+tag_id_score.shape[0]+cast_id_score.shape[0] == len(id_score)


# In[27]:


# for these 3 features: father_id_score, cast_score, tag_score,
# choose top 3 preferences
# e.g top_3_id = [(id, freq), (id, freq), (id, freq)]
# score = weight_avg(top_3_id), which values are id_score, weights are frequency
# if the id_list is all null, then return null
def get_id_score(id_list):
    x = sorted(id_list.split(";"))
    x_count = {k: sum(1 for _ in g) for k, g in groupby(x) if k != "nan"}
    if x_count:
        x_sort = sorted(x_count.items(), key=lambda k: -k[1])
        top_x = x_sort[:3]
        res = [(n, id_score.get(k, 0)) for k, n in top_x]
        res = sum(n*v for n, v in res) / sum(n for n, v in res)
        return res
    else:
        return np.nan


# In[28]:


playback_grp["father_id_score"] = playback_grp.father_id_list.apply(get_id_score)


# In[29]:


playback_grp["cast_id_score"] = playback_grp.cast_list.apply(get_id_score)


# In[30]:


playback_grp["tag_score"] = playback_grp.tag_list.apply(get_id_score)
playback_grp


# In[31]:


data = data.merge(
    playback_grp[["user_id", "end_date", "label", "playtime_seq", "duration_prefer", "father_id_score", "cast_id_score", "tag_score"]],
    on=["user_id", "end_date", "label"],
    how="left"
)
data


# # user portrait data process

# In[32]:


portrait = pd.read_csv(input_dir + "user_portrait_data.csv", dtype={"territory_code": str})
portrait = pd.merge(data[["user_id", "label"]], portrait, how="left", on="user_id")
portrait


# In[33]:


# for territory_code, using target encoding again
df = portrait.loc[(portrait.label >= 0) & (portrait.territory_code.notna()), ["territory_code", "label"]]
territory_score = target_encoding("territory_code", df)
territory_score


# In[34]:


# add territory_code score into id_score
n1 = len(id_score)
id_score.update({x[1]: x[5] for x in territory_score.itertuples()})
n1 + territory_score.shape[0] == len(id_score)


# In[35]:


# get territory score, retain null value
portrait["territory_score"] = portrait.territory_code.apply(lambda x: id_score.get(x, 0) if isinstance(x, str) else np.nan)
portrait


# In[36]:


# for multi values of device_ram and device_rom, choose the first one
portrait["device_ram"] = portrait.device_ram.apply(lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)
portrait["device_rom"] = portrait.device_rom.apply(lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)
portrait


# In[37]:


# add portrait features into data
data = data.merge(portrait.drop("territory_code", axis=1), how="left", on=["user_id", "label"])
data


# # interaction data process

# In[38]:


# only use interact_type preference
# use all interaction data to calculate interact_type preference
interact = pd.read_csv(input_dir + "user_interaction_data.csv")
interact.interact_type.min(), interact.interact_type.max()


# In[39]:


interact_grp = interact.groupby("user_id").agg(
    interact_type=("interact_type", list)
).reset_index()
interact_grp


# In[40]:


# similar to duration preference, the interact_type preference could be a 11-dimentional vector
def get_interact_prefer(interact_type):
    x = sorted(interact_type)
    x_count = {k: sum(1 for _ in g) for k, g in groupby(x)}
    x_max = max(x_count.values())
    res = [round(x_count.get(i, 0)/x_max, 4) for i in range(1, 12)]
    return res
interact_grp["interact_prefer"] = interact_grp.interact_type.apply(get_interact_prefer)
interact_grp


# In[41]:


data = data.merge(interact_grp[["user_id", "interact_prefer"]], on="user_id", how="left")
data


# # feature normalization and save data

# In[42]:


# the following features should be normalized
# method: x = (x - mean(x)) / std(x)
norm_cols = ["father_id_score", "cast_id_score", "tag_score", 
            "device_type", "device_ram", "device_rom", "sex",
            "age", "education", "occupation_status", "territory_score"]
for col in norm_cols:
    mean = data[col].mean()
    std = data[col].std()
    data[col] = (data[col] - mean) / std
data


# In[43]:


# filling null vector features with zero-vectors
data.fillna({
    "playtime_seq": str([0]*32),
    "duration_prefer": str([0]*16),
    "interact_prefer": str([0]*11)
}, inplace=True)
data


# In[44]:


# filling null numeric features with 0
data.fillna(0, inplace=True)
data


# In[45]:


# finally
data.loc[data.label >= 0].to_csv(output_dir + "train_data.txt", sep="\t", index=False)
data.loc[data.label < 0].to_csv(output_dir + "test_data.txt", sep="\t", index=False)

