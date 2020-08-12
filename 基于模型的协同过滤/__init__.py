import pandas as pd
import numpy as np
dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
dataset = pd.read_csv("ml-latest-small/ratings.csv", usecols=range(3), dtype=dict(dtype))

# 用户评分数据  groupby 分组  groupby('userId') 根据用户id分组 agg（aggregation聚合）
users_ratings = dataset.groupby('userId').agg([list])
# 物品评分数据
items_ratings = dataset.groupby('movieId').agg([list])
# 计算全局平均分
global_mean = dataset['rating'].mean()
# 初始化bu bi
bu = dict(zip(users_ratings.index, np.zeros(len(users_ratings))))
bi = dict(zip(items_ratings.index, np.zeros(len(items_ratings))))