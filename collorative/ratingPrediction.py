#稠密矩阵
#构建数据 缺失使用None代替 设置为0会被当做评分0对待
users = ["User1", "User2", "User3", "User4", "User5"]
items = ["Item A", "Item B", "Item C", "Item D", "Item E"]
# 用户购买记录数据集
datasets = [
    [5,3,4,4,None],
    [3,1,2,3,3],
    [4,3,4,3,5],
    [3,3,1,5,4],
    [1,5,5,2,1],
]
import pandas as pd
import numpy as np
from pprint import pprint

df = pd.DataFrame(datasets,
                  columns=items,
                  index=users)

#使用皮尔逊相关系数计算相似度，-1表示强负相关，1表示强正相关
print("用户之间的两两相似度：")
# 直接计算皮尔逊相关系数
# 默认是按列进行计算，因此如果计算用户间的相似度，当前需要进行转置
user_similar = df.T.corr()
print(user_similar.round(4))

print("物品之间的两两相似度：")
item_similar = df.corr()
print(item_similar.round(4))
topN_users = {}

#遍历每一行数据

for i in user_similar.index:
    # 取出每一列数据，并删除自身，然后排序数据
    _df = user_similar.loc[i].drop([i])
    #sort_values 排序 按照相似度降序排列
    _df_sorted = _df.sort_values(ascending=False)
    # 从排序之后的结果中切片 取出前两条（相似度最高的两个）
    top2 = list(_df_sorted.index[:2])
    topN_users[i] = top2
print("Top2相似用户：")
pprint(topN_users)