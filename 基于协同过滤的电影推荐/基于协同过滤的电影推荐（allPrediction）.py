import os

import pandas as pd
import numpy as np

DATA_PATH = "D:\\推荐系统项目\\评分数据\\ratings.csv"

dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
# 加载数据，我们只用前三列数据，分别是用户ID，电影ID，已经用户对电影的对应评分
ratings = pd.read_csv(DATA_PATH, dtype=dtype, usecols=range(3))
# 透视表，将电影ID转换为列名称，转换成为一个User-Movie的评分矩阵
ratings_matrix = ratings.pivot_table(index=["userId"], columns=["movieId"],values="rating")
#计算用户之间相似度
item_similar = ratings_matrix.corr()
user_similar = ratings_matrix.T.corr()

def predict(uid, iid, ratings_matrix, user_similar):
    '''
    预测给定用户对给定物品的评分值
    :param uid: 用户ID
    :param iid: 物品ID
    :param ratings_matrix: 用户-物品评分矩阵
    :param user_similar: 用户两两相似度矩阵
    :return: 预测的评分值
    '''
    print("开始预测用户<%d>对电影<%d>的评分..."%(uid, iid))
    # 1. 找出uid用户的相似用户
    similar_users = user_similar[uid].drop([uid]).dropna()
    # 相似用户筛选规则：正相关的用户
    similar_users = similar_users.where(similar_users>0).dropna()
    if similar_users.empty is True:
        raise Exception("用户<%d>没有相似的用户" % uid)

    # 2. 从uid用户的近邻相似用户中筛选出对iid物品有评分记录的近邻用户
    ids = set(ratings_matrix[iid].dropna().index)&set(similar_users.index)
    finally_similar_users = similar_users.loc[list(ids)]

    # 3. 结合uid用户与其近邻用户的相似度预测uid用户对iid物品的评分
    numerator = 0    # 评分预测公式的分子部分的值
    denominator = 0    # 评分预测公式的分母部分的值
    for sim_uid, similarity in finally_similar_users.iteritems():
        # 近邻用户的评分数据
        sim_user_rated_movies = ratings_matrix.ix[sim_uid].dropna()
        # 近邻用户对iid物品的评分
        sim_user_rating_for_item = sim_user_rated_movies[iid]
        # 计算分子的值
        numerator += similarity * sim_user_rating_for_item
        # 计算分母的值
        denominator += similarity

    # 计算预测的评分值并返回
    predict_rating = numerator/denominator
    print("预测出用户<%d>对电影<%d>的评分：%0.2f" % (uid, iid, predict_rating))
    return round(predict_rating, 2)

def predict_all(uid, ratings_matrix, user_similar):
    '''
    预测全部评分
    :param uid: 用户id
    :param ratings_matrix: 用户-物品打分矩阵
    :param user_similar: 用户两两间的相似度
    :return: 生成器，逐个返回预测评分
    '''
    # 准备要预测的物品的id列表
    item_ids = ratings_matrix.columns
    # 逐个预测
    for iid in item_ids:
        try:
            rating = predict(uid, iid, ratings_matrix, user_similar)
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating
if __name__ == '__main__':
    for i in predict_all(1, ratings_matrix, user_similar):
        pass