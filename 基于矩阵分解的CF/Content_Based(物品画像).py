import pandas as pd
import numpy as np
'''
- 利用tags.csv中每部电影的标签作为电影的候选关键词
- 利用TF·IDF计算每部电影的标签的tfidf值，选取TOP-N个关键词作为电影画像标签
- 并将电影的分类词直接作为每部电影的画像标签
'''

def get_movie_dataset():
    # 加载基于所有电影的标签
    # all-tags.csv来自ml-latest数据集中
    # 由于ml-latest-small中标签数据太多，因此借助其来扩充
    _tags = pd.read_csv("datasets/ml-latest-small/all-tags.csv", usecols=range(1, 3)).dropna()
    tags = _tags.groupby("movieId").agg(list)

    # 加载电影列表数据集
    movies = pd.read_csv("datasets/ml-latest-small/movies.csv", index_col="movieId")
    # 将类别词分开
    movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))
    # 为每部电影匹配对应的标签数据，如果没有将会是NAN
    movies_index = set(movies.index) & set(tags.index)
    new_tags = tags.loc[list(movies_index)]
    ret = movies.join(new_tags)

    # 构建电影数据集，包含电影Id、电影名称、类别、标签四个字段
    # 如果电影没有标签数据，那么就替换为空列表
    # map(fun,可迭代对象)
    movie_dataset = pd.DataFrame(
        map(
            lambda x: (x[0], x[1], x[2], x[2]+x[3]) if x[3] is not np.nan else (x[0], x[1], x[2], []), ret.itertuples())
        , columns=["movieId", "title", "genres","tags"]
    )

    movie_dataset.set_index("movieId", inplace=True)
    return movie_dataset

movie_dataset = get_movie_dataset()
#print(movie_dataset)

from gensim.models import TfidfModel

import pandas as pd
import numpy as np

from pprint import pprint

# ......

def create_movie_profile(movie_dataset):
    '''
    使用tfidf，分析提取topn关键词
    :param movie_dataset:
    :return:
    '''
    dataset = movie_dataset["tags"].values

    from gensim.corpora import Dictionary
    # 根据数据集建立词袋，并统计词频，将所有词放入一个词典，使用索引进行获取
    dct = Dictionary(dataset)
    # 根据将每条数据，返回对应的词索引和词频
    corpus = [dct.doc2bow(line) for line in dataset]
    # 训练TF-IDF模型，即计算TF-IDF值
    model = TfidfModel(corpus)

    _movie_profile = []
    for i, data in enumerate(movie_dataset.itertuples()):
        mid = data[0]
        title = data[1]
        genres = data[2]
        vector = model[corpus[i]]
        movie_tags = sorted(vector, key=lambda x: x[1], reverse=True)[:30]
        topN_tags_weights = dict(map(lambda x: (dct[x[0]], x[1]), movie_tags))
        # 将类别词的添加进去，并设置权重值为1.0
        for g in genres:
            topN_tags_weights[g] = 1.0
        topN_tags = [i[0] for i in topN_tags_weights.items()]
        _movie_profile.append((mid, title, topN_tags, topN_tags_weights))

    movie_profile = pd.DataFrame(_movie_profile, columns=["movieId", "title", "profile", "weights"])
    movie_profile.set_index("movieId", inplace=True)
    return movie_profile

movie_dataset = get_movie_dataset()
movie_profile = create_movie_profile(movie_dataset)
pprint(create_movie_profile(movie_dataset))

# ......

'''
建立tag-物品的倒排索引
'''

def create_inverted_table(movie_profile):
    inverted_table = {}
    for mid, weights in movie_profile["weights"].iteritems():
        for tag, weight in weights.items():
            #到inverted_table dict 用tag作为Key去取值 如果取不到就返回[]
            _ = inverted_table.get(tag, [])
            #将电影的id 和 权重 放到一个tuple中 添加到list中
            _.append((mid, weight))
            #将修改后的值设置回去
            inverted_table.setdefault(tag, _)
    return inverted_table

inverted_table = create_inverted_table(movie_profile)
pprint(inverted_table)
import pandas as pd
import numpy as np
from gensim.models import TfidfModel

from functools import reduce
import collections

from pprint import pprint

# ......

'''
user profile画像建立：
1. 提取用户观看列表
2. 根据观看列表和物品画像为用户匹配关键词，并统计词频
3. 根据词频排序，最多保留TOP-k个词，这里K设为100，作为用户的标签
'''

def create_user_profile():
    watch_record = pd.read_csv("datasets/ml-latest-small/ratings.csv", usecols=range(2), dtype={"userId":np.int32, "movieId": np.int32})

    watch_record = watch_record.groupby("userId").agg(list)
    # print(watch_record)

    movie_dataset = get_movie_dataset()
    movie_profile = create_movie_profile(movie_dataset)

    user_profile = {}
    for uid, mids in watch_record.itertuples():
        record_movie_prifole = movie_profile.loc[list(mids)]
        counter = collections.Counter(reduce(lambda x, y: list(x)+list(y), record_movie_prifole["profile"].values))
        # 取出出现次数最多的前50个词
        interest_words = counter.most_common(50)
        # 取出出现次数最多的词 出现的次数
        maxcount = interest_words[0][1]
        # 利用次数计算权重 出现次数最多的词权重为1
        interest_words = [(w,round(c/maxcount, 4)) for w,c in interest_words]
        user_profile[uid] = interest_words

    return user_profile

user_profile = create_user_profile()
pprint(user_profile)


from collections import Counter
names = ["Stanley", "Lily", "Bob", "Well", "Peter", "Bob", "Well", "Peter", "Well", "Peter", "Bob","Stanley", "Lily", "Bob", "Well", "Peter", "Bob", "Bob", "Well", "Peter", "Bob", "Well"]
names_counts = Counter(names)

# ......

user_profile = create_user_profile()

watch_record = pd.read_csv("datasets/ml-latest-small/ratings.csv", usecols=range(2),dtype={"userId": np.int32, "movieId": np.int32})

watch_record = watch_record.groupby("userId").agg(list)

for uid, interest_words in user_profile.items():
    result_table = {} # 电影id:[0.2,0.5,0.7]
    for interest_word, interest_weight in interest_words:
        related_movies = inverted_table[interest_word]
        for mid, related_weight in related_movies:
            _ = result_table.get(mid, [])
            _.append(interest_weight)    # 只考虑用户的兴趣程度
            # _.append(related_weight)    # 只考虑兴趣词与电影的关联程度
            # _.append(interest_weight*related_weight)    # 二者都考虑
            result_table.setdefault(mid, _)

    rs_result = map(lambda x: (x[0], sum(x[1])), result_table.items())
    rs_result = sorted(rs_result, key=lambda x:x[1], reverse=True)[:100]
    print(uid)
    pprint(rs_result)
    break


