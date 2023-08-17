import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# 加载中文停用词数据集
nltk.download('stopwords')
stopwords_chinese = set(nltk.corpus.stopwords.words('chinese'))

# 加载英文停用词数据集
nltk.download('stopwords')
stopwords_english = set(nltk.corpus.stopwords.words('english'))

# 合并中文和英文停用词集合
stopwords = stopwords_chinese.union(stopwords_english)

# 读取 CSV 文件
df = pd.read_csv('your_csv_file.csv')

# 分词处理函数
def tokenize_text(text):
    tokens = word_tokenize(str(text)) # 分词
    tokens = [word.lower() for word in tokens if word.lower() not in stopwords] # 过滤停用词并转换为小写
    return tokens

# 对标题、描述和解决方案字段进行分词处理
df['标题'] = df['标题'].apply(tokenize_text)
df['描述'] = df['描述'].apply(tokenize_text)
df['解决方案'] = df['解决方案'].apply(tokenize_text)

# 打印分词后的数据
print(df)



from sklearn.feature_extraction.text import CountVectorizer

# 将标题、描述和解决方案字段的分词结果进行拼接，并转换为文本列表
text_list = [' '.join(title + description + solution) for title, description, solution in zip(df['标题'], df['描述'], df['解决方案'])]

# 创建一个 CountVectorizer 对象
count_vectorizer = CountVectorizer()

# 使用 CountVectorizer 进行拟合和变换
count_matrix = count_vectorizer.fit_transform(text_list)

# 获取所有特征词
features = count_vectorizer.get_feature_names()

# 构建词频矩阵的 DataFrame
freq_matrix = pd.DataFrame(count_matrix.toarray(), columns=features)

# 打印词频矩阵
print(freq_matrix)


from sklearn.feature_extraction.text import TfidfTransformer

# 创建一个 TfidfTransformer 对象
tfidf_transformer = TfidfTransformer()

# 使用 TfidfTransformer 对词频矩阵进行拟合和转换
tfidf_matrix = tfidf_transformer.fit_transform(freq_matrix)

# 将稀疏矩阵转换为常规矩阵
tfidf_matrix_dense = tfidf_matrix.toarray()

# 创建包含特征词的列表
feature_names = freq_matrix.columns.tolist()

# 构建 TF-IDF 值矩阵的 DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix_dense, columns=feature_names)

# 打印 TF-IDF 值矩阵
print(tfidf_df)


import pandas as pd

# 读取之前的CSV文件
df = pd.read_csv('之前的CSV文件路径')

# 根据 TF-IDF 值对特征词进行排序
sorted_features = tfidf_df.sum().sort_values(ascending=False)

# 设置关键词的数量（取排名靠前的前几个词）
num_keywords = 5

# 提取关键词
keywords = sorted_features.head(num_keywords).index.tolist()

# 添加关键词列到DataFrame
df['关键词'] = ', '.join(keywords)

# 保存修改后的DataFrame到新的CSV文件
df.to_csv('新的CSV文件路径', index=False)

