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
