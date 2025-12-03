from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr



def combined2(x_train1, x_train2):
    x_train1 = x_train1.reset_index(drop=True)
    x_train2 = x_train2.reset_index(drop=True)
    x_train = pd.concat([x_train1, x_train2], axis=1)
    #x_train = x_train.T.reset_index(drop=True).T
    #x_train = x_train.values
    #print(x_train)
    return x_train

def binary_to_top3(binary_matrix, label_names):
    top3_indices = np.argsort(-binary_matrix, axis=1)[:, :3]
    return [[label_names[idx] for idx in row] for row in top3_indices]

def main(path):
    df = pd.read_excel(path)
    # 假设原始数据是这样的格式（每行包含3个标签）：
    original_labels = df.loc[:,'正式选科1':'正式选科3'].values
    data = df.loc[:,'物理思维T分数':'内省']
    print("总数据量为",len(original_labels),"条")

    # 所有可能的标签列表
    all_possible_labels = ['物理', '化学', '生物', '政治', '历史', '地理']

    # 创建转换器
    mlb = MultiLabelBinarizer(classes=all_possible_labels)

    # 转换为6列的二值形式
    binary_labels = pd.DataFrame(mlb.fit_transform(original_labels),columns=all_possible_labels)
    data_all = combined2(data, binary_labels)

    x_train, x_test, y_train, y_test = train_test_split(data, binary_labels, test_size=0.3, random_state=58)
    
    features_group1 = x_train
    features_group2 = y_train

    # 计算斯皮尔曼相关系数和p值
    corr_matrix = pd.DataFrame(index=features_group1.columns, columns=features_group2.columns)
    pvalue_matrix = pd.DataFrame(index=features_group1.columns, columns=features_group2.columns)

    for col1 in features_group1.columns:
        for col2 in features_group2.columns:
            corr, pvalue = spearmanr(features_group1[col1], features_group2[col2])
            corr_matrix.loc[col1, col2] = corr
            pvalue_matrix.loc[col1, col2] = pvalue

    # 转换为数值类型
    corr_matrix = corr_matrix.astype(float)
    pvalue_matrix = pvalue_matrix.astype(float)

    # 绘制热力图
    plt.figure(figsize=(10, 20))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    heatmap = sns.heatmap(
        corr_matrix,
        annot=False,                # 显示相关系数值
        fmt=".2f",                # 数值格式
        cmap="coolwarm",           # 颜色映射
        vmin=-1, vmax=1,          # 颜色范围固定为[-1,1]
        cbar_kws={"label": "Spearman Correlation"},
    )

    # 标注显著性（*表示p<0.05，**表示p<0.01）
    for i in range(len(features_group1.columns)):
        for j in range(len(features_group2.columns)):
            pvalue = pvalue_matrix.iloc[i, j]
            if pvalue < 0.001:
                text = "***"
            elif pvalue < 0.01:
                text = "**"
            elif pvalue < 0.05:
                text = "*"
            else:
                text = ""
            plt.text(j + 0.5, i + 0.5, text, ha="center", va="center", color="black",
                fontsize=15)

    plt.title("Spearman Correlation Heatmap\n*: p<0.05, **: p<0.01, **: p<0.001")
    plt.xticks(rotation=0,fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.tight_layout()
    plt.savefig('D:\\职业推荐\\相关性.png')
    plt.show()




if __name__ == '__main__':
    main(path="D:\职业推荐\职业生涯带标签数据-0327.xlsx")