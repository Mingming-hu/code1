from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import torch 
import torch.nn as nn
import torch.nn.functional as F


def binary_to_top3(binary_matrix, label_names):
    top3_indices = np.argsort(-binary_matrix, axis=1)[:, :3]
    return [[label_names[idx] for idx in row] for row in top3_indices]



def main(path):
    df = pd.read_excel(path)
    # 假设原始数据是这样的格式（每行包含3个标签）：
    original_labels = df.loc[:,'正式选科1':'正式选科3'].values
    
    data = df.loc[:,'物理思维T分数':'内省']
    #data = df.loc[:,'物理思维T分数':'自然认知']
    #data = data.drop(['社交型', '传统型'], axis=1)
    print("总数据量为",len(original_labels),"条")

    # 所有可能的标签列表
    all_possible_labels = ['物理', '化学', '生物', '政治', '历史', '地理']

    # 创建转换器
    mlb = MultiLabelBinarizer(classes=all_possible_labels)

    # 转换为6列的二值形式
    binary_labels = pd.DataFrame(mlb.fit_transform(original_labels),columns=all_possible_labels)

    x_train, x_test, y_train, y_test = train_test_split(data, binary_labels, test_size=0.3, random_state=58)

    print("测试集数据量为",len(x_test),"条")

    #model = MultiOutputClassifier(XGBClassifier(n_estimators=256, learning_rate=0.012, max_depth=6, gamma=0.01, 
    #                          colsample_bytree = .9, subsample=0.6, reg_alpha=0.05,
    #                          reg_lambda=4.5, random_state=0))
    
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=7,max_depth=6))
    
    model_svm_linear = MultiOutputClassifier(SVC(kernel='linear', random_state=7, probability=True))

    # 径向基函数(RBF)核SVM
    model_svm_rbf = MultiOutputClassifier(SVC(kernel='rbf', random_state=7, probability=True))

    model_nb = MultiOutputClassifier(GaussianNB())
    model_lr = MultiOutputClassifier(LogisticRegression(random_state=7, max_iter=1000))
    model_knn = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))

    model_knn.fit(x_train, y_train)

    # 获取每个标签的概率预测
    probabilities = model_knn.predict_proba(x_test)  # X_new是新样本

    # 对于每个样本，选择概率最高的3个标签
    top3_labels = []
    for sample_probs in zip(*[p[:, 1] for p in probabilities]):
        top3 = sorted(zip(y_train.columns, sample_probs), key=lambda x: -x[1])[:3]
        top3_labels.append([label for label, prob in top3])

    y_prob = np.array([p[:, 1] for p in probabilities]).T
    lr_score = label_ranking_average_precision_score(y_test, y_prob)
    print("测试集LRAP精度 = ",lr_score)


    # 将真实标签和预测标签都转换为标签集合
    y_true_sets = [set(np.where(row)[0]) for row in y_test.values]  # 真实标签集合
    y_pred_sets = [set(np.argsort(row)[-3:]) for row in y_prob]  # 预测Top3集合

    # 计算完全匹配准确率（完全匹配Top3集合）
    exact_match_acc = np.mean([true == pred for true, pred in zip(y_true_sets, y_pred_sets)])

    # 计算部分匹配指标（Jaccard相似度）
    jaccard_scores = [len(true & pred)/len(true) 
    #jaccard_scores = [len(true & pred)/len(true | pred) 
                     for true, pred in zip(y_true_sets, y_pred_sets)]
    #print(jaccard_scores)
    mean_jaccard = np.mean(jaccard_scores)
    print(f"测试集准确率 = {exact_match_acc:.4f}")
    print(f"测试集命中率 = {mean_jaccard:.4f}")


'''

    # 获取训练集每个标签的概率预测
    probabilities_train = model.predict_proba(x_train)  # X_new是新样本

    # 对于每个样本，选择概率最高的3个标签
    top3_labels_train = []
    for sample_probs1 in zip(*[q[:, 1] for q in probabilities_train]):
        top3_train = sorted(zip(y_train.columns, sample_probs1), key=lambda x: -x[1])[:3]
        top3_labels_train.append([label for label, prob in top3_train])

    y_prob_train = np.array([q[:, 1] for q in probabilities_train]).T
    lr_train_score = label_ranking_average_precision_score(y_train, y_prob_train)
    print("训练集精度 = ", lr_train_score)

    # 将标签编码转回具体预选学科

    test_labels = binary_to_top3(y_prob, all_possible_labels)

    prod_df = pd.DataFrame(test_labels, columns=['预选学科1', '预选学科2', '预选学科3'])
    print(prod_df)
'''

if __name__ == '__main__':
    main(path="D:\职业推荐\职业生涯带标签数据-0327.xlsx")

