
import jieba

text = "我的痛苦没有减少"
words = list(jieba.cut(text, cut_all=False))
print(words)


def loadDict(fileName, score):
    wordDict = {}
    with open(fileName, encoding='utf-8') as fin:
        for line in fin:
            word = line.strip()
            wordDict[word] = score
    return wordDict


def appendDict(wordDict, fileName, score):
    with open(fileName) as fin:
        for line in fin:
            word = line.strip()
            wordDict[word] = score


def loadExtentDict(fileName, level):
    extentDict = {}
    for i in range(level):
        with open(fileName + str(i + 1) + ".txt", encoding='utf-8') as fin:
            for line in fin:
                word = line.strip()
                extentDict[word] = i + 1
    return extentDict


import jieba



def getScore(content):
    postDict = loadDict(u"D:/日常办公/支撑性工作/智慧心育/数据探索/线上回流数据/情感词典/正面情感词语（中文）.txt", 1)  # 积极情感词典
    negDict = loadDict(u"D:/日常办公/支撑性工作/智慧心育/数据探索/线上回流数据/情感词典/负面情感词语（中文）.txt", -1)  # 消极情感词典
    inverseDict = loadDict(u"D:/日常办公/支撑性工作/智慧心育/数据探索/线上回流数据/情感词典/否定词语.txt", -1)  # 否定词词典
    extentDict = loadExtentDict(u"D:/日常办公/支撑性工作/智慧心育/数据探索/线上回流数据/情感词典/程度级别词语（中文）1", 6)
    punc = loadDict(u"D:/日常办公/支撑性工作/智慧心育/数据探索/线上回流数据/情感词典/标点符号.txt", 1)
    exclamation = {"!": 2, "！": 2}

    words = jieba.cut(content)
    wordList = list(words)
    # print(wordList)

    totalScore = 0  # 记录最终情感得分
    lastWordPos = 0  # 记录情感词的位置
    lastPuncPos = 0  # 记录标点符号的位置
    i = 0  # 记录扫描到的词的位置

    for word in wordList:
        if word in punc:
            lastPuncPos = i

        if word in postDict:
            if lastWordPos > lastPuncPos:
                start = lastWordPos
            else:
                start = lastPuncPos

            score = 1
            for word_before in wordList[start:i]:
                if word_before in extentDict:
                    score = score * extentDict[word_before]
                if word_before in inverseDict:
                    score = score * -1
            for word_after in wordList[i + 1:]:
                if word_after in inverseDict:
                    score = score * -1
                if word_after in punc:
                    if word_after in exclamation:
                        score = score + 2
                    else:
                        break
            lastWordPos = i
            totalScore += score
        elif word in negDict:
            if lastWordPos > lastPuncPos:
                start = lastWordPos
            else:
                start = lastPuncPos
            score = -1
            for word_before in wordList[start:i]:
                if word_before in extentDict:
                    score = score * extentDict[word_before]
                if word_before in inverseDict:
                    score = score * -1
            for word_after in wordList[i + 1:]:
                if word_after in punc:
                    if word_after in exclamation:
                        score = score - 2
                    else:
                        break
            lastWordPos = i
            totalScore += score
        i = i + 1

    return totalScore


import pandas as pd

# 读取Excel文件
df = pd.read_excel('c:/Users/mmhu6/Desktop/code/duihua.xlsx')

# 计算A列数据的平均值
content = df['A']

# 将结果保存到B列
df['B'] = getScore(content)

# 将结果保存到新的Excel文件
df.to_excel('c:/Users/mmhu6/Desktop/code/output.xlsx', index=False)



#content = u"我的痛苦没有减少"
#print(getScore(content))  


