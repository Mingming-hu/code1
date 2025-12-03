from snownlp import SnowNLP
from snownlp import sentiment
s=SnowNLP(u'这个东西真的很赞')
print(s.words)
print(s.sentiments)