from sklearn.feature_extraction import DictVectorizer
import csv
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO
from sqlalchemy.sql.expression import false

# Read in the csv file and put features into list of dict and list of class label
# allElectronicsData = open(r'F:\BaiduNetdiskDownload\代码与素材\代码与素材(1)\01DTree\AllElectronics.csv', 'rt')
allElectronicsData = open(r'AllElectronics.csv', 'rt')
allData = pd.read_csv('AllElectronics.csv')
print(allData)

reader = csv.reader(allElectronicsData)
headers = next(reader)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)
 
# Vetorize features
#添加 sparse=False转化后的矩阵为非稀疏矩阵，不用讲转化结果toarray了
vec = DictVectorizer(sparse=False)

dummyX = vec.fit_transform(featureList)

print("dummyX: " + str(dummyX))
print(vec.get_feature_names())
 
print("labelList: " + str(labelList))
 
# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))
  
# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))
   
   
# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)
    
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))
     
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))
     
print(newRowX.shape)
print(type(newRowX)   )
#predictedY = clf.predict(newRowX)
predictedY = clf.predict(newRowX.reshape(1,-1))
print((newRowX.reshape(1,-1).shape))

print("predictedY: " + str(predictedY))


