import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn import preprocessing
df=pd.read_excel('titanic.xls')
df.drop(['name','body'],axis=1,inplace=True)
#dsf=df.copy()
f=df['sex']=='male'
df['sex']=f.astype(int)
#j=df['home.dest']=='Montreal, PQ / Chesterville, ON'
def numers(dff):
    column=dff.columns.values
    text_={}
    def mfg(val):
        return text_[val]
    for col in column:
        if dff[col].dtype!=np.int64 or dff[col].dtype!=np.float64:
         saul=dff[col].values
         x=0
         k=set(saul)
         for l in k:
          if l not in text_:
            text_[l]=x;
            x+=1
        dff[col]=list(map(mfg,dff[col]))
    return dff

df.fillna(0,inplace=True)
df=numers(df)
print df.head()
X=np.array(df.drop(['survived'],axis=1).astype(float))
X=preprocessing.scale(X)
Y=np.array(df['survived'])
clf=KMeans(n_clusters=2)
clf.fit(X)
corr=0
for i in range(len(X)):
    predict_=np.array(X[i].astype(float))
    predict_ = predict_.reshape(-1, len(predict_))
    prediction=clf.predict(predict_)
    if(prediction==Y[i]):
        corr+=1

accuracy=float(corr)
print "Accuracy:",(accuracy/len(X)*100)
#plt.plot(X,Y)
#plt.show()
