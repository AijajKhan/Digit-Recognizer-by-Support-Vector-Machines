#!/usr/bin/env python
# coding: utf-8

# In[197]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as py
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
sns.set(style="whitegrid",color_codes=True)


# In[198]:


digits=load_digits()
a=digits.images
b=digits.target
#df=pd.DataFrame["Images": a, "Labels": b]
mak=list(zip(digits.images,digits.target))
for i, (imz, var) in enumerate(mak[:5]):
    plt.subplot(2,5,i+1)
    plt.axis('off')
    plt.gray()
    plt.imshow(imz,interpolation="nearest")
    print(var)
    plt.title("Training: %i"%var)


# In[199]:


len(digits.images)


# In[200]:


len(digits.target)


# In[201]:


samples=len(digits.data)
test_size=3*(samples//4)
digitimages=digits.images.reshape(samples,-1)


# In[202]:


model=SVC(gamma=0.0001,kernel="rbf",max_iter=2000,random_state=1)
model.fit(digitimages[:test_size],digits.target[:test_size])


# In[227]:


x_test=digitimages[test_size:]
rev_x_test=digits.images[test_size:]
y_test=digits.target[test_size:]


# In[204]:


predicted_y_test=model.predict(x_test)


# In[205]:


print(classification_report(y_test,predicted_y_test))


# In[206]:


a=confusion_matrix(y_test,predicted_y_test)
sns.heatmap(a,annot=True,cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")


# In[207]:


ax=accuracy_score(y_test,predicted_y_test)
#x=precision_score(y_test,predicted_y_test)
#y=recall_score(y_test,predicted_y_test)
#z=f1_score(y_test,predicted_y_test)
print("Accuracy Score is: {}".format(ax))


# In[212]:


'''a=np.array([2,3,4,5,6])
b=a.reshape(len(a),1)
print(b.reshape(1,len(a)))'''


# The score is same with Linear Kernel and Radial Basis Function kernel.

# In[229]:


makpredict=list(zip(rev_x_test,predicted_y_test))
for i , (imz,var) in enumerate(makpredict[:5]):
    plt.subplot(2,5,i+1)
    plt.gray()
    plt.axis("off")
    plt.imshow(imz,interpolation="nearest")
    plt.title("Prediction %i"%var)
'''xtrain=digitimages[test_size,:]
print(xtrain.shape)'''


# In[ ]:





# In[ ]:




