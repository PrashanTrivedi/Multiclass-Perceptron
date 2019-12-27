import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
### Glass Identification Data Set ###        
data=pd.read_table('glass_norm.txt')
data.head()
data=shuffle(data)
X=np.array(data.loc[:,'Refractive Index':'Iron'])
Y=np.array(data.Y)


def norm(x):
    s=0
    for i in range(len(x)):
        s=s+pow(x[i],2)
    t=s**0.5
    return t

def multiclass():
    c=0 
    w=[]
    W=[]
    for i in range(len(set(Y))):
        w.append(np.ones(len(X[1])))

##diff=[]
    for t in range(len(X)):
        #print w
        x=X[t]
        #print Y[t]*x
    
        r=np.argmax([np.dot(w[i],x) for i in range(len(w))])
        #print 'the predicted label is' ,r+1
        y=Y[t]
        if (r+1)!=y:
            w[r]=w[r]-x
            w[y-1]=w[y-1]+x
            c=c+1
        else:
            w=w
        W.append(sum([np.dot(w[i],w[i]) for i in range(len(w))]))
    return W
def prop_algo():
    start=2
    W=[]
    c=0 
    prob=np.zeros(len(set(Y)))
    count=np.zeros(len(set(Y)))
    w=[]
    for i in range(len(set(Y))):
        w.append(np.ones(len(X[1])))
        
    for t in range(start):
        y=Y[t]
        count[y-1]=count[y-1]+1
    prob=count/start
    ##diff=[]
    for t in range(start+1,len(X)):
        #print w
        x=X[t]
        #print Y[t]*x
        
        r=np.argmax([np.dot(w[i],x) for i in range(len(w))])
        #print 'the predicted label is' ,r+1
        y=Y[t]
        #print 'true',y

        #print 'prob',prob
        if (r+1)!=y:
            w[r]=w[r]-prob[r]*x
            w[y-1]=w[y-1]+prob[y-1]*x
            c=c+1
        else:
            w=w
        count[y-1]=count[y-1]+1
        prob=count/t
        W.append(sum([np.dot(w[i],w[i]) for i in range(len(w))]))
    return W

def prop_algo2():
    start=2
    W=[]
    c=0 
    prob=np.zeros(len(set(Y)))
    count=np.zeros(len(set(Y)))
    print count
    w=[]
    for i in range(len(set(Y))):
        w.append(np.ones(len(X[1])))
        
    for t in range(start):
        y=Y[t]
        #print 'y',y
        count[y-1]=count[y-1]+1
        #print 'count',count
        
    prob=count/start
    for t in range(start+1,len(X)):
        x=X[t]
        #print 'x',x
        r=np.argmax([np.dot(w[i],x) for i in range(len(w))])
        score=[]
        ind=[]
        for i in range(len(count)):
            if count[i]!=0:
                score.append(np.dot(w[i],x))
                ind.append(i)
                #print score ,'score'
                #print ind,'ind'
        y=Y[t]
        #print 'y',y
        s=np.dot(w[y-1],x)
        #print s,'s'
        if (y-1) in ind:
            ind.remove(y-1)
        #print ind
        #print s,'s'
        for i in range(len(ind)):
            if s<score[i]:
                w[ind[i]]=w[ind[i]]-prob[ind[i]]*x
        if s<=max(score):
                #print 'w[ind[i]]',w[ind[i]]
            w[y-1]=w[y-1]+prob[y-1]*x
        count[y-1]=count[y-1]+1
        prob=count/t
        W.append(sum([np.dot(w[i],w[i]) for i in range(len(w))]))
    return W
        
x=[i+1 for i in range(len(X))]
y=[i+1 for i in range(3,len(X))]
s=multiclass()
z=prop_algo()
z1=prop_algo2()
plt.plot(x,s,'r-',label='Multiclass Algorithm')
plt.hold(True)
plt.plot(y,z,'b-',label='Modified Multiclass Algorithm')
plt.hold(True)
plt.plot(y,z1,'k-',label='Modified Multiclass Algorithm 2')
plt.xlabel('Number of iterations')
plt.ylabel('Norm of Weights')
plt.title('Glass Identification data')
plt.legend(loc='best')
plt.show()


##print 'The number of mistake of multiclass perceptron is', c
##print 'The final weights are',w
 
    

