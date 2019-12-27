import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

### Glass Identification Data Set ###  
data=pd.read_table('glass_norm.txt')
data.head()


#number of runs
n=50


def multiclass(X,Y):
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
        #W.append(sum([np.dot(w[i],w[i]) for i in range(len(w))]))
    return c
def prop_algo(X,Y):
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
        #W.append(sum([np.dot(w[i],w[i]) for i in range(len(w))]))
    return c
def prop_algo2(X,Y):
    start=2
    W=[]
    c=0 
    prob=np.zeros(len(set(Y)))
    count=np.zeros(len(set(Y)))
    #print count
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
        if (r+1)!=y:
            c=c+1
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
        #W.append(sum([np.dot(w[i],w[i]) for i in range(len(w))]))
    return c
m1=[]
m2=[]
m3=[]
for i in range(n):
    data=shuffle(data)
    X=np.array(data.loc[:,'Refractive Index':'Iron'])
    Y=np.array(data.Y)
    m1.append(multiclass(X,Y))
    m2.append(prop_algo(X,Y))
    m3.append(prop_algo2(X,Y))
    
#print 'm1',m1
#print 'm2',m2

print ' the number of mistakes for multiclass algorithm is ', np.mean(m1)
print ' the number of mistakes for modified multiclass algorithm is ', np.mean(m2)
print ' the number of mistakes for modified multiclass algorithm 2 is ', np.mean(m3)


