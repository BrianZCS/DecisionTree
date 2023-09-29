### Run it: python3 DY.py Filename
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
import random
from scipy.interpolate import lagrange

class treenode:
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent
        self.left = None
        self.right = None
        self.best_split = None
        self.classlabel = None
    def leftnode(self, left):
        self.left = left
    def rightnode(self, right):
        self.right = right

def DetermineCandidateSplits(data):
    candidate = []
    for i in range(len(data['x1'])):
        temp = (1,data.iloc[i,0])
        if temp not in candidate:
            candidate.append(temp)
    for i in range(len(data['x2'])):    
        temp = (2,data.iloc[i,1])
        if temp not in candidate:
            candidate.append(temp)
    return candidate

def cal_entropy(data):
    num1 = len(data[data.y==0].y)
    num2 = len(data[data.y==1].y)
    if(num1+num2==0):
        return 0
    p0 = num1/(num1+num2)
    p1 = num2/(num1+num2)
    if(p0==0 or p1==0):
        return 0
    return -(p0*math.log2(p0)+p1*math.log2(p1))

def cal_split_gain_ratio(data, split, debug):
    if(split[0]==1):
        temp_data_left = data[data.x1>=split[1]]
        temp_data_right = data[data.x1<split[1]]
    else:
        temp_data_left = data[(data.x2>=split[1])]
        temp_data_right = data[(data.x2<split[1])]
    num1 = len(temp_data_left.y)
    num2 = len(temp_data_right.y)
    p0 = num1/(num1+num2)
    p1 = num2/(num1+num2)
    if(p0==0 or p1==0):
        split_info = 0
    else:
        split_info = -(p0*math.log2(p0)+p1*math.log2(p1))
    conditional_entropy = p0*cal_entropy(temp_data_left) + p1*cal_entropy(temp_data_right)
    info_gain = cal_entropy(data) - conditional_entropy
    if(split_info == 0):
        if(debug):
            print("Information gain:", info_gain)
        return(split_info,0)
    gain_ratio = info_gain/split_info
    if(debug):
        print("Information gain:", info_gain, "\tGain ratio:", gain_ratio)
    return (split_info,gain_ratio)

def MakeSubtree(node, debug):
    data = node.data
    candidate = DetermineCandidateSplits(data)
    #the node is empty
    if(len(data['y'])==0):
        node.classlabel = 1
        return 
    best_gain = 0
    best_split_entropy = 0
    if debug:
        print("For the node:")
    for candid in candidate:
        if debug:
            print("for split:" + str(candid))
        split_gain_ratio = cal_split_gain_ratio(data, candid,debug)
        split_info = split_gain_ratio[0]
        gain_ratio = split_gain_ratio[1]
        if(split_info>best_split_entropy):
            best_split_entropy = split_info
        ## find best split
        if(gain_ratio>best_gain):
            best_gain = gain_ratio
            best_split = candid
    #the entropy of any candidates split is zero        
    if(best_split_entropy==0):
        count = data['y'].value_counts()
        ## whenever there is no majority class in a leaf, let it predict y = 1
        if (count.shape[0]>1 and count.iloc[0]==count.iloc[1]):
            node.classlabel = 1
        else:
            node.classlabel = count.index[0]
        return
    #all splits have zero gain ratio (if the entropy of the split is non-zero)
    if(best_gain==0):
        count = data['y'].value_counts()
        ## whenever there is no majority class in a leaf, let it predict y = 1
        if (count.shape[0]>1 and count.iloc[0]==count.iloc[1]):
            node.classlabel = 1
        else:
            node.classlabel = count.index[0]
        return
    node.best_split = best_split
    if(best_split[0]==1):
        temp_data_left = data[data.x1>=best_split[1]]
        temp_data_right = data[data.x1<best_split[1]]
    else:
        temp_data_left = data[data.x2>=best_split[1]]
        temp_data_right = data[data.x2<best_split[1]]
    left_node = treenode(temp_data_left,node)
    MakeSubtree(left_node,debug)
    right_node = treenode(temp_data_right,node)
    MakeSubtree(right_node, debug)
    node.left = left_node
    node.right = right_node
    return node

def traverseTree(node,string,print_tree):
    if(node.classlabel is not None):
        if(print_tree):
            print(string + "label:" + str(node.classlabel))
        return 1
    else:
        if(print_tree):
            print(string + "y=0: " + str(len(node.data.y[node.data.y==0])) + "\ty=1: " +str(len(node.data.y[node.data.y==1]))+
              "\tBest split: "+str(node.best_split))       
        return traverseTree(node.left, string + "-", print_tree)+traverseTree(node.right,string + "-", print_tree)+1


def plot_data(data):
    ##data.y = data.y.astype('category')
    plt.scatter(data.x1[data.y==0], data.x2[data.y==0], color = 'r', label = "y=0")
    plt.scatter(data.x1[data.y==1], data.x2[data.y==1], color = 'b', label = "y=1")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

def pred_data(node, x1, x2):
    while(node.classlabel == None):
        if(node.best_split[0] == 1):
            if(x1 >= node.best_split[1]):
                node = node.left
            else:
                node = node.right
        else:
            if(x2 >= node.best_split[1]):
                node = node.left
            else:
                node = node.right
    return node.classlabel

def draw_boundary(node,min, max):
    x1 = np.arange(min, max, 0.01)##can be changed to 0.05
    x2 = np.arange(min, max, 0.01)
    l1 = len(x1)
    l2 = len(x2)
    x1 = np.repeat(x1, l2)
    x2 = np.tile(x2, l1)
    for i in range(len(x1)):
        if(pred_data(node, x1[i], x2[i])==0):
            plt.scatter(x1[i], x2[i], color = 'r',alpha = 0.1,s=50)
        else:
            plt.scatter(x1[i], x2[i], color = 'b',alpha = 0.1,s=50)

f = open(sys.argv[1], "r")
x1 = []
x2 = []
y = []

a = 0
for line in f:
    a = a+1
    temp = line.split()
    x1.append(temp[0])
    x2.append(temp[1])
    y.append(temp[2])
data = pd.DataFrame(list(zip(x1, x2, y)), columns =['x1', 'x2','y'], dtype = float)
data.y = data.y.astype(int)

plot_data(data)
root = treenode(data, None)
MakeSubtree(root, False)
print("Number of node:", traverseTree(root,"-",True))
draw_boundary(root,0,1)
plt.show()

#Q7
# data = data.sample(n = 10000, random_state=1)
# train_data = data.iloc[:8192,]
# test_data = data.iloc[8192:,]
# D32 = train_data.iloc[:32,]
# D128 = train_data.iloc[:128,]
# D512 = train_data.iloc[:512,]
# D2048 = train_data.iloc[:2048,]
# D8192 = train_data.iloc[:8192,]

# n = []
# error_list = []

# root = treenode(D32, None)
# MakeSubtree(root, False)
# print("Number of node:", traverseTree(root,"-",False))
# error = 0
# for i in range(len(test_data.y)):
#     error = error + (test_data.y.iloc[i]!=pred_data(root,test_data.x1.iloc[i],test_data.x2.iloc[i]))
# print("Error:", error/len(test_data.y))
# error_list.append(error/len(test_data.y))
# n.append(32)
# plot_data(D32)
# draw_boundary(root,-2,2)
# plt.show()


# root = treenode(D128, None)
# MakeSubtree(root, False)
# print("Number of node:", traverseTree(root,"-",False))
# error = 0
# for i in range(len(test_data.y)):
#     error = error + (test_data.y.iloc[i]!=pred_data(root,test_data.x1.iloc[i],test_data.x2.iloc[i]))
# print("Error:", error/len(test_data.y))
# error_list.append(error/len(test_data.y))
# n.append(128)
# plot_data(D128)
# draw_boundary(root,-2,2)
# plt.show()

# root = treenode(D512, None)
# MakeSubtree(root, False)
# print("Number of node:", traverseTree(root,"-",False))
# error = 0
# for i in range(len(test_data.y)):
#     error = error + (test_data.y.iloc[i]!=pred_data(root,test_data.x1.iloc[i],test_data.x2.iloc[i]))
# print("Error:", error/len(test_data.y))
# error_list.append(error/len(test_data.y))
# n.append(512)
# plot_data(D512)
# draw_boundary(root,-2,2)
# plt.show()

# root = treenode(D2048, None)
# MakeSubtree(root, False)
# print("Number of node:", traverseTree(root,"-",False))
# error = 0
# for i in range(len(test_data.y)):
#     error = error + (test_data.y.iloc[i]!=pred_data(root,test_data.x1.iloc[i],test_data.x2.iloc[i]))
# print("Error:", error/len(test_data.y))
# error_list.append(error/len(test_data.y))
# n.append(2048)
# plot_data(D2048)
# draw_boundary(root,-2,2)
# plt.show()

# root = treenode(D8192, None)
# MakeSubtree(root, False)
# print("Number of node:", traverseTree(root,"-",False))
# error = 0
# for i in range(len(test_data.y)):
#     error = error + (test_data.y.iloc[i]!=pred_data(root,test_data.x1.iloc[i],test_data.x2.iloc[i]))
# print("Error:", error/len(test_data.y))
# error_list.append(error/len(test_data.y))
# n.append(8192)
# plot_data(D8192)
# draw_boundary(root,-2,2)
# plt.show()

# plt.plot(n, error_list, "ro")
# plt.plot(n, error_list, "-b")
# plt.xlabel('Number of nodes')
# plt.ylabel('Test set error errn')
# plt.title('Learning Curve')
# plt.show()

# 3. sklearn

# n = []
# error_list = []
# X = D32[['x1','x2']]
# y = D32['y']
# clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=1)
# clf = clf.fit(X, y)
# print("Number of node:", clf.tree_.node_count)
# print("Error:", 1-clf.score(test_data[['x1','x2']],test_data['y']))
# error_list.append(1-clf.score(test_data[['x1','x2']],test_data['y']))
# n.append(32)

# X = D128[['x1','x2']]
# y = D128['y']
# clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=1)
# clf = clf.fit(X, y)
# print("Number of node:", clf.tree_.node_count)
# print("Error:", 1-clf.score(test_data[['x1','x2']],test_data['y']))
# error_list.append(1-clf.score(test_data[['x1','x2']],test_data['y']))
# n.append(128)

# X = D512[['x1','x2']]
# y = D512['y']
# clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=1)
# clf = clf.fit(X, y)
# print("Number of node:", clf.tree_.node_count)
# print("Error:", 1-clf.score(test_data[['x1','x2']],test_data['y']))
# error_list.append(1-clf.score(test_data[['x1','x2']],test_data['y']))
# n.append(512)

# X = D2048[['x1','x2']]
# y = D2048['y']
# clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=1)
# clf = clf.fit(X, y)
# print("Number of node:", clf.tree_.node_count)
# print("Error:", 1-clf.score(test_data[['x1','x2']],test_data['y']))
# error_list.append(1-clf.score(test_data[['x1','x2']],test_data['y']))
# n.append(2048)

# X = D8192[['x1','x2']]
# y = D8192['y']
# clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=1)
# clf = clf.fit(X, y)
# print("Number of node:", clf.tree_.node_count)
# print("Error:", 1-clf.score(test_data[['x1','x2']],test_data['y']))
# error_list.append(1-clf.score(test_data[['x1','x2']],test_data['y']))
# n.append(8192)

# plt.plot(n, error_list, "ro")
# plt.plot(n, error_list, "-b")
# plt.xlabel('Number of nodes')
# plt.ylabel('Test set error errn')
# plt.title('Learning Curve')
# plt.show()

#Part 4 Lagrange Interpolation 
# np.random.seed(0)

# x = np.random.uniform(0, 2*np.pi,100)
# y = np.sin(x)
# x_test = np.random.uniform(0, 2*np.pi,100)
# y_test = np.sin(x_test)

# poly = lagrange(x, y)
# print("MSE_train:",np.log(sum((poly(x)-y)**2)/len(x)))
# print("MSE_test:",np.log(sum((poly(x_test)-y_test)**2)/len(x)))

# sigmas = [0,0.5,1,10,50,100,1000]
# a = np.random.normal(0, 1, size = len(x))
# for sigma in sigmas:
#     x_noise = x + np.random.normal(0, sigma, size = len(x))
#     y_noise = np.sin(x_noise)
#     poly = lagrange(x_noise, y_noise)
#     print("For standard deviation=", sigma, "\tMSE_train:",np.log(sum((poly(x_noise)-y_noise)**2)/len(x)))
#     print("For standard deviation=", sigma, "\tMSE_test:",np.log(sum((poly(x_test)-y_test)**2)/len(x)))


