# How to Run

Training the decision on a dataset:
```
python3 DY.py Filename
```

Example:
```
python3 DT.py D1.txt
```

# Functions in the code
Two important lines in DT.Py. It helps train the decision tree.
```
root = treenode(data, None)
MakeSubtree(root, False)
```

Print the decision tree:
```
traverseTree(root,"-",False)
```

Test the decision tree:
```
for i in range(len(test_data.y)):
     error = error + (test_data.y.iloc[i]!=pred_data(root,test_data.x1.iloc[i],test_data.x2.iloc[i]))
```
