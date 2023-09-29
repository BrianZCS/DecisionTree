# How to Run

Training the decision on a dataset:
```
python3 DT.py Filename
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
# Requirements

* Candidate splits (j, c) for numeric features should use a threshold c in feature dimension j in the form of
xj ≥ c.
* c should be on values of that dimension present in the training data; i.e. the threshold is on training points,
not in between training points. You may enumerate all features, and for each feature, use all possible values
for that dimension.
* You may skip those candidate splits with zero split information (i.e. the entropy of the split), and continue
the enumeration.
* The left branch of such a split is the “then” branch, and the right branch is “else”.
* Splits should be chosen using information gain ratio. If there is a tie you may break it arbitrarily.
* The stopping criteria (for making a node into a leaf) are that
– the node is empty, or
– all splits have zero gain ratio (if the entropy of the split is non-zero), or
– the entropy of any candidates split is zero
* To simplify, whenever there is no majority class in a leaf, let it predict y = 1.