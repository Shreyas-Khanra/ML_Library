import numpy as np


class Node():
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
    
    def _is_leaf_node(self):
        if self.value != None:
            return True
    

class DecisionTree():

    def __init__(self,min_samples_split=2,max_depth=10,n_features=None,n_bins=20):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.n_bins=n_bins
        self.root=None
    
    def fit(self,X,y):
        
        self.n_features = X.shape[1]

        self.root=self._grow_tree(X,y)


    def _grow_tree(self,X,y,depth=0):
    
        #check stopping criteria 
        n_samples,n_features=X.shape
        n_labels=len(np.unique(y))

        if n_samples == 0:
          return None
        
        #n_labels==1 checks if it is a pure node
        if(depth>self.max_depth or n_samples<self.min_samples_split or n_labels==1):
           leaf_value=self._most_common_label(y)
           return Node(value=leaf_value)
        
    
        
        #decide best split
        
        feat_ids=np.arange(n_features)

        best_feature,best_thresh=self._best_split(X,y,feat_ids)

        #create child nodes
        left_idxs,right_idxs=self._split(X[:,best_feature],best_thresh)
        left=self._grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right=self._grow_tree(X[right_idxs,:],y[right_idxs],depth+1)
        return Node(best_feature,best_thresh,left,right)

    
    def _most_common_label(self, y):
        labels, counts = np.unique(y, return_counts=True)
    
        index = np.argmax(counts)
        
        return labels[index]
        
    def _best_split(self,X,y,feat_ids):
        best_gain=0
        split_idx,split_threshold=None,None
            
        for feat_id in feat_ids:
            X_column=X[ : ,feat_id]
            uniques=np.unique(X_column)

            if len(uniques)<=self.n_bins:
                threshold=uniques
            
            else:
                threshold=np.percentile(X_column, np.linspace(0, 100, self.n_bins))
                
            for thr in threshold:
                gain=self._information_gain(X_column,y,thr)
                if gain>best_gain:
                    best_gain=gain
                    split_idx=feat_id
                    split_threshold=thr 
                
        return split_idx,split_threshold
        
    def _information_gain(self,X_column,y,threshold):
        #parent entropy
        parent_entropy=self._entropy(y)
        
        #create child nodes 
        left_idx,right_idx=self._split(X_column,threshold)

        if len(left_idx)==0 or len(right_idx)==0:
                return 0
            
        #weighted entropy of children
        n=len(y)
        w_l,w_r=len(left_idx)/n ,len(right_idx)/n
        e_l,e_r=self._entropy(y[left_idx]),self._entropy(y[right_idx])
        weighted_average=w_l*e_l+ w_r*e_r
        #information gain
        information_gain=parent_entropy - weighted_average
        return information_gain
         
    def _split(self,X_column,split_threshold):

        left_idx=np.argwhere(X_column<=split_threshold).flatten()
        right_idx=np.argwhere(X_column>split_threshold).flatten()
        return left_idx,right_idx
    
    def _entropy(self,y):
        if len(y) == 0: return 0
        _, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        return -np.sum(ps * np.log2(ps))    
            

    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    def _traverse_tree(self,x,node):
        if node._is_leaf_node():
            return node.value
        
        if(x[node.feature])<=node.threshold:
            return self._traverse_tree(x,node.left)
        else:
            return self._traverse_tree(x,node.right)
        
