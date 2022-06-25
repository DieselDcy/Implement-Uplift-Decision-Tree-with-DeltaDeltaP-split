

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterable



class Node:

    def __init__(self, n_samples, ddp, ate_value):
        self.n_samples = n_samples
        self.ddp = ddp  # delta delta p
        self.ate_value = ate_value # average treatment effect
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None

    
    

class UpliftTreeRegressor:
    def __init__(
       self,
       max_depth: int =3,  # max tree depth
       min_samples_leaf: int = 6000, # min number of values in leaf
       min_samples_leaf_treated: int = 2500, # min number of treatment values in leaf
       min_samples_leaf_control: int = 2500,  # min number of control values in leaf
       ):

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control

    '''
    Create root with all values
    Recursively run function Build:
    '''
    def build_tree(self, x, treatment, y, depth: int = 0) -> Node:
        
        node = Node(n_samples=x.shape[0], ddp=0, ate_value=self.cal_ate(treatment,y))
        
        if depth < self.max_depth:
            
            feat_n, thres = self.best_split(x, treatment, y)
            
            if feat_n is not None:
                
                indices_left = x[:, feat_n] <= thres
                left_x, left_treatment, left_y = \
                    x[indices_left], treatment[indices_left], y[indices_left]
                right_x, right_treatment, right_y = \
                    x[~indices_left], treatment[~indices_left], y[~indices_left]
                
                node.split_feature = feat_n
                node.split_threshold = thres
                node.left = self.build_tree(left_x, left_treatment, left_y, depth + 1)
                node.right = self.build_tree(right_x, right_treatment, right_y, depth + 1)
        
        return node
    
    def fit(
       self,
       x: np.ndarray, # (n * k) array with features
       treatment: np.ndarray,  # (n) array with treatment flag
       y: np.ndarray  # (n) array with the target
    ) -> None:
       
        # fit the model
        self.n_features = len(x[0,:])
        self.tree = self.build_tree(x, treatment, y)
    
    
    
    
    def recur(self, row: np.ndarray) -> float:
        # recursively find ate
        node = self.tree
        while node.left:
            if row[node.split_feature] < node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node.ate_value

    def predict(self, x: np.ndarray) -> Iterable[float]:
        return np.array([self.recur(row) for row in x])
        

    
    
    def threshold_options(self, column_values):
        # Threshold algorithm:
        unique_values = np.unique(column_values)
        if len(unique_values) > 10:
            percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])
        return np.unique(percentiles)

    def cal_ate(self, treatment: np.ndarray, y: np.ndarray) -> float:
        #calculate average treatment effect

        ate = y[treatment == 1].mean() - y[treatment == 0].mean()
    
        return ate

    def check_min(self, treatment: np.ndarray) -> bool:
        if (len(treatment) >= self.min_samples_leaf and
            len(treatment[treatment == 1]) >= self.min_samples_leaf_treated and
            len(treatment[treatment == 0]) >= self.min_samples_leaf_control):
            return True
        return False


    def best_split(self, x, treatment, y) -> (int, float):
        
        best_ddp = 0.0
        split_feat = None
        split_thres = None

        for feat_n in range(self.n_features):

            thres_op = list(self.threshold_options(x[:, feat_n]))
            
            for thres in thres_op:
                left_x,left_treatment,left_y,\
                right_x, right_treatment, right_y\
                = self.split_first(x[:, feat_n], treatment, y, threshold=thres)
                
                if self.check_min(left_treatment) and self.check_min(right_treatment):
                    ddp = abs(self.cal_ate(left_treatment, left_y) - self.cal_ate(right_treatment, right_y))
    

                
                    if ddp > best_ddp:
                        best_ddp = ddp
                        split_feat = feat_n
                        split_thres = thres

        return split_feat, split_thres
    
    def split_first(self, x, treatment, y, threshold):
        left_x,left_treatment,left_y,\
        right_x, right_treatment, right_y\
        = x[x <= threshold], treatment[x <= threshold], y[x <= threshold], \
          x[x > threshold], treatment[x > threshold], y[x > threshold]
        return left_x, left_treatment, left_y, right_x, right_treatment, right_y
    

if __name__ == '__main__':

    x = np.load('example_X.npy')
    y = np.load('example_y.npy')
    treatment = np.load('example_treatment.npy')
    pred = np.load('example_preds.npy')
    
    model = UpliftTreeRegressor()
    model.fit(x, treatment, y)
    predicts = model.predict(x)
    



    