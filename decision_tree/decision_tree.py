import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = {}
        self.depth = 0

    def fit(self, X, y):
        self.tree = self._fit(X, y, depth = 0) 
    
    def _fit(self, X, y, depth = 0):
        # Initialize a node dictionary
        node = {}
        
        # Check if all instances in the current node belong to the same class
        if len(np.unique(y)) == 1:
            node['leaf'] = True
            node['label'] = np.unique(y)[0]
            return node

        # Also check for maximum depth and min samples
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            node['leaf'] = True
            node['label'] = y.mode()[0]  # using mode to find most frequent label
            return node
        
        # Calculate overall entropy
        overall_entropy = entropy(pd.Series(y).value_counts().to_numpy())

        
        # Calculate the information gains
        info_gain_dict = {}
        for column in X.columns: 
            conditional_entropy = 0
            for value in X[column].unique():
                subgroup = y[X[column] == value]
                subgroup_entropy = entropy(subgroup.value_counts().to_numpy())
                weight = len(subgroup) / len(y)
                conditional_entropy += weight * subgroup_entropy
            info_gain = overall_entropy - conditional_entropy
            info_gain_dict[column] = info_gain

        # Choose the feature with the maximum information gain
        best_feature = max(info_gain_dict, key=info_gain_dict.get)
        
        # Implement a minimum Information Gain threshold
        if info_gain_dict[best_feature] < 1e-6:  
            node['leaf'] = True
            node['label'] = y.mode()[0]
            return node

        # Initialize the node as a non-leaf node
        node['leaf'] = False
        node['feature'] = best_feature
        node['children'] = {}
        node['majority_label'] = y.mode()[0]  # Store majority label at each node

        
        # Split the dataset and call `fit` recursively
        unique_values = X[best_feature].unique()
        for value in unique_values:
            mask = X[best_feature] == value
            node['children'][value] = self._fit(X[mask], y[mask], depth + 1)
            
        return node

                

            

    def prune(self, node, validation_data):
        if node['leaf']:  # If leaf node, nothing to prune
            return
        
        # First prune all children
        for child_node in node['children'].values():
            self.prune(child_node, validation_data)
        
        # Temporarily convert this node to a leaf node
        original_type = node['leaf']
        original_label = node.get('label', None)  # store the original label
        node['leaf'] = True
        node['label'] = pd.Series(validation_data['y']).mode()[0]  # Most frequent class in validation set
        
        # Check if pruning reduced accuracy
        pruned_accuracy = accuracy(validation_data['y'], self.predict(validation_data['X']))
        node['leaf'] = False  # revert to original type for accurate original prediction
        original_accuracy = accuracy(validation_data['y'], self.predict(validation_data['X']))
        
        # If pruned_accuracy is greater than or equal to original_accuracy, keep the node pruned
        if pruned_accuracy >= original_accuracy:
            node['leaf'] = True
        else:
            node['leaf'] = original_type
            node['label'] = original_label  # revert to original label if we decide not to prune this node
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        # TODO: Implement
        result = [] 
        for _, row in X.iterrows():
            node = self.tree 
            while node['leaf'] == False:
                feature_value = row[node['feature']]
                if feature_value in node['children']:
                    node = node['children'][feature_value]
                else:
                    # Use the majority label if feature value is not found
                    node = {'leaf': True, 'label': node['majority_label']}
            result.append(node['label'])
        return np.array(result)


            


        
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        rules = []

        def traverse_tree(tree, path):
            if tree['leaf']:
                rules.append((path, tree['label']))
                return
            for value, subtree in tree['children'].items():
                new_path = path + [(tree['feature'], value)]
                traverse_tree(subtree, new_path)

        traverse_tree(self.tree, [])
        return rules


# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))



