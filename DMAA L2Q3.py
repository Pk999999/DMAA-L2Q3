import pandas as pd
from math import log2
from collections import Counter

def entropy(target):
    total = len(target)
    return -sum((count / total) * log2(count / total) for count in Counter(target).values())

def information_gain(data, attribute, target):
    total_entropy = entropy(data[target])
    weighted_entropy = 0
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy

def best_attribute(data, attributes, target):
    return max(attributes, key=lambda attr: information_gain(data, attr, target))

class DecisionTree:
    def __init__(self, data, target, attributes):
        self.data = data
        self.target = target
        self.attributes = attributes
        self.tree = self.build_tree(data, target, attributes)

    def build_tree(self, data, target, attributes):
        if len(data[target].unique()) == 1:
            return data[target].iloc[0]

        if len(attributes) == 0:
            return data[target].mode().iloc[0]

        best_attr = best_attribute(data, attributes, target)

        tree = {best_attr: {}}

        for value in data[best_attr].unique():
            subset = data[data[best_attr] == value]
            new_attributes = [attr for attr in attributes if attr != best_attr]
            subtree = self.build_tree(subset, target, new_attributes)
            tree[best_attr][value] = subtree

        return tree

data = pd.read_csv('road_transport_records.csv')

target = 'AccidentRisk'
attributes = ['Length', 'Numberof_Bends', 'Trafficvolume']

dt = DecisionTree(data, target, attributes)
import pprint
pprint.pprint(dt.tree)
print("Prithvi Kathuria, 21BBS0158")