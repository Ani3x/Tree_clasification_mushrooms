import pandas as pd



class Node:
    def __init__(self, feature=None, children=None, label=None):
        self.feature = feature        # cecha do podziału
        self.children = children or {}  # dict: wartość -> Node
        self.label = label            # jeśli liść


def gini_index(dane_frame, cecha : str):
    n = dane_frame.shape[0]
    grouped = dane_frame.groupby(cecha)
    
    gini_value = 0

    for x, dataframe in grouped:
        rows = dataframe.shape[0]

        sum_edible = (dataframe['class'] == 'e').sum()
        sum_poisonous = (dataframe['class'] == 'p').sum()

        g = 1 - pow(sum_edible/rows, 2) - pow(sum_poisonous/rows, 2)
        w = rows/n
        gini_value += w*g
    
    return gini_value, cecha


def best_feature(df, features):
    best = None
    best_gini = float('inf')

    for feature in features:
        g = gini_index(df, feature)[0]
        if g < best_gini:
            best_gini = g
            best = feature

    return best


def predict(node, sample):
    if node.label is not None:
        return node.label

    value = sample[node.feature]

    if value in node.children:
        return predict(node.children[value], sample)
    else:
        return None  # albo fallback
    

def build_tree(df, features, depth=0, max_depth=5):
    # jeśli wszystkie klasy takie same → liść
    if len(df['class'].unique()) == 1:
        return Node(label=df['class'].iloc[0])

    # warunek stopu
    if depth >= max_depth or len(features) == 0:
        return Node(label=df['class'].mode()[0])

    # wybór najlepszej cechy
    best = best_feature(df, features)
    node = Node(feature=best)

    for value, subset in df.groupby(best):
        if subset.empty:
            node.children[value] = Node(label=df['class'].mode()[0])
        else:
            node.children[value] = build_tree(
                subset,
                [f for f in features if f != best],
                depth + 1,
                max_depth
            )

    return node
