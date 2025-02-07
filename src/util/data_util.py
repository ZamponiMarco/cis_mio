import numpy as np


def gini_index(output_points) -> float:
    _, counts = np.unique(output_points, return_counts=True, axis=0)
    total = sum(counts)
    return 1 - sum([(count / total) ** 2 for count in counts])

def label_to_class(label):
    return int("".join(map(str, label)), 2)

def labels_to_classes(y):
    return np.array([label_to_class(label) for label in y])

def class_to_label(clazz, label_length):
    return list(map(int, format(clazz, f'0{label_length}b')))

def classes_to_labels(classes, label_length):
    return np.array([class_to_label(clazz, label_length) for clazz in classes])