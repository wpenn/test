############################################################
# CIS 521: Perceptrons Homework
############################################################

student_name = "Wesley Penn"

############################################################
# Imports
############################################################

import perceptrons_data as data
from collections import defaultdict
# import matplotlib.pyplot as plt

# Include your imports here, if any are used.


############################################################
# Section 1: Perceptrons
############################################################

def dict_dot(x_dict, w_dict):
    val = 0
    for feature in x_dict:
        val += x_dict[feature] * w_dict[feature]
    return val


class BinaryPerceptron(object):

    def __init__(self, examples, iterations):
        self.w_dict = defaultdict(int)
        for _ in range(iterations):
            for x, y in examples:
                pred = self.predict(x)
                if pred != y:
                    for feature in x:
                        self.w_dict[feature] += (-1 if pred else 1) * x[feature]

    def predict(self, x):
        return dict_dot(x, self.w_dict) >= 0


# train = [({"x1": 1}, True), ({"x2": 1}, True), ({"x1": -1}, False), ({"x2": -1}, False)]
# test = [{"x1": 1}, {"x1": 1, "x2": 1}, {"x1": -1, "x2": 1.5}, {"x1": -0.5, "x2": -2}]
# p = BinaryPerceptron(train, 1)
# print([p.predict(x) for x in test])

class MulticlassPerceptron(object):
    def __init__(self, examples, iterations):
        self.class_w_dict = defaultdict(lambda: defaultdict(lambda: 0))
        for _ in range(iterations):
            for x_dict, y in examples:
                _ = self.class_w_dict[y] # initialize
                pred = self.predict(x_dict)
                if pred != y:
                    for feature in x_dict:
                        self.class_w_dict[y][feature] += x_dict[feature]
                        self.class_w_dict[pred][feature] -= x_dict[feature]

    # Argmax
    def predict(self, x):
        max_val = float("-inf")
        best_class = None
        for cl in self.class_w_dict:
            val = dict_dot(x, self.class_w_dict[cl])
            if val > max_val:
                max_val = val
                best_class = cl
        return best_class


# train = [({"x1": 1}, 1), ({"x1": 1, "x2": 1}, 2), ({"x2": 1}, 3), ({"x1": -1, "x2": 1}, 4),
#          ({"x1": -1}, 5), ({"x1": -1, "x2": -1}, 6), ({"x2": -1}, 7), ({"x1": 1, "x2": -1}, 8)]
# p = MulticlassPerceptron(train, 10)
# print(p.class_w_dict)
# print("Predicted: ", [p.predict(x) for x, y in train])
# print("Expected: ", [1, 2, 3, 4, 5, 6, 7, 8])

############################################################
# Section 2: Applications
############################################################
def get_labeled_features(x):
    labeled_features = {
        f"x{i + 1}": x[i]
        for i in range(len(x))
    }
    return labeled_features


def get_sparse_dataset(datum):
    return [
        (get_labeled_features(x), y)
        for x, y in datum
    ]


class IrisClassifier(object):

    def __init__(self, data):
        formatted_data = get_sparse_dataset(data)
        self.classifier = MulticlassPerceptron(formatted_data, 100)

    def classify(self, instance):
        return self.classifier.predict(get_labeled_features(instance))


# c = IrisClassifier(data.iris)
# print(c.classify((5.1, 3.5, 1.4, 0.2)))
# print(c.classify((7.0, 3.2, 4.7, 1.4)))


class DigitClassifier(object):

    def __init__(self, data):
        self.classifier = MulticlassPerceptron(get_sparse_dataset(data), 30)

    def classify(self, instance):
        return self.classifier.predict(get_labeled_features(instance))


# c = DigitClassifier(data.digits)
# print(c.classify((0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0)))


class BiasClassifier(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron(self.augment_data(data), 30)

    def classify(self, instance):
        return self.classifier.predict(self.augment_val(instance))

    def augment_data(self, data):
        return [
            (self.augment_val(val), y)
            for val, y in data
        ]

    def augment_val(self, val):
        return {
            "x": val,
            "bias": 1
        }


# c = BiasClassifier(data.bias)
# print([c.classify(x) for x in (-1, 0, 0.5, 1.5, 2)])
#[False, False, False, True, True]



class MysteryClassifier1(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron(self.augment_data(data), 30)

    def classify(self, instance):
        return self.classifier.predict(self.augment_val(instance))

    def augment_data(self, data):
        return [
            (self.augment_val(val), y)
            for val, y in data
        ]

    def augment_val(self, instance):
        return {
            "x": instance[0],
            "y": instance[1],
            "map": instance[0] ** 2 + instance[1] ** 2,
            "bias": 1
        }

# c = MysteryClassifier1(data.mystery1)
# print([c.classify(x) for x in ((0, 0), (0, 1), (-1, 0), (1, 2), (-3, -4))])
 # [False, False, False, True, True]

# false_x, false_y, true_x, true_y = [], [], [], []
# for (x, y), label in data.mystery1:
#     if label:
#         true_x.append(x**2 + y**2)
#         true_y.append(y)
#     else:
#         false_x.append(x)
#         false_y.append(y)
# plt.scatter(true_x, true_y, color='green')
# plt.scatter(false_x, false_y, color='red')
# plt.show()

class MysteryClassifier2(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron(self.augment_data(data), 100)

    def classify(self, instance):
        return self.classifier.predict(self.augment_val(instance))

    def augment_data(self, data):
        return [
            (self.augment_val(val), y)
            for val, y in data
        ]

    def augment_val(self, instance):
        return {
            "x": instance[0],
            "y": instance[1],
            "z": instance[2],
            "map": instance[0] * instance[1] * instance[2],
            "bias": 1
        }

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# false_x, false_y, false_z, true_x, true_y, true_z= [], [], [], [], [], []
# for (x, y, z), label in data.mystery2:
#     if label:
#         true_x.append(x)
#         true_y.append(y)
#         true_z.append(z)
#     else:
#         false_x.append(x)
#         false_y.append(y)
#         false_z.append(z)
# ax.scatter(true_x, true_y, true_z, marker="o",  s=20, color='green')
# ax.scatter(false_x, false_y, false_z, marker="^", s=20, color='red')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

# c = MysteryClassifier2(data.mystery2)
# print([c.classify(x) for x in ((1, 1, 1), (-1, -1, -1), (1, 2, -3), (-1, -2, 3))])
# [True, False, False, True]


############################################################
# Section 3: Feedback
############################################################

feedback_question_1 = 10

feedback_question_2 = """
I found debugging my multiclass classifier difficult. I was getting the autograder correct but the application section was
taking too long and was not working.
"""

feedback_question_3 = """
I liked figuring out how to classify the mystery dataset. First having to plot out the data, figuring out what mappings 
made it linearly separable, and then finetuning it to work with the test grader was cool.
"""
