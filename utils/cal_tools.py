import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl

mpl.rcParams['font.sans-serif']=["Microsoft YaHei"]
mpl.rcParams['axes.unicode_minus']=False # in case minus sign is shown as box


def matrixs(pre, true, model_name, classes):
    n = len(classes)
    matrix = np.zeros((n, n), dtype="float")
    for i in range(len(pre)):
        matrix[int(true[i])][int(pre[i])] += 1
    print(matrix)
    print("acc: ", np.sum(np.diagonal(matrix))/np.sum(matrix))
    MakeMatrix(matrix=matrix, name=model_name, classes=classes).draw()


class MakeMatrix(object):
    def __init__(self, matrix, name, classes):
        self.matrix = matrix
        self.classes = classes
        self.classes2 = classes
        self.name = name

    def draw(self):
        plt.figure(figsize=(10, 8), facecolor='#FFFFFF')
        self.plot_confusion_matrix(self.matrix, self.classes, normalize=False,
                                   title=self.name)

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # print(type(cm))

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, size=24)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=16)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, self.classes2, rotation=90, size=16)
        plt.yticks(tick_marks, classes, size=16)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, int(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", size=20)

        plt.ylabel('True', size="24")
        plt.xlabel('Predict', size="24")
        plt.tight_layout()
        plt.savefig(self.name + ".png")
        plt.show()