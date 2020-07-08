import tensorflow as tf
import pandas as pd
import numpy as np
import re
import glob
import matplotlib as plt
import seaborn as sns
import time
from tensorflow.python import keras as tfkrs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE


tf.enable_eager_execution()
start_time = time.time()

files_normal = glob.glob("/home/gon/Desktop/0622/dataset/1.normal/*")
files_inner = glob.glob("/home/gon/Desktop/0622/dataset/2.inner/*")
files_outer = glob.glob("/home/gon/Desktop/0622/dataset/3.outer/*")
files_roller = glob.glob("/home/gon/Desktop/0622/dataset/4.roller/*")

my_model = tfkrs.models.load_model('/home/gon/Desktop/untitled/IMS_final.h5')


_, train_normal = train_test_split(files_normal, test_size=50, random_state=23)
_, train_inner = train_test_split(files_inner, test_size=50, random_state=34)
_, train_outer = train_test_split(files_outer, test_size=50, random_state=232)
_, train_roller = train_test_split(files_roller, test_size=50, random_state=12)


train_files = train_normal + train_inner + train_outer + train_roller
test_files = files_normal[650:750] + files_inner[650:750] + files_outer[650:750] + files_roller[650:750]

print(len(train_files), len(test_files))

def data_create(file_names):
    data = []
    labels = []
    patterns = tf.constant([".*(normal)", ".*(inner)", ".*(outer)", ".*(roller)"])
    for file in file_names:
        temp = pd.read_csv(open(file, 'r'), sep="\s+", header=None)
        fault_columns = [0, 4, 2, 6]
        num = np.int(np.floor(len(temp[0])/1024))
        j = 0
        for pattern in patterns:
            if re.match(pattern.numpy(), tf.constant(file).numpy()):
                labels = labels + list(np.repeat(j, num))
                column_number = fault_columns[j]
                break
            j = j + 1
        data = data + list(temp[column_number][0:num*1024].values.reshape(num, 32, 32, 1))

    data = np.asarray(data).reshape(-1, 32, 32, 1)
    labels = np.asarray(labels)
    return data, labels



train_data, train_labels = data_create(train_files)
train_data, train_labels = train_data.astype(np.float32), train_labels.astype(np.float32)
test_data, test_labels = data_create(test_files)
test_data, test_labels = test_data.astype(np.float32), test_labels.astype(np.float32)
print("Train data\n", train_data.shape, train_labels.shape)
print("Test data\n", test_data.shape, test_labels.shape)


print(my_model)

outputs_train = my_model(train_data)
outputs_train = outputs_train.numpy()


outputs_test = my_model(test_data)
outputs_test = outputs_test.numpy()


index = np.random.permutation(len(outputs_train))

outputs_train, train_labels = outputs_train[index], train_labels[index]


# SVC
svm_fit = SVC(C=1, kernel= "rbf", gamma=0.01)
svm_fit.fit(outputs_train, train_labels)

train_res = svm_fit.predict(outputs_train)


# Confusion matrix
c_matrix = confusion_matrix(train_labels, train_res)
print("State Confusion matrix\n", c_matrix)

test_res = svm_fit.predict(outputs_test)


#
mat = confusion_matrix(test_labels, test_res)
print("SVM fit prediction Confusion matrix\n", mat)



#confu

confu = sns.heatmap(mat, annot=True, fmt="d", cbar=False,
                    xticklabels=["Normal", "Inner", "Outer", "Roller"],
                    yticklabels=["Normal", "Inner", "Outer", "Roller"],
                    linewidths=0.1, linecolor="lightblue")

confu.set(xlabel="Predicted", ylabel="True")
confu.set_yticklabels(confu.get_yticklabels(), va="center", rotation=90)

# setting heatmap

#plt.savefig("confusion_matrix_ims_iai.png")

ac_score = accuracy_score(test_labels, test_res)
print("Accuracy score :",ac_score)
print("Total time :", time.time() - start_time)

fault_type = pd.Categorical(np.repeat(["Normal", "Inner", "Outer", "Roller"], 2000))
tsne_res = TSNE(n_components=2, random_state=4).fit_transform(outputs_test)
plt.figure()
sns.scatterplot(
    x=tsne_res[:0], y=tsne_res[:1],
    hue=fault_type,
    data=pd.DataFrame(tsne_res)
)
plt.legend(title="Fault type")
plt.xlabel("First t-SNE Component")
plt.ylabel("Second t-SNE Component")
#plt.savefig("tsne_test_ims.png")

