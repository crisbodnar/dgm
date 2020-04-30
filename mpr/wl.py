"""
=========================================================================
Graph classification on MUTAG using the Weisfeiler-Lehman subtree kernel.
=========================================================================
Script makes use of :class:`grakel.WeisfeilerLehman`, :class:`grakel.VertexHistogram`
"""
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram

# Loads the dataset
dataset = fetch_dataset("REDDIT-MULTI-5K", verbose=False, produce_labels_nodes=True)

# Splits the dataset into a training and a test set
kf = StratifiedKFold(n_splits=10, shuffle=False)
curr_fold = 0

for train_val_idxs, test_idxs in kf.split(dataset.data, dataset.target):
    curr_fold += 1
    print('>>> 10-fold cross-validation --- fold %d' % curr_fold)
    kf2 = StratifiedKFold(n_splits=9, shuffle=False)

    train_val_data = [dataset.data[i] for i in train_val_idxs]
    train_val_targets = [dataset.target[i] for i in train_val_idxs]

    for train_idxs, _ in kf2.split(train_val_data, train_val_targets):
        print(len(train_idxs), len(dataset.data))
        train_dataset_data = [train_val_data[i] for i in train_idxs]
        train_dataset_target = [train_val_targets[i] for i in train_idxs]
        break

    test_data = [dataset.data[i] for i in test_idxs]
    test_targets = [dataset.target[i] for i in test_idxs]

    # Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
    gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    K_train = gk.fit_transform(train_dataset_data)
    K_test = gk.transform(test_data)

    # Uses the SVM classifier to perform classification
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, train_dataset_target)
    y_pred = clf.predict(K_test)

    # Computes and prints the classification accuracy
    acc = accuracy_score(test_targets, y_pred)
    print("Accuracy:", str(round(acc*100, 2)) + "%")
