import numpy as np
from sklearn.linear_model import LogisticRegression

def weighted_fusion(probs_list, weights=None):
    probs = np.vstack(probs_list).T
    if weights is None:
        w = np.ones(probs.shape[1])/probs.shape[1]
    else:
        w = np.array(weights)/np.sum(weights)
    return (probs * w).sum(axis=1)

def stacking_fusion(train_probs_list, y_train, test_probs_list):
    Xtr = np.vstack(train_probs_list).T
    Xte = np.vstack(test_probs_list).T
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtr, y_train)
    return lr.predict_proba(Xte)[:,1]
