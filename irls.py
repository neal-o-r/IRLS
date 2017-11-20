import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def sigmoid(x): 
        return 1 / (1 + np.exp(-x))

def fit(X, y, maxiter=100, eps=1e-4, l=1):

        n, f = X.shape

        p = sigmoid(np.zeros(f).dot(X.T))
        W = np.eye(n) * p * (1 - p)
        r = l * np.eye(f)
        
        beta = np.linalg.inv(X.T.dot(W).dot(X) + 
                        r).dot(X.T.dot(y))

        delta = 1

        i = 0
        while (delta > eps) and i < maxiter:
                
                p = sigmoid(beta.dot(X.T))
                W = np.eye(n) * p * (1 - p)
                z = X.dot(beta) + np.linalg.pinv(W).dot(y - p)

                r = l * np.linalg.norm(beta) * np.eye(f)
                beta_n = np.linalg.pinv(X.T.dot(W).dot(X)
                                        + r).dot(X.T).dot(W).dot(z)

                delta = np.linalg.norm(beta_n - beta)
                beta = beta_n
                i += 1

        return beta

if __name__ == '__main__':

        X, y = datasets.make_classification(n_samples=1000, n_features=20,
                        n_informative=2, n_redundant=10, random_state=42)

        X = np.hstack((np.ones(X.shape[0]).reshape(-1,1),  X))

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                test_size=0.66, random_state=42)


        beta = fit(X_train, y_train)

        p = sigmoid(beta.dot(X_test.T))

        print(roc_auc_score(y_test, p))

