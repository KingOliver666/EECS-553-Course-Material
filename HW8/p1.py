import numpy as np
import sklearn
import sklearn.linear_model as sk_linear

def load_data():
    # Label 0 are pullovers, label 1 are coats
    # X matrices are shape (num samples, num pixels)
    data = np.load("p3f_data.npy", allow_pickle=True).item()
    X_train = data["X_train"] # (1500, 784)
    X_val = data["X_val"]     # (1400, 784)
    X_test = data["X_test"]   # (700, 784)
    y_train = data["y_train"] # (1500,)
    y_val = data["y_val"]     # (1400,)
    y_test = data["y_test"]   # (700,)
    return X_train, X_val, X_test, y_train, y_val, y_test

def prediction_metrics(y_true, y_pred):
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)
    prec = sklearn.metrics.precision_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    return acc, f1, prec, recall

def EM_adjust_posterior(classifier, X_train, y_train, train_prior, X_val):
    # classifier is an sklearn.linear_model.LogisticRegression class
    # TODO: complete this function for estimating class priors after
    # the label shift. 
    # X_val contains sample feature vectors from after the shift.
    # Hint: the method LogisticRegression.predict_proba()
    # may be useful.
    new_prior = train_prior

    for t in range(100):
        # TODO: iteratively update new_prior
        continue
        
    return new_prior

def update_predictions(classifier, train_prior, new_prior, X_test):
    # TODO: complete this function for updating the predictions
    # on X_test using new_prior, an estimate of the after-shift priors.
    # This function should return class predictions, not class probabilities.

    # default return so code runs
    return np.ones(X_test.shape[0])

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    classifier = sk_linear.LogisticRegression(max_iter=500)
    classifier.fit(X_train, y_train)

    pi0 = (y_train == 0).mean()
    pi1 = (y_train == 1).mean()
    train_prior = np.asarray([pi0, pi1])

    y_pred_unadjust = classifier.predict(X_test)
    acc, f1, prec, recall = prediction_metrics(y_test, y_pred_unadjust)
    print(f"Unadjusted LR: Accuracy: {acc:.2f}, F1-score: {f1:.2f}, Precision: {prec:.2f}, Recall: {recall:.2f}")

    EM_prior = EM_adjust_posterior(classifier, X_train, y_train, train_prior, X_val)
    y_pred_EM = update_predictions(classifier, train_prior, EM_prior, X_test)

    acc, f1, prec, recall = prediction_metrics(y_test, y_pred_EM)
    print(f"EM-adjusted LR: Accuracy: {acc:.2f}, F1-score: {f1:.2f}, Precision: {prec:.2f}, Recall: {recall:.2f}")

    test_ML_priors = np.asarray([(y_test==0).mean(), (y_test==1).mean()])
    y_pred_ML = update_predictions(classifier, train_prior, test_ML_priors, X_test) 

    acc, f1, prec, recall = prediction_metrics(y_test, y_pred_ML)
    print(f"CLairvoyant (ML) adjusted LR: Accuracy: {acc:.2f}, F1-score: {f1:.2f}, Precision: {prec:.2f}, Recall: {recall:.2f}")