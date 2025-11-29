from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

def evaluate_models(models,X_test,y_test):
    best=None;best_f1=0
    for name,model in models.items():
        p=model.predict(X_test)
        f=f1_score(y_test,p)
        if f>best_f1:
            best_f1=f;best=model
    return best
