import joblib
def save_model(model,path='best_model.pkl'):
    joblib.dump(model,path)
