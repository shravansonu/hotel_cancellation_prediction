from data_loader import load_data
from preprocess import clean_data, preprocess_data
from feature_engineering import create_features
from train import build_preprocessor, train_models
from evaluate import evaluate_models
from model_save import save_model

def main():
    df=load_data()
    df=clean_data(df)
    df=create_features(df)
    X_train,X_test,y_train,y_test=preprocess_data(df)
    pre=build_preprocessor(X_train)
    models=train_models(X_train,y_train,pre)
    best=evaluate_models(models,X_test,y_test)
    save_model(best)

if __name__=='__main__':
    main()
