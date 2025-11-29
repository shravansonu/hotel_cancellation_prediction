from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np

def build_preprocessor(X):
    num=X.select_dtypes(include=np.number).columns.tolist()
    cat=X.select_dtypes(exclude=np.number).columns.tolist()
    pre=ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
    ])
    return pre

def train_models(X_train,y_train,pre):
    models={
        "log_reg":LogisticRegression(max_iter=1000),
        "rf":RandomForestClassifier(),
        "gb":GradientBoostingClassifier()
    }
    trained={}
    for n,m in models.items():
        pipe=Pipeline([('pre',pre),('smote',SMOTE()),('clf',m)])
        pipe.fit(X_train,y_train)
        trained[n]=pipe
    return trained
