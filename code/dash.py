import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_squared_log_error, make_scorer
import shap
import sklearn
import joblib
import uuid
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from vader_sentiment.vader_sentiment import SentimentIntensityAnalyzer
from collections import OrderedDict
import sys
import os
from datetime import datetime




def preprocess(train):
    names = train["track_name"].values
    scores = get_sent(names)
    train["track_name_sent_score"] = scores
    le = LabelEncoder()
    cols_for_encode = ["track_genre","explicit"]
    train[cols_for_encode] = train[cols_for_encode].apply(le.fit_transform)
    train.drop(columns=["track_name", "artists", "track_id"], inplace=True)
    return train


def get_split(train):
    target = "popularity"
    ft = train.columns.tolist()
    ft.remove(target)
    print(ft, len(ft))
    x_train, x_test, y_train, y_test = train_test_split(train[ft],train[target], test_size=0.2,random_state= 10)
    print(x_train.shape)
    print(y_train.shape)
    return x_train,x_test, y_train, y_test, ft


def get_sent(list_of_sents):
    analyzer = SentimentIntensityAnalyzer()
    list_of_sent = []
    for sentence in list_of_sents:
        vs = analyzer.polarity_scores(sentence)
        list_of_sent.append(vs["compound"])
    return list_of_sent


def get_grid_params(hp_list, hp):
    default = {"reg__n_estimators": 100}
    req_hp = {"reg__" + str(k):v for k,v in hp.items() if k in hp_list}
    if len(req_hp) < 1:
        return default
    return req_hp

def get_error(y_true, y_pred):
    err = np.sqrt(mean_squared_error(y_true, y_pred))
    return err


def get_model(hp_list,flag="random"):
    my_scorer = make_scorer(get_error, greater_is_better=False)
    pipe_rf = Pipeline([('rs', RobustScaler()),
        ('reg', RandomForestRegressor())])
    
    hp = {'bootstrap': [True, False], 'ccp_alpha':[0.0,2,0.5], 'criterion': ["squared_error", "absolute_error", "friedman_mse", "poisson"], 'max_depth':[1,10,2],
      'max_features': ["sqrt", "log2", None], 'max_leaf_nodes': None, 
      'max_samples':[0.0, 1, 0.05], 'min_impurity_decrease':0.0, 'min_samples_leaf': 1, 'min_samples_split': [2,4,1], 'min_weight_fraction_leaf':0.0, 
      'n_estimators':[50], 'n_jobs':[-1,1], "random_state":10}
    grid_param = get_grid_params(hp_list,hp)
    print(f"Hyer Parameters used are {grid_param}")
    if flag == "random":
        gs_rf = (GridSearchCV(estimator=pipe_rf, param_grid=grid_param, 
                          cv=2,
                          scoring = my_scorer,
                          n_jobs = 1,
                          verbose=4))
    elif flag == "grid":
        gs_rf = (RandomizedSearchCV(estimator=pipe_rf, param_distributions=grid_param, 
                              cv=2,
                              scoring = my_scorer,
                              n_jobs = 1,  
                              verbose=4))
    return gs_rf


def save_model(model, m_path):
    joblib.dump(model,m_path)
    print(f"Model saved at {m_path}")

def gen_uuid():
    x = uuid.uuid4()
    return str(x)

def get_shap_features(model, features, sample, rec_index, path, flag="bar"):
    ex_tree = shap.TreeExplainer(model.best_estimator_[1])
    shap_values = ex_tree.shap_values(sample)
    if flag == "summary":
        plt.clf()
        shap.summary_plot(shap_values, sample,feature_names=list(features), plot_type='bar', show=False)
        plt.savefig(path + "shap_summary_bar.png", dpi='figure')
    elif flag == "force":
        plt.clf()
        shap.force_plot(ex_tree.expected_value[0], shap_values[rec_index,:], sample.iloc[rec_index,:],matplotlib=True, show=False)
        plt.savefig(path + "shap_force.png",dpi=150, bbox_inches='tight')
    elif flag == "both":
        plt.clf()
        shap.summary_plot(shap_values, sample,feature_names=list(features), plot_type='bar', show=False)
        plt.savefig(path + "shap_summary_bar.png", dpi='figure')
        plt.clf()
        shap.force_plot(ex_tree.expected_value[0], shap_values[rec_index,:], sample.iloc[rec_index,:],matplotlib=True, show=False)
        plt.savefig(path + "shap_force.png",dpi=150, bbox_inches='tight')


def training(train, hp_list, path, flag="random",save=False, shap=False, shap_params=None): 
    start = datetime.now()
    model = get_model(hp_list, flag="random")
    x_train, x_test, y_train, y_test, features = get_split(train)
    print(f"Training started on {x_train.shape} samples")
    model.fit(x_train, y_train)
    end0 = datetime.now()
    time1 = end0-start
    print(f"Time taken to complete the training {time1}")
    eval_metric = -model.score(x_test, y_test)
    # Scoring model in terms of mean_squared_log_error(Note it comes negative due to inbuilt gridsearchcv's minimization procedure for greater_is_better False flag)
    # For detail check the link (https://stackoverflow.com/questions/21050110/sklearn-gridsearchcv-with-pipeline)
    print(f"root_mean_squared_error is {eval_metric}")
    print('Five samples of actual target',y_test[:5])
    y_pred_svr = model.predict(x_test[:5])
    print('Five samples of prediction',y_pred_svr)
    # 
    if save:
        m_p = "../models/model.pkl"
        save_model(model, m_p)
    if shap:
        sample = x_train[:500]
        rec_index = shap_params["fi"]
        shap_flag = shap_params["type"]
        get_shap_features(model, features, sample, rec_index, path, flag=shap_flag)
        end = datetime.now()
        time = end - start
        print(f"Time taken to complete the training and shap is {time}")
    return model, eval_metric, path, features


    
def get_inference(model, test_feature_val, feature_list):
    new_dict = OrderedDict((k, test_feature_val.get(k)) for k in feature_list)
    arr = np.array(list(new_dict.values()))
    arr = arr.reshape(1, -1)
    preds = model.predict(arr)
    return preds


def all_plots(train_pr, target, path):
    ft = train_pr.columns.tolist()
    ft.remove(target)
    corr(train_pr, path)
    get_scatter_all(train_pr, target, ft, path)
    get_dist_plot(train_pr, target, path)
    get_prob_plot(train_pr, target, path)
    


def get_prob_plot(train, target, path):
    p_ = os.path.join(path, "prob.png")
    fig1 = plt.figure()
    res = stats.probplot(train[target], plot=plt)
    fig1.savefig(p_)


def get_dist_plot(train, target, path):
    fig2 = plt.figure()
    sns.distplot(train[target], fit=norm)
    p_ = os.path.join(path, "dist.png")
    fig2.savefig(p_)
    

def get_scatter_all(train, target, ft, path):
    
    for idx, var in enumerate(ft):
        data = pd.concat([train[target], train[var]], axis=1)
        # fig = plt.figure()
        data.plot.scatter(x=var, y=target, ylim=(0,150))
        var_ext = str(idx) + ".png"
        new_path = os.path.join(path, var_ext)
        plt.savefig(new_path)


def corr(train, path):
    new_path = os.path.join(path, "corr.png")
    corr = train.corr()
    fig7 = plt.figure(figsize=(12, 10))
    plt.grid(False)
    sns.heatmap(corr[(corr >= 0.4) | (corr <= -0.4)], 
            cmap='crest', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8});
    plt.xticks(minor=True);
    #Apply yticks
    plt.yticks(minor=True)
    #show plot
    fig7.savefig(new_path)





def create_output(path):
    try: 
        os.mkdir(path) 
    except OSError as error: 
        print(error)






'''
['bootstrap','ccp_alpha','criterion',
 'max_depth','max_features','max_samples',
 'min_samples_split', 'n_estimators','n_jobs']
'''


if __name__ == "__main__":
    inference = True
    inf_input = {'duration_ms': 291000.0,'explicit': 0.0,'danceability': 0.419,
             'energy': 0.341,'key': 11.0,'loudness': -13.72,'mode': 0.0,'speechiness': 0.0317,
             'acousticness': 0.322, 'instrumentalness': 0.914,'liveness': 0.0786,'valence': 0.179,'tempo': 152.958,
             'time_signature': 3.0,'track_genre': 45.0,'track_name_sent_score': 0.0}
    train = pd.read_csv("../data/input/final.csv")
    target = "popularity"
    model_id = gen_uuid()
    path = os.path.join("../data/output/" ,model_id)
    path = path + "/"
    hp_list = ["bootstrap"]

    create_output(path)
    train_after_preproc = preprocess(train)
    all_plots(train_after_preproc, target, path)
    model, eval_metric, path, features = training(train_after_preproc, hp_list,path, save=True, shap=True ,shap_params={"type":"both", "fi":0})
    if inference:
        pred = get_inference(model, inf_input, features)
        print(pred)

    