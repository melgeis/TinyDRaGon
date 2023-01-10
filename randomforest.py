from types import SimpleNamespace
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import pandas as pd
import time
import pickle
from datetime import datetime
import math
import os
import joblib

class TinyDRaGonRF():
    def __init__(self, num_workers=1) -> None:
        self.rf = None
        self.num_workers = num_workers


    def train(self, trees, max_depth, X_train, y_train):
        # initalize random forest
        rf = RandomForestRegressor(n_estimators=trees, max_depth=max_depth, n_jobs=self.num_workers)
        
        # train random forest
        start = time.time()
        rf.fit(X_train, y_train)
        end = time.time()

        duration = end - start
        print("duration: " + str(duration))

        self.rf = rf
        self.trees = trees


    def save_model(self, file_name, compression=None):
        if self.rf:
            # store trained model
            start = time.time()
            if compression is None:
                pickle.dump(self.rf, open(file_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                joblib.dump(self.rf, file_name, compress=3)
            end = time.time()
            store_duration = end - start
            print("storing model: " + str(store_duration))
            model_size_MB = os.path.getsize(file_name) / 1e6
            print("model_size: " + str(model_size_MB))
        else:
            raise Exception("no model available")


    def load_model(self, file_name):
        # load trained model
        self.rf = pickle.load(open("model.pkl", 'rb'))
        #rf = joblib.load("model.pkl")


    def predict(self, X_test):
        # predict y 
        y_pred = self.rf.predict(X_test)
        return y_pred

    def feature_importance(self, header):
        importances = self.rf.feature_importances_
        d = {c: list(importances)[i] for i, c in enumerate(list(header))}

        x0 = ["frequency", "bandwidth", "ref"]
        x1 = ["dist3d", "cell_height", "height", "delta_elevation", "delta_lon", "delta_lat"]
        x2 = ["indoorIntersections", "indoorDist", "terrainIntersections", "terrainDist"]
        x3 = ["sv_h_0_b", "sv_h_1_b", "sv_h_2_b", "sv_h_3_b", "sv_h_0_t", "sv_h_1_t", "sv_h_2_t", "sv_h_3_t", "sv_v_0_b", "sv_v_1_b", "sv_v_2_b", "sv_v_3_b", "sv_v_0_t", "sv_v_1_t", "sv_v_2_t", "sv_v_3_t"]
        x4 = ["tv_h_0_b", "tv_h_1_b", "tv_h_2_b", "tv_h_3_b", "tv_h_4_b"]
        
        x_com = 0
        x_geo = 0
        x_env = 0
        x_side = 0
        x_top = 0

        for i, elem in enumerate(list(header)):
            # print(elem + ": " + str(list(importances)[i]))

            if elem in x0:
                # x com
                x_com += list(importances)[i]
            elif elem in x1:
                # x geo
                x_geo += list(importances)[i]
            elif elem in x2:
                # x env
                x_env += list(importances)[i]
            elif elem in x3:
                # x_side
                x_side += list(importances)[i]
            elif elem in x4:
                # x_top
                x_top += list(importances)[i]

        print("x_com: " + str(x_com))
        print("x_geo: " + str(x_geo))
        print("x_env: " + str(x_env))
        print("x_side: " + str(x_side))
        print("x_top: " + str(x_top))





def hyperparameters():
    args = SimpleNamespace()
    args.trees = 80
    args.max_depth = 23


if __name__ == "__main__":
    # load training data
    data = pd.read_csv("resources/dortmund__train.csv")
    X = data[data.columns[1::]].to_numpy()
    y = data[data.columns[0]].to_numpy()

    # load test data
    test = pd.read_csv("resources/dortmund__test.csv")
    X_test = test[test.columns[1::]].to_numpy()
    y_test = test[test.columns[0]].to_numpy()

    num_workers = int(128*0.9)
    num_trees = 80 #80
    max_depth = 22 #23
    rf_model = TinyDRaGonRF(num_workers)
    rf_model.train(num_trees, max_depth, X,y)
    rf_model.save_model("test_model.pkl")
    
    rf_model.load_model("test_model.pkl")
    rf_model.feature_importance(test.columns[1::])
    y_pred = rf_model.predict(X_test)
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
    print("test RMSE: " + str(RMSE))
