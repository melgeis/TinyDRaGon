from numpy import average
from generator import Generator
import pandas as pd
import time
import os

if __name__ == "__main__":
    scenarios = ["kopenhagen", "wuppertal", "dortmund", "aarhus"]

    # path where data is stored
    base_path = "../Data/DRaGon/"

    # used range for image samples
    size = int(300)

    dir_existed = True

    for scenario in scenarios:
        
        path = base_path + scenario + "/"

        if not os.path.isdir(path):
            dir_existed = False
            os.makedirs(path)
            os.makedirs(path + "dataset/", exist_ok=True)
            os.makedirs(path + "dataset/sv/", exist_ok=True)
            os.makedirs(path + "dataset/tv/", exist_ok=True)

    if not dir_existed:
        print("add files and restart script")
        exit()
    
    save_images = True
    generator = Generator()

    for i in range(0,10):
        for scenario in scenarios:
            path = base_path + scenario + "/"

            osm = path + scenario + "_height.osm"
            elev = path + "hp_" + scenario + "_converted.csv"
            features = pd.read_pickle(path + scenario + ".pkl")
            print(len(features))
            target_path = path + "dataset/"

            generator.run(osm, elev, features, target_path, save_images)
