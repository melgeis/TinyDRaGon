from numpy import average
from generator import Generator
import pandas as pd
import time
import os

if __name__ == "__main__":
    scenarios = ["kopenhagen", "wuppertal", "dortmund", "aarhus"]

    # path where data is stored
    base_path = "/home/geis/Documents/Data/DRaGon/"#"/mnt/e/geis/Documents/Data/DRaGon/"

    # used range for image samples
    size = int(300)

    dir_existed = True

    for scenario in scenarios:
        
        path = base_path + scenario + "/"

        if not os.path.isdir(path):
            dir_existed = False
            os.makedirs(path)

        if scenario == "aarhus":
            for area in ["August_10m", "August_15m", "August_20m", "August_25m", "August_30m", "August_35m", "August_40m", "October_15m", "October_40m", "October_100m", "October_DT"]:
                tmp_path = path + area + "/"
                if not os.path.isdir(tmp_path):
                    dir_existed = False
                    os.makedirs(tmp_path)
                    os.makedirs(tmp_path + "dataset/")
                    os.makedirs(tmp_path + "dataset/sv/")
                    os.makedirs(tmp_path + "dataset/tv/")
        elif scenario == "dortmund":
            for area in ["campus", "highway", "suburban", "urban"]:
                tmp_path = path + area + "/"
                if not os.path.isdir(tmp_path):
                    dir_existed = False
                    os.makedirs(tmp_path)
                    os.makedirs(tmp_path + "dataset/")
                    os.makedirs(tmp_path + "dataset/sv/")
                    os.makedirs(tmp_path + "dataset/tv/")

        else:
            os.makedirs(path + "dataset/", exist_ok=True)
            os.makedirs(path + "dataset/sv/", exist_ok=True)
            os.makedirs(path + "dataset/tv/", exist_ok=True)

    if not dir_existed:
        print("add files and restart script")
        exit()
    
    scenarios = ["kopenhagen", "wuppertal", "dortmund", "aarhus"]
    save_images = True

    # path where data is stored
    base_path = "/home/geis/Documents/Data/DRaGon/"#'/mnt/e/geis/Documents/Data/DRaGon/'
    file_object = open('time_measure.txt', 'a')

    # used range for image samples
    size = int(300)

    generator = Generator()

    for i in range(0,10):
        for scenario in scenarios:
            path = base_path + scenario + "/"

            if scenario == "aarhus":
                osm = path + scenario + "_height.osm"
                elev = path + "hp_" + scenario + "_converted.csv"
                for area in ["August_10m", "August_15m", "August_20m", "August_25m", "August_30m", "August_35m", "August_40m", "October_15m", "October_40m", "October_100m", "October_DT"]:
                    print(area)
                    features = pd.read_pickle(path + area + "/" + area + "_offset_per_cell_REM_5.pkl")
                    print(len(features))
                    target_path = path + area + "/dataset/"
                    
                    start = time.time()
                    generator.run(osm, elev, features, target_path, save_images, file_object)
                    end = time.time()
                    file_object.write("finished aarhus " + area)
                    print("finished aarhus " + area)
                    file_object.write("total time")
                    print("total time")
                    file_object.write(str(end-start))
                    print(end - start)

            elif scenario == "dortmund":
                for area in ["campus", "highway", "suburban", "urban"]:
                    osm = path + area + "/" + area + "_height.osm"
                    elev = path + area + "/" + "hp_" + area + "_converted.csv"
                    features = pd.read_pickle(path + area + "/" + area + "_offset_per_cell_REM_5.pkl")
                    print(len(features))
                    target_path = path + area + "/dataset/"

                    start = time.time()
                    generator.run(osm, elev, features, target_path, save_images, file_object)
                    end = time.time()
                    file_object.write("finished dortmund " + area)
                    print("finished dortmund " + area)
                    file_object.write("total time")
                    print("total time")
                    file_object.write(str(end-start))
                    print(end - start)

            elif scenario == "wuppertal":
                osm = path + scenario + ".osm"
                elev = path + "hp_" + scenario + "_converted.csv"
                features = pd.read_pickle(path + scenario + "_offset_per_cell_REM_5.pkl")
                print(len(features))
                target_path = path + "dataset/"

                start = time.time()
                generator.run(osm, elev, features, target_path, save_images, file_object)
                end = time.time()
                file_object.write("finished wuppertal \n")
                print("finished wuppertal ")
                file_object.write("total time \n")
                print("total time")
                file_object.write(str(end-start))
                print(end - start)

            else:
                osm = path + scenario + "_height.osm"
                elev = path + "hp_" + scenario + "_converted.csv"
                features = pd.read_pickle(path + scenario + "_offset_per_cell_REM_5.pkl")
                print(len(features))
                target_path = path + "dataset/"

                start = time.time()
                generator.run(osm, elev, features, target_path, save_images, file_object)
                end = time.time()
                file_object.write("finished copenhagen \n")
                print("finished copenhagen ")
                file_object.write("total time \n")
                print("total time")
                file_object.write(str(end-start))
                print(end - start)