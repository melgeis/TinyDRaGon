import pylimosim
import multiprocessing
import xml.etree.ElementTree as ET
from worker import Worker
import pandas as pd
import time
import extract_features
from types import SimpleNamespace

limosim_instance = pylimosim.PyLIMoSim()

class Generator():

    def __init__(self) -> None:
        pass

    def run(self, osm:str, elev:str, features:pd.DataFrame, target_path:str, save_images):
        self.worker_instance = Worker() 
        self.generate_datasamples(osm, elev, features, target_path, save_images)


    def generate_datasamples(self, osm:str, elev:str, features:pd.DataFrame, target_path:str, save_images, file_object=None):
        """
        generates datasample images based on existing feature vectors

        Args:
            osm (str): path to open street map file (*.osm)
            elev (str): path to elevation profile matrix (*.csv)
            features (pd.DataFrame): pandas dataframe holding the data samples' features
            target_path (str): path where the image samples are stored
        """
        start = time.time()
        # get scenario boundaries
        root = ET.parse(osm).getroot()
        for child in root:
            if child.tag == 'bounds':
                bounds = child.attrib
                min_lat = float(bounds["minlat"])
                min_lon = float(bounds["minlon"])
                max_lat = float(bounds["maxlat"])
                max_lon = float(bounds["maxlon"])
                break
        del root

        end = time.time()
        print("got boundaries in")
        file_object.write("got boundaries in \n")
        file_object.write(str(end-start))
        print(end - start)


        # load environmental model
        start = time.time()
        limosim_instance.load_3d_model(osm, elev, 15, 3)
        end = time.time()
        print("load 3d environment in")
        file_object.write("load 3d environment in \n")
        file_object.write(str(end-start))
        print(end - start)

        num_workers = 100

        task_list = []
        i = 0
        for index, row in features.iterrows():
            task_list.append((i, row, target_path, min_lat, min_lon, max_lat, max_lon, save_images))
            i += 1
            
        num_workers = 120
        manager = multiprocessing.Manager()

        data_tmp = []

        with multiprocessing.Pool(processes=num_workers) as pool:
            if not save_images:
                for i in pool.imap_unordered(self.worker_function, task_list):
                    data_tmp.append(i)
            else:
                pool.imap_unordered(self.worker_function2, task_list)

            pool.close()
            pool.join()


        if not save_images:
            return_value = dict()
            return_value["id"] = []
            return_value = extract_features.init_dict_split(return_value)
            feature_df = pd.DataFrame(data_tmp, columns = return_value.keys())
            feature_df.to_pickle(target_path + "extracted_features_split.pkl")


    def worker_function(self, elem):
        return self.worker_instance.process(elem[0], elem[1], elem[2], elem[3], elem[4], elem[5], elem[6], elem[7], limosim_instance)

    def worker_function2(self, elem):
        self.worker_instance.process2(elem[0], elem[1], elem[2], elem[3], elem[4], elem[5], elem[6], elem[7], limosim_instance)