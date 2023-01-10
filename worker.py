
import pylimosim
import pickle
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import numpy as np
import extract_features

limosim_instance = pylimosim.PyLIMoSim()


class Worker():

    def __init__(self) -> None:
        pass


    def process(self, index, row, target_path, min_lat, min_lon, max_lat, max_lon, save_images, limosim_instance):
        if not save_images:
            return_value = dict()
            return_value["id"] = []
            return_value = extract_features.init_dict_split(return_value)
        
        lat = row["lat"]
        lon = row["lon"]
        altitude = row["altitude"]
        cell_lat = row["cell_lat"]
        cell_lon = row["cell_lon"]
        cell_altitude = row["cell_altitude"]
        
        # generate raw image data using limosim
        datasample = limosim_instance.generate_imagesample(lat, lon, altitude, cell_lat, cell_lon, cell_altitude, 300, min_lat, min_lon, max_lat, max_lon)
        
        # save sv image sample
        building_data = list(datasample.getSvData().getBuildingData())
        elevation_data = list(datasample.getSvData().getElevationData())

        image = Image.new("L", (250, 250), color = 255)
        draw = ImageDraw.Draw(image)
        try:
            draw.polygon((building_data), fill = "black")
        except TypeError:
            pass
        try:
            draw.polygon((elevation_data), fill = "#a0a0a4")
        except TypeError:
            pass
        image = image.resize((64,64))
        image = ImageOps.flip(image)
        
        if save_images:
            image.save(target_path + "sv/" + str(row["id"]) + ".png")
        else:
            image = np.array(image)
            image = extract_features.transform_image(image)
            tmp = [None] * len(return_value.keys())
            tmp[0] = int(str(row["id"]).split('.')[0])
            extracted_features = extract_features.extract_features_sv(image, False, num_sections=4)
            tmp_dict = extract_features.insert_extracted_features_split(return_value, extracted_features, "sv")
            for i, key in enumerate(return_value.keys()):
                if key == "id":
                    continue
                if tmp_dict[key] != []:
                    tmp[i] = tmp_dict[key][0]


        # save tv image sample
        tv_data = list(datasample.getTvData())

        image = Image.new("L", (250, 250), color = 255)
        draw = ImageDraw.Draw(image)

        for building in tv_data:
            color = building.getColor()
            if color == 'b':
                color = 'black'
            else:
                color = 'white'

            data = list(building.getBuildingData())
            draw.polygon((data), fill = color)
        
        image = image.resize((64,64))
        image = ImageOps.flip(image)
        
        if save_images:
            image.save(target_path + "tv/" + str(row["id"]) + ".png")
        else:
            image = np.array(image)
            image = extract_features.transform_image(image)
            extracted_features = extract_features.extract_features_tv(image, False, num_sections=5)
            tmp_dict = extract_features.insert_extracted_features_split(return_value, extracted_features, "tv")
            for i, key in enumerate(return_value.keys()):
                if key == "id":
                    continue
                if tmp_dict[key] != []:
                    tmp[i] = tmp_dict[key][0]
            
            return tmp


    def process2(self, index, row, target_path, min_lat, min_lon, max_lat, max_lon, save_images, limosim_instance):
        lat = row["lat"]
        lon = row["lon"]
        altitude = row["altitude"]
        cell_lat = row["cell_lat"]
        cell_lon = row["cell_lon"]
        cell_altitude = row["cell_altitude"]
        
        # generate raw image data using limosim
        datasample = limosim_instance.generate_imagesample(lat, lon, altitude, cell_lat, cell_lon, cell_altitude, 300, min_lat, min_lon, max_lat, max_lon)
        
        # save sv image sample
        building_data = list(datasample.getSvData().getBuildingData())
        elevation_data = list(datasample.getSvData().getElevationData())

        image = Image.new("L", (250, 250), color = 255)
        draw = ImageDraw.Draw(image)
        try:
            draw.polygon((building_data), fill = "black")
        except TypeError:
            pass
        try:
            draw.polygon((elevation_data), fill = "#a0a0a4")
        except TypeError:
            pass
        image = image.resize((64,64))
        image = ImageOps.flip(image)
        
        if save_images:
            image.save(target_path + "sv/" + str(row["id"]) + ".png")


        # save tv image sample
        tv_data = list(datasample.getTvData())

        image = Image.new("L", (250, 250), color = 255)
        draw = ImageDraw.Draw(image)

        for building in tv_data:
            color = building.getColor()
            if color == 'b':
                color = 'black'
            else:
                color = 'white'

            data = list(building.getBuildingData())
            draw.polygon((data), fill = color)
        
        image = image.resize((64,64))
        image = ImageOps.flip(image)
        
        if save_images:
            image.save(target_path + "tv/" + str(row["id"]) + ".png")
