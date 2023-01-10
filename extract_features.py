import torch
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from skimage import io
from progress.bar import IncrementalBar

from helpers import edit_columns

# colors
black = 0
white = 1
grey = 0.627451

def extract_features_sv(image, tube:bool = True, num_sections:int = 4, tube_sizes:list = [100, 50, 25, 10]):
    dim = image.shape
    num_pixel = int(dim[0] / num_sections)
    tmp = dict()

    if tube:
        list.sort(tube_sizes, reverse=True)
        for i, elem in enumerate(tube_sizes):
            tmp[str(elem)] = dict()

            if i == len(tube_sizes) - 1:
                split = image[int(dim[0]/2 - elem/100/2*dim[0]) : int(dim[0]/2 + elem/100/2*dim[0]),:]
            else:
                bounds_outer = (int(dim[0]/2 - elem/100/2*dim[0]), int(dim[0]/2 + elem/100/2*dim[0]))
                bounds_inner = (int(dim[0]/2 - tube_sizes[i+1]/100/2*dim[0]), int(dim[0]/2 +  tube_sizes[i+1]/100/2*dim[0]))
                split1 = image[bounds_outer[0] : bounds_inner[0],:]
                split2 = image[bounds_inner[1] : bounds_outer[1],:]
                split = np.concatenate((split1, split2), axis=0)
                
            size = split.shape[0] * split.shape[1]

            # buildings
            tmp[str(elem)]["buildings"] = np.count_nonzero(split <= (grey - black) / 2) / size
            
            # terrain
            tmp[str(elem)]["terrain"] = np.count_nonzero(np.logical_and(split > (grey - black) / 2, split <= grey + (white - grey) / 2)) / size

            # nothing
            tmp[str(elem)]["white"] = np.count_nonzero(split > grey + (white - grey) / 2) / size

    else:        
        tmp["horizontal"] = dict()
        tmp["vertical"] = dict()

        # horizontal sections
        for i in range(num_sections):
            tmp["horizontal"][i] = dict()
            if i == num_sections-1:
                split = image[i*num_pixel::,:]
            else:
                split = image[i*num_pixel:(i+1)*num_pixel,:]

            size = split.shape[0] * split.shape[1]

            # buildings
            tmp["horizontal"][i]["buildings"] = np.count_nonzero(split <= (grey - black) / 2) / size
            
            # terrain
            tmp["horizontal"][i]["terrain"] = np.count_nonzero(np.logical_and(split > (grey - black) / 2, split <= grey + (white - grey) / 2)) / size

            # nothing
            tmp["horizontal"][i]["white"] = np.count_nonzero(split > grey + (white - grey) / 2) / size


        # vertical sections
        for i in range(num_sections):
            tmp["vertical"][i] = dict()
            if i == num_sections-1:
                split = image[:,i*num_pixel::]
            else:
                split = image[:,i*num_pixel:(i+1)*num_pixel]

            size = split.shape[0] * split.shape[1]

            # buildings
            tmp["vertical"][i]["buildings"] = np.count_nonzero(split <= (grey - black) / 2) / size
            
            # terrain
            tmp["vertical"][i]["terrain"] = np.count_nonzero(np.logical_and(split > (grey - black) / 2, split <= grey + (white - grey) / 2)) / size

            # nothing
            tmp["vertical"][i]["white"] = np.count_nonzero(split > grey + (white - grey) / 2) / size


    return tmp


def extract_features_tv(image, tube:bool = True, num_sections:int = 4, tube_sizes:list = [100, 50, 25, 10]):
    dim = image.shape
    num_pixel = int(dim[0] / num_sections)
    tmp = dict()

    if tube:
        list.sort(tube_sizes, reverse=True)

        for i, elem in enumerate(tube_sizes):
            tmp[str(elem)] = dict()

            if i == len(tube_sizes) - 1:
                split = image[int(dim[0]/2 - elem/100/2*dim[0]) : int(dim[0]/2 + elem/100/2*dim[0]),:]
            else:
                bounds_outer = (int(dim[0]/2 - elem/100/2*dim[0]), int(dim[0]/2 + elem/100/2*dim[0]))
                bounds_inner = (int(dim[0]/2 - tube_sizes[i+1]/100/2*dim[0]), int(dim[0]/2 +  tube_sizes[i+1]/100/2*dim[0]))
                split1 = image[bounds_outer[0] : bounds_inner[0],:]
                split2 = image[bounds_inner[1] : bounds_outer[1],:]
                split = np.concatenate((split1, split2), axis=0)

            split = image[int(dim[0]/2 - elem/100/2*dim[0]) : int(dim[0]/2 + elem/100/2*dim[0]),:]
            size = split.shape[0] * split.shape[1]

            # buildings
            tmp[str(elem)]["buildings"] = np.count_nonzero(split <= (white - black) / 2) / size
            
            # nothing
            tmp[str(elem)]["white"] = np.count_nonzero(split > (white - black) / 2) / size
    
    else:
        tmp["horizontal"] = dict()
        tmp["vertical"] = dict()

        # horizontal sections
        for i in range(num_sections):
            tmp["horizontal"][i] = dict()
            if i == num_sections-1:
                split = image[i*num_pixel::,:]
            else:
                split = image[i*num_pixel:(i+1)*num_pixel,:]

            size = split.shape[0] * split.shape[1]

            # buildings
            tmp["horizontal"][i]["buildings"] = np.count_nonzero(split <= (white - black) / 2) / size
            
            # nothing
            tmp["horizontal"][i]["white"] = np.count_nonzero(split > (white - black) / 2) / size


        # vertical sections
        for i in range(num_sections):
            tmp["vertical"][i] = dict()
            if i == num_sections-1:
                split = image[:,i*num_pixel::]
            else:
                split = image[:,i*num_pixel:(i+1)*num_pixel]

            size = split.shape[0] * split.shape[1]

            # buildings
            tmp["vertical"][i]["buildings"] = np.count_nonzero(split <= (white - black) / 2) / size
            
            # nothing
            tmp["vertical"][i]["white"] = np.count_nonzero(split > (white - black) / 2) / size

    return tmp


def read_image(image_path:str):
    image = io.imread(image_path)
    return transform_image(image)

def transform_image(image):
    composed = transforms.Compose([transforms.ToPILImage(),  transforms.Grayscale(), transforms.ToTensor()])
    image = image / 255
    tmp = torch.from_numpy(image).float()
    tmp = composed(tmp)
    return np.array(tmp[0])


def init_dict_tube(new_features, tube_sizes):
    for tube_size in tube_sizes:
        new_features["sv_" + str(tube_size) + "_b"] = []
        new_features["sv_" + str(tube_size) + "_t"] = []
        new_features["sv_" + str(tube_size) + "_w"] = []

        new_features["tv_" + str(tube_size) + "_b"] = []
        new_features["tv_" + str(tube_size) + "_w"] = []

    return new_features


def init_dict_split(new_features, num_splits_sv=4, num_splits_tv=5):
    for j in ["h", "v"]:
        for i in range(num_splits_sv):
            new_features["sv_" + j + "_" + str(i) + "_b"] = []
            new_features["sv_" + j + "_" + str(i) + "_t"] = []
            new_features["sv_" + j + "_" + str(i) + "_w"] = []

        for i in range(num_splits_tv):
            new_features["tv_" + j + "_" + str(i) + "_b"] = []
            new_features["tv_" + j + "_" + str(i) + "_w"] = []

    return new_features

def insert_extracted_features_split(new_features, extracted_features, im_type):
    for a in extracted_features.keys():
        key = im_type + "_"
        if a == "vertical":
            key += "v_"
        else:
            key += "h_" 
        
        for b in extracted_features[a].keys():
            _key = key + str(b) + "_"
            for c in extracted_features[a][b].keys():
                new_features[_key + c[0]] = new_features[_key + c[0]] + [extracted_features[a][b][c]]

    return new_features


def generate_feature_dict(target_path:str, tube:bool = True, num_splits:int = 4, tube_sizes:list = [100, 50, 25, 10]) -> dict:
    new_features = dict()
    new_features["id"] = []
    bar = IncrementalBar(target_path, max = len(os.listdir(target_path + "sv/")))

    if tube:
        new_features = init_dict_tube(tube_sizes, new_features)

        for elem in os.listdir(target_path + "sv/"):
            new_features["id"].append(int(elem.split('.')[0]))

            for im_type in ["sv", "tv"]:
                sample = read_image(target_path + im_type + "/" + elem)

                if im_type == "sv":
                    extracted_features = extract_features_sv(sample, True)
                else:
                    extracted_features = extract_features_tv(sample, True)

                for a in extracted_features.keys():
                    key = im_type + "_" + a + "_"
                    
                    for b in extracted_features[a].keys():
                        new_features[key + b[0]].append(extracted_features[a][b])
            
            bar.next()

    else:
        new_features = init_dict_split(new_features)

        for elem in os.listdir(target_path + "sv/"):
            new_features["id"].append(int(elem.split('.')[0]))

            for im_type in ["sv", "tv"]:
                sample = read_image(target_path + im_type + "/" + elem)

                if im_type == "sv":
                    extracted_features = extract_features_sv(sample, False, num_sections=4)
                else:
                    extracted_features = extract_features_tv(sample, False, num_sections=5)

                new_features = insert_extracted_features_split(new_features, extracted_features, im_type)
                            
            bar.next()

    return new_features



if __name__ == "__main__":
    scenarios = ["dortmund", "kopenhagen", "wuppertal", "aarhus"]

    # path where data is stored
    base_path = "../Data/DRaGon/"

    # number of splits
    num_splits = 4


    for scenario in scenarios:
        path = base_path + scenario + "/"

        features = pd.read_pickle(path + scenario + ".pkl")
        features = edit_columns(features)
        features.to_pickle("data/features/" + scenario + "/" + scenario + "_edited.pkl")
        print(len(features))
        target_path = path + "dataset/"

        new_features = generate_feature_dict(target_path, False)
        new_features_df = pd.DataFrame.from_dict(new_features)
        new_features_df.to_pickle("data/features/" + scenario + "/" + scenario + "_extracted_features_split.pkl")


# image_path = "data/sv_test.png"
# sv = read_image(image_path)
# a = extract_features_sv(sv)

# image_path = "data/tv_test.png"
# tv = read_image(image_path)
# b = extract_features_tv(tv)
