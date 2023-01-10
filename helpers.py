from scipy.constants import speed_of_light
import numpy as np
import pandas as pd
import math

# UMa LOS
def UMa_LOS_pathloss(d_2D, d_3D, f_Hz, h_UT=1.5, h_BS=25):
    f_GHz = f_Hz/pow(10,9)
    d_BP = 4 * h_BS * h_UT * f_Hz/speed_of_light
    PL_LOS = np.zeros(len(d_2D))

    for i in range(len(d_2D)):
        if d_2D[i] <= d_BP:
            PL_LOS[i] += 28 + 22*np.log10(d_3D[i]) + 20*np.log10(f_GHz)
        else:
            PL_LOS[i] += 28 + 40*np.log10(d_3D[i]) + 20*np.log10(f_GHz) - 9*np.log10(pow(d_BP, 2) + pow(h_BS - h_UT,2))

    return PL_LOS

# UMa B NLOS
def UMa_B_NLOS_pathloss(d_2D, d_3D, f_Hz, h_UT=1.5, h_BS=30):
    f_GHz = f_Hz/pow(10,9)
    PL_NLOS = 13.54 + 39.08*np.log10(d_3D) + 20*np.log10(f_GHz) - 0.6*(h_UT - 1.5)
    PL_LOS = UMa_LOS_pathloss(d_2D, d_3D, f_Hz, h_UT, h_BS)
    return np.maximum(PL_NLOS, PL_LOS)

# Winner II C2 NLOS (Urban Macro cell)
def Winner2_C2_NLOS_pathloss(d_2D, d_3D, f_Hz, h_UT=1.5, h_BS=30):
    f_GHz = f_Hz / np.power(10,9)
    PL = (44.9 - 6.55*np.log10(h_BS)) * np.log10(d_3D) + 34.46 + 5.83*np.log10(h_BS) + 23*np.log10(f_GHz/5.0)
    return PL

# Friis model (freespace)
def friis_pathloss(d_3D, f_Hz, alpha=2):
    PL = np.power(speed_of_light / (4 * 3.14 * f_Hz), 2) * (1 / np.power(d_3D, alpha))
    return - 10 * np.log10(PL)

# obstacle model
def obstacle_model_pathloss(d_3D, f_Hz, number_IS_indoor, d_OLOS_indoor, number_IS_terrain, d_OLOS_terrain, alpha=2, beta_indoor=2.6, gamma_indoor=0.63, beta_terrain=2.6, gamma_terrain=0.63):
    PL_friis = friis_pathloss(d_3D, f_Hz, alpha)
    PL_obs = number_IS_indoor * beta_indoor + d_OLOS_indoor * gamma_indoor + number_IS_terrain * beta_terrain + d_OLOS_terrain * gamma_terrain    
    return PL_friis + PL_obs

# two-ray ground model
def two_ray_pathloss(d_2D, d_3D, f_Hz, h_UT=1.5, h_BS=30):
    d_BP = 4 * math.pi * h_BS * h_UT * f_Hz/speed_of_light
    PL = np.zeros(np.shape(d_3D))

    c = d_2D < d_BP
    indices = np.argwhere(c)
    PL[indices] = friis_pathloss(d_3D[indices], f_Hz[indices], 2)

    indices = np.argwhere(np.logical_not(c))
    tmp = 40 * np.log10(d_3D[indices]) - 20 * np.log10(h_UT[indices]) - 20 * np.log10(h_BS[indices])
    PL[indices] = tmp

    return PL


def __erlang(k, mean, num):
    tmp = 0
    for i in range(k): #mean, bound = 0
        v = np.random.uniform(size=num)
        r = -mean * np.log(v)
        tmp += r

    return tmp

def nakagami_pathloss(d_3D, f_Hz, m=2, alpha=2):
    PL_friis = friis_pathloss(d_3D, f_Hz, alpha)

    P_W = np.power(10, (40-30) / 10)
    P_dBm = 10 * np.log10(__erlang(2, P_W / 2, len(d_3D))) + 30
    
    return -(P_dBm - 40 - PL_friis)



# RSRP reference basend on bandwidth
def get_rsrp_ref(B_MHz):
    if B_MHz == 1.4:
        RB = 6
    else:
        RB = B_MHz / 0.2

    return 10*np.log10(12*RB)


# compute cartesian distance 
def compute_distance(_lat: float, _lon: float, _ref_lat: float, _ref_lon: float) -> float:    
    pi = 3.141592654
    dlat = (_lat - _ref_lat) / 180 * pi
    dlon = (_lon - _ref_lon) / 180 * pi

    a = pow(math.sin(dlat/2), 2) + math.cos(_lat / 180 * pi) * math.cos(_ref_lat / 180 * pi) * pow(math.sin(dlon/2), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return 6367 * c * 1000



def edit_columns(samples):
    """
    adapt raw dataframe features:
    - delete unnecessary columns
    - calculate path loss
    - calculate delta values
    - calculate target values

    Args:
        samples (pd.DataFrame): samples to adapt

    Returns:
        pd.DataFrame: adapted samples
    """

    B = samples["bandwidth"].to_numpy()
    f = samples["frequency"].to_numpy()
    offset = samples["offset"].to_numpy()
    dist = samples["dist3d"].to_numpy()
    lat_dist = samples["lat_dist"].to_numpy()
    lon_dist = samples["lon_dist"].to_numpy()
    UE_height = samples["height"].to_numpy()

    dist_2d = np.sqrt(np.power(lat_dist, 2) + np.power(lon_dist, 2))
    Ps = np.ones(B.shape) * 40
    L = np.zeros(Ps.shape)
    ref = np.zeros(Ps.shape)

    for i in range(len(Ps)):
        L[i] = UMa_B_NLOS_pathloss([dist_2d[i]], [dist[i]], f[i] * 1e6, h_UT=UE_height[i]) - offset[i]
        ref[i] = Ps[i] - get_rsrp_ref(B[i])

    samples["pathloss"] = pd.Series(ref - L, index=samples.index) # estimation of RSRP using UMa B channel model
    samples["ref"] = pd.Series(ref, index=samples.index)
    samples["dist2d"] = pd.Series(dist_2d, index=samples.index)

    # switch to delta values
    samples["delta_lat"] = pd.Series(np.abs(samples["cell_lat"].to_numpy() - samples["lat"].to_numpy()),
                                        index=samples.index)
    samples["delta_lon"] = pd.Series(np.abs(samples["cell_lon"].to_numpy() - samples["lon"].to_numpy()),
                                        index=samples.index)
    samples["delta_elevation"] = pd.Series((samples["cell_altitude"].to_numpy() - samples["cell_height"].to_numpy()) - (
                samples["altitude"].to_numpy() - samples["height"].to_numpy()), index=samples.index)

    # drop columns
    samples.drop(columns=["lat_dist", "lon_dist", "lat", "cell_lat", "lon", "cell_lon", "cell_altitude", "altitude"],
                    inplace=True)

    return samples