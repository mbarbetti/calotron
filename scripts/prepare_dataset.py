import pickle
from glob import glob
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from tqdm import tqdm
from utils_argparser import argparser_preprocessing

ECAL_W = 8000
ECAL_H = 6500
MAX_OVX = 50
MAX_OVY = 50
MAX_OVZ = 150
MAX_MATCH_DISTANCE = 0.01

PADDING_VALUE = 0.0
MAX_INPUT_PHOTONS = 96
MAX_OUTPUT_CLUSTERS = 96
MAX_INPUT_PHOTONS_DEMO = 32
MAX_OUTPUT_CLUSTERS_DEMO = 32


# +------------------+
# |   Parser setup   |
# +------------------+

parser = argparser_preprocessing(description="dataset preparation setup")
args = parser.parse_args()

# +-------------------+
# |   Initial setup   |
# +-------------------+

if "*" in args.filename:
    data_fnames = np.array(glob(args.filename))
else:
    data_fnames = np.array([args.filename])

max_files = int(args.max_files)
chunk_size = int(args.chunk_size)

indices = np.random.permutation(len(data_fnames))
data_fnames = data_fnames[indices][:max_files]

with open("config/directories.yml") as file:
    config_dir = yaml.full_load(file)

export_data_dir = config_dir["data_dir"]
images_dir = config_dir["images_dir"]
models_dir = config_dir["models_dir"]

# +------------------+
# |   Data loading   |
# +------------------+

start = time()
photon_list = list()
cluster_list = list()

for fname in data_fnames:
    with uproot.open(fname) as file:
        photon_list.append(
            file["CaloTupler/calo_true"]
            .arrays(library="pd")
            .query("pz > 750 and abs(ovx) < 50 and abs(ovy) < 50 and abs(ovz) < 150")
        )
        cluster_list.append(
            file["CaloTupler/neutral_protos"].arrays(library="pd").query("E > 1500")
        )

print(f"[INFO] Data correctly loaded in {time()-start:.2f} s")

photon_df = pd.concat(photon_list, ignore_index=True).dropna()
photon_df = shuffle(photon_df).reset_index(drop=True)[:chunk_size]
print(f"[INFO] DataFrame of {len(photon_df)} generated photons correctly created")

cluster_df = pd.concat(cluster_list, ignore_index=True).dropna()
cluster_df = shuffle(cluster_df).reset_index(drop=True)[:chunk_size]
print(
    f"[INFO] DataFrame of {len(cluster_df)} reconstructed calo-clusters correctly created"
)

# +----------------------+
# |   Photon DataFrame   |
# +----------------------+

photon_df["x"] = photon_df["ecal_x"]
photon_df["y"] = photon_df["ecal_y"]
photon_df["p"] = np.linalg.norm(photon_df[["px", "py", "pz"]], axis=1)
photon_df["E"] = photon_df["p"]
photon_df["logE"] = np.log(photon_df.E)
photon_df["tx"] = photon_df.px / photon_df.pz
photon_df["ty"] = photon_df.py / photon_df.pz
photon_df["NotPadding"] = 1.0

true_vars = ["x", "y", "logE", "tx", "ty", "ovx", "ovy", "ovz"]
true_vars += ["NotPadding"]

if args.verbose:
    print(photon_df[true_vars].describe())

# +---------------------------+
# |   Photons preprocessing   |
# +---------------------------+

p_photon_df = photon_df.copy()

photon_scaler_logE = MinMaxScaler()

start = time()
p_photon_df["x"] = p_photon_df.x / ECAL_W * 2
p_photon_df["y"] = p_photon_df.y / ECAL_H * 2
p_photon_df["logE"] = photon_scaler_logE.fit_transform(np.c_[p_photon_df.logE])
p_photon_df["ovx"] = p_photon_df.ovx / MAX_OVX
p_photon_df["ovy"] = p_photon_df.ovy / MAX_OVY
p_photon_df["ovz"] = p_photon_df.ovz / MAX_OVZ
print(f"[INFO] Generated photons preprocessing completed in {time()-start:.2f} s")

if args.verbose:
    print(p_photon_df[true_vars].describe())

# +-----------------------+
# |   Cluster DataFrame   |
# +-----------------------+

cluster_df["logE"] = np.log(cluster_df.E)
cluster_df["NotPadding"] = 1.0

reco_vars = ["x", "y", "logE"]

if not args.demo:
    bool_vars = ["PhotonFromMergedPi0", "Pi0Merged", "Photon"]
    pid_vars = ["PhotonID", "IsNotE", "IsNotH"]
    reco_vars += bool_vars + pid_vars
reco_vars += ["NotPadding"]

if args.verbose:
    print(cluster_df[reco_vars].describe())

# +----------------------------+
# |   Clusters preprocessing   |
# +----------------------------+

p_cluster_df = cluster_df.copy()

cluster_scaler_logE = MinMaxScaler()
cluster_scaler_pid = StandardScaler()

start = time()
p_cluster_df["x"] = p_cluster_df.x / ECAL_W * 2
p_cluster_df["y"] = p_cluster_df.y / ECAL_H * 2
p_cluster_df["logE"] = cluster_scaler_logE.fit_transform(np.c_[p_cluster_df.logE])
if not args.demo:
    p_cluster_df[pid_vars] = cluster_scaler_pid.fit_transform(
        cluster_df[pid_vars].values
    )
print(
    f"[INFO] Reconstructed calo-clusters preprocessing completed in {time()-start:.2f} s"
)

if args.verbose:
    print(p_cluster_df[reco_vars].describe())

# +--------------------------------+
# |   Data processing per events   |
# +--------------------------------+

true_photons = list()
reco_clusters = list()

events = np.unique(p_cluster_df.evtNumber)
nEvents = len(events)

for iEvent, event in tqdm(enumerate(events), total=nEvents, desc="Processing events"):
    true = (
        p_photon_df[p_photon_df.evtNumber == event]
        .query("mcID == 22")  # photons
        .sort_values("p", ascending=False)
    )
    reco = p_cluster_df[p_cluster_df.evtNumber == event].sort_values(
        "E", ascending=False
    )
    true_photons.append(true[true_vars].values)
    reco_clusters.append(reco[reco_vars].values)

# +-------------------+
# |   Event example   |
# +-------------------+

event_number = 42
photon = np.array(true_photons[event_number])
cluster = np.array(reco_clusters[event_number])

plt.figure(figsize=(8, 6), dpi=300)
plt.xlabel("Preprocessed $x$-coordinate [a.u.]", fontsize=12)
plt.ylabel("Preprocessed $y$-coordinate [a.u.]", fontsize=12)
plt.scatter(
    photon[:, 0],
    photon[:, 1],
    s=50.0 * photon[:, 2] / cluster[:, 2].max(),
    marker="o",
    facecolors="none",
    edgecolors="#d7191c",
    lw=0.75,
    label="Generated photon",
)
plt.scatter(
    cluster[:, 0],
    cluster[:, 1],
    s=50.0 * cluster[:, 2] / cluster[:, 2].max(),
    marker="s",
    facecolors="none",
    edgecolors="#2b83ba",
    lw=0.75,
    label="Reconstructed cluster",
)
plt.legend(loc="upper left", fontsize=10)
plt.savefig(fname=f"{images_dir}/evt-example.png")
plt.close()

# +----------------------------------+
# |   Event multiplicity histogram   |
# +----------------------------------+

plt.figure(figsize=(8, 5), dpi=300)
plt.xlabel("Event multipilcity", fontsize=12)
plt.ylabel("Number of events", fontsize=12)
bins = np.linspace(0, 150, 51)
plt.hist(
    np.array([len(photon) for photon in true_photons]),
    bins=bins,
    color="#3288bd",
    label="Generated photons",
)
plt.hist(
    np.array([len(cluster) for cluster in reco_clusters]),
    bins=bins,
    histtype="step",
    color="#fc8d59",
    lw=2,
    label="Reconstructed clusters",
)
plt.legend(loc="upper right", fontsize=10)
plt.savefig(fname=f"{images_dir}/evt-multi-hist.png")
plt.close()

# +------------------------------+
# |   Training set preparation   |
# +------------------------------+

max_input_photons = MAX_INPUT_PHOTONS_DEMO if args.demo else MAX_INPUT_PHOTONS
max_output_clusters = MAX_OUTPUT_CLUSTERS_DEMO if args.demo else MAX_OUTPUT_CLUSTERS

pad_photons = PADDING_VALUE * np.ones(
    shape=(nEvents, max_input_photons, len(true_vars))
)
pad_clusters = PADDING_VALUE * np.ones(
    shape=(nEvents, max_output_clusters, len(reco_vars))
)

for iRow, photon in enumerate(true_photons):
    photons_trunkated = photon[:max_input_photons]
    pad_photons[iRow, : len(photons_trunkated)] = photons_trunkated

for iRow, cluster in enumerate(reco_clusters):
    clusters_trunkated = cluster[:max_output_clusters]
    pad_clusters[iRow, : len(clusters_trunkated)] = clusters_trunkated

# +--------------------------+
# |   Calorimeter deposits   |
# +--------------------------+

plt.figure(figsize=(18, 5), dpi=300)

plt.subplot(1, 2, 1)
plt.title("Generated photons", fontsize=14)
plt.xlabel("Preprocessed $x$-coordinate [a.u.]", fontsize=12)
plt.ylabel("Preprocessed $y$-coordinate [a.u.]", fontsize=12)
x_bins = np.linspace(-0.4, 0.4, 41)
y_bins = np.linspace(-0.4, 0.4, 41)
plt.hist2d(
    pad_photons[:, :, 0].flatten(),
    pad_photons[:, :, 1].flatten(),
    weights=pad_photons[:, :, 2].flatten(),
    bins=(x_bins, y_bins),
    cmin=0,
    cmap="magma",
)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Reconstructed calo-clusters", fontsize=14)
plt.xlabel("Preprocessed $x$-coordinate [a.u.]", fontsize=12)
plt.ylabel("Preprocessed $y$-coordinate [a.u.]", fontsize=12)
x_bins = np.linspace(-0.4, 0.4, 41)
y_bins = np.linspace(-0.4, 0.4, 41)
plt.hist2d(
    pad_clusters[:, :, 0].flatten(),
    pad_clusters[:, :, 1].flatten(),
    weights=pad_clusters[:, :, 2].flatten(),
    bins=(x_bins, y_bins),
    cmin=0,
    cmap="magma",
)
plt.colorbar()

img_name = "calo-deposits-demo" if args.demo else "calo-deposits"
plt.savefig(fname=f"{images_dir}/{img_name}.png")
plt.close()

# +----------------------+
# |   Energy sequences   |
# +----------------------+

photon_energy = pad_photons[:64, :, 2]
cluster_energy = pad_clusters[:64, :, 2]
vmax = max(photon_energy.max(), cluster_energy.max())

plt.figure(figsize=(18, 10), dpi=300)

plt.subplot(1, 2, 1)
plt.title("Generated photons", fontsize=14)
plt.xlabel("Photon preprocessed energy [a.u.]", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(photon_energy, aspect="auto", vmin=0.0, vmax=vmax, cmap="magma")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Reconstructed calo-clusters", fontsize=14)
plt.xlabel("Cluster preprocessed energy [a.u.]", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(cluster_energy, aspect="auto", vmin=0.0, vmax=vmax, cmap="magma")
plt.colorbar()

img_name = "energy-seq-demo" if args.demo else "energy-seq"
plt.savefig(fname=f"{images_dir}/{img_name}.png")
plt.close()

# +----------------------+
# |   Matching weights   |
# +----------------------+

photons_xy = np.tile(pad_photons[:, None, :, :2], (1, pad_clusters.shape[1], 1, 1))
clusters_xy = np.tile(pad_clusters[:, :, None, :2], (1, 1, pad_photons.shape[1], 1))

pairwise_distance = np.linalg.norm(clusters_xy - photons_xy, axis=-1)
min_pairwise_distance = np.min(pairwise_distance, axis=-1)

match_weights = MAX_MATCH_DISTANCE / np.maximum(
    min_pairwise_distance, MAX_MATCH_DISTANCE
)
match_weights *= pad_clusters[:, :, -1]  # NotPadding boolean

# +---------------------------+
# |   Reduced event example   |
# +---------------------------+

x_photon = pad_photons[event_number, :, 0].flatten()
y_photon = pad_photons[event_number, :, 1].flatten()
energy_photon = pad_photons[event_number, :, 2].flatten()

x_cluster = pad_clusters[event_number, :, 0].flatten()
y_cluster = pad_clusters[event_number, :, 1].flatten()
energy_cluster = pad_clusters[event_number, :, 2].flatten()

w = match_weights[event_number, :].flatten()

plt.figure(figsize=(8, 6), dpi=300)
plt.xlabel("Preprocessed $x$-coordinate [a.u.]", fontsize=12)
plt.ylabel("Preprocessed $y$-coordinate [a.u.]", fontsize=12)
plt.scatter(
    x_photon,
    y_photon,
    s=50.0 * energy_photon / energy_cluster.max(),
    marker="o",
    facecolors="none",
    edgecolors="#d7191c",
    lw=0.75,
    label="Generated photon",
    zorder=2,
)
plt.scatter(
    x_cluster,
    y_cluster,
    s=50.0 * energy_cluster / energy_cluster.max(),
    marker="s",
    facecolors="none",
    edgecolors="#2b83ba",
    lw=0.75,
    label="Reconstructed cluster",
    zorder=3,
)
plt.scatter(
    np.where(w == 1.0, x_cluster, 0.0),
    np.where(w == 1.0, y_cluster, 0.0),
    s=50.0 * np.where(w == 1.0, energy_cluster / energy_cluster.max(), 0.0),
    marker="s",
    facecolors="yellow",
    edgecolors="#2b83ba",
    lw=0.75,
    label="Photon-matched cluster",
    zorder=1,
)
plt.legend(loc="upper left", fontsize=10)
plt.savefig(fname=f"{images_dir}/reduced-evt-example.png")
plt.close()

# +--------------------------+
# |   Training data export   |
# +--------------------------+

export_data_fname = "calotron-dataset"
if args.demo:
    export_data_fname += "-demo"
npz_fname = f"{export_data_dir}/{export_data_fname}.npz"
np.savez(
    file=npz_fname,
    photon=pad_photons[:, :, :-1],  # avoid NotPadding boolean
    cluster=pad_clusters[:, :, :-1],  # avoid NotPadding boolean
    weight=match_weights,
)
print(
    f"[INFO] Training data of {len(pad_photons)} instances correctly saved to {npz_fname}"
)

# +---------------------------------+
# |   Preprocessing models export   |
# +---------------------------------+

pkl_fname = f"{models_dir}/photon-scaler-logE.pkl"
pickle.dump(photon_scaler_logE, open(pkl_fname, "wb"))
print(f"[INFO] Photon energy scaler correctly saved to {pkl_fname}")

pkl_fname = f"{models_dir}/cluster-scaler-logE.pkl"
pickle.dump(cluster_scaler_logE, open(pkl_fname, "wb"))
print(f"[INFO] Cluster energy scaler correctly saved to {pkl_fname}")

if not args.demo:
    pkl_fname = f"{models_dir}/cluster-scaler-pid.pkl"
    pickle.dump(cluster_scaler_pid, open(pkl_fname, "wb"))
    print(f"[INFO] Cluster PID scaler correctly saved to {pkl_fname}")
