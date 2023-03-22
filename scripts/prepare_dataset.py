import pickle
from argparse import ArgumentParser
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

ECAL_W = 8000
ECAL_H = 6500
ENERGY_MIN = 1.0
PADDING_VALUE = 0.0
MAX_INPUT_PHOTONS = 128
MAX_OUTPUT_CLUSTERS = 64
MAX_INPUT_PHOTONS_DEMO = 32
MAX_OUTPUT_CLUSTERS_DEMO = 16

# +------------------+
# |   Parser setup   |
# +------------------+

parser = ArgumentParser(description="dataset preparation setup")

parser.add_argument("-f", "--filename", required=True)
parser.add_argument("-m", "--max_files", default=10)
parser.add_argument("-c", "--chunk_size", default=-1)

parser.add_argument("--demo", action="store_true")
parser.add_argument("--no-demo", dest="saving", action="store_false")
parser.set_defaults(demo=False)

parser.add_argument("--verbose", action="store_true")
parser.add_argument("--no-verbose", dest="saving", action="store_false")
parser.set_defaults(verbose=False)

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
        photon_list.append(file["CaloTupler/calo_true"].arrays(library="pd"))
        cluster_list.append(file["CaloTupler/neutral_protos"].arrays(library="pd"))

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

photon_df["p"] = np.linalg.norm(photon_df[["px", "py", "pz"]], axis=1)
photon_df["log_p"] = np.log(np.maximum(photon_df.p, ENERGY_MIN))
photon_df["tx"] = photon_df.px / photon_df.pz
photon_df["ty"] = photon_df.py / photon_df.pz

true_vars = ["ecal_x", "ecal_y", "log_p", "tx", "ty"]

if args.verbose:
    print(photon_df[true_vars].describe())

# +---------------------------+
# |   Photons preprocessing   |
# +---------------------------+

p_photon_df = photon_df.copy()

p_scaler = MinMaxScaler()

start = time()
p_photon_df["ecal_x"] = p_photon_df.ecal_x / ECAL_W * 2
p_photon_df["ecal_y"] = p_photon_df.ecal_y / ECAL_H * 2
p_photon_df["log_p"] = p_scaler.fit_transform(np.c_[p_photon_df.log_p])
print(f"[INFO] Generated photons preprocessing completed in {time()-start:.2f} s")

if args.verbose:
    print(p_photon_df[true_vars].describe())

# +-----------------------+
# |   Cluster DataFrame   |
# +-----------------------+

cluster_df["log_E"] = np.log(np.maximum(cluster_df.E, ENERGY_MIN))
cluster_df["NotPadding"] = 1.0

reco_vars = ["x", "y", "log_E"]

if not args.demo:
    bool_vars = ["PhotonFromMergedPi0", "Pi0Merged", "Photon", "NotPadding"]
    pid_vars = ["PhotonID", "IsNotE", "IsNotH"]
    reco_vars += bool_vars + pid_vars

if args.verbose:
    print(cluster_df[reco_vars].describe())

# +----------------------------+
# |   Clusters preprocessing   |
# +----------------------------+

p_cluster_df = cluster_df.copy()

e_scaler = MinMaxScaler()
pid_scaler = StandardScaler()

start = time()
p_cluster_df["x"] = p_cluster_df.x / ECAL_W * 2
p_cluster_df["y"] = p_cluster_df.y / ECAL_H * 2
p_cluster_df["log_E"] = e_scaler.fit_transform(np.c_[p_cluster_df.log_E])
if not args.demo:
    p_cluster_df[pid_vars] = pid_scaler.fit_transform(cluster_df[pid_vars].values)
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
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("$y$ coordinate", fontsize=12)
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
    label="Reconstructed calo-cluster",
)
plt.xlim([-1.0, 1.0])
plt.ylim([-1.0, 1.0])
plt.legend(loc="upper left", fontsize=10)
plt.savefig(fname=f"{images_dir}/evt-example.png")
plt.close()

# +----------------------------------+
# |   Event multiplicity histogram   |
# +----------------------------------+

plt.figure(figsize=(8, 5), dpi=300)
plt.xlabel("Event multipilcity", fontsize=12)
plt.ylabel("Number of events", fontsize=12)
bins = np.linspace(0, 400, 51)
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
plt.savefig(fname=f"{images_dir}/evt-multiplicity-hist.png")
plt.close()

# +------------------------------+
# |   Training set preparation   |
# +------------------------------+

max_input_photons = MAX_INPUT_PHOTONS_DEMO if args.demo else MAX_INPUT_PHOTONS
max_output_clusters = MAX_OUTPUT_CLUSTERS_DEMO if args.demo else MAX_OUTPUT_CLUSTERS

pad_photons = PADDING_VALUE * np.ones((nEvents, max_input_photons, len(true_vars)))
pad_clusters = PADDING_VALUE * np.ones((nEvents, max_output_clusters, len(reco_vars)))

for iRow, photon in enumerate(true_photons):
    photons_trunkated = photon[:max_input_photons]
    pad_photons[iRow, : len(photons_trunkated)] = photons_trunkated

for iRow, cluster in enumerate(reco_clusters):
    clusters_trunkated = cluster[:max_output_clusters]
    pad_clusters[iRow, : len(clusters_trunkated)] = clusters_trunkated

# +--------------------------+
# |   Calorimeter deposits   |
# +--------------------------+

plt.figure(figsize=(16, 5), dpi=300)

plt.subplot(1, 2, 1)
plt.title("Generated photons", fontsize=14)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("$y$ coordinate", fontsize=12)
x_bins = np.linspace(-1.0, 1.0, 101)
y_bins = np.linspace(-1.0, 1.0, 101)
plt.hist2d(
    pad_photons[:, :, 0].flatten(),
    pad_photons[:, :, 1].flatten(),
    weights=pad_photons[:, :, 2].flatten(),
    bins=(x_bins, y_bins),
    cmin=0,
    cmap="gist_heat",
)

plt.subplot(1, 2, 2)
plt.title("Reconstructed calo-clusters", fontsize=14)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("$y$ coordinate", fontsize=12)
x_bins = np.linspace(-1.0, 1.0, 101)
y_bins = np.linspace(-1.0, 1.0, 101)
plt.hist2d(
    pad_clusters[:, :, 0].flatten(),
    pad_clusters[:, :, 1].flatten(),
    weights=pad_clusters[:, :, 2].flatten(),
    bins=(x_bins, y_bins),
    cmin=0,
    cmap="gist_heat",
)

img_name = "calo-deposits-demo" if args.demo else "calo-deposits"
plt.savefig(fname=f"{images_dir}/{img_name}.png")
plt.close()

# +----------------------+
# |   Energy sequences   |
# +----------------------+

plt.figure(figsize=(16, 10), dpi=300)

plt.subplot(1, 2, 1)
plt.title("Generated photons", fontsize=14)
plt.xlabel("Photon energy", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(pad_photons[:64, :, 2], aspect="auto", cmap="gist_heat")

plt.subplot(1, 2, 2)
plt.title("Reconstructed calo-clusters", fontsize=14)
plt.xlabel("Cluster energy deposits", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(pad_clusters[:64, :, 2], aspect="auto", cmap="gist_heat")

img_name = "energy-seq-demo" if args.demo else "energy-seq"
plt.savefig(fname=f"{images_dir}/{img_name}.png")
plt.close()

# +--------------------------+
# |   Training data export   |
# +--------------------------+

export_data_fname = "calo-train-data-demo" if args.demo else "calo-train-data"
npz_fname = f"{export_data_dir}/{export_data_fname}.npz"
np.savez(npz_fname, photon=pad_photons, cluster=pad_clusters)
print(
    f"[INFO] Training data of {len(pad_photons)} instances correctly saved to {npz_fname}"
)

# +---------------------------------+
# |   Preprocessing models export   |
# +---------------------------------+

export_scaler_fname = "photon-energy-scaler"
pkl_fname = f"{models_dir}/{export_scaler_fname}.pkl"
pickle.dump(p_scaler, open(pkl_fname, "wb"))
print(f"[INFO] Photon energy scaler correctly saved to {pkl_fname}")

export_scaler_fname = "cluster-energy-scaler"
pkl_fname = f"{models_dir}/{export_scaler_fname}.pkl"
pickle.dump(e_scaler, open(pkl_fname, "wb"))
print(f"[INFO] Cluster energy scaler correctly saved to {pkl_fname}")

if not args.demo:
    export_scaler_fname = "cluster-pid-scaler"
    pkl_fname = f"{models_dir}/{export_scaler_fname}.pkl"
    pickle.dump(pid_scaler, open(pkl_fname, "wb"))
    print(f"[INFO] Cluster PID scaler correctly saved to {pkl_fname}")
