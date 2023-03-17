from argparse import ArgumentParser
from glob import glob
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import yaml
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.utils import shuffle
from tqdm import tqdm

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
photon_df["tx"] = photon_df.px / photon_df.pz
photon_df["ty"] = photon_df.py / photon_df.pz

true_vars = ["ecal_x", "ecal_y", "p", "tx", "ty"]
if args.verbose:
    print(photon_df[true_vars].describe())

# +---------------------------+
# |   Photons preprocessing   |
# +---------------------------+

p_photon_df = photon_df.copy()

ECAL_W = 8e3
ECAL_H = 5e3
p_scaler = QuantileTransformer()

start = time()
p_photon_df["ecal_x"] = p_photon_df.ecal_x / ECAL_W * 2
p_photon_df["ecal_y"] = p_photon_df.ecal_y / ECAL_H * 2
p_photon_df["p"] = p_scaler.fit_transform(
    np.c_[np.log(np.where(p_photon_df.p != 0.0, p_photon_df.p, 1e-12))]
)
print(f"[INFO] Generated photons preprocessing completed in {time()-start:.2f} s")

if args.verbose:
    print(p_photon_df[true_vars].describe())

# +-----------------------+
# |   Cluster DataFrame   |
# +-----------------------+

cluster_df["NotPadding"] = 1.0

if args.demo:
    reco_vars = ["x", "y", "E"]
else:
    reco_vars = [
        "x",
        "y",
        "E",
        "PhotonFromMergedPi0",
        "Pi0Merged",
        "Photon",
        "NotPadding",
        "PhotonID",
        "IsNotE",
        "IsNotH",
    ]
if args.verbose:
    print(cluster_df[reco_vars].describe())

# +----------------------------+
# |   Clusters preprocessing   |
# +----------------------------+

p_cluster_df = cluster_df.copy()

e_scaler = QuantileTransformer()
rec_scaler = StandardScaler()

start = time()
p_cluster_df["x"] = p_cluster_df.x / ECAL_W * 2
p_cluster_df["y"] = p_cluster_df.y / ECAL_H * 2
p_cluster_df["E"] = e_scaler.fit_transform(
    np.c_[np.log(np.where(p_cluster_df.E != 0.0, p_cluster_df.E, 1e-12))]
)
if not args.demo:
    p_cluster_df[reco_vars[7:]] = rec_scaler.fit_transform(
        cluster_df[reco_vars[7:]].values
    )
print(
    f"[INFO] Reconstructed calo-clusters preprocessing completed in {time()-start:.2f} s"
)

if args.verbose:
    print(p_cluster_df[reco_vars].describe())

# +--------------------------------+
# |   Data processing per events   |
# +--------------------------------+

X = list()
Y = list()

events = np.unique(p_cluster_df.evtNumber)

start = time()
for iEvent, event in enumerate(events):
    true = (
        p_photon_df[p_photon_df.evtNumber == event]
        .query("mcID == 22")  # photons
        .sort_values("p", ascending=False)
    )
    reco = p_cluster_df[
        (p_cluster_df.evtNumber == event) & (~p_cluster_df.E.isnull())
    ].sort_values("E", ascending=False)
    X.append(true[true_vars].values)
    Y.append(reco[reco_vars].values)

print(
    f"[INFO] Photons and clusters from {len(events)} events collected in {time()-start:.2f} s"
)

# +------------------------+
# |   Event example plot   |
# +------------------------+

event_number = 42
photon = np.array(X[event_number])
cluster = np.array(Y[event_number])

plt.figure(figsize=(8, 6), dpi=100)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("$y$ coordinate", fontsize=12)
plt.scatter(
    photon[:, 0].flatten(),
    photon[:, 1].flatten(),
    s=50 * photon[:, 2].flatten() / cluster[:, 2].flatten().max(),
    marker="o",
    facecolors="none",
    edgecolors="#d7191c",
    lw=0.75,
    label="Generated photon",
)
plt.scatter(
    cluster[:, 0].flatten(),
    cluster[:, 1].flatten(),
    s=50 * cluster[:, 2].flatten() / cluster[:, 2].flatten().max(),
    marker="s",
    facecolors="none",
    edgecolors="#2b83ba",
    lw=0.75,
    label="Reconstructed calo-cluster",
)
plt.legend(loc="upper left", fontsize=10)
plt.savefig(fname=f"{images_dir}/evt-example.png")
plt.close()

# +----------------------------------+
# |   Event multiplicity histogram   |
# +----------------------------------+

plt.figure(figsize=(8, 5), dpi=100)
plt.xlabel("Event multipilcity", fontsize=12)
plt.ylabel("Number of events", fontsize=12)
bins = np.linspace(0, 400, 51)
plt.hist(
    np.array([len(x) for x in X]), bins=bins, color="#3288bd", label="Generated photons"
)
plt.hist(
    np.array([len(y) for y in Y]),
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

nEvents = len(events)
max_input_photons = MAX_INPUT_PHOTONS_DEMO if args.demo else MAX_INPUT_PHOTONS
max_output_clusters = MAX_OUTPUT_CLUSTERS_DEMO if args.demo else MAX_OUTPUT_CLUSTERS

pad_X = np.zeros((nEvents, max_input_photons, len(true_vars)))
pad_Y = np.zeros((nEvents, max_output_clusters, len(reco_vars)))

for iRow, x in tqdm(enumerate(X), total=len(X), desc="Padding X"):
    x_trunkated = x[:max_input_photons]
    pad_X[iRow, : len(x_trunkated)] = x_trunkated

for iRow, y in tqdm(enumerate(Y), total=len(Y), desc="Padding Y"):
    y_trunkated = y[:max_output_clusters]
    pad_Y[iRow, : len(y_trunkated)] = y_trunkated

# +---------------------+
# |   xy 2D-histogram   |
# +---------------------+

plt.figure(figsize=(16, 5), dpi=100)
plt.subplot(1, 2, 1)
plt.title("Generated photons", fontsize=14)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("$y$ coordinate", fontsize=12)
x_min = pad_X[:, :, 0].flatten().min()
x_max = pad_X[:, :, 0].flatten().max()
x_bins = np.linspace(x_min, x_max, 101)
y_min = pad_X[:, :, 1].flatten().min()
y_max = pad_X[:, :, 1].flatten().max()
y_bins = np.linspace(y_min, y_max, 101)
plt.hist2d(
    pad_X[:, :, 0].flatten(),
    pad_X[:, :, 1].flatten(),
    weights=pad_X[:, :, 2].flatten(),
    bins=(x_bins, y_bins),
    cmin=0,
    cmap="gist_heat",
)
plt.subplot(1, 2, 2)
plt.title("Reconstructed calo-clusters", fontsize=14)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("$y$ coordinate", fontsize=12)
x_min = pad_Y[:, :, 0].flatten().min()
x_max = pad_Y[:, :, 0].flatten().max()
x_bins = np.linspace(x_min, x_max, 101)
y_min = pad_Y[:, :, 1].flatten().min()
y_max = pad_Y[:, :, 1].flatten().max()
y_bins = np.linspace(y_min, y_max, 101)
plt.hist2d(
    pad_Y[:, :, 0].flatten(),
    pad_Y[:, :, 1].flatten(),
    weights=pad_Y[:, :, 2].flatten(),
    bins=(x_bins, y_bins),
    cmin=0,
    cmap="gist_heat",
)
img_name = "xy-hist2d-demo" if args.demo else "xy-hist2d"
plt.savefig(fname=f"{images_dir}/{img_name}.png")
plt.close()

# +-------------------------+
# |   Energy batches plot   |
# +-------------------------+

plt.figure(figsize=(16, 10), dpi=100)
plt.subplot(1, 2, 1)
plt.title("Generated photons", fontsize=14)
plt.xlabel("Photon energy", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(pad_X[:64, :, 2], aspect="auto", interpolation="none")
plt.subplot(1, 2, 2)
plt.title("Reconstructed calo-clusters", fontsize=14)
plt.xlabel("Cluster energy deposits", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(pad_Y[:64, :, 2], aspect="auto", interpolation="none")
img_name = "energy-batches-demo" if args.demo else "energy-batches"
plt.savefig(fname=f"{images_dir}/{img_name}.png")
plt.close()

# +--------------------------+
# |   Training data export   |
# +--------------------------+

export_data_fname = "calo-train-data-demo" if args.demo else "calo-train-data"
npz_fname = f"{export_data_dir}/{export_data_fname}.npz"
np.savez(npz_fname, photon=pad_X, cluster=pad_Y)
print(f"[INFO] Training data of {len(pad_X)} instances correctly saved to {npz_fname}")
