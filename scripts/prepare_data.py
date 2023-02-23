import yaml
import uproot
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time
from sklearn.preprocessing import StandardScaler, QuantileTransformer


VERBOSE = False
MAX_INPUT_PHOTONS = 64
MAX_OUTPUT_CLUSTERS = 32


# +-------------------+
# |   Initial setup   |
# +-------------------+

with open("config/directories.yml") as file:
  config_dir = yaml.full_load(file)

data_dir = config_dir["data_dir"]
images_dir = config_dir["images_dir"]

data_fname = f"{data_dir}/LamarrTraining.root"

# +------------------------------+
# |   Generated photons import   |
# +------------------------------+

start = time()
calo_true = (
               uproot.open(data_fname)["CaloTupler/calo_true"]
               .arrays(library="pd")   # pandas DataFrame
            )
print(f"[INFO] Generated photons DataFrame ({len(calo_true)} "
      f"rows) correctly loaded in {time()-start:.2f} s")

calo_true["p"] = np.linalg.norm(calo_true[["px", "py", "pz"]], axis=1)
calo_true["tx"] = calo_true.px/calo_true.pz
calo_true["ty"] = calo_true.py/calo_true.pz

true_vars = ["ecal_x", "ecal_y", "p", "tx", "ty"]
if VERBOSE:
  print(calo_true[true_vars].describe())

# +--------------------------------+
# |   Photons data preprocessing   |
# +--------------------------------+

p_calo_true = calo_true.copy()

ECAL_W = 8e3
ECAL_H = 5e3
p_scaler = QuantileTransformer()

start = time()
p_calo_true["ecal_x"] = p_calo_true.ecal_x / ECAL_W * 2
p_calo_true["ecal_y"] = p_calo_true.ecal_y / ECAL_H * 2
p_calo_true["p"] = p_scaler.fit_transform(np.c_[np.log(p_calo_true.p)])
print(f"[INFO] Generated photons preprocessing completed in {time()-start:.2f} s")

if VERBOSE:
  print(p_calo_true[true_vars].describe())

# +-----------------------------------+
# |   Reconstructed clusters import   |
# +-----------------------------------+

start = time()
neutral_protos = (
                    uproot.open(data_fname)["CaloTupler/neutral_protos"]
                    .arrays(library="pd")   # pandas DataFrame
                    .query("Pi0Merged == 0 and E > 0")
                 )
print(f"[INFO] Reconstructed calo-clusters DataFrame ({len(neutral_protos)} "
      f"rows) correctly loaded in {time()-start:.2f} s")

neutral_protos["NotPadding"] = 1.0

reco_vars = ["x", "y", "E", "PhotonFromMergedPi0", "Pi0Merged",
             "Photon", "NotPadding", "PhotonID", "IsNotE", "IsNotH"]
if VERBOSE:
  print(neutral_protos[reco_vars].describe())

# +---------------------------------+
# |   Clusters data preprocessing   |
# +---------------------------------+

p_neutral_protos = neutral_protos.copy()

e_scaler = QuantileTransformer()
rec_scaler = StandardScaler()

start = time()
p_neutral_protos["x"] = p_neutral_protos.x / ECAL_W * 2
p_neutral_protos["y"] = p_neutral_protos.y / ECAL_H * 2
p_neutral_protos["E"] = e_scaler.fit_transform(np.c_[np.log(p_neutral_protos.E)])
p_neutral_protos[reco_vars[7:]] = rec_scaler.fit_transform(neutral_protos[reco_vars[7:]].values)
print(f"[INFO] Reconstructed calo-clusters preprocessing completed in {time()-start:.2f} s")

if VERBOSE:
  print(p_neutral_protos[reco_vars].describe())

# +--------------------------------+
# |   Data processing per events   |
# +--------------------------------+

X = list()
Y = list()

events = np.unique(p_neutral_protos.evtNumber)

start = time()
for iEvent, event in enumerate(events):
  true = (
            p_calo_true[p_calo_true.evtNumber == event]
            .query("mcID == 22")   # photons
            .sort_values("p", ascending=False)
         )
  reco = (
            p_neutral_protos[
              (p_neutral_protos.evtNumber == event) & (~p_neutral_protos.E.isnull())
            ]
            .sort_values("E", ascending=False)
         )
  X.append(true[true_vars].values)
  Y.append(reco[reco_vars].values)

print(f"[INFO] Photons and clusters from {len(events)} events collected in {time()-start:.2f} s")

# +-----------------------------------+
# |   Photon/cluster positions plot   |
# +-----------------------------------+

event_number = 42
photon = np.array(X[event_number])
cluster = np.array(Y[event_number])

plt.figure(figsize=(8,8), dpi=100)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("$y$ coordinate", fontsize=12)
plt.scatter(
              photon[:,0].flatten(),
              photon[:,1].flatten(),
              s=50 * photon[:,2].flatten() / cluster[:,2].flatten().max(),
              marker="o", facecolors="none", edgecolors="#d7191c",
              lw=0.75, label="Generated photon"
           )
plt.scatter(
              cluster[:,0].flatten(),
              cluster[:,1].flatten(),
              s=50 * cluster[:,2].flatten() / cluster[:,2].flatten().max(),
              marker="s", facecolors="none", edgecolors="#2b83ba", 
              lw=0.75, label="Reconstructed calo-cluster"
           )
plt.legend()
plt.savefig(f"{images_dir}/event-example.png")
plt.close()

# +-----------------------------+
# |   Event multiplicity plot   |
# +-----------------------------+

plt.figure(figsize=(8,5), dpi=100)
plt.xlabel("Event multipilcity", fontsize=12)
plt.ylabel("Number of events", fontsize=12)
bins = np.linspace(0, 512, 65)
plt.hist(
           np.array([len(x) for x in X]),
           bins=bins, color="#3288bd",
           label="Generated photons"
        )
plt.hist(
           np.array([len(y) for y in Y]),
           bins=bins, histtype="step", color="#fc8d59",
           lw=2, label="Reconstructed clusters"
        )
plt.legend(loc="upper right", fontsize=10)
plt.savefig(fname=f"{images_dir}/events-multiplicity.png")
plt.close()

# +------------------------------+
# |   Training set preparation   |
# +------------------------------+

nEvents = len(events)

pad_X = np.zeros((nEvents, MAX_INPUT_PHOTONS, len(true_vars)))
pad_Y = np.zeros((nEvents, MAX_OUTPUT_CLUSTERS, len(reco_vars)))

for iRow, x in tqdm(enumerate(X), total=len(X), desc="Padding X"):
  x_trunkated = x[:MAX_INPUT_PHOTONS]
  pad_X[iRow, :len(x_trunkated)] = x_trunkated

for iRow, y in tqdm(enumerate(Y), total=len(Y), desc="Padding Y"):
  y_trunkated = y[:MAX_OUTPUT_CLUSTERS]
  pad_Y[iRow, :len(y_trunkated)] = y_trunkated

# +------------------------+
# |   Energy matrix plot   |
# +------------------------+

plt.figure(figsize=(16,10), dpi=100)
plt.subplot(1,2,1)
plt.title("Generated photons", fontsize=14)
plt.xlabel("Photon energy", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(pad_X[:64,:,2], aspect="auto", interpolation="none")
plt.subplot(1,2,2)
plt.title("Reconstructed calo-clusters", fontsize=14)
plt.xlabel("Cluster energy deposits", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(pad_Y[:64,:,2], aspect="auto", interpolation="none")
plt.savefig(fname=f"{images_dir}/energy-matrix.png")
plt.close()

# +--------------------------+
# |   Training data export   |
# +--------------------------+

data_fname = f"{data_dir}/train-data.npz"
np.savez(data_fname, photon=pad_X, cluster=pad_Y)
print(f"[INFO] Training data correctly saved to {data_fname}")
