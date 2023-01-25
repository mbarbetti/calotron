import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from html_reports import Report

from calotron.losses import CaloLoss
from calotron.simulators import Simulator
from calotron.utils import initHPSingleton, getModelSummary
from calotron.models import Transformer, Discriminator, Calotron
from calotron.callbacks.schedulers import PolynomialDecay, ExponentialDecay


ALPHA = 0.1
ORDER = True
POSITION = True
BATCHSIZE = 128
EPOCHS = 500
DTYPE = tf.float32

# +-------------------+
# |   Initial setup   |
# +-------------------+

hp = initHPSingleton()

with open("config/directories.yml") as file:
  config = yaml.full_load(file)

data_dir = config["data_dir"]
export_dir = config["export_dir"]
images_dir = config["images_dir"]
report_dir = config["report_dir"]

# +------------------+
# |   Data loading   |
# +------------------+

npzfile = np.load(f"{data_dir}/train-data-demo.npz")
photon = npzfile["photon"][:, ::-1, :]
cluster = npzfile["cluster"]
cluster_labels = npzfile["cluster_labels"]

#print(f"photon {photon.shape}\n", photon)
#print(f"cluster {cluster.shape}\n", cluster)
#print(f"cluster labels {cluster_labels.shape}\n", cluster_labels)

chunk_size = photon.shape[0]

photon_shuffled = np.zeros_like(photon)
cluster_shuffled = np.zeros_like(cluster)
cluster_labels_shuffled = np.zeros_like(cluster_labels)
for i in range(chunk_size):
  new_photon_order = np.random.permutation(photon.shape[1])
  photon_shuffled[i] = photon[i, new_photon_order, :]
  new_order = np.random.permutation(cluster.shape[1]-1)
  new_cluster_order = [0] + list(new_order+1)
  cluster_shuffled[i] = cluster[i, new_cluster_order, :]
  new_cluster_order = list(new_order) + [cluster.shape[1]-1]
  cluster_labels_shuffled[i] = cluster_labels[i, new_cluster_order, :]

#print(f"photon SHUFFLED {photon_shuffled.shape}\n", photon_shuffled)
#print(f"cluster SHUFFLED {cluster_shuffled.shape}\n", cluster_shuffled)
#print(f"cluster labels SHUFFLED {cluster_labels_shuffled.shape}\n", cluster_labels_shuffled)

if ORDER:
  photon_final = photon
  cluster_final = cluster
  cluster_labels_final = cluster_labels
else:
  photon_final = photon_shuffled
  cluster_final = cluster_shuffled
  cluster_labels_final = cluster_labels_shuffled

# +-------------------------+
# |   Dataset preparation   |
# +-------------------------+

with tf.device("/gpu:0"):
  X = tf.cast(photon_final, dtype=DTYPE)
  # Y = tf.cast(cluster_final, dtype=DTYPE)
  Y = tf.cast(cluster_labels_final, dtype=DTYPE)

train_ds = (tf.data.Dataset.from_tensor_slices((X, Y))
            .shuffle(hp.get("chunk_size", chunk_size))
            .batch(hp.get("batch_size", BATCHSIZE), drop_remainder=True)
            .cache()
            .prefetch(tf.data.AUTOTUNE))

# +------------------------+
# |   Model construction   |
# +------------------------+

transformer = Transformer(output_depth=hp.get("t_output_depth", Y.shape[2]),
                          encoder_depth=hp.get("t_encoder_depth", 32),
                          decoder_depth=hp.get("t_decoder_depth", 32),
                          num_layers=hp.get("t_num_layers", 5),
                          num_heads=hp.get("t_num_heads", 4),
                          key_dim=hp.get("t_key_dim", 64),
                          encoder_pos_dim=hp.get("t_encoder_pos_dim", 16),
                          decoder_pos_dim=hp.get("t_decoder_pos_dim", 16),
                          encoder_pos_normalization=hp.get("t_encoder_pos_normalization", 128),
                          decoder_pos_normalization=hp.get("t_decoder_pos_normalization", 128),
                          encoder_max_length=hp.get("t_encoder_max_length", X.shape[1]),
                          decoder_max_length=hp.get("t_decoder_max_length", Y.shape[1]),
                          ff_units=hp.get("t_ff_units", 256),
                          dropout_rate=hp.get("t_dropout_rate", 0.1),
                          pos_sensitive=hp.get("t_pos_sensitive", POSITION),
                          residual_smoothing=hp.get("t_residual_smoothing", True),
                          output_activations=hp.get("t_output_activations", ["linear", "linear", "sigmoid"]),
                          dtype=DTYPE)

discriminator = Discriminator(latent_dim=hp.get("d_latent_dim", 64),
                              output_units=hp.get("d_output_units", 1),
                              output_activation=hp.get("d_output_activation", "sigmoid"),
                              hidden_layers=hp.get("d_hidden_layers", 5),
                              hidden_units=hp.get("d_hidden_units", 256),
                              dropout_rate=hp.get("d_dropout_rate", 0.1),
                              dtype=DTYPE)

model = Calotron(transformer=transformer, discriminator=discriminator)

output = model((X, Y))
model.summary()

# +----------------------+
# |   Optimizers setup   |
# +----------------------+

t_lr0 = 1e-3
t_opt = tf.keras.optimizers.RMSprop(t_lr0)
hp.get("t_optimizer", "RMSprop")
hp.get("t_lr0", t_lr0)

d_lr0 = 1e-4
d_opt = tf.keras.optimizers.RMSprop(d_lr0)
hp.get("d_optimizer", "RMSprop")
hp.get("d_lr0", d_lr0)

# +----------------------------+
# |   Training configuration   |
# +----------------------------+

loss = CaloLoss(alpha=ALPHA)
hp.get("loss", loss.name)
hp.get("loss_alpha", loss._alpha)

model.compile(loss=loss,
              metrics=hp.get("metrics", ["accuracy"]),
              transformer_optimizer=t_opt,
              discriminator_optimizer=d_opt,
              transformer_upds_per_batch=hp.get("transformer_upds_per_batch", 5),
              discriminator_upds_per_batch=hp.get("discriminator_upds_per_batch", 1))

# +------------------------------+
# |   Learning rate scheduling   |
# +------------------------------+

t_sched = PolynomialDecay(model.transformer_optimizer,
                          decay_steps=10000,
                          end_learning_rate=1e-4)
hp.get("t_sched", "PolynomialDecay")

d_sched = ExponentialDecay(model.discriminator_optimizer,
                           decay_rate=0.10,
                           decay_steps=20000)
hp.get("d_sched", "ExponentialDecay")

# +------------------------+
# |   Training procedure   |
# +------------------------+

start = datetime.now()
train = model.fit(
  train_ds,
  epochs=hp.get("epochs", EPOCHS),
  callbacks=[t_sched, d_sched]
)
stop = datetime.now()

duration = str(stop-start).split(".")[0].split(":")   # [HH, MM, SS]
duration = f"{duration[0]}h {duration[1]}min {duration[2]}s"
print(f"[INFO] Model training completed in {duration}")

# +------------------+
# |   Model output   |
# +------------------+

start_token = model.get_start_token(Y)
sim = Simulator(model.transformer, start_token=start_token)
out = sim(X, max_length=Y.shape[1])

# +---------------------+
# |   Training report   |
# +---------------------+

report = Report()
report.add_markdown('<h1 align="center">Calotron training report</h1>')

timestamp = str(datetime.now())
date, hour = timestamp.split(" ")
date = date.replace("-", "/")
hour = hour.split(".")[0]

prefix = ""
timestamp = timestamp.split(".")[0].replace("-", "").replace(" ", "-")
for time, unit in zip(timestamp.split(":"), ["h", "m", "s"]):
  prefix += time + unit   # YYYYMMDD-HHhMMmSSs

report.add_markdown(
  f"""
    - Report generated on {date} at {hour}
    - Model training completed in {duration}
  """
)
report.add_markdown("---")

## Hyperparameters and other details
report.add_markdown('<h2 align="center">Hyperparameters and other details</h2>')
content = ""
for k, v in hp.get_dict().items():
  content += f"\t- {k} : {v}\n"
report.add_markdown(content)
report.add_markdown("---")

## Transformer architecture
report.add_markdown('<h2 align="center">Transformer architecture</h2>')
html_table, num_params = getModelSummary(model.transformer)
report.add_markdown(html_table)
report.add_markdown(f"**Total params** : {num_params}")
report.add_markdown("---")

## Discriminator architecture
report.add_markdown('<h2 align="center">Discriminator architecture</h2>')
html_table, num_params = getModelSummary(model.discriminator)
report.add_markdown(html_table)
report.add_markdown(f"**Total params** : {num_params}")
report.add_markdown("---")

## Training plots
report.add_markdown('<h2 align="center">Training plots</h2>')

prefix = ""
timestamp = timestamp.split(".")[0].replace("-", "").replace(" ", "-")
for time, unit in zip(timestamp.split(":"), ["h", "m", "s"]):
  prefix += time + unit   # YYYYMMDD-HHhMMmSSs
prefix += "_calotron"

#### Learning curves
plt.figure(figsize=(8,5), dpi=100)
plt.title("Learning curves", fontsize=14)
plt.xlabel("Training epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.plot(
  np.array(train.history["t_calo_loss"]), 
  lw=1.5, color="dodgerblue", label="transformer [x1]"
)
plt.plot(
  np.array(train.history["d_calo_loss"])/10,
  lw=1.5, color="coral", label="discriminator [x10$^{-1}$]"
)
plt.legend(loc="lower left", fontsize=10)
plt.savefig(fname=f"{images_dir}/{prefix}_learn-curves.png")
report.add_figure(options="width=45%")
plt.close()

#### Learning rate scheduling
plt.figure(figsize=(8,5), dpi=100)
plt.title("Learning rate scheduling", fontsize=14)
plt.xlabel("Training epochs", fontsize=12)
plt.ylabel("Learning rate", fontsize=12)
plt.plot(
  np.array(train.history["t_lr"]),
  lw=1.5, color="dodgerblue", label="transformer"
)
plt.plot(
  np.array(train.history["d_lr"]),
  lw=1.5, color="coral", label="discriminator"
)
plt.yscale("log")
plt.legend(loc="upper right", fontsize=10)
plt.savefig(fname=f"{images_dir}/{prefix}_lr-sched.png")
report.add_figure(options="width=45%")
plt.close()

#### Metric curves
plt.figure(figsize=(8,5), dpi=100)
plt.title("Metric curves", fontsize=14)
plt.xlabel("Training epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.plot(
  np.array(train.history["accuracy"]),
  lw=1.5, color="forestgreen", label="training set"
)
plt.legend(loc="lower right", fontsize=10)
plt.savefig(fname=f"{images_dir}/{prefix}_metric-curves.png")
report.add_figure(options="width=45%")
plt.close()

report.add_markdown("---")

## Validation plots
report.add_markdown('<h2 align="center">Validation plots</h2>')

#### X coordinate
plt.figure(figsize=(8,5), dpi=100)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("Candidates", fontsize=12)
plt.hist(
  Y[:,:,0].numpy().flatten(),
  bins=100, label="Training data"
)
plt.hist(
  out[:,:,0].numpy().flatten(),
  bins=100, histtype="step",
  lw=2, label="Calotron output"
)
plt.yscale("log")
plt.legend(loc="upper left", fontsize=10)
plt.savefig(f"{images_dir}/{prefix}_x-coord.png")
report.add_figure(options="width=45%")
plt.close()

#### Y coordinate
plt.figure(figsize=(8,5), dpi=100)
plt.xlabel("$y$ coordinate", fontsize=12)
plt.ylabel("Candidates", fontsize=12)
plt.hist(
  Y[:,:,1].numpy().flatten(), 
  bins=100, label="Training data"
)
plt.hist(
  out[:,:,1].numpy().flatten(),
  bins=100, histtype="step",
  lw=2, label="Calotron output"
)
plt.yscale("log")
plt.legend(loc="upper left", fontsize=10)
plt.savefig(f"{images_dir}/{prefix}_y-coord.png")
report.add_figure(options="width=45%")
plt.close()

#### Energy
plt.figure(figsize=(8,5), dpi=100)
plt.xlabel("Preprocessed energy [a.u]", fontsize=12)
plt.ylabel("Candidates", fontsize=12)
plt.hist(
  Y[:,:,2].numpy().flatten(),
  bins=100, label="Training data"
)
plt.hist(
  out[:,:,2].numpy().flatten(),
  bins=100, histtype="step", 
  lw=2, label="Calotron output"
)
plt.yscale("log")
plt.legend(loc="upper left", fontsize=10)
plt.savefig(f"{images_dir}/{prefix}_energy.png")
report.add_figure(options="width=45%")
plt.close()

#### Energy matrix
plt.figure(figsize=(16,10), dpi=100)
plt.subplot(1,2,1)
plt.title("Training data", fontsize=14)
plt.xlabel("Cluster energy deposits", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(
  Y[:100,:,2].numpy(),
  aspect="auto", interpolation="none"
)
plt.subplot(1,2,2)
plt.title("Calotron output", fontsize=14)
plt.xlabel("Cluster energy deposits", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(
  out[:100,:,2].numpy(),
  aspect="auto", interpolation="none"
)
plt.savefig(f"{images_dir}/{prefix}_energy-matrix.png")
report.add_figure(options="width=95%")
plt.close()

report.add_markdown("---")

report_fname = f"{report_dir}/{prefix}_train-report.html"
report.write_report(filename=report_fname)
print(f"[INFO] Training report correctly exported to {report_fname}")
