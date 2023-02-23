import os
import yaml
import socket
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from html_reports import Report
from sklearn.utils import shuffle
from argparse import ArgumentParser

from calotron.losses import CaloLoss
from calotron.models import Transformer, Discriminator, Calotron
from calotron.callbacks.schedulers import ExponentialDecay
from calotron.simulators import Simulator, ExportSimulator
from calotron.utils import initHPSingleton, getSummaryHTML


DTYPE = tf.float32
TRAIN_RATIO = 1.0
BATCHSIZE = 128
ALPHA = 0.02
EPOCHS = 500


# +------------------+
# |   Parser setup   |
# +------------------+

parser = ArgumentParser(description="scripts configuration")

parser.add_argument("--saving", action="store_true")
parser.add_argument("--no-saving", dest="saving", action="store_false")
parser.set_defaults(saving=True)

args = parser.parse_args()

# +-------------------+
# |   Initial setup   |
# +-------------------+

hp = initHPSingleton()

with open("config/directories.yml") as file:
  config_dir = yaml.full_load(file)

data_dir = config_dir["data_dir"]
export_dir = config_dir["export_dir"]
images_dir = config_dir["images_dir"]
report_dir = config_dir["report_dir"]

# +------------------+
# |   Data loading   |
# +------------------+

npzfile = np.load(f"{data_dir}/train-data-demo-medium-1.npz")
photon = npzfile["photon"][:,::-1]
cluster = npzfile["cluster"][:,::-1]

#print(f"photon {photon.shape}\n", photon)
#print(f"cluster {cluster.shape}\n", cluster)

photon, cluster = shuffle(photon, cluster)

chunk_size = photon.shape[0]
train_size = int(TRAIN_RATIO * chunk_size)

# +-------------------------+
# |   Dataset preparation   |
# +-------------------------+

X = tf.cast(photon, dtype=DTYPE)
Y = tf.cast(cluster, dtype=DTYPE)

train_ds = (tf.data.Dataset.from_tensor_slices((X[:train_size], Y[:train_size]))
              .batch(hp.get("batch_size", BATCHSIZE), drop_remainder=True)
              .cache()
              .prefetch(tf.data.AUTOTUNE))

if TRAIN_RATIO != 1.0:
  val_ds = (tf.data.Dataset.from_tensor_slices((X[train_size:], Y[train_size:]))
              .batch(BATCHSIZE, drop_remainder=True)
              .cache()
              .prefetch(tf.data.AUTOTUNE))
else:
  val_ds = None

# +------------------------+
# |   Model construction   |
# +------------------------+

transformer = Transformer(output_depth=hp.get("t_output_depth", Y.shape[2]),
                          encoder_depth=hp.get("t_encoder_depth", 32),
                          decoder_depth=hp.get("t_decoder_depth", 32),
                          num_layers=hp.get("t_num_layers", 5),
                          num_heads=hp.get("t_num_heads", 4),
                          key_dim=hp.get("t_key_dim", 64),
                          encoder_pos_dim=hp.get("t_encoder_pos_dim", 32),
                          decoder_pos_dim=hp.get("t_decoder_pos_dim", 32),
                          encoder_pos_normalization=hp.get("t_encoder_pos_normalization", 128),
                          decoder_pos_normalization=hp.get("t_decoder_pos_normalization", 128),
                          encoder_max_length=hp.get("t_encoder_max_length", X.shape[1]),
                          decoder_max_length=hp.get("t_decoder_max_length", Y.shape[1]),
                          ff_units=hp.get("t_ff_units", 256),
                          dropout_rate=hp.get("t_dropout_rate", 0.1),
                          pos_sensitive=hp.get("t_pos_sensitive", True),
                          residual_smoothing=hp.get("t_residual_smoothing", True),
                          output_activations=hp.get("t_output_activations", ["linear", "linear", "sigmoid"]),
                          start_token_initializer=hp.get("t_start_toke_initializer", "zeros"),
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

d_lr0 = 1e-3
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
              metrics=hp.get("metrics", ["accuracy", "bce"]),
              transformer_optimizer=t_opt,
              discriminator_optimizer=d_opt,
              transformer_upds_per_batch=hp.get("transformer_upds_per_batch", 4),
              discriminator_upds_per_batch=hp.get("discriminator_upds_per_batch", 1))

# +------------------------------+
# |   Learning rate scheduling   |
# +------------------------------+

t_decay_rate = 0.10
t_decay_steps = 50_000
t_sched = ExponentialDecay(model.transformer_optimizer,
                           decay_rate=t_decay_rate,
                           decay_steps=t_decay_steps)
hp.get("t_sched", "ExponentialDecay")
hp.get("t_decay_rate", t_decay_rate)
hp.get("t_decay_steps", t_decay_steps)

d_decay_rate = 0.20
d_decay_steps = 20_000
d_sched = ExponentialDecay(model.discriminator_optimizer,
                           decay_rate=d_decay_rate,
                           decay_steps=d_decay_steps)
hp.get("d_sched", "ExponentialDecay")
hp.get("d_decay_rate", d_decay_rate)
hp.get("d_decay_steps", d_decay_steps)

# +------------------------+
# |   Training procedure   |
# +------------------------+

start = datetime.now()
train = model.fit(train_ds,
                  epochs=hp.get("epochs", EPOCHS),
                  validation_data=val_ds,
                  callbacks=[t_sched, d_sched])
stop = datetime.now()

duration = str(stop-start).split(".")[0].split(":")   # [HH, MM, SS]
duration = f"{duration[0]}h {duration[1]}min {duration[2]}s"
print(f"[INFO] Model training completed in {duration}")

# +------------------+
# |   Model export   |
# +------------------+

start_token = model.get_start_token(Y)
sim = Simulator(model.transformer, start_token=start_token)
exp_sim = ExportSimulator(sim, max_length=Y.shape[1])
out = exp_sim(X)

timestamp = str(datetime.now())
date, hour = timestamp.split(" ")
date = date.replace("-", "/")
hour = hour.split(".")[0]

prefix = ""
timestamp = timestamp.split(".")[0].replace("-", "").replace(" ", "-")
for time, unit in zip(timestamp.split(":"), ["h", "m", "s"]):
  prefix += time + unit   # YYYYMMDD-HHhMMmSSs
prefix += "_calotron"

if args.saving:
  export_model_fname = f"{export_dir}/{prefix}_model"
  tf.saved_model.save(exp_sim, export_dir=export_model_fname)
  hp.dump(f"{export_model_fname}/hyperparams.yml")   # export also list of hyperparams
  print(f"[INFO] Trained model correctly exported to {export_model_fname}")
  export_img_fname = f"{images_dir}/{prefix}_img"
  os.makedirs(export_img_fname)   # need to save images

# +---------------------+
# |   Training report   |
# +---------------------+

report = Report()
report.add_markdown('<h1 align="center">Calotron training report</h1>')

report.add_markdown(
  f"""
    - Script executed on {socket.gethostname()}
    - Model training completed in {duration}
    - Report generated on {date} at {hour}
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
html_table, num_params = getSummaryHTML(model.transformer)
report.add_markdown(html_table)
report.add_markdown(f"**Total params** : {num_params}")
report.add_markdown("---")

## Discriminator architecture
report.add_markdown('<h2 align="center">Discriminator architecture</h2>')
html_table, num_params = getSummaryHTML(model.discriminator)
report.add_markdown(html_table)
report.add_markdown(f"**Total params** : {num_params}")
report.add_markdown("---")

## Training plots
report.add_markdown('<h2 align="center">Training plots</h2>')

#### Learning curves
plt.figure(figsize=(8,5), dpi=100)
plt.title("Learning curves", fontsize=14)
plt.xlabel("Training epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.plot(
  np.array(train.history["t_loss"]), 
  lw=1.5, color="#3288bd", label="transformer"
)
plt.plot(
  np.array(train.history["d_loss"]),
  lw=1.5, color="#fc8d59", label="discriminator"
)
plt.yscale("log")
plt.legend(loc="upper left", fontsize=10)
if args.saving:
  plt.savefig(fname=f"{export_img_fname}/learn-curves.png")
report.add_figure(options="width=45%")
plt.close()

#### Learning rate scheduling
plt.figure(figsize=(8,5), dpi=100)
plt.title("Learning rate scheduling", fontsize=14)
plt.xlabel("Training epochs", fontsize=12)
plt.ylabel("Learning rate", fontsize=12)
plt.plot(
  np.array(train.history["t_lr"]),
  lw=1.5, color="#3288bd", label="transformer"
)
plt.plot(
  np.array(train.history["d_lr"]),
  lw=1.5, color="#fc8d59", label="discriminator"
)
plt.yscale("log")
plt.legend(loc="upper right", fontsize=10)
if args.saving:
  plt.savefig(fname=f"{export_img_fname}/lr-sched.png")
report.add_figure(options="width=45%")
plt.close()

#### Accuracy curves
plt.figure(figsize=(8,5), dpi=100)
plt.title("Metric curves", fontsize=14)
plt.xlabel("Training epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.plot(
  np.array(train.history["accuracy"]),
  lw=1.5, color="#4dac26", label="training set"
)
if TRAIN_RATIO != 1.0:
  plt.plot(
    np.array(train.history["val_accuracy"]),
    lw=1.5, color="#d01c8b", label="validation set"
  )
plt.legend(loc="upper right", fontsize=10)
if args.saving:
  plt.savefig(fname=f"{export_img_fname}/metric-curves-0.png")
report.add_figure(options="width=45%")
plt.close()

#### BCE curves
plt.figure(figsize=(8,5), dpi=100)
plt.title("Metric curves", fontsize=14)
plt.xlabel("Training epochs", fontsize=12)
plt.ylabel("Binary cross-entropy", fontsize=12)
plt.plot(
  np.array(train.history["bce"]),
  lw=1.5, color="#4dac26", label="training set"
)
if TRAIN_RATIO != 1.0:
  plt.plot(
    np.array(train.history["val_bce"]),
    lw=1.5, color="#d01c8b", label="validation set"
  )
plt.legend(loc="upper right", fontsize=10)
if args.saving:
  plt.savefig(fname=f"{export_img_fname}/metric-curves-1.png")
report.add_figure(options="width=45%")
plt.close()

report.add_markdown("---")

## Validation plots
report.add_markdown('<h2 align="center">Validation plots</h2>')

#### X coordinate
plt.figure(figsize=(8,5), dpi=100)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("Candidates", fontsize=12)
x_min = Y[:,:,0].numpy().flatten().min()
x_max = Y[:,:,0].numpy().flatten().max()
bins = np.linspace(x_min, x_max, 101)
plt.hist(
  Y[:,:,0].numpy().flatten(),
  bins=bins, color="#3288bd",
  label="Training data"
)
plt.hist(
  out[:,:,0].numpy().flatten(),
  bins=bins, histtype="step", color="#fc8d59",
  lw=2, label="Calotron output"
)
plt.yscale("log")
plt.legend(loc="upper left", fontsize=10)
if args.saving:
  plt.savefig(f"{export_img_fname}/x-coord.png")
report.add_figure(options="width=45%")
plt.close()

#### Y coordinate
plt.figure(figsize=(8,5), dpi=100)
plt.xlabel("$y$ coordinate", fontsize=12)
plt.ylabel("Candidates", fontsize=12)
y_min = Y[:,:,1].numpy().flatten().min()
y_max = Y[:,:,1].numpy().flatten().max()
bins = np.linspace(y_min, y_max, 101)
plt.hist(
  Y[:,:,1].numpy().flatten(), 
  bins=bins, color="#3288bd",
  label="Training data"
)
plt.hist(
  out[:,:,1].numpy().flatten(),
  bins=bins, histtype="step", color="#fc8d59",
  lw=2, label="Calotron output"
)
plt.yscale("log")
plt.legend(loc="upper left", fontsize=10)
if args.saving:
  plt.savefig(f"{export_img_fname}/y-coord.png")
report.add_figure(options="width=45%")
plt.close()

#### X-Y correlations
plt.figure(figsize=(8,5), dpi=100)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("$y$ coordinate", fontsize=12)
plt.scatter(
  Y[:,:,0].numpy().flatten(),
  Y[:,:,1].numpy().flatten(), 
  s=0.75, alpha=0.2, color="#3288bd",
  label="Training data"
)
plt.scatter(
  out[:,:,0].numpy().flatten(),
  out[:,:,1].numpy().flatten(),
  s=0.75, alpha=0.2, color="#fc8d59",
  label="Calotron output"
)
# plt.legend(loc="upper left", fontsize=10)
if args.saving:
  plt.savefig(f"{export_img_fname}/x-y-corr.png")
report.add_figure(options="width=45%")
plt.close()

#### Energy
plt.figure(figsize=(8,5), dpi=100)
plt.xlabel("Preprocessed energy [a.u]", fontsize=12)
plt.ylabel("Candidates", fontsize=12)
e_min = Y[:,:,2].numpy().flatten().min()
e_max = Y[:,:,2].numpy().flatten().max()
bins = np.linspace(e_min, e_max, 101)
plt.hist(
  Y[:,:,2].numpy().flatten(),
  bins=bins, color="#3288bd",
  label="Training data"
)
plt.hist(
  out[:,:,2].numpy().flatten(),
  bins=bins, histtype="step", color="#fc8d59",
  lw=2, label="Calotron output"
)
plt.yscale("log")
plt.legend(loc="upper left", fontsize=10)
if args.saving:
  plt.savefig(f"{export_img_fname}/energy.png")
report.add_figure(options="width=45%")
plt.close()

#### X-energy correlations
plt.figure(figsize=(8,5), dpi=100)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("Preprocessed energy [a.u]", fontsize=12)
plt.scatter(
  Y[:,:,0].numpy().flatten(),
  Y[:,:,2].numpy().flatten(), 
  s=0.75, alpha=0.2, color="#3288bd",
  label="Training data"
)
plt.scatter(
  out[:,:,0].numpy().flatten(),
  out[:,:,2].numpy().flatten(),
  s=0.75, alpha=0.2, color="#fc8d59",
  label="Calotron output"
)
# plt.legend(loc="upper left", fontsize=10)
if args.saving:
  plt.savefig(f"{export_img_fname}/x-energy-corr.png")
report.add_figure(options="width=45%")
plt.close()

#### Y-energy correlations
plt.figure(figsize=(8,5), dpi=100)
plt.xlabel("$y$ coordinate", fontsize=12)
plt.ylabel("Preprocessed energy [a.u]", fontsize=12)
plt.scatter(
  Y[:,:,1].numpy().flatten(),
  Y[:,:,2].numpy().flatten(), 
  s=0.75, alpha=0.2, color="#3288bd",
  label="Training data"
)
plt.scatter(
  out[:,:,1].numpy().flatten(),
  out[:,:,2].numpy().flatten(),
  s=0.75, alpha=0.2, color="#fc8d59",
  label="Calotron output"
)
# plt.legend(loc="upper left", fontsize=10)
if args.saving:
  plt.savefig(f"{export_img_fname}/y-energy-corr.png")
report.add_figure(options="width=45%")
plt.close()

#### Photon/cluster coordinates
for i in range(4):
  evt = int(np.random.uniform(0, chunk_size))
  plt.figure(figsize=(8,5), dpi=100)
  plt.xlabel("$x$ coordinate", fontsize=12)
  plt.ylabel("$y$ coordinate", fontsize=12)
  plt.scatter(
    X[evt,:,0].numpy().flatten(),
    X[evt,:,1].numpy().flatten(),
    s=50 * X[evt,:,2].numpy().flatten() / Y[evt,:,2].numpy().flatten().max(),
    marker="o", facecolors="none", edgecolors="#d7191c",
    lw=0.75, label="True photon"
  )
  plt.scatter(
    Y[evt,:,0].numpy().flatten(),
    Y[evt,:,1].numpy().flatten(),
    s=50 * Y[evt,:,2].numpy().flatten() / Y[evt,:,2].numpy().flatten().max(),
    marker="s", facecolors="none", edgecolors="#2b83ba", 
    lw=0.75, label="Calo neutral cluster"
  )
  plt.scatter(
    out[evt,:,0].numpy().flatten(),
    out[evt,:,1].numpy().flatten(),
    s=50 * out[evt,:,2].numpy().flatten() / Y[evt,:,2].numpy().flatten().max(),
    marker="^", facecolors="none", edgecolors="#1a9641",
    lw=0.75, label="Calotron output"
  )
  plt.legend()
  if args.saving:
    plt.savefig(f"{export_img_fname}/photon-cluster-coord-{i}.png")
  report.add_figure(options="width=45%")
plt.close()

#### Energy matrix
plt.figure(figsize=(16,10), dpi=100)
plt.subplot(1,2,1)
plt.title("Training data", fontsize=14)
plt.xlabel("Cluster energy deposits", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(
  Y[:128,:,2].numpy(),
  aspect="auto", interpolation="none"
)
plt.subplot(1,2,2)
plt.title("Calotron output", fontsize=14)
plt.xlabel("Cluster energy deposits", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(
  out[:128,:,2].numpy(),
  aspect="auto", interpolation="none"
)
if args.saving:
  plt.savefig(f"{export_img_fname}/energy-matrix.png")
report.add_figure(options="width=95%")
plt.close()

report.add_markdown("---")

report_fname = f"{report_dir}/{prefix}_train-report.html"
report.write_report(filename=report_fname)
print(f"[INFO] Training report correctly exported to {report_fname}")
