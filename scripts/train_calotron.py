import os
import socket
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from html_reports import Report
from sklearn.utils import shuffle

from calotron.callbacks.schedulers import ExponentialDecay
from calotron.losses import PrimaryPhotonMatch
from calotron.models import Calotron, Discriminator, Transformer
from calotron.simulators import ExportSimulator, Simulator
from calotron.utils import getSummaryHTML, initHPSingleton

DTYPE = np.float32
TRAIN_RATIO = 0.7
BATCHSIZE = 128
ALPHA = 0.2
BETA = 0.75
EPOCHS = 500

# +------------------+
# |   Parser setup   |
# +------------------+

parser = ArgumentParser(description="calotron training setup")

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
models_dir = config_dir["models_dir"]
images_dir = config_dir["images_dir"]
reports_dir = config_dir["reports_dir"]

# +------------------+
# |   Data loading   |
# +------------------+

npzfile = np.load(f"{data_dir}/train-data-demo.npz")
photon = npzfile["photon"].astype(DTYPE)
cluster = npzfile["cluster"].astype(DTYPE)

print(f"[INFO] Generated photons - shape: {photon.shape}")
print(f"[INFO] Reconstructed clusters - shape: {cluster.shape}")

photon, cluster = shuffle(photon, cluster)

chunk_size = photon.shape[0]
train_size = int(TRAIN_RATIO * chunk_size)

# +-------------------------+
# |   Dataset preparation   |
# +-------------------------+

photon_train = photon[:train_size]
cluster_train = cluster[:train_size]
train_ds = (
    tf.data.Dataset.from_tensor_slices((photon_train, cluster_train))
    .batch(hp.get("batch_size", BATCHSIZE), drop_remainder=True)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

if TRAIN_RATIO != 1.0:
    photon_val = photon[train_size:]
    cluster_val = cluster[train_size:]
    val_ds = (
        tf.data.Dataset.from_tensor_slices((photon_val, cluster_val))
        .batch(BATCHSIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
else:
    photon_val = photon_train
    cluster_val = cluster_train
    val_ds = None

# +------------------------+
# |   Model construction   |
# +------------------------+

transformer = Transformer(
    output_depth=hp.get("t_output_depth", cluster.shape[2]),
    encoder_depth=hp.get("t_encoder_depth", 32),
    decoder_depth=hp.get("t_decoder_depth", 32),
    num_layers=hp.get("t_num_layers", 5),
    num_heads=hp.get("t_num_heads", 4),
    key_dims=hp.get("t_key_dims", 64),
    fnn_units=hp.get("t_fnn_units", 128),
    dropout_rates=hp.get("t_dropout_rates", 0.1),
    seq_ord_latent_dims=hp.get("t_seq_ord_latent_dims", 16),
    seq_ord_max_lengths=hp.get(
        "t_seq_ord_max_lengths", [photon.shape[1], cluster.shape[1]]
    ),
    seq_ord_normalizations=hp.get("t_seq_ord_normalizations", 10_000),
    residual_smoothing=hp.get("t_residual_smoothing", True),
    output_activations=hp.get("t_output_activations", ["linear", "linear", "sigmoid"]),
    start_token_initializer=hp.get("t_start_toke_initializer", "ones"),
    dtype=DTYPE,
)

discriminator = Discriminator(
    latent_dim=hp.get("d_latent_dim", 64),
    output_units=hp.get("d_output_units", 1),
    output_activation=hp.get("d_output_activation", "sigmoid"),
    deepsets_num_layers=hp.get("d_deepsets_num_layers", 5),
    deepsets_hidden_units=hp.get("d_deepsets_hidden_units", 256),
    dropout_rate=hp.get("d_dropout_rate", 0.1),
    dtype=DTYPE,
)

model = Calotron(transformer=transformer, discriminator=discriminator)

output = model((photon, cluster))
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

loss = PrimaryPhotonMatch(alpha=ALPHA, beta=BETA)
hp.get("loss", loss.name)
hp.get("loss_alpha", loss.alpha)
hp.get("loss_beta", loss.beta)

model.compile(
    loss=loss,
    metrics=hp.get("metrics", ["accuracy", "bce"]),
    transformer_optimizer=t_opt,
    discriminator_optimizer=d_opt,
    transformer_upds_per_batch=hp.get("transformer_upds_per_batch", 1),
    discriminator_upds_per_batch=hp.get("discriminator_upds_per_batch", 1),
)

# +------------------------------+
# |   Learning rate scheduling   |
# +------------------------------+

t_decay_rate = 0.10
t_decay_steps = 350_000
t_sched = ExponentialDecay(
    model.transformer_optimizer, decay_rate=t_decay_rate, decay_steps=t_decay_steps
)
hp.get("t_sched", "ExponentialDecay")
hp.get("t_decay_rate", t_decay_rate)
hp.get("t_decay_steps", t_decay_steps)

d_decay_rate = 0.20
d_decay_steps = 150_000
d_sched = ExponentialDecay(
    model.discriminator_optimizer, decay_rate=d_decay_rate, decay_steps=d_decay_steps
)
hp.get("d_sched", "ExponentialDecay")
hp.get("d_decay_rate", d_decay_rate)
hp.get("d_decay_steps", d_decay_steps)

# +------------------------+
# |   Training procedure   |
# +------------------------+

start = datetime.now()
train = model.fit(
    train_ds,
    epochs=hp.get("epochs", EPOCHS),
    validation_data=val_ds,
    callbacks=[t_sched, d_sched],
)
stop = datetime.now()

duration = str(stop - start).split(".")[0].split(":")  # [HH, MM, SS]
duration = f"{duration[0]}h {duration[1]}min {duration[2]}s"
print(f"[INFO] Model training completed in {duration}")

# +---------------------+
# |   Model inference   |
# +---------------------+

start_token = model.get_start_token(cluster_train)
start_token = np.mean(start_token, axis=0)

sim = Simulator(model.transformer, start_token=start_token)
exp_sim = ExportSimulator(sim, max_length=cluster.shape[1])

dataset = (
    tf.data.Dataset.from_tensor_slices(photon_val)
    .batch(512, drop_remainder=True)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

output = exp_sim(dataset).numpy()
photon_val = photon_val[: len(output)]
cluster_val = cluster_val[: len(output)]

# +------------------+
# |   Model export   |
# +------------------+

timestamp = str(datetime.now())
date, hour = timestamp.split(" ")
date = date.replace("-", "/")
hour = hour.split(".")[0]

prefix = ""
timestamp = timestamp.split(".")[0].replace("-", "").replace(" ", "-")
for time, unit in zip(timestamp.split(":"), ["h", "m", "s"]):
    prefix += time + unit  # YYYYMMDD-HHhMMmSSs
prefix += "_calotron"

if args.saving:
    export_model_fname = f"{models_dir}/{prefix}_model"
    tf.saved_model.save(exp_sim, models_dir=export_model_fname)
    hp.dump(f"{export_model_fname}/hyperparams.yml")  # export also list of hyperparams
    print(f"[INFO] Trained model correctly exported to {export_model_fname}")
    export_img_dirname = f"{images_dir}/{prefix}_img"
    os.makedirs(export_img_dirname)  # need to save images

# +---------------------+
# |   Training report   |
# +---------------------+

report = Report()
report.add_markdown('<h1 align="center">Calotron training report</h1>')

info = [
    f"- Script executed on **{socket.gethostname()}**",
    f"- Model training completed in **{duration}**",
    f"- Report generated on **{date}** at **{hour}**",
]
report.add_markdown("\n".join([i for i in info]))

report.add_markdown("---")

## Hyperparameters and other details
report.add_markdown('<h2 align="center">Hyperparameters and other details</h2>')
hyperparams = ""
for k, v in hp.get_dict().items():
    hyperparams += f"- **{k}:** {v}\n"
report.add_markdown(hyperparams)

report.add_markdown("---")

## Transformer architecture
report.add_markdown('<h2 align="center">Transformer architecture</h2>')
html_table, num_params = getSummaryHTML(model.transformer)
report.add_markdown(html_table)
report.add_markdown(f"**Total params:** {num_params}")

report.add_markdown("---")

## Discriminator architecture
report.add_markdown('<h2 align="center">Discriminator architecture</h2>')
html_table, num_params = getSummaryHTML(model.discriminator)
report.add_markdown(html_table)
report.add_markdown(f"**Total params:** {num_params}")

report.add_markdown("---")

## Training plots
report.add_markdown('<h2 align="center">Training plots</h2>')

start_epoch = int(EPOCHS / 20)

#### Learning curves
plt.figure(figsize=(8, 5), dpi=100)
plt.title("Learning curves", fontsize=14)
plt.xlabel("Training epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
ratio = (
    np.array(train.history["d_loss"])[start_epoch:]
    / np.array(train.history["t_loss"])[start_epoch:]
)
ratio = int(np.mean(ratio)) + 1
plt.plot(
    np.arange(EPOCHS)[start_epoch:],
    np.array(train.history["t_loss"])[start_epoch:],
    lw=1.5,
    color="#3288bd",
    label=f"transformer [x {1:.1f}]",
)
plt.plot(
    np.arange(EPOCHS)[start_epoch:],
    np.array(train.history["d_loss"])[start_epoch:] / ratio,
    lw=1.5,
    color="#fc8d59",
    label=f"discriminator [x {1/ratio:.1f}]",
)
# plt.yscale("log")
plt.legend(loc="upper right", fontsize=10)
if args.saving:
    plt.savefig(fname=f"{export_img_dirname}/learn-curves.png")
report.add_figure(options="width=45%")
plt.close()

#### Learning rate scheduling
plt.figure(figsize=(8, 5), dpi=100)
plt.title("Learning rate scheduling", fontsize=14)
plt.xlabel("Training epochs", fontsize=12)
plt.ylabel("Learning rate", fontsize=12)
plt.plot(
    np.arange(EPOCHS),
    np.array(train.history["t_lr"]),
    lw=1.5,
    color="#3288bd",
    label="transformer",
)
plt.plot(
    np.arange(EPOCHS),
    np.array(train.history["d_lr"]),
    lw=1.5,
    color="#fc8d59",
    label="discriminator",
)
plt.yscale("log")
plt.legend(loc="upper right", fontsize=10)
if args.saving:
    plt.savefig(fname=f"{export_img_dirname}/lr-sched.png")
report.add_figure(options="width=45%")
plt.close()

#### Accuracy curves
plt.figure(figsize=(8, 5), dpi=100)
plt.title("Metric curves", fontsize=14)
plt.xlabel("Training epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.plot(
    np.arange(EPOCHS)[start_epoch:],
    np.array(train.history["accuracy"])[start_epoch:],
    lw=1.5,
    color="#4dac26",
    label="training set",
)
if TRAIN_RATIO != 1.0:
    plt.plot(
        np.arange(EPOCHS)[start_epoch:],
        np.array(train.history["val_accuracy"])[start_epoch:],
        lw=1.5,
        color="#d01c8b",
        label="validation set",
    )
plt.legend(loc="upper left", fontsize=10)
if args.saving:
    plt.savefig(fname=f"{export_img_dirname}/metric-curves-0.png")
report.add_figure(options="width=45%")
plt.close()

#### BCE curves
plt.figure(figsize=(8, 5), dpi=100)
plt.title("Metric curves", fontsize=14)
plt.xlabel("Training epochs", fontsize=12)
plt.ylabel("Binary cross-entropy", fontsize=12)
plt.plot(
    np.arange(EPOCHS)[start_epoch:],
    np.array(train.history["bce"])[start_epoch:],
    lw=1.5,
    color="#4dac26",
    label="training set",
)
if TRAIN_RATIO != 1.0:
    plt.plot(
        np.arange(EPOCHS)[start_epoch:],
        np.array(train.history["val_bce"])[start_epoch:],
        lw=1.5,
        color="#d01c8b",
        label="validation set",
    )
plt.legend(loc="upper right", fontsize=10)
if args.saving:
    plt.savefig(fname=f"{export_img_dirname}/metric-curves-1.png")
report.add_figure(options="width=45%")
plt.close()

report.add_markdown("---")

## Validation plots
report.add_markdown('<h2 align="center">Validation plots</h2>')

#### X histogram
plt.figure(figsize=(8, 5), dpi=100)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("Candidates", fontsize=12)
x_min = cluster_val[:, :, 0].flatten().min()
x_max = cluster_val[:, :, 0].flatten().max()
x_bins = np.linspace(x_min, x_max, 101)
plt.hist(
    cluster_val[:, :, 0].flatten(), bins=x_bins, color="#3288bd", label="Training data"
)
plt.hist(
    output[:, :, 0].flatten(),
    bins=x_bins,
    histtype="step",
    color="#fc8d59",
    lw=2,
    label="Calotron output",
)
plt.yscale("log")
plt.legend(loc="upper left", fontsize=10)
if args.saving:
    plt.savefig(f"{export_img_dirname}/x-hist.png")
report.add_figure(options="width=45%")
plt.close()

#### Y histogram
plt.figure(figsize=(8, 5), dpi=100)
plt.xlabel("$y$ coordinate", fontsize=12)
plt.ylabel("Candidates", fontsize=12)
y_min = cluster_val[:, :, 1].flatten().min()
y_max = cluster_val[:, :, 1].flatten().max()
y_bins = np.linspace(y_min, y_max, 101)
plt.hist(
    cluster_val[:, :, 1].flatten(), bins=y_bins, color="#3288bd", label="Training data"
)
plt.hist(
    output[:, :, 1].flatten(),
    bins=y_bins,
    histtype="step",
    color="#fc8d59",
    lw=2,
    label="Calotron output",
)
plt.yscale("log")
plt.legend(loc="upper left", fontsize=10)
if args.saving:
    plt.savefig(f"{export_img_dirname}/y-hist.png")
report.add_figure(options="width=45%")
plt.close()

#### XY 2D-histogram
plt.figure(figsize=(16, 5), dpi=100)
plt.subplot(1, 2, 1)
plt.title("Training data", fontsize=14)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("$y$ coordinate", fontsize=12)
plt.hist2d(
    cluster_val[:, :, 0].flatten(),
    cluster_val[:, :, 1].flatten(),
    weights=cluster_val[:, :, 2].flatten(),
    bins=(x_bins, y_bins),
    cmin=0,
    cmap="gist_heat",
)
plt.subplot(1, 2, 2)
plt.title("Calotron output", fontsize=14)
plt.xlabel("$x$ coordinate", fontsize=12)
plt.ylabel("$y$ coordinate", fontsize=12)
plt.hist2d(
    output[:, :, 0].flatten(),
    output[:, :, 1].flatten(),
    weights=output[:, :, 2].flatten(),
    bins=(x_bins, y_bins),
    cmin=0,
    cmap="gist_heat",
)
if args.saving:
    plt.savefig(f"{export_img_dirname}/xy-hist2d.png")
report.add_figure(options="width=95%")
plt.close()

#### Event examples
for i in range(4):
    evt = int(np.random.uniform(0, len(cluster_val)))
    plt.figure(figsize=(8, 6), dpi=100)
    plt.xlabel("$x$ coordinate", fontsize=12)
    plt.ylabel("$y$ coordinate", fontsize=12)
    plt.scatter(
        photon_val[evt, :, 0].flatten(),
        photon_val[evt, :, 1].flatten(),
        s=50 * photon_val[evt, :, 2].flatten() / cluster_val[evt, :, 2].flatten().max(),
        marker="o",
        facecolors="none",
        edgecolors="#d7191c",
        lw=0.75,
        label="True photon",
    )
    plt.scatter(
        cluster_val[evt, :, 0].flatten(),
        cluster_val[evt, :, 1].flatten(),
        s=50
        * cluster_val[evt, :, 2].flatten()
        / cluster_val[evt, :, 2].flatten().max(),
        marker="s",
        facecolors="none",
        edgecolors="#2b83ba",
        lw=0.75,
        label="Calo neutral cluster",
    )
    plt.scatter(
        output[evt, :, 0].flatten(),
        output[evt, :, 1].flatten(),
        s=50 * output[evt, :, 2].flatten() / cluster_val[evt, :, 2].flatten().max(),
        marker="^",
        facecolors="none",
        edgecolors="#1a9641",
        lw=0.75,
        label="Calotron output",
    )
    plt.legend()
    if args.saving:
        plt.savefig(f"{export_img_dirname}/evt-example-{i}.png")
    report.add_figure(options="width=45%")
plt.close()

#### Energy histogram
plt.figure(figsize=(8, 5), dpi=100)
plt.xlabel("Preprocessed energy [a.u]", fontsize=12)
plt.ylabel("Candidates", fontsize=12)
e_min = cluster_val[:, :, 2].flatten().min()
e_max = cluster_val[:, :, 2].flatten().max()
e_bins = np.linspace(e_min, e_max, 101)
plt.hist(
    cluster_val[:, :, 2].flatten(), bins=e_bins, color="#3288bd", label="Training data"
)
plt.hist(
    output[:, :, 2].flatten(),
    bins=e_bins,
    histtype="step",
    color="#fc8d59",
    lw=2,
    label="Calotron output",
)
plt.yscale("log")
plt.legend(loc="upper left", fontsize=10)
if args.saving:
    plt.savefig(f"{export_img_dirname}/energy-hist.png")
report.add_figure(options="width=45%")
plt.close()

#### Energy batches plot
plt.figure(figsize=(16, 10), dpi=100)
plt.subplot(1, 2, 1)
plt.title("Training data", fontsize=14)
plt.xlabel("Cluster energy deposits", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(cluster_val[:128, :, 2], aspect="auto", interpolation="none")
plt.subplot(1, 2, 2)
plt.title("Calotron output", fontsize=14)
plt.xlabel("Cluster energy deposits", fontsize=12)
plt.ylabel("Events", fontsize=12)
plt.imshow(output[:128, :, 2], aspect="auto", interpolation="none")
if args.saving:
    plt.savefig(f"{export_img_dirname}/energy-batches.png")
report.add_figure(options="width=95%")
plt.close()

report.add_markdown("---")

report_fname = f"{reports_dir}/{prefix}_train-report.html"
report.write_report(filename=report_fname)
print(f"[INFO] Training report correctly exported to {report_fname}")
