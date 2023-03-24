import os
import socket
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import tensorflow as tf
import yaml
from html_reports import Report
from sklearn.utils import shuffle
from utils import (
    calorimeter_deposits,
    energy_sequences,
    event_example,
    learn_rate_scheduling,
    learning_curves,
    metric_curves,
    validation_histogram,
)

from calotron.callbacks.schedulers import ExponentialDecay
from calotron.losses import PrimaryPhotonMatch
from calotron.models import AuxClassifier, Calotron, Discriminator, Transformer
from calotron.simulators import ExportSimulator, Simulator
from calotron.utils import getSummaryHTML, initHPSingleton

DTYPE = np.float32
TRAIN_RATIO = 0.7
BATCHSIZE = 128
ALPHA = 0.2
BETA = 0.5
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

npzfile = np.load(f"{data_dir}/calo-train-data-demo.npz")
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
    output_activations=hp.get("t_output_activations", ["tanh", "tanh", "sigmoid"]),
    start_token_initializer=hp.get("t_start_toke_initializer", "ones"),
    dtype=DTYPE,
)

discriminator = Discriminator(
    latent_dim=hp.get("d_latent_dim", 128),
    output_units=hp.get("d_output_units", 1),
    output_activation=hp.get("d_output_activation", "sigmoid"),
    deepsets_num_layers=hp.get("d_deepsets_num_layers", 5),
    deepsets_hidden_units=hp.get("d_deepsets_hidden_units", 256),
    dropout_rate=hp.get("d_dropout_rate", 0.1),
    dtype=DTYPE,
)

aux_classifier = AuxClassifier(
    transformer=transformer,
    output_depth=hp.get("a_output_depth", 1),
    output_activation=hp.get("a_output_activation", "sigmoid"),
    dropout_rate=hp.get("a_dropout", 0.1),
    dtype=DTYPE,
)

model = Calotron(
    transformer=transformer, discriminator=discriminator, aux_classifier=aux_classifier
)

output = model((photon[:BATCHSIZE], cluster[:BATCHSIZE]))
model.summary()

# +----------------------+
# |   Optimizers setup   |
# +----------------------+

t_lr0 = 5e-4
t_opt = tf.keras.optimizers.RMSprop(t_lr0)
hp.get("t_optimizer", "RMSprop")
hp.get("t_lr0", t_lr0)

d_lr0 = 1e-4
d_opt = tf.keras.optimizers.RMSprop(d_lr0)
hp.get("d_optimizer", "RMSprop")
hp.get("d_lr0", d_lr0)

a_lr0 = 1e-3
a_opt = tf.keras.optimizers.RMSprop(a_lr0)
hp.get("a_optimizer", "RMSprop")
hp.get("a_lr0", a_lr0)

# +----------------------------+
# |   Training configuration   |
# +----------------------------+

loss = PrimaryPhotonMatch(alpha=ALPHA, beta=BETA, label_smoothing=0.1)
hp.get("loss", loss.name)
hp.get("loss_alpha", loss.alpha)
hp.get("loss_beta", loss.beta)
hp.get("loss_label_smoothing", loss.label_smoothing)

model.compile(
    loss=loss,
    metrics=hp.get("metrics", ["accuracy", "bce"]),
    transformer_optimizer=t_opt,
    discriminator_optimizer=d_opt,
    aux_classifier_optimizer=a_opt,
    transformer_upds_per_batch=hp.get("transformer_upds_per_batch", 1),
    discriminator_upds_per_batch=hp.get("discriminator_upds_per_batch", 1),
    aux_classifier_upds_per_batch=hp.get("aux_classifier_upds_per_batch", 1),
)

# +------------------------------+
# |   Learning rate scheduling   |
# +------------------------------+

t_decay_rate = 0.10
t_decay_steps = 200_000
t_sched = ExponentialDecay(
    model.transformer_optimizer, decay_rate=t_decay_rate, decay_steps=t_decay_steps
)
hp.get("t_sched", "ExponentialDecay")
hp.get("t_decay_rate", t_decay_rate)
hp.get("t_decay_steps", t_decay_steps)

d_decay_rate = 0.10
d_decay_steps = 350_000
d_sched = ExponentialDecay(
    model.discriminator_optimizer, decay_rate=d_decay_rate, decay_steps=d_decay_steps
)
hp.get("d_sched", "ExponentialDecay")
hp.get("d_decay_rate", d_decay_rate)
hp.get("d_decay_steps", d_decay_steps)

a_decay_rate = 0.10
a_decay_steps = 150_000
a_sched = ExponentialDecay(
    model.aux_classifier_optimizer, decay_rate=a_decay_rate, decay_steps=a_decay_steps
)
hp.get("a_sched", "ExponentialDecay")
hp.get("a_decay_rate", a_decay_rate)
hp.get("a_decay_steps", a_decay_steps)

# +------------------------+
# |   Training procedure   |
# +------------------------+

start = datetime.now()
train = model.fit(
    train_ds,
    epochs=hp.get("epochs", EPOCHS),
    validation_data=val_ds,
    callbacks=[t_sched, d_sched, a_sched],
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
prefix += "_calotron_aux"

export_model_fname = f"{models_dir}/{prefix}_model"
export_img_dirname = f"{images_dir}/{prefix}_img"

if args.saving:
    tf.saved_model.save(exp_sim, models_dir=export_model_fname)
    hp.dump(f"{export_model_fname}/hyperparams.yml")  # export also list of hyperparams
    print(f"[INFO] Trained model correctly exported to {export_model_fname}")
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

## Auxiliary Classifier architecture
report.add_markdown('<h2 align="center">Auxiliary Classifier architecture</h2>')
html_table, num_params = getSummaryHTML(model.aux_classifier)
report.add_markdown(html_table)
report.add_markdown(f"**Total params:** {num_params}")

report.add_markdown("---")

## Training plots
report.add_markdown('<h2 align="center">Training plots</h2>')

start_epoch = int(EPOCHS / 20)

#### Learning curves
learning_curves(
    report=report,
    history=train,
    start_epoch=start_epoch,
    keys=["t_loss", "d_loss", "a_loss"],
    colors=["#3288bd", "#fc8d59", "#4dac26"],
    labels=["transformer", "discriminator", "aux-classifier"],
    legend_loc="upper right",
    save_figure=args.saving,
    scale_curves=True,
    export_fname=f"{export_img_dirname}/learn-curves.png",
)

#### Learning rate scheduling
learn_rate_scheduling(
    report=report,
    history=train,
    start_epoch=0,
    keys=["t_lr", "d_lr", "a_lr"],
    colors=["#3288bd", "#fc8d59", "#4dac26"],
    labels=["transformer", "discriminator", "aux-classifier"],
    legend_loc="upper right",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/lr-sched.png",
)

#### Accuracy curves
metric_curves(
    report=report,
    history=train,
    start_epoch=start_epoch,
    key="accuracy",
    ylabel="Accuracy",
    validation_set=(TRAIN_RATIO != 1.0),
    colors=["#d01c8b", "#4dac26"],
    labels=["training set", "validation set"],
    legend_loc="upper left",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/accuracy-curves.png",
)

#### BCE curves
metric_curves(
    report=report,
    history=train,
    start_epoch=start_epoch,
    key="bce",
    ylabel="Binary cross-entropy",
    validation_set=(TRAIN_RATIO != 1.0),
    colors=["#d01c8b", "#4dac26"],
    labels=["training set", "validation set"],
    legend_loc="upper right",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/bce-curves.png",
)

report.add_markdown("---")

## Validation plots
report.add_markdown('<h2 align="center">Validation plots</h2>')

#### X histogram
validation_histogram(
    report=report,
    ref_data=cluster_val[:, :, 0].flatten(),
    gen_data=output[:, :, 0].flatten(),
    scaler=None,
    xlabel="$x$ coordinate",
    ref_label="Training data",
    gen_label="Calotron output",
    legend_loc="upper right",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/x-hist.png",
)

#### Y histogram
validation_histogram(
    report=report,
    ref_data=cluster_val[:, :, 1].flatten(),
    gen_data=output[:, :, 1].flatten(),
    scaler=None,
    xlabel="$y$ coordinate",
    ref_label="Training data",
    gen_label="Calotron output",
    legend_loc="upper right",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/y-hist.png",
)

#### Calorimeter deposits
calorimeter_deposits(
    report=report,
    ref_coords=(cluster_val[:, :, 0].flatten(), cluster_val[:, :, 1].flatten()),
    gen_coords=(output[:, :, 0].flatten(), output[:, :, 1].flatten()),
    ref_energy=cluster_val[:, :, 2].flatten(),
    gen_energy=output[:, :, 2].flatten(),
    scaler=None,
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/calo-deposits.png",
)

#### Event examples
for i in range(4):
    evt = int(np.random.uniform(0, len(cluster_val)))
    event_example(
        report=report,
        photon_coords=(
            photon_val[evt, :, 0].flatten(),
            photon_val[evt, :, 1].flatten(),
        ),
        cluster_coords=(
            cluster_val[evt, :, 0].flatten(),
            cluster_val[evt, :, 1].flatten(),
        ),
        output_coords=(output[evt, :, 0].flatten(), output[evt, :, 1].flatten()),
        photon_energy=photon_val[evt, :, 2].flatten(),
        cluster_energy=cluster_val[evt, :, 2].flatten(),
        output_energy=output[evt, :, 2].flatten(),
        photon_scaler=None,
        cluster_scaler=None,
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/evt-example-{i}.png",
    )

#### Energy histogram
validation_histogram(
    report=report,
    ref_data=cluster_val[:, :, 2].flatten(),
    gen_data=output[:, :, 2].flatten(),
    scaler=None,
    xlabel="Preprocessed energy [a.u]",
    ref_label="Training data",
    gen_label="Calotron output",
    legend_loc="upper right",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/energy-hist.png",
)

#### Energy sequences
energy_sequences(
    report=report,
    ref_energy=cluster_val[:128, :, 2],
    gen_energy=output[:128, :, 2],
    scaler=None,
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/energy-seq.png",
)

report.add_markdown("---")

report_fname = f"{reports_dir}/{prefix}_train-report.html"
report.write_report(filename=report_fname)
print(f"[INFO] Training report correctly exported to {report_fname}")
