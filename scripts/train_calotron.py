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
    attention_plot,
    calorimeter_deposits,
    energy_sequences,
    event_example,
    learn_rate_scheduling,
    learning_curves,
    metric_curves,
    photon2cluster_corr,
    validation_histogram,
)

from calotron.callbacks.schedulers import ExponentialDecay
from calotron.losses import PhotonClusterMatch
from calotron.models import Calotron
from calotron.models.discriminators import Discriminator
from calotron.models.transformers import MaskedTransformer
from calotron.simulators import ExportSimulator, Simulator
from calotron.utils import getSummaryHTML, initHPSingleton

VERSION = "v0"
DTYPE = np.float32
ADV_METRIC = "wass"
TRAIN_RATIO = 0.7
BATCHSIZE = 256
EPOCHS = 400
ALPHA = 0.7

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

npzfile = np.load(f"{data_dir}/trainset-demo-{VERSION}.npz")
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

transformer = MaskedTransformer(
    output_depth=hp.get("t_output_depth", cluster.shape[2]),
    encoder_depth=hp.get("t_encoder_depth", 32),
    decoder_depth=hp.get("t_decoder_depth", 32),
    num_layers=hp.get("t_num_layers", 5),
    num_heads=hp.get("t_num_heads", 4),
    key_dims=hp.get("t_key_dims", 64),
    mlp_units=hp.get("t_mlp_units", 128),
    dropout_rates=hp.get("t_dropout_rates", 0.1),
    seq_ord_latent_dims=hp.get("t_seq_ord_latent_dims", 64),
    seq_ord_max_lengths=hp.get(
        "t_seq_ord_max_lengths", [photon.shape[1], cluster.shape[1]]
    ),
    seq_ord_normalizations=hp.get("t_seq_ord_normalizations", 10_000),
    attn_mask_init_nonzero_size=hp.get("t_attn_mask_init_nonzero_size", 8),
    enable_residual_smoothing=hp.get("t_enable_residual_smoothing", True),
    enable_attention_mask=hp.get("t_enable_attention_mask", False),
    output_activations=hp.get("t_output_activations", ["tanh", "tanh", "sigmoid"]),
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

model = Calotron(
    transformer=transformer, discriminator=discriminator, aux_classifier=None
)

output = model((photon[:BATCHSIZE], cluster[:BATCHSIZE]))
model.summary()

# +----------------------+
# |   Optimizers setup   |
# +----------------------+

hp.get("t_optimizer", "RMSprop")
t_opt = tf.keras.optimizers.RMSprop(hp.get("t_lr0", 7e-4))

hp.get("d_optimizer", "RMSprop")
d_opt = tf.keras.optimizers.RMSprop(hp.get("d_lr0", 3e-5))

# +----------------------------+
# |   Training configuration   |
# +----------------------------+

adv_metric = "binary-crossentropy" if ADV_METRIC == "bce" else "wasserstein-distance"
metrics = ["accuracy", "bce"] if ADV_METRIC == "bce" else ["wass_dist"]

hp.get("loss", "PhotonClusterMatch")
loss = PhotonClusterMatch(
    alpha=hp.get("loss_alpha", ALPHA),
    max_match_distance=hp.get("loss_max_match_distance", 1e-3),
    adversarial_metric=hp.get("loss_adversarial_metric", adv_metric),
    bce_options=hp.get(
        "loss_bce_options",
        {
            "injected_noise_stddev": 0.01,
            "label_smoothing": 0.1,
            "ignore_padding": False,
        },
    ),
    wass_options=hp.get(
        "loss_wass_options",
        {
            "lipschitz_penalty": 100.0,
            "virtual_direction_upds": 1,
            "ignore_padding": False,
        },
    ),
)

model.compile(
    loss=loss,
    metrics=hp.get("metrics", metrics),
    transformer_optimizer=t_opt,
    discriminator_optimizer=d_opt,
    aux_classifier_optimizer=None,
    transformer_upds_per_batch=hp.get("transformer_upds_per_batch", 1),
    discriminator_upds_per_batch=hp.get("discriminator_upds_per_batch", 1),
    aux_classifier_upds_per_batch=None,
)

# +--------------------------+
# |   Callbacks definition   |
# +--------------------------+

t_sched = ExponentialDecay(
    model.transformer_optimizer,
    decay_rate=hp.get("t_decay_rate", 0.10),
    decay_steps=hp.get("t_decay_steps", 20_000),
    min_learning_rate=hp.get("t_min_learning_rate", 1e-6),
)
hp.get("t_sched", "ExponentialDecay")

d_sched = ExponentialDecay(
    model.discriminator_optimizer,
    decay_rate=hp.get("d_decay_rate", 0.10),
    decay_steps=hp.get("d_decay_steps", 130_000),
    min_learning_rate=hp.get("d_min_learning_rate", 1e-8),
)
hp.get("d_sched", "ExponentialDecay")

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

output, attn_weights = [exp_out.numpy() for exp_out in exp_sim(dataset)]
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
prefix += f"_calotron_{ADV_METRIC}"

export_model_fname = f"{models_dir}/{prefix}_model"
export_img_dirname = f"{images_dir}/{prefix}_img"

if args.saving:
    tf.saved_model.save(exp_sim, export_dir=export_model_fname)
    hp.dump(f"{export_model_fname}/hyperparams.yml")  # export also list of hyperparams
    print(f"[INFO] Trained model correctly exported to {export_model_fname}")
    os.makedirs(export_img_dirname)  # need to save images

np.savez(
    f"{data_dir}/calotron-export-{VERSION}.npz",
    photon=photon_val,
    cluster=cluster_val,
    output=output,
)

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
report.add_markdown(f"**Model name:** {model.transformer.name}")
html_table, num_params = getSummaryHTML(model.transformer)
report.add_markdown(html_table)
report.add_markdown(f"**Total params:** {num_params}")

report.add_markdown("---")

## Discriminator architecture
report.add_markdown('<h2 align="center">Discriminator architecture</h2>')
report.add_markdown(f"**Model name:** {model.discriminator.name}")
html_table, num_params = getSummaryHTML(model.discriminator)
report.add_markdown(html_table)
report.add_markdown(f"**Total params:** {num_params}")

report.add_markdown("---")

## Training plots
report.add_markdown('<h2 align="center">Training plots</h2>')

start_epoch = int(EPOCHS / 20)

#### Learning curves
learning_curves(
    report=report,
    history=train.history,
    start_epoch=start_epoch,
    keys=["t_loss", "d_loss"],
    colors=["#3288bd", "#fc8d59"],
    labels=["transformer", "discriminator"],
    legend_loc=None,
    save_figure=args.saving,
    scale_curves=True,
    export_fname=f"{export_img_dirname}/learn-curves.png",
)

#### Learning rate scheduling
learn_rate_scheduling(
    report=report,
    history=train.history,
    start_epoch=0,
    keys=["t_lr", "d_lr"],
    colors=["#3288bd", "#fc8d59"],
    labels=["transformer", "discriminator"],
    legend_loc="upper right",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/lr-sched.png",
)

#### Transformer loss
metric_curves(
    report=report,
    history=train.history,
    start_epoch=start_epoch,
    key="t_loss",
    ylabel="Transformer loss",
    title="Learning curves",
    validation_set=(TRAIN_RATIO != 1.0),
    colors=["#d01c8b", "#4dac26"],
    labels=["training set", "validation set"],
    legend_loc=None,
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/transf-loss.png",
)

#### Discriminator loss
metric_curves(
    report=report,
    history=train.history,
    start_epoch=start_epoch,
    key="d_loss",
    ylabel="Discriminator loss",
    title="Learning curves",
    validation_set=(TRAIN_RATIO != 1.0),
    colors=["#d01c8b", "#4dac26"],
    labels=["training set", "validation set"],
    legend_loc=None,
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/disc-loss.png",
)

if ADV_METRIC == "bce":
    #### Accuracy curves
    metric_curves(
        report=report,
        history=train.history,
        start_epoch=start_epoch,
        key="accuracy",
        ylabel="Accuracy",
        title="Metric curves",
        validation_set=(TRAIN_RATIO != 1.0),
        colors=["#d01c8b", "#4dac26"],
        labels=["training set", "validation set"],
        legend_loc=None,
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/accuracy-curves.png",
    )

    #### BCE curves
    metric_curves(
        report=report,
        history=train.history,
        start_epoch=start_epoch,
        key="bce",
        ylabel="Binary cross-entropy",
        title="Metric curves",
        validation_set=(TRAIN_RATIO != 1.0),
        colors=["#d01c8b", "#4dac26"],
        labels=["training set", "validation set"],
        legend_loc=None,
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/bce-curves.png",
    )
else:
    #### Wass curves
    metric_curves(
        report=report,
        history=train.history,
        start_epoch=start_epoch,
        key="wass_dist",
        ylabel="Wasserstein distance",
        title="Metric curves",
        validation_set=(TRAIN_RATIO != 1.0),
        colors=["#d01c8b", "#4dac26"],
        labels=["training set", "validation set"],
        legend_loc=None,
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/wass-curves.png",
    )

report.add_markdown("<br>")

#### Attention weights
for head_id in range(attn_weights.shape[1]):
    attention_plot(
        report=report,
        attn_weights=attn_weights,
        head_id=head_id,
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/attn-plot.png",
    )

report.add_markdown("---")

## Validation plots
report.add_markdown('<h2 align="center">Validation plots</h2>')

photon_xy = np.tile(photon_val[:, None, :, :2], (1, cluster_val.shape[1], 1, 1))
cluster_xy = np.tile(cluster_val[:, :, None, :2], (1, 1, photon_val.shape[1], 1))
pairwise_distance = np.linalg.norm(cluster_xy - photon_xy, axis=-1)
cluster_matched = pairwise_distance.min(axis=-1) <= loss.max_match_distance

photon_not_padded = (photon_val[:, :, 2] > 0.0)
cluster_not_padded = (cluster_val[:, :, 2] > 0.0)
output_not_padded = (output[:, :, 2] > 1e-8)

#### X histogram
cluster_x = cluster_val[:, :, 0][cluster_not_padded].flatten()
output_x = output[:, :, 0][output_not_padded].flatten()
for log_scale in [False, True]:
    validation_histogram(
        report=report,
        ref_data=cluster_x,
        gen_data=output_x,
        xlabel="$x$ coordinate",
        ref_label="Training data",
        gen_label="Calotron output",
        log_scale=log_scale,
        legend_loc="upper right",
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/x-hist.png",
    )

#### X histogram (matched clusters)
cluster_x_matched = cluster_val[:, :, 0][cluster_not_padded & cluster_not_padded].flatten()
output_x_matched = output[:, :, 0][output_not_padded & output_not_padded].flatten()
for log_scale in [False, True]:
    validation_histogram(
        report=report,
        ref_data=cluster_x_matched,
        gen_data=output_x_matched,
        xlabel="$x$ coordinate of matched clusters",
        ref_label="Training data",
        gen_label="Calotron output",
        log_scale=log_scale,
        legend_loc="upper right",
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/x-match-hist.png",
    )

#### Y histogram
cluster_y = cluster_val[:, :, 1][cluster_not_padded].flatten()
output_y = output[:, :, 1][output_not_padded].flatten()
for log_scale in [False, True]:
    validation_histogram(
        report=report,
        ref_data=cluster_y,
        gen_data=output_y,
        xlabel="$y$ coordinate",
        ref_label="Training data",
        gen_label="Calotron output",
        log_scale=log_scale,
        legend_loc="upper right",
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/y-hist.png",
    )

#### Y histogram (matched clusters)
cluster_y_matched = cluster_val[:, :, 1][cluster_not_padded & cluster_not_padded].flatten()
output_y_matched = output[:, :, 1][output_not_padded & output_not_padded].flatten()
for log_scale in [False, True]:
    validation_histogram(
        report=report,
        ref_data=cluster_y_matched,
        gen_data=output_y_matched,
        xlabel="$y$ coordinate of matched clusters",
        ref_label="Training data",
        gen_label="Calotron output",
        log_scale=log_scale,
        legend_loc="upper right",
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/y-match-hist.png",
    )

#### Calorimeter deposits
calorimeter_deposits(
    report=report,
    ref_xy=(cluster_val[:, :, 0].flatten(), cluster_val[:, :, 1].flatten()),
    gen_xy=(output[:, :, 0].flatten(), output[:, :, 1].flatten()),
    ref_energy=cluster_val[:, :, 2].flatten(),
    gen_energy=output[:, :, 2].flatten(),
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/calo-deposits.png",
)

#### Event examples
for i in range(4):
    evt = int(np.random.uniform(0, len(cluster_val)))
    event_example(
        report=report,
        photon_xy=(photon_val[evt, :, 0].flatten(), photon_val[evt, :, 1].flatten()),
        cluster_xy=(cluster_val[evt, :, 0].flatten(), cluster_val[evt, :, 1].flatten()),
        output_xy=(output[evt, :, 0].flatten(), output[evt, :, 1].flatten()),
        photon_energy=photon_val[evt, :, 2].flatten(),
        cluster_energy=cluster_val[evt, :, 2].flatten(),
        output_energy=output[evt, :, 2].flatten(),
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/evt-example-{i}.png",
    )

#### Energy histogram
cluster_energy = cluster_val[:, :, 2][cluster_not_padded].flatten()
output_energy = output[:, :, 2][output_not_padded].flatten()
for log_scale in [False, True]:
    validation_histogram(
        report=report,
        ref_data=cluster_energy,
        gen_data=output_energy,
        xlabel="Preprocessed energy [a.u]",
        ref_label="Training data",
        gen_label="Calotron output",
        log_scale=log_scale,
        legend_loc="upper right",
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/energy-hist.png",
    )

#### Energy histogram (matched clusters)
cluster_energy_matched = cluster_val[:, :, 2][cluster_not_padded & cluster_not_padded].flatten()
output_energy_matched = output[:, :, 2][output_not_padded & output_not_padded].flatten()
for log_scale in [False, True]:
    validation_histogram(
        report=report,
        ref_data=cluster_energy_matched,
        gen_data=output_energy_matched,
        xlabel="Preprocessed energy of matched clusters [a.u]",
        ref_label="Training data",
        gen_label="Calotron output",
        log_scale=log_scale,
        legend_loc="upper right",
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/energy-match-hist.png",
    )

#### Energy sequences
energy_sequences(
    report=report,
    ref_energy=cluster_val[:64, :, 2],
    gen_energy=output[:64, :, 2],
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/energy-seq.png",
)

#### Photon-to-cluster correlations
photon2cluster_corr(
    report,
    photon_energy=(0.5 * (photon_val[:, 0::2, 2] + photon_val[:, 1::2, 2])).flatten(),
    cluster_energy=cluster_val[:, :, 2].flatten(),
    output_energy=output[:, :, 2].flatten(),
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/gamma2calo-corr.png",
)

report.add_markdown("---")

report_fname = f"{reports_dir}/{prefix}_train-report.html"
report.write_report(filename=report_fname)
print(f"[INFO] Training report correctly exported to {report_fname}")
