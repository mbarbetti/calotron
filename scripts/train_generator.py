import os
import socket
from datetime import datetime

import numpy as np
import tensorflow as tf
import yaml
from html_reports import Report
from sklearn.utils import shuffle
from utils_argparser import argparser_training
from utils_training import prepare_training_plots, prepare_validation_plots

from calotron.callbacks.schedulers import LearnRateExpDecay
from calotron.models.transformers import GigaGenerator
from calotron.simulators import ExportSimulator, Simulator
from calotron.utils.reports import getSummaryHTML, initHPSingleton

DTYPE = np.float32
BATCHSIZE = 512
EPOCHS = 100

# +------------------+
# |   Parser setup   |
# +------------------+

parser = argparser_training(
    model="Generator", adv_learning=False, description="Generator training setup"
)
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

chunk_size = int(args.chunk_size)
train_ratio = float(args.train_ratio)

# +------------------+
# |   Data loading   |
# +------------------+

npzfile = np.load(f"{data_dir}/calotron-dataset-demo.npz")

photon = npzfile["photon"].astype(DTYPE)[:chunk_size]
cluster = npzfile["cluster"].astype(DTYPE)[:chunk_size]
weight = npzfile["weight"].astype(DTYPE)[:chunk_size]

print(f"[INFO] Generated photons - shape: {photon.shape}")
print(f"[INFO] Reconstructed clusters - shape: {cluster.shape}")
print(f"[INFO] Matching weights - shape: {weight.shape}")

if not args.weights:
    weight = (weight > 0.0).astype(DTYPE)

photon, cluster, weight = shuffle(photon, cluster, weight)

chunk_size = photon.shape[0]
train_size = int(train_ratio * chunk_size)

# +-------------------------+
# |   Dataset preparation   |
# +-------------------------+

photon_train = photon[:train_size]
cluster_train = cluster[:train_size]
weight_train = weight[:train_size]
train_ds = (
    tf.data.Dataset.from_tensor_slices(
        ((photon_train, cluster_train), cluster_train, weight_train)
    )
    .batch(hp.get("batch_size", BATCHSIZE), drop_remainder=True)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

if train_ratio != 1.0:
    photon_val = photon[train_size:]
    cluster_val = cluster[train_size:]
    weight_val = weight[train_size:]
    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            ((photon_val, cluster_val), cluster_val, weight_val)
        )
        .batch(BATCHSIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
else:
    photon_val = photon_train
    cluster_val = cluster_train
    weight_val = weight_train
    val_ds = None

# +------------------------+
# |   Model construction   |
# +------------------------+

model = GigaGenerator(
    output_depth=hp.get("t_output_depth", cluster.shape[2]),
    encoder_depth=hp.get("t_encoder_depth", 32),
    mapping_latent_dim=hp.get("mapping_latent_dim", 64),
    synthesis_depth=hp.get("t_synthesis_depth", 32),
    num_layers=hp.get("t_num_layers", 5),
    num_heads=hp.get("t_num_heads", 4),
    key_dim=hp.get("t_key_dim", 64),
    admin_res_scale=hp.get("t_admin_res_scale", "O(n)"),
    mlp_units=hp.get("t_mlp_units", 128),
    dropout_rate=hp.get("t_dropout_rate", 0.1),
    seq_ord_latent_dim=hp.get("t_seq_ord_latent_dim", 64),
    seq_ord_max_length=hp.get(
        "t_seq_ord_max_length", max(photon.shape[1], cluster.shape[1])
    ),
    seq_ord_normalization=hp.get("t_seq_ord_normalization", 10_000),
    enable_res_smoothing=hp.get("t_enable_res_smoothing", True),
    output_activations=hp.get("t_output_activations", ["tanh", "tanh", "sigmoid"]),
    start_token_initializer=hp.get("t_start_toke_initializer", "ones"),
    dtype=DTYPE,
)

output = model((photon[:BATCHSIZE], cluster[:BATCHSIZE]))
model.summary()

# +----------------------+
# |   Optimizers setup   |
# +----------------------+

opt = tf.keras.optimizers.Adam(learning_rate=hp.get("lr0", 1e-4))
hp.get("optimizer", opt.name)

# +----------------------------+
# |   Training configuration   |
# +----------------------------+

loss = tf.keras.losses.MeanSquaredError()
hp.get("loss", loss.name)

metrics = hp.get("metrics", ["mae"])

model.compile(loss=loss, optimizer=opt, weighted_metrics=metrics)

# +--------------------------+
# |   Callbacks definition   |
# +--------------------------+

callbacks = list()

lr_sched = LearnRateExpDecay(
    model.optimizer,
    decay_rate=hp.get("decay_rate", 0.10),
    decay_steps=hp.get("decay_steps", 100_000),
    min_learning_rate=hp.get("min_learning_rate", 1e-6),
    verbose=True,
)
hp.get("lr_sched", lr_sched.name)
callbacks.append(lr_sched)

# +------------------------+
# |   Training procedure   |
# +------------------------+

start = datetime.now()
train = model.fit(
    train_ds,
    epochs=hp.get("epochs", EPOCHS),
    validation_data=val_ds,
    callbacks=callbacks,
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

sim = Simulator(model, start_token=start_token)
exp_sim = ExportSimulator(sim, max_length=cluster.shape[1])

dataset = (
    tf.data.Dataset.from_tensor_slices(photon_val)
    .batch(BATCHSIZE, drop_remainder=True)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

output, attn_weights = [exp_out.numpy() for exp_out in exp_sim(dataset)]

photon_val = photon_val[: len(output)]
cluster_val = cluster_val[: len(output)]
weight_val = weight_val[: len(output)]

# +------------------+
# |   Model export   |
# +------------------+

timestamp = str(datetime.now())
date, hour = timestamp.split(" ")
date = date.replace("-", "/")
hour = hour.split(".")[0]

if args.test:
    prefix = "test"
else:
    prefix = ""
    timestamp = timestamp.split(".")[0].replace("-", "").replace(" ", "-")
    for time, unit in zip(timestamp.split(":"), ["h", "m", "s"]):
        prefix += time + unit  # YYYYMMDD-HHhMMmSSs
prefix += f"_generator"

export_model_fname = f"{models_dir}/{prefix}_model"
export_img_dirname = f"{images_dir}/{prefix}_img"

if args.saving:
    tf.saved_model.save(exp_sim, export_dir=export_model_fname)
    print(f"[INFO] Trained model correctly exported to {export_model_fname}")
    hp.dump(f"{export_model_fname}/hyperparams.yml")  # export also list of hyperparams
    np.savez(
        f"{export_model_fname}/results.npz",
        photon=photon_val,
        cluster=cluster_val,
        output=output,
    )  # export training results
    if not os.path.exists(export_img_dirname):
        os.makedirs(export_img_dirname)  # need to save images

# +---------------------+
# |   Training report   |
# +---------------------+

report = Report()
report.add_markdown('<h1 align="center">Generator training report</h1>')

info = [
    f"- Script executed on **{socket.gethostname()}**",
    f"- Model training completed in **{duration}**",
    f"- Report generated on **{date}** at **{hour}**",
]

if args.weights:
    info += ["- Model trained with **matching weights**"]
else:
    info += ["- Model trained **avoiding padded values**"]

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
report.add_markdown(f"**Model name:** {model.name}")
html_table, params_details = getSummaryHTML(model)
model_weights = ""
for k, n in zip(["Total", "Trainable", "Non-trainable"], params_details):
    model_weights += f"- **{k} params:** {n}\n"
report.add_markdown(html_table)
report.add_markdown(model_weights)

report.add_markdown("---")

## Training plots
prepare_training_plots(
    report=report,
    history=train.history,
    metrics=metrics,
    num_epochs=EPOCHS,
    attn_weights=attn_weights,
    show_discriminator_curves=False,
    is_from_validation_set=(train_ratio != 1.0),
    save_images=args.saving,
    images_dirname=export_img_dirname,
)

## Validation plots
prepare_validation_plots(
    report=report,
    photon=photon_val,
    cluster=cluster_val,
    output=output,
    weight=weight_val,
    model_name="Generator",
    save_images=args.saving,
    images_dirname=export_img_dirname,
)

report_fname = f"{reports_dir}/{prefix}_train-report.html"
report.write_report(filename=report_fname)
print(f"[INFO] Training report correctly exported to {report_fname}")
