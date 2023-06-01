import os
import socket
from argparse import ArgumentParser
from datetime import datetime

import hopaas_client as hpc
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

from calotron.callbacks.schedulers import LearnRateExpDecay
from calotron.losses import MeanSquaredError
from calotron.models import Calotron
from calotron.models.discriminators import Discriminator
from calotron.models.transformers import MaskedTransformer
from calotron.optimization.callbacks import HopaasPruner
from calotron.simulators import ExportSimulator, Simulator
from calotron.utils import getSummaryHTML, initHPSingleton

STUDY_NAME = "Calotron::ModArch::Testv0"
DTYPE = np.float32
TRAIN_RATIO = 0.7
BATCHSIZE = 256
EPOCHS = 500

# +------------------+
# |   Parser setup   |
# +------------------+

parser = ArgumentParser(description="calotron optimization setup")

parser.add_argument("--saving", action="store_true")
parser.add_argument("--no-saving", dest="saving", action="store_false")
parser.set_defaults(saving=False)

parser.add_argument("--weights", action="store_true")
parser.add_argument("--no-weights", dest="weights", action="store_false")
parser.set_defaults(weights=True)

address = socket.gethostbyname(socket.gethostname())
parser.add_argument("-n", "--node_name", default=f"{address}")

args = parser.parse_args()

# +---------------+
# |   GPU setup   |
# +---------------+

avail_gpus = tf.config.list_physical_devices("GPU")

if len(avail_gpus) == 0:
    raise RuntimeError("No GPUs available for the optimization study")

# +-------------------+
# |   Initial setup   |
# +-------------------+

hp = initHPSingleton()

with open("config/directories.yml") as file:
    config_dir = yaml.full_load(file)

data_dir = config_dir["data_dir"]
models_dir = f"{config_dir['models_dir']}/opt_studies"
images_dir = f"{config_dir['images_dir']}/opt_studies"
reports_dir = f"{config_dir['reports_dir']}/opt_studies"

# +-----------------------------+
# |    Client initialization    |
# +-----------------------------+

with open("config/hopaas.yml") as file:
    config_hopaas = yaml.full_load(file)

server = config_hopaas["server"]
token = config_hopaas["token"]

client = hpc.Client(server=server, token=token)

# +----------------------+
# |    Study creation    |
# +----------------------+

properties = {
    "enc_d": hpc.suggestions.Int(4, 32, step=4),
    "dec_d": hpc.suggestions.Int(4, 32, step=4),
    "lat_d": hpc.suggestions.Int(4, 64, step=4),
    "so_norm": hpc.suggestions.Float(100, 10_000),
    "res_s": hpc.suggestions.Categorical(["true", "false"]),
}

properties.update(
    {"train_ratio": TRAIN_RATIO, "batch_size": BATCHSIZE, "epochs": EPOCHS}
)

study = hpc.Study(
    name=STUDY_NAME,
    properties=properties,
    special_properties={"address": address, "node_name": str(args.node_name)},
    direction="minimize",
    pruner=hpc.pruners.MedianPruner(
        n_startup_trials=20, n_warmup_steps=50, interval_steps=1, n_min_trials=10
    ),
    sampler=hpc.samplers.TPESampler(n_startup_trials=50),
    client=client,
)

with study.trial() as trial:
    print(f"\n{'< ' * 30} Trial n. {trial.id} {' >' * 30}\n")

    # +------------------+
    # |   Data loading   |
    # +------------------+

    npzfile = np.load(f"{data_dir}/calotron-dataset-demo.npz")
    photon = npzfile["photon"].astype(DTYPE)[:150_000]
    cluster = npzfile["cluster"].astype(DTYPE)[:150_000]
    weight = npzfile["weight"].astype(DTYPE)[:150_000]

    print(f"[INFO] Generated photons - shape: {photon.shape}")
    print(f"[INFO] Reconstructed clusters - shape: {cluster.shape}")
    print(f"[INFO] Matching weights - shape: {weight.shape}")

    if not args.weights:
        weight = (weight > 0.0).astype(DTYPE)

    photon, cluster, weight = shuffle(photon, cluster, weight)

    chunk_size = photon.shape[0]
    train_size = int(TRAIN_RATIO * chunk_size)

    # +-------------------------+
    # |   Dataset preparation   |
    # +-------------------------+

    photon_train = photon[:train_size]
    cluster_train = cluster[:train_size]
    weight_train = weight[:train_size]
    train_ds = (
        tf.data.Dataset.from_tensor_slices((photon_train, cluster_train, weight_train))
        .batch(hp.get("batch_size", BATCHSIZE), drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    if TRAIN_RATIO != 1.0:
        photon_val = photon[train_size:]
        cluster_val = cluster[train_size:]
        weight_val = weight[train_size:]
        val_ds = (
            tf.data.Dataset.from_tensor_slices((photon_val, cluster_val, weight_val))
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

    encoder_depth = int(trial.enc_d) if trial.res_s == "true" else int(trial.lat_d)
    decoder_depth = int(trial.dec_d) if trial.res_s == "true" else int(trial.lat_d)
    seq_ord_latent_dims = int(trial.lat_d)
    seq_ord_normalizations = float(trial.so_norm)
    enable_residual_smoothing = trial.res_s == "true"

    transformer = MaskedTransformer(
        output_depth=hp.get("output_depth", cluster.shape[2]),
        encoder_depth=hp.get("encoder_depth", encoder_depth),
        decoder_depth=hp.get("decoder_depth", decoder_depth),
        num_layers=hp.get("num_layers", 5),
        num_heads=hp.get("num_heads", 4),
        key_dims=hp.get("key_dims", 64),
        mlp_units=hp.get("mlp_units", 128),
        dropout_rates=hp.get("dropout_rates", 0.1),
        seq_ord_latent_dims=hp.get("seq_ord_latent_dims", seq_ord_latent_dims),
        seq_ord_max_lengths=hp.get(
            "seq_ord_max_lengths", [photon.shape[1], cluster.shape[1]]
        ),
        seq_ord_normalizations=hp.get("seq_ord_normalizations", seq_ord_normalizations),
        attn_mask_init_nonzero_size=hp.get("attn_mask_init_nonzero_size", 8),
        enable_residual_smoothing=hp.get(
            "enable_residual_smoothing", enable_residual_smoothing
        ),
        enable_source_baseline=hp.get("enable_source_baseline", True),
        enable_attention_mask=hp.get("enable_attention_mask", False),
        output_activations=hp.get("output_activations", None),
        start_token_initializer=hp.get("start_toke_initializer", "ones"),
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

    output = model((photon[:BATCHSIZE], cluster[:BATCHSIZE]))
    model.summary()

    # +----------------------+
    # |   Optimizers setup   |
    # +----------------------+

    hp.get("t_optimizer", "RMSprop")
    t_opt = tf.keras.optimizers.RMSprop(hp.get("t_lr0", 1e-3))

    hp.get("d_optimizer", "RMSprop")
    d_opt = tf.keras.optimizers.RMSprop(hp.get("d_lr0", 1e-4))

    # +----------------------------+
    # |   Training configuration   |
    # +----------------------------+

    loss = MeanSquaredError(
        warmup_energy=hp.get("warmup_energy", 0.0),
        alpha=hp.get("loss_alpha", 0.0),
        adversarial_metric=hp.get("loss_adversarial_metric", "wasserstein-distance"),
        bce_options=hp.get(
            "loss_bce_options", {"injected_noise_stddev": 0.01, "label_smoothing": 0.1}
        ),
        wass_options=hp.get(
            "loss_wass_options",
            {"lipschitz_penalty": 100.0, "virtual_direction_upds": 1},
        ),
    )
    hp.get("loss", loss.name)

    model.compile(
        loss=loss,
        metrics=hp.get("metrics", ["wass_dist"]),
        transformer_optimizer=t_opt,
        discriminator_optimizer=d_opt,
        transformer_upds_per_batch=hp.get("Calotron_upds_per_batch", 1),
        discriminator_upds_per_batch=hp.get("discriminator_upds_per_batch", 1),
    )

    # +--------------------------+
    # |   Callbacks definition   |
    # +--------------------------+

    callbacks = list()

    t_sched = LearnRateExpDecay(
        model.transformer_optimizer,
        decay_rate=hp.get("t_decay_rate", 0.10),
        decay_steps=hp.get("t_decay_steps", 100_000),
        min_learning_rate=hp.get("t_min_learning_rate", 1e-6),
    )
    hp.get("t_sched", "LearnRateExpDecay")
    callbacks.append(t_sched)

    d_sched = LearnRateExpDecay(
        model.discriminator_optimizer,
        decay_rate=hp.get("d_decay_rate", 0.10),
        decay_steps=hp.get("d_decay_steps", 50_000),
        min_learning_rate=hp.get("d_min_learning_rate", 1e-6),
    )
    hp.get("d_sched", "LearnRateExpDecay")
    callbacks.append(d_sched)

    pruner = HopaasPruner(
        trial=trial, loss_name="val_wass_dist", report_frequency=1, enable_pruning=True
    )
    callbacks.append(pruner)

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

    sim = Simulator(model.transformer, start_token=start_token)
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

    prefix = f"suid{study.study_id[:8]}-trial{trial.id:04d}_calotron"
    export_model_fname = f"{models_dir}/{prefix}_model"
    export_img_dirname = f"{images_dir}/{prefix}_img"

    if args.saving:
        tf.saved_model.save(exp_sim, export_dir=export_model_fname)
        hp.dump(
            f"{export_model_fname}/hyperparams.yml"
        )  # export also list of hyperparams
        print(f"[INFO] Trained model correctly exported to {export_model_fname}")
        os.makedirs(export_img_dirname)  # need to save images

    # +---------------------+
    # |   Training report   |
    # +---------------------+

    report = Report()
    report.add_markdown('<h1 align="center">Calotron training report</h1>')

    timestamp = str(datetime.now())
    date, hour = timestamp.split(" ")
    date = date.replace("-", "/")
    hour = hour.split(".")[0]

    info = [
        f"- Script executed on **{socket.gethostname()}** (address: {args.node_name})",
        f"- Trial **#{trial.id:04d}** (suid: {study.study_id})",
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
        export_fname=f"{export_img_dirname}/learn-curves",
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
        export_fname=f"{export_img_dirname}/lr-sched",
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
        yscale="linear",
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/transf-loss",
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
        yscale="linear",
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/disc-loss",
    )

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
        yscale="linear",
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/wass-curves",
    )

    report.add_markdown("<br>")

    #### Attention weights
    for head_id in range(attn_weights.shape[1]):
        attention_plot(
            report=report,
            attn_weights=attn_weights,
            head_id=head_id,
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/attn-plot",
        )

    report.add_markdown("---")

    ## Validation plots
    report.add_markdown('<h2 align="center">Validation plots</h2>')

    photon_x = photon_val[:, :, 0].flatten()
    photon_y = photon_val[:, :, 1].flatten()
    photon_energy = photon_val[:, :, 2].flatten()

    cluster_x = cluster_val[:, :, 0].flatten()
    cluster_y = cluster_val[:, :, 1].flatten()
    cluster_energy = cluster_val[:, :, 2].flatten()

    output_x = output[:, :, 0].flatten()
    output_y = output[:, :, 1].flatten()
    output_energy = output[:, :, 2].flatten()

    w = weight_val.flatten()

    #### X histogram
    for log_scale in [False, True]:
        validation_histogram(
            report=report,
            data_ref=cluster_x,
            data_gen=output_x,
            weights_ref=w,
            weights_gen=w,
            xlabel="Preprocessed $x$-coordinate [a.u.]",
            density=False,
            ref_label="Training data",
            gen_label="Calotron output",
            log_scale=log_scale,
            legend_loc="upper right",
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/x-hist",
        )

    #### Y histogram
    for log_scale in [False, True]:
        validation_histogram(
            report=report,
            data_ref=cluster_y,
            data_gen=output_y,
            weights_ref=w,
            weights_gen=w,
            xlabel="Preprocessed $y$-coordinate [a.u.]",
            density=False,
            ref_label="Training data",
            gen_label="Calotron output",
            log_scale=log_scale,
            legend_loc="upper right",
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/y-hist",
        )

    #### Energy histogram
    for log_scale in [False, True]:
        validation_histogram(
            report=report,
            data_ref=cluster_energy,
            data_gen=output_energy,
            weights_ref=w,
            weights_gen=w,
            xlabel="Preprocessed energy [a.u.]",
            density=False,
            ref_label="Training data",
            gen_label="Calotron output",
            log_scale=log_scale,
            legend_loc="upper right",
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/energy-hist",
        )

    #### Calorimeter deposits
    for log_scale in [False, True]:
        calorimeter_deposits(
            report=report,
            ref_xy=(cluster_x, cluster_y),
            gen_xy=(output_x, output_y),
            ref_energy=cluster_energy * w,
            gen_energy=output_energy * w,
            min_energy=0.0,
            model_name="Calotron",
            log_scale=log_scale,
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/calo-deposits",
        )

    #### Event examples
    cluster_mean_energy = np.mean(cluster_val[:, :, 2], axis=-1)
    events = np.argsort(cluster_mean_energy)[::-1][:4]
    for i, evt in enumerate(events):
        event_example(
            report=report,
            photon_xy=(
                photon_val[evt, :, 0].flatten(),
                photon_val[evt, :, 1].flatten(),
            ),
            cluster_xy=(
                cluster_val[evt, :, 0].flatten(),
                cluster_val[evt, :, 1].flatten(),
            ),
            output_xy=(output[evt, :, 0].flatten(), output[evt, :, 1].flatten()),
            photon_energy=photon_val[evt, :, 2].flatten(),
            cluster_energy=cluster_val[evt, :, 2].flatten(),
            output_energy=output[evt, :, 2].flatten(),
            min_energy=0.0,
            model_name="Calotron",
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/evt-example-{i}",
        )

    #### Energy sequences
    energy_sequences(
        report=report,
        ref_energy=cluster_val[:64, :, 2],
        gen_energy=output[:64, :, 2],
        min_energy=0.0,
        model_name="Calotron",
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/energy-seq",
    )

    #### Photon-to-cluster correlations
    photon2cluster_corr(
        report,
        photon=photon_val,
        cluster=cluster_val,
        output=output,
        min_energy=0.0,
        log_scale=True,
        save_figure=args.saving,
        export_fname=f"{export_img_dirname}/gamma2calo-corr",
    )

    report.add_markdown("---")

    report_fname = f"{reports_dir}/{prefix}_train-report.html"
    report.write_report(filename=report_fname)
    print(f"[INFO] Training report correctly exported to {report_fname}")
    print(
        f"[INFO] The trained model of Trial n. {trial.id} scored {train.history['val_wass_dist'][-1]:.3f}"
    )
