import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

my_cmap = copy.copy(mpl.cm.get_cmap("magma"))
my_cmap.set_bad((0, 0, 0))


def learning_curves(
    report,
    history,
    start_epoch=0,
    keys=["loss"],
    colors=None,
    labels=None,
    legend_loc="upper right",
    save_figure=False,
    scale_curves=True,
    export_fname="./images/learn-curves",
) -> None:
    if scale_curves:
        if "t_loss" in keys:
            scale_key = "t_loss"
        else:
            id_min = np.argmin(
                [np.abs(np.mean(np.array(history[k])[start_epoch:])) for k in keys]
            )
            scale_key = keys[id_min]
        ratios = [
            np.array(history[k])[start_epoch:]
            / np.array(history[scale_key])[start_epoch:]
            for k in keys
        ]
        scales = np.mean(ratios, axis=-1)
        for i in range(len(scales)):
            if np.abs(scales[i]) >= 1.0:
                scales[i] = 1 / np.around(scales[i])
            else:
                scales[i] = np.around(1 / scales[i])
        for i in range(len(labels)):
            labels[i] += f" [x {scales[i]:.0e}]"
    else:
        scales = [1.0 for _ in keys]

    if colors is None:
        colors = [None for _ in range(len(keys))]
    else:
        assert len(colors) == len(keys)

    if labels is None:
        labels = [None for _ in range(len(keys))]
    else:
        assert len(labels) == len(keys)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.title("Learning curves", fontsize=14)
    plt.xlabel("Training epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    for i, (k, l, c) in enumerate(zip(keys, labels, colors)):
        num_epochs = np.arange(len(history[k]))[start_epoch:]
        loss = np.array(history[k])[start_epoch:] * scales[i]
        plt.plot(num_epochs, loss, lw=1.5, color=c, label=l)
    plt.legend(loc=legend_loc, fontsize=10)
    if save_figure:
        plt.savefig(f"{export_fname}.png")
    report.add_figure(options="width=45%")
    plt.close()


def learn_rate_scheduling(
    report,
    history,
    start_epoch=0,
    keys=["lr"],
    colors=None,
    labels=None,
    legend_loc="upper right",
    save_figure=False,
    export_fname="./images/lr-sched",
) -> None:
    if colors is None:
        colors = [None for _ in range(len(keys))]
    else:
        assert len(colors) == len(keys)

    if labels is None:
        labels = [None for _ in range(len(keys))]
    else:
        assert len(labels) == len(keys)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.title("Learning rate scheduling", fontsize=14)
    plt.xlabel("Training epochs", fontsize=12)
    plt.ylabel("Learning rate", fontsize=12)
    for k, l, c in zip(keys, labels, colors):
        num_epochs = np.arange(len(history[k]))[start_epoch:]
        lr = np.array(history[k])[start_epoch:]
        plt.plot(num_epochs, lr, lw=1.5, color=c, label=l)
    plt.legend(loc=legend_loc, fontsize=10)
    plt.yscale("log")
    if save_figure:
        plt.savefig(f"{export_fname}.png")
    report.add_figure(options="width=45%")
    plt.close()


def metric_curves(
    report,
    history,
    start_epoch=0,
    key="metric",
    ylabel="Metric",
    title="Metric curves",
    validation_set=False,
    colors=None,
    labels=None,
    legend_loc="upper right",
    yscale="linear",
    save_figure=False,
    export_fname="./images/metric-curves",
) -> None:
    keys = [key]
    if validation_set:
        keys += [f"val_{key}"]

    if colors is None:
        colors = [None for _ in range(len(keys))]
    else:
        colors = colors[: len(keys)]

    if labels is None:
        labels = [None for _ in range(len(keys))]
    else:
        labels = labels[: len(keys)]

    zorder = 1
    plt.figure(figsize=(8, 5), dpi=300)
    plt.title(title, fontsize=14)
    plt.xlabel("Training epochs", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    for k, l, c in zip(keys, labels, colors):
        num_epochs = np.arange(len(history[k]))[start_epoch:]
        metric = np.array(history[k])[start_epoch:]
        plt.plot(num_epochs, metric, lw=1.5, color=c, label=l, zorder=zorder)
        zorder -= 1
    plt.legend(loc=legend_loc, fontsize=10)
    plt.yscale(yscale)
    if save_figure:
        plt.savefig(f"{export_fname}.png")
    report.add_figure(options="width=45%")
    plt.close()


def validation_histogram(
    report,
    data_ref,
    data_gen,
    weights_ref=None,
    weights_gen=None,
    xlabel=None,
    density=False,
    ref_label=None,
    gen_label=None,
    log_scale=False,
    legend_loc="upper left",
    save_figure=False,
    export_fname="./images/val-hist",
) -> None:
    min_ = data_ref.min()
    max_ = data_ref.max()
    bins = np.linspace(min_, max_, 101)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Candidates", fontsize=12)
    plt.hist(
        data_ref,
        bins=bins,
        density=density,
        weights=weights_ref,
        color="#3288bd",
        label=ref_label,
    )
    plt.hist(
        data_gen,
        bins=bins,
        density=density,
        weights=weights_gen,
        histtype="step",
        color="#fc8d59",
        lw=2,
        label=gen_label,
    )
    if log_scale:
        plt.yscale("log")
        export_fname += "-log"
    plt.legend(loc=legend_loc, fontsize=10)
    if save_figure:
        plt.savefig(f"{export_fname}.png")
    report.add_figure(options="width=45%")
    plt.close()


def calorimeter_deposits(
    report,
    ref_xy,
    gen_xy,
    ref_energy=None,
    gen_energy=None,
    min_energy=0.0,
    model_name="Calotron",
    log_scale=False,
    save_figure=False,
    export_fname="./images/calo-deposits",
) -> None:
    coords = [ref_xy, gen_xy]
    weights = [ref_energy, gen_energy]
    bins = np.linspace(-0.4, 0.4, 81)
    vmin = 1.0 if log_scale else 0.0
    vmax = 0.0
    for xy, w in zip(coords, weights):
        x, y = xy
        w = np.where(w > min_energy, w, 0.0)

        h, _, _ = np.histogram2d(x, y, bins=bins, weights=w)
        vmax = max(h.max(), vmax)

    titles = ["Training data", f"{model_name} output"]

    plt.figure(figsize=(18, 5), dpi=300)
    for i, (xy, w, title) in enumerate(zip(coords, weights, titles)):
        x, y = xy
        w = np.where(w > min_energy, w, 0.0)
        plt.subplot(1, 2, i + 1)
        plt.title(title, fontsize=14)
        plt.xlabel("$x$ coordinate", fontsize=12)
        plt.ylabel("$y$ coordinate", fontsize=12)
        plt.hist2d(
            x,
            y,
            norm=mpl.colors.LogNorm(vmin=vmin) if log_scale else None,
            weights=w,
            bins=bins,
            cmap=my_cmap,
        )
        plt.clim(vmin=vmin, vmax=vmax)
        plt.colorbar()
    if save_figure:
        if log_scale:
            export_fname += "-log"
        plt.savefig(f"{export_fname}.png")
    report.add_figure(options="width=95%")
    plt.close()


def event_example(
    report,
    photon_xy,
    cluster_xy,
    output_xy,
    photon_energy=None,
    cluster_energy=None,
    output_energy=None,
    match_weights=None,
    min_energy=0.0,
    model_name="Calotron",
    show_errors=True,
    save_figure=False,
    export_fname="./images/evt-example",
) -> None:
    x_photon, y_photon = photon_xy
    x_cluster, y_cluster = cluster_xy
    x_output, y_output = output_xy

    if (
        np.all(photon_energy is not None)
        & np.all(cluster_energy is not None)
        & np.all(output_energy is not None)
    ):
        photon_energy = np.where(photon_energy > min_energy, photon_energy, 0.0)
        cluster_energy = np.where(cluster_energy > min_energy, cluster_energy, 0.0)
        output_energy = np.where(output_energy > min_energy, output_energy, 0.0)

        photon_size = 50.0 * photon_energy / cluster_energy.max()
        cluster_size = 50.0 * cluster_energy / cluster_energy.max()
        output_size = 50.0 * output_energy / cluster_energy.max()
    else:
        photon_size, cluster_size, output_size = [30.0 for _ in range(3)]

    plt.figure(figsize=(8, 6), dpi=300)
    plt.xlabel("$x$ coordinate", fontsize=12)
    plt.ylabel("$y$ coordinate", fontsize=12)
    plt.scatter(
        x_photon,
        y_photon,
        s=photon_size,
        marker="o",
        facecolors="none",
        edgecolors="#d7191c",
        lw=0.75,
        label="Generated photon",
        zorder=2,
    )
    plt.scatter(
        x_cluster,
        y_cluster,
        s=cluster_size,
        marker="s",
        facecolors="none",
        edgecolors="#2b83ba",
        lw=0.75,
        label="Reconstructed cluster",
        zorder=3,
    )
    if np.all(match_weights is not None):
        plt.scatter(
            np.where(match_weights == 1.0, x_cluster, 0.0),
            np.where(match_weights == 1.0, y_cluster, 0.0),
            s=np.where(match_weights == 1.0, cluster_size, 0.0),
            marker="s",
            facecolors="yellow",
            edgecolors="#2b83ba",
            lw=0.75,
            label="Photon-matched cluster",
            zorder=1,
        )
    plt.scatter(
        x_output,
        y_output,
        s=output_size,
        marker="^",
        facecolors="none",
        edgecolors="#1a9641",
        lw=0.75,
        label=f"{model_name} output",
        zorder=4,
    )
    if np.all(cluster_energy is not None):
        weights = cluster_energy.copy()
    else:
        weights = np.ones_like(x_cluster)
    if show_errors:
        for x_true, y_true, x_pred, y_pred, w in zip(
            x_cluster, y_cluster, x_output, y_output, weights
        ):
            plt.plot(
                [x_true, x_pred],
                [y_true, y_pred],
                color="#1a9641",
                lw=0.5 * w / weights.max(),
                zorder=5,
            )
    plt.legend(fontsize=10)
    if save_figure:
        plt.savefig(f"{export_fname}.png")
    report.add_figure(options="width=45%")
    plt.close()


def energy_sequences(
    report,
    ref_energy,
    gen_energy,
    min_energy=0.0,
    model_name="Calotron",
    save_figure=False,
    export_fname="./images/energy-seq",
) -> None:
    ref_energy = np.where(ref_energy > min_energy, ref_energy, 0.0)
    gen_energy = np.where(gen_energy > min_energy, gen_energy, 0.0)
    vmax = max(ref_energy.max(), gen_energy.max())

    plt.figure(figsize=(18, 10), dpi=300)
    plt.subplot(1, 2, 1)
    plt.title("Reconstructed clusters", fontsize=14)
    plt.xlabel("Preprocessed energy deposits [a.u.]", fontsize=12)
    plt.ylabel("Events", fontsize=12)
    plt.imshow(ref_energy, aspect="auto", vmin=0.0, vmax=vmax, cmap=my_cmap)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title(f"{model_name} output", fontsize=14)
    plt.xlabel("Preprocessed energy deposits [a.u.]", fontsize=12)
    plt.ylabel("Events", fontsize=12)
    plt.imshow(gen_energy, aspect="auto", vmin=0.0, vmax=vmax, cmap=my_cmap)
    plt.colorbar()
    if save_figure:
        plt.savefig(f"{export_fname}.png")
    report.add_figure(options="width=95%")
    plt.close()


def photon2cluster_corr(
    report,
    photon,
    cluster,
    output,
    min_energy=0.0,
    log_scale=False,
    save_figure=False,
    export_fname="./images/gamma2calo-corr",
) -> None:
    reco_objects = [cluster, output]
    energies = list()
    bins = np.linspace(min_energy, 0.8 - min_energy, 76)
    vmin = 1.0 if log_scale else 0.0
    vmax = 0.0
    for object in reco_objects:
        photon_xy = np.tile(photon[:, None, :, :2], (1, object.shape[1], 1, 1))
        object_xy = np.tile(object[:, :, None, :2], (1, 1, photon.shape[1], 1))

        pairwise_distance = np.linalg.norm(object_xy - photon_xy, axis=-1)
        match_photon_idx = np.argmin(pairwise_distance, axis=-1)
        matched_photon = np.take_along_axis(
            photon, match_photon_idx[:, :, None], axis=1
        )

        photon_energy = matched_photon[:, :, 2].flatten()
        object_energy = object[:, :, 2].flatten()

        photon_energy = np.where(photon_energy > min_energy, photon_energy, 0.0)
        object_energy = np.where(object_energy > min_energy, object_energy, 0.0)
        energies.append((photon_energy, object_energy))

        h, _, _ = np.histogram2d(photon_energy, object_energy, bins=bins)
        vmax = max(h.max(), vmax)

    titles = ["Photon-to-cluster correlations", "Photon-to-output correlations"]
    ylabels = [
        "Cluster preprocessed energy [a.u.]",
        "Output preprocessed energy [a.u.]",
    ]

    plt.figure(figsize=(14, 5), dpi=300)
    for i, (energy, title, ylabel) in enumerate(zip(energies, titles, ylabels)):
        photon_energy, object_energy = energy
        plt.subplot(1, 2, i + 1)
        plt.title(title, fontsize=14)
        plt.xlabel("Photon preprocessed energy [a.u.]", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.hist2d(
            photon_energy,
            object_energy,
            bins=bins,
            norm=mpl.colors.LogNorm(vmin=vmin) if log_scale else None,
            cmap=my_cmap,
        )
        plt.clim(vmin=vmin, vmax=vmax)
        plt.colorbar()
    if save_figure:
        if log_scale:
            export_fname += "-log"
        plt.savefig(f"{export_fname}.png")
    report.add_figure(options="width=95%")
    plt.close()


def attention_plot(
    report,
    attn_weights,
    head_id=0,
    save_figure=False,
    export_fname="./images/attn-plot",
) -> None:
    if len(attn_weights.shape) == 4:
        attn_weights = np.mean(attn_weights, axis=0)
    if attn_weights.shape[1] == attn_weights.shape[2]:
        figsize = (7, 5)
    else:
        figsize = (8, 4)
    plt.figure(figsize=figsize, dpi=300)
    plt.title(f"Last attention weights of head #{head_id+1}", fontsize=14)
    plt.xlabel("Generated photons", fontsize=12)
    plt.ylabel("Reconstructed clusters", fontsize=12)
    plt.imshow(attn_weights[head_id], aspect="auto", cmap=my_cmap)
    plt.colorbar()
    if save_figure:
        plt.savefig(f"{export_fname}.png")
    report.add_figure(options="width=45%")
    plt.close()


def energy_center_dist(
    report,
    cluster,
    output,
    weights=None,
    density=False,
    model_name="Calotron",
    log_scale=False,
    legend_loc="upper left",
    save_figure=False,
    export_fname="./images/energy-center-dist",
) -> None:
    cluster_xy_center = np.sum(
        cluster[:, :, :2] * cluster[:, :, 2][:, :, None], axis=1
    ) / np.sum(cluster[:, :, 2], axis=1, keepdims=True)
    output_xy_center = np.sum(
        output[:, :, :2] * output[:, :, 2][:, :, None], axis=1
    ) / np.sum(output[:, :, 2], axis=1, keepdims=True)

    cluster_distances = np.linalg.norm(
        cluster[:, :, :2] - cluster_xy_center[:, None, :], axis=-1
    ).flatten()
    output_distances = np.linalg.norm(
        output[:, :, :2] - output_xy_center[:, None, :], axis=-1
    ).flatten()

    if np.any(weights is None):
        weights = np.ones_like(cluster[:, :, 0])

    min_ = cluster_distances.min()
    max_ = cluster_distances.max()
    bins = np.linspace(min_, max_, 101)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.xlabel("Distance from center of energy [a.u.]", fontsize=12)
    plt.ylabel("Candidates", fontsize=12)
    plt.hist(
        cluster_distances,
        bins=bins,
        density=density,
        weights=weights.flatten(),
        color="#3288bd",
        label="Training data",
    )
    plt.hist(
        output_distances,
        bins=bins,
        density=density,
        weights=weights.flatten(),
        histtype="step",
        color="#fc8d59",
        lw=2,
        label=f"{model_name} model",
    )
    if log_scale:
        plt.yscale("log")
        export_fname += "-log"
    plt.legend(loc=legend_loc, fontsize=10)
    if save_figure:
        plt.savefig(f"{export_fname}.png")
    report.add_figure(options="width=45%")
    plt.close()
