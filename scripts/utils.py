import matplotlib.pyplot as plt
import numpy as np


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
    export_fname="./images/learn-curves.png",
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
                scales[i] = 1/np.around(scales[i])
            else:
                scales[i] = np.around(1/scales[i]) 
        for i in range(len(labels)):
            labels[i] += f" [x {scales[i]:.1f}]"
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
        plt.savefig(export_fname)
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
    export_fname="./images/lr-sched.png",
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
        plt.savefig(export_fname)
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
    save_figure=False,
    export_fname="./images/metric-curves.png",
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
    if save_figure:
        plt.savefig(export_fname)
    report.add_figure(options="width=45%")
    plt.close()


def validation_histogram(
    report,
    ref_data,
    gen_data,
    scaler=None,
    xlabel=None,
    ref_label=None,
    gen_label=None,
    legend_loc="upper left",
    save_figure=False,
    export_fname="./images/val-hist.png",
) -> None:
    if scaler is not None:
        ref_data = scaler.inverse_transform(np.c_[ref_data]).flatten()
        gen_data = scaler.inverse_transform(np.c_[gen_data]).flatten()

    min_ = ref_data.min()
    max_ = ref_data.max()
    bins = np.linspace(min_, max_, 101)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Candidates", fontsize=12)
    plt.hist(ref_data, bins=bins, color="#3288bd", label=ref_label)
    plt.hist(
        gen_data, bins=bins, histtype="step", color="#fc8d59", lw=2, label=gen_label
    )
    plt.yscale("log")
    plt.legend(loc=legend_loc, fontsize=10)
    if save_figure:
        plt.savefig(export_fname)
    report.add_figure(options="width=45%")
    plt.close()


def calorimeter_deposits(
    report,
    ref_xy,
    gen_xy,
    ref_energy=None,
    gen_energy=None,
    scaler=None,
    save_figure=False,
    export_fname="./images/calo-deposits.png",
) -> None:
    x_ref, y_ref = ref_xy
    x_gen, y_gen = gen_xy

    if scaler is not None:
        ref_energy = scaler.inverse_transform(np.c_[ref_energy]).flatten()
        gen_energy = scaler.inverse_transform(np.c_[gen_energy]).flatten()

    x_bins = np.linspace(x_ref.min(), x_ref.max(), 101)
    y_bins = np.linspace(y_ref.min(), y_ref.max(), 101)

    plt.figure(figsize=(18, 5), dpi=300)
    plt.subplot(1, 2, 1)
    plt.title("Training data", fontsize=14)
    plt.xlabel("$x$ coordinate", fontsize=12)
    plt.ylabel("$y$ coordinate", fontsize=12)
    plt.hist2d(
        x_ref,
        y_ref,
        weights=ref_energy,
        bins=(x_bins, y_bins),
        cmin=0,
        cmap="gist_heat",
    )
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Calotron output", fontsize=14)
    plt.xlabel("$x$ coordinate", fontsize=12)
    plt.ylabel("$y$ coordinate", fontsize=12)
    plt.hist2d(
        x_gen,
        y_gen,
        weights=gen_energy,
        bins=(x_bins, y_bins),
        cmin=0,
        cmap="gist_heat",
    )
    plt.colorbar()
    if save_figure:
        plt.savefig(export_fname)
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
    photon_scaler=None,
    cluster_scaler=None,
    save_figure=False,
    export_fname="./images/evt-example.png",
) -> None:
    x_photon, y_photon = photon_xy
    x_cluster, y_cluster = cluster_xy
    x_output, y_output = output_xy

    if (
        (photon_energy is not None)
        & (cluster_energy is not None)
        & (output_energy is not None)
    ):
        if photon_scaler is not None:
            photon_energy = photon_scaler.inverse_transform(
                np.c_[photon_energy]
            ).flatten()
        if cluster_scaler is not None:
            cluster_energy = cluster_scaler.inverse_transform(
                np.c_[cluster_energy]
            ).flatten()
            output_energy = cluster_scaler.inverse_transform(
                np.c_[output_energy]
            ).flatten()
        photon_size = 50.0 * photon_energy / cluster_energy.max()
        cluster_size = 50.0 * cluster_energy / cluster_energy.max()
        output_size = 50.0 * output_energy / cluster_energy.max()
    else:
        photon_size, cluster_size, output_size = [10.0 for _ in range(3)]

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
        label="True photon",
    )
    plt.scatter(
        x_cluster,
        y_cluster,
        s=cluster_size,
        marker="s",
        facecolors="none",
        edgecolors="#2b83ba",
        lw=0.75,
        label="Calo neutral cluster",
    )
    plt.scatter(
        x_output,
        y_output,
        s=output_size,
        marker="^",
        facecolors="none",
        edgecolors="#1a9641",
        lw=0.75,
        label="Calotron output",
    )
    plt.legend(fontsize=10)
    if save_figure:
        plt.savefig(export_fname)
    report.add_figure(options="width=45%")
    plt.close()


def energy_sequences(
    report,
    ref_energy,
    gen_energy,
    scaler=None,
    save_figure=False,
    export_fname="./images/energy-seq.png",
) -> None:
    if scaler is not None:
        batch_size, length = ref_energy.shape
        ref_energy = scaler.inverse_transform(
            ref_energy.reshape(batch_size * length, 1)
        )
        gen_energy = scaler.inverse_transform(
            gen_energy.reshape(batch_size * length, 1)
        )
        ref_energy = ref_energy.reshape(batch_size, length)
        gen_energy = gen_energy.reshape(batch_size, length)

    plt.figure(figsize=(18, 10), dpi=300)
    plt.subplot(1, 2, 1)
    plt.title("Reconstructed clusters", fontsize=14)
    plt.xlabel("Preprocessed energy deposits [a.u.]", fontsize=12)
    plt.ylabel("Events", fontsize=12)
    plt.imshow(ref_energy, aspect="auto", vmin=0.0, vmax=1.0, cmap="gist_heat")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Calotron output", fontsize=14)
    plt.xlabel("Preprocessed energy deposits [a.u.]", fontsize=12)
    plt.ylabel("Events", fontsize=12)
    plt.imshow(gen_energy, aspect="auto", vmin=0.0, vmax=1.0, cmap="gist_heat")
    plt.colorbar()
    if save_figure:
        plt.savefig(export_fname)
    report.add_figure(options="width=95%")
    plt.close()


def photon2cluster_corr(
    report,
    photon_energy,
    cluster_energy,
    output_energy,
    photon_scaler=None,
    cluster_scaler=None,
    save_figure=False,
    export_fname="./images/gamma2calo-corr.png",
) -> None:
    if photon_scaler is not None:
        photon_energy = photon_scaler.inverse_transform(np.c_[photon_energy]).flatten()
    if cluster_scaler is not None:
        cluster_energy = cluster_scaler.inverse_transform(
            np.c_[cluster_energy]
        ).flatten()
        output_energy = cluster_scaler.inverse_transform(np.c_[output_energy]).flatten()

    energies = [cluster_energy, output_energy]
    titles = ["Photon-to-cluster correlations", "Photon-to-output correlations"]
    ylabels = [
        "Cluster preprocessed energy [a.u.]",
        "Output preprocessed energy [a.u.]",
    ]

    plt.figure(figsize=(18, 5), dpi=300)
    for i, (energy, title, ylabel) in enumerate(zip(energies, titles, ylabels)):
        h, x_edges, y_edges = np.histogram2d(
            photon_energy, energy, bins=np.linspace(0, 1, 76)
        )
        nonzero_x, nonzero_y = np.nonzero(h)
        colors = np.ones_like(photon_energy) / (np.sum(h) + 1e-12)
        for id_x, id_y in zip(nonzero_x, nonzero_y):
            colors[
                ((photon_energy >= x_edges[id_x]) & (photon_energy < x_edges[id_x + 1]))
                & ((energy >= y_edges[id_y]) & (energy < y_edges[id_y + 1]))
            ] *= h[id_x, id_y]
        plt.subplot(1, 2, i + 1)
        plt.title(title, fontsize=14)
        plt.xlabel("Photon preprocessed energy [a.u.]", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.scatter(photon_energy, energy, s=1, c=colors, cmap="gist_heat")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.colorbar()
    if save_figure:
        plt.savefig(export_fname)
    report.add_figure(options="width=95%")
    plt.close()
