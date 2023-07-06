import numpy as np
from utils_plot import (
    attention_plot,
    calorimeter_deposits,
    energy_center_dist,
    energy_sequences,
    event_example,
    learn_rate_scheduling,
    learning_curves,
    metric_curves,
    photon2cluster_corr,
    validation_histogram,
)

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "bce": "Binary cross-entropy",
    "js_div": "Jensen-Shannon divergence",
    "kl_div": "Kullback-Leibler divergence",
    "mae": "Mean Absolute Error",
    "mse": "Mean Squared Error",
    "rmse": "Root Mean Squared Error",
    "wass_dist": "Wasserstein distance",
}
MIN_ENERGY = 1e-4


def prepare_training_plots(
    report,
    history,
    metrics,
    num_epochs,
    attn_weights,
    show_discriminator_curves=True,
    is_from_validation_set=True,
    save_images=False,
    images_dirname="./images",
) -> None:
    report.add_markdown('<h2 align="center">Training plots</h2>')

    start_epoch = int(num_epochs / 20)

    learning_curves(
        report=report,
        history=history,
        start_epoch=start_epoch,
        keys=["t_loss", "d_loss"] if show_discriminator_curves else ["loss"],
        colors=["#3288bd", "#fc8d59"] if show_discriminator_curves else ["#3288bd"],
        labels=["transformer", "discriminator"]
        if show_discriminator_curves
        else ["transformer"],
        legend_loc=None,
        save_figure=save_images,
        scale_curves=show_discriminator_curves,
        export_fname=f"{images_dirname}/learn-curves",
    )

    learn_rate_scheduling(
        report=report,
        history=history,
        start_epoch=0,
        keys=["t_lr", "d_lr"] if show_discriminator_curves else ["lr"],
        colors=["#3288bd", "#fc8d59"] if show_discriminator_curves else ["#3288bd"],
        labels=["transformer", "discriminator"]
        if show_discriminator_curves
        else ["transformer"],
        legend_loc="upper right",
        save_figure=save_images,
        export_fname=f"{images_dirname}/lr-sched",
    )

    metric_curves(
        report=report,
        history=history,
        start_epoch=start_epoch,
        key="t_loss" if show_discriminator_curves else "loss",
        ylabel="Transformer loss",
        title="Learning curves",
        validation_set=is_from_validation_set,
        colors=["#d01c8b", "#4dac26"],
        labels=["training set", "validation set"],
        legend_loc=None,
        yscale="linear",
        save_figure=save_images,
        export_fname=f"{images_dirname}/transf-loss",
    )

    if show_discriminator_curves:
        metric_curves(
            report=report,
            history=history,
            start_epoch=start_epoch,
            key="d_loss",
            ylabel="Discriminator loss",
            title="Learning curves",
            validation_set=is_from_validation_set,
            colors=["#d01c8b", "#4dac26"],
            labels=["training set", "validation set"],
            legend_loc=None,
            yscale="linear",
            save_figure=save_images,
            export_fname=f"{images_dirname}/disc-loss",
        )

    for metric in metrics:
        metric_curves(
            report=report,
            history=history,
            start_epoch=start_epoch,
            key=metric,
            ylabel=METRIC_LABELS[metric],
            title="Metric curves",
            validation_set=is_from_validation_set,
            colors=["#d01c8b", "#4dac26"],
            labels=["training set", "validation set"],
            legend_loc=None,
            yscale="linear",
            save_figure=save_images,
            export_fname=f"{images_dirname}/{metric}-curves",
        )

    report.add_markdown("<br>")

    for head_id in range(attn_weights.shape[1]):
        attention_plot(
            report=report,
            attn_weights=attn_weights,
            head_id=head_id,
            save_figure=save_images,
            export_fname=f"{images_dirname}/attn-plot",
        )

    report.add_markdown("---")


def prepare_validation_plots(
    report,
    photon,
    cluster,
    output,
    weight,
    model_name="Calotron",
    save_images=False,
    images_dirname="./images",
) -> None:
    report.add_markdown('<h2 align="center">Validation plots</h2>')

    photon_x = photon[:, :, 0].flatten()
    photon_y = photon[:, :, 1].flatten()
    photon_energy = photon[:, :, 2].flatten()

    cluster_x = cluster[:, :, 0].flatten()
    cluster_y = cluster[:, :, 1].flatten()
    cluster_energy = cluster[:, :, 2].flatten()

    output_x = output[:, :, 0].flatten()
    output_y = output[:, :, 1].flatten()
    output_energy = output[:, :, 2].flatten()

    w = weight.flatten()

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
            gen_label=f"{model_name} output",
            log_scale=log_scale,
            legend_loc="upper right",
            save_figure=save_images,
            export_fname=f"{images_dirname}/x-hist",
        )

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
            gen_label=f"{model_name} output",
            log_scale=log_scale,
            legend_loc="upper right",
            save_figure=save_images,
            export_fname=f"{images_dirname}/y-hist",
        )

    for log_scale in [False, True]:
        energy_center_dist(
            report=report,
            cluster=cluster,
            output=output,
            weights=weight,
            density=False,
            model_name=f"{model_name}",
            log_scale=log_scale,
            legend_loc="upper right",
            save_figure=save_images,
            export_fname=f"{images_dirname}/energy-center-dist",
        )

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
            gen_label=f"{model_name} output",
            log_scale=log_scale,
            legend_loc="upper right",
            save_figure=save_images,
            export_fname=f"{images_dirname}/energy-hist",
        )

    for log_scale in [False, True]:
        calorimeter_deposits(
            report=report,
            ref_xy=(cluster_x, cluster_y),
            gen_xy=(output_x, output_y),
            ref_energy=cluster_energy * w,
            gen_energy=output_energy * w,
            min_energy=MIN_ENERGY,
            model_name=f"{model_name}",
            log_scale=log_scale,
            save_figure=save_images,
            export_fname=f"{images_dirname}/calo-deposits",
        )

    cluster_mean_energy = np.mean(cluster[:, :, 2], axis=-1)
    events = np.random.choice(np.argsort(cluster_mean_energy)[::-1][:1000], size=4)
    for i, evt in enumerate(events):
        event_example(
            report=report,
            photon_xy=(photon[evt, :, 0].flatten(), photon[evt, :, 1].flatten()),
            cluster_xy=(cluster[evt, :, 0].flatten(), cluster[evt, :, 1].flatten()),
            output_xy=(output[evt, :, 0].flatten(), output[evt, :, 1].flatten()),
            photon_energy=photon[evt, :, 2].flatten(),
            cluster_energy=cluster[evt, :, 2].flatten(),
            output_energy=output[evt, :, 2].flatten(),
            match_weights=weight[evt, :].flatten(),
            min_energy=MIN_ENERGY,
            model_name=f"{model_name}",
            show_errors=True,
            save_figure=save_images,
            export_fname=f"{images_dirname}/evt-example-{i}",
        )

    energy_sequences(
        report=report,
        ref_energy=cluster[:64, :, 2],
        gen_energy=output[:64, :, 2],
        min_energy=MIN_ENERGY,
        model_name=f"{model_name}",
        save_figure=save_images,
        export_fname=f"{images_dirname}/energy-seq",
    )

    for log_scale in [False, True]:
        photon2cluster_corr(
            report,
            photon=photon,
            cluster=cluster,
            output=output,
            min_energy=MIN_ENERGY,
            log_scale=log_scale,
            save_figure=save_images,
            export_fname=f"{images_dirname}/gamma2calo-corr",
        )

    report.add_markdown("---")
