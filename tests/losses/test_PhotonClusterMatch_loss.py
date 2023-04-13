import pytest
import tensorflow as tf

from calotron.models.transformers import Transformer
from calotron.models.discriminators import Discriminator
from calotron.models.auxiliaries import AuxClassifier

CHUNK_SIZE = int(1e4)
source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))
weight = tf.random.uniform(shape=(CHUNK_SIZE, target.shape[1], 1))

transf = Transformer(
    output_depth=target.shape[2],
    encoder_depth=8,
    decoder_depth=8,
    num_layers=2,
    num_heads=4,
    key_dims=32,
    mlp_units=16,
    dropout_rates=0.1,
    seq_ord_latent_dims=16,
    seq_ord_max_lengths=[source.shape[1], target.shape[1]],
    seq_ord_normalizations=10_000,
    residual_smoothing=True,
    output_activations="relu",
    start_token_initializer="ones",
)

disc = Discriminator(
    latent_dim=8,
    output_units=1,
    output_activation="sigmoid",
    deepsets_num_layers=2,
    deepsets_hidden_units=32,
    dropout_rate=0.1,
)

aux = AuxClassifier(
    transformer=transf, output_depth=1, output_activation="sigmoid", dropout_rate=0.1
)


@pytest.fixture
def loss():
    from calotron.losses import PhotonClusterMatch

    loss_ = PhotonClusterMatch(
        lambda_adv=0.1,
        lambda_geom=1.0,
        lambda_global=0.0,
        max_match_distance=0.005,
        adversarial_metric="binary-crossentropy",
        bce_options={
            "injected_noise_stddev": 0.0,
            "from_logits": False,
            "label_smoothing": 0.0,
        },
        wass_options={"lipschitz_penalty": 100.0, "virtual_direction_upds": 1},
        aux_bce_options={"from_logits": False, "label_smoothing": 0.0},
    )
    return loss_


###########################################################################


def test_loss_configuration(loss):
    from calotron.losses import PhotonClusterMatch

    assert isinstance(loss, PhotonClusterMatch)
    assert isinstance(loss.lambda_adv, float)
    assert isinstance(loss.lambda_geom, float)
    assert isinstance(loss.lambda_global, float)
    assert isinstance(loss.max_match_distance, float)
    assert isinstance(loss.adversarial_metric, str)
    assert isinstance(loss.bce_options, dict)
    assert isinstance(loss.wass_options, dict)
    assert isinstance(loss.aux_bce_options, dict)
    assert isinstance(loss.name, str)


@pytest.mark.parametrize(
    "adversarial_metric", ["binary-crossentropy", "wasserstein-distance"]
)
def test_loss_use_no_weights(adversarial_metric):
    from calotron.losses import PhotonClusterMatch

    loss = PhotonClusterMatch(
        lambda_adv=0.1,
        lambda_global=0.0,
        max_match_distance=0.005,
        adversarial_metric=adversarial_metric,
        bce_options={"injected_noise_stddev": 0.1},
        wass_options={"lipschitz_penalty": 100.0},
    )
    out = loss.discriminator_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=None,
        training=False,
    )
    assert out.numpy()
    out = loss.transformer_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=None,
        training=False,
    )
    assert out.numpy()
    out = loss.aux_classifier_loss(
        aux_classifier=aux,
        source=source,
        target=target,
        sample_weight=None,
        training=False,
    )
    assert out.numpy()


@pytest.mark.parametrize(
    "adversarial_metric", ["binary-crossentropy", "wasserstein-distance"]
)
def test_loss_use_with_weights(adversarial_metric):
    from calotron.losses import PhotonClusterMatch

    loss = PhotonClusterMatch(
        lambda_adv=0.1,
        lambda_global=0.0,
        max_match_distance=0.005,
        adversarial_metric=adversarial_metric,
        bce_options={"injected_noise_stddev": 0.1},
        wass_options={"lipschitz_penalty": 100.0},
    )
    out = loss.discriminator_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=weight,
        training=False,
    )
    assert out.numpy()
    out = loss.transformer_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=weight,
        training=False,
    )
    assert out.numpy()
    out = loss.aux_classifier_loss(
        aux_classifier=aux,
        source=source,
        target=target,
        sample_weight=weight,
        training=False,
    )
    assert out.numpy()
