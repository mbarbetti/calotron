import pytest
import tensorflow as tf

from calotron.models.discriminators import Discriminator
from calotron.models.transformers import Transformer

CHUNK_SIZE = int(1e4)
source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))
weight = tf.random.uniform(shape=(CHUNK_SIZE, target.shape[1]))

transf = Transformer(
    output_depth=target.shape[2],
    encoder_depth=8,
    decoder_depth=8,
    num_layers=2,
    num_heads=4,
    key_dim=32,
    admin_res_scale="O(n)",
    mlp_units=128,
    dropout_rate=0.1,
    seq_ord_latent_dim=16,
    seq_ord_max_length=max(source.shape[1], target.shape[1]),
    seq_ord_normalization=10_000,
    enable_res_smoothing=True,
    output_activations="linear",
    start_token_initializer="ones",
)

disc = Discriminator(
    output_units=1,
    latent_dim=8,
    deepsets_num_layers=2,
    deepsets_hidden_units=32,
    output_activation=None,
    dropout_rate=0.1,
)


@pytest.fixture
def loss():
    from calotron.losses import Huber

    loss_ = Huber(
        delta=0.1,
        alpha=0.5,
        adversarial_metric="binary-crossentropy",
        bce_options={
            "injected_noise_stddev": 0.01,
            "from_logits": False,
            "label_smoothing": 0.1,
        },
        wass_options={
            "lipschitz_regularizer": "alp",
            "lipschitz_penalty": 100.0,
            "lipschitz_penalty_strategy": "one-sided",
        },
        warmup_energy=1e-8,
    )
    return loss_


###########################################################################


def test_loss_configuration(loss):
    from calotron.losses import Huber

    assert isinstance(loss, Huber)
    assert isinstance(loss.delta, float)
    assert isinstance(loss.alpha, float)
    assert isinstance(loss.adversarial_metric, str)
    assert isinstance(loss.bce_options, dict)
    assert isinstance(loss.wass_options, dict)
    assert isinstance(loss.warmup_energy, float)
    assert isinstance(loss.name, str)


@pytest.mark.parametrize(
    "adversarial_metric", ["binary-crossentropy", "wasserstein-distance"]
)
@pytest.mark.parametrize("sample_weight", [weight, None])
def test_loss_use(adversarial_metric, sample_weight):
    from calotron.losses import Huber

    loss = Huber(
        delta=0.1,
        alpha=0.5,
        adversarial_metric=adversarial_metric,
        bce_options={"injected_noise_stddev": 0.01},
        wass_options={"lipschitz_penalty": 100.0},
        warmup_energy=1e-8,
    )
    out = loss.transformer_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=sample_weight,
        training=False,
    )
    assert out.numpy() + 1e-12
    out = loss.discriminator_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=sample_weight,
        training=False,
    )
    assert out.numpy() + 1e-12
