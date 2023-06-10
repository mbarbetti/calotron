import pytest
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer, RMSprop

CHUNK_SIZE = int(1e4)

source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))
weight = tf.random.uniform(shape=(CHUNK_SIZE, target.shape[1]))


@pytest.fixture
def model():
    from calotron.models import Calotron
    from calotron.models.discriminators import Discriminator
    from calotron.models.transformers import Transformer

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
        latent_dim=8,
        output_units=1,
        output_activation="sigmoid",
        deepsets_dense_num_layers=2,
        deepsets_dense_units=32,
        dropout_rate=0.1,
    )

    calo = Calotron(transformer=transf, discriminator=disc)
    return calo


###########################################################################


def test_model_configuration(model):
    from calotron.models import Calotron
    from calotron.models.discriminators import Discriminator
    from calotron.models.transformers import Transformer

    assert isinstance(model, Calotron)
    assert isinstance(model.transformer, Transformer)
    assert isinstance(model.discriminator, Discriminator)


def test_model_use(model):
    outputs = model((source, target))
    t_output, d_output_true, d_output_pred = outputs
    model.summary()

    test_t_shape = list(target.shape)
    test_t_shape[-1] = model.transformer.output_depth
    assert t_output.shape == tuple(test_t_shape)

    test_d_shape = [target.shape[0]]
    test_d_shape.append(model.discriminator.output_units)
    assert d_output_true.shape == tuple(test_d_shape)
    assert d_output_pred.shape == tuple(test_d_shape)


@pytest.mark.parametrize("metrics", [["bce"], None])
def test_model_compilation(model, metrics):
    from calotron.losses import MeanSquaredError

    loss = MeanSquaredError(
        warmup_energy=0.0, alpha=0.1, adversarial_metric="binary-crossentropy"
    )
    t_opt = RMSprop(learning_rate=0.001)
    d_opt = RMSprop(learning_rate=0.001)
    model.compile(
        loss=loss,
        metrics=metrics,
        transformer_optimizer=t_opt,
        discriminator_optimizer=d_opt,
        transformer_upds_per_batch=1,
        discriminator_upds_per_batch=1,
    )
    assert isinstance(model.metrics, list)
    assert isinstance(model.transformer_optimizer, Optimizer)
    assert isinstance(model.discriminator_optimizer, Optimizer)
    assert isinstance(model.transformer_upds_per_batch, int)
    assert isinstance(model.discriminator_upds_per_batch, int)


@pytest.mark.parametrize(
    "adversarial_metrics", ["binary-crossentropy", "wasserstein-distance"]
)
@pytest.mark.parametrize("sample_weight", [weight, None])
def test_model_train(model, adversarial_metrics, sample_weight):
    if sample_weight is not None:
        slices = (source, target, weight)
    else:
        slices = (source, target)
    dataset = (
        tf.data.Dataset.from_tensor_slices(slices)
        .batch(batch_size=512, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    from calotron.losses import MeanSquaredError

    loss = MeanSquaredError(
        warmup_energy=0.0, alpha=0.1, adversarial_metric=adversarial_metrics
    )
    t_opt = RMSprop(learning_rate=0.001)
    d_opt = RMSprop(learning_rate=0.001)
    model.compile(
        loss=loss,
        metrics=None,
        transformer_optimizer=t_opt,
        discriminator_optimizer=d_opt,
        transformer_upds_per_batch=1,
        discriminator_upds_per_batch=1,
    )
    model.fit(dataset, epochs=2)


@pytest.mark.parametrize("sample_weight", [weight, None])
def test_model_eval(model, sample_weight):
    from calotron.losses import MeanSquaredError

    loss = MeanSquaredError(
        warmup_energy=0.0, alpha=0.1, adversarial_metric="binary-crossentropy"
    )
    t_opt = RMSprop(learning_rate=0.001)
    d_opt = RMSprop(learning_rate=0.001)
    model.compile(
        loss=loss,
        metrics=None,
        transformer_optimizer=t_opt,
        discriminator_optimizer=d_opt,
        transformer_upds_per_batch=1,
        discriminator_upds_per_batch=1,
    )
    if sample_weight is not None:
        model.evaluate(source, target, sample_weight=weight)
    else:
        model.evaluate(source, target)
