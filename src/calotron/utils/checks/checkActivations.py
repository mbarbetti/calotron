from tensorflow import keras


def checkActivations(
    activations, output_length, dtype=None
):  # TODO: add Union[list, None]
    if activations is None:
        return None
    elif isinstance(activations, str):
        return [
            keras.layers.Activation(activations, dtype=dtype)
            for _ in range(output_length)
        ]
    elif isinstance(activations, keras.layers.Layer):
        return [activations for _ in range(output_length)]
    elif isinstance(activations, list):
        if len(activations) != output_length:
            raise ValueError(
                f"`activations` list should have "
                f"{output_length} elements, instead "
                f"{len(activations)} passed"
            )
        checked_activations = list()
        for activation in activations:
            if isinstance(activation, str):
                checked_activations.append(
                    keras.layers.Activation(activation, dtype=dtype)
                )
            elif isinstance(activation, keras.layers.Layer):
                checked_activations.append(activation)
            else:
                raise ValueError(
                    f"`activations` elements should be strings "
                    f"or TensorFlow `Activation` layers, "
                    f"instead {type(activation)} passed"
                )
        return checked_activations
    else:
        raise TypeError(
            f"`activations` should be a string, a TensorFlow "
            f"`Activation` layer, or a list of such objects, "
            f"instead {type(activations)} passed"
        )
