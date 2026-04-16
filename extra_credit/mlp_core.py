"""Inference-only MLP forward pass (copied from mlp_model.ipynb). Used by the Streamlit app."""

from __future__ import annotations

import numpy as np


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


ACTIVATIONS = {
    "relu": relu,
    "tanh": tanh,
    "sigmoid": sigmoid,
}


def forward_pass(X, weights, biases, activations):
    layers_inputs = [X]
    layers_logits = []
    next_layer_inputs = X
    for layer in range(len(weights) - 1):
        logits = next_layer_inputs.dot(weights[layer]) + biases[layer]
        layers_logits.append(logits)
        next_layer_inputs = activations[layer](logits)
        layers_inputs.append(next_layer_inputs)
    predictions = next_layer_inputs.dot(weights[-1]) + biases[-1]
    layers_logits.append(predictions)
    return layers_inputs, layers_logits, predictions


def MLP_predict_regression(X, weights, biases, activations):
    return forward_pass(X, weights, biases, activations)[2]


KPB_FEATURES = ["KADJ O", "KADJ D", "BARTHAG", "SEED", "EXP", "TALENT", "ELITE SOS", "BADJ T"]


def feature_diff_names() -> list[str]:
    return [f"DIFF_{c}" for c in KPB_FEATURES]
