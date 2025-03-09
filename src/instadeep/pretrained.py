# Copyright 2024 InstaDeep Ltd
#
# Licensed under the Creative Commons BY-NC-SA 4.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
from typing import Callable

import haiku as hk
import jax.numpy as jnp
import joblib

from config import BulkRNABertConfig
from model import BulkRNABert
from tokenizer import BinnedExpressionTokenizer

CHECKPOINT_DIRECTORY = "../../models/"


def get_pretrained_model(
    model_name: str,
    compute_dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    output_dtype: jnp.dtype = jnp.float32,
    embeddings_layers_to_save: tuple[int, ...] = (),
    checkpoint_directory: str = CHECKPOINT_DIRECTORY,
) -> tuple[hk.Params, Callable, BinnedExpressionTokenizer, BulkRNABertConfig]:
    """
    Create a Haiku Nucleotide Transformer
    model by downloading pre-trained weights and hyperparameters.
    Nucleotide Transformer Models have ESM-like architectures.

    Args:
        model_name: Name of the model.
        compute_dtype: the type of the activations. fp16 runs faster and is lighter in
            memory. bf16 handles better large int, and is hence more stable ( it avoids
            float overflows ).
        param_dtype: if compute_dtype is fp16, the model weights will be cast to fp16
            during the forward pass anyway. So in inference mode ( not training mode ),
            it is better to use params in fp16 if compute_dtype is fp16 too. During
            training, it is preferable to keep parameters in float32 for better
            numerical stability.
        output_dtype: the output type of the model. it determines the float precision
            of the gradient when training the model.
        embeddings_layers_to_save: Intermediate embeddings to return in the output.
        checkpoint_directory: name of the folder where checkpoints are stored.

    Returns:
        Model parameters.
        Haiku function to call the model.
        Tokenizer.
        Model config (hyperparameters).

    """
    checkpoint_path = pathlib.Path(checkpoint_directory) / model_name
    print(f"🔬 Loading pre-trained {model_name} model, {checkpoint_path}")

    import json

    with open(checkpoint_path / "config.json", "r") as f:
        config_data = json.load(f)

    config = BulkRNABertConfig(**config_data)  # ✅ Directly pass as keyword arguments
    tokenizer = BinnedExpressionTokenizer(
        n_expressions_bins=config.n_expressions_bins,
        use_max_normalization=config.use_max_normalization,
        normalization_factor=config.normalization_factor,
        prepend_cls_token=False,
    )

    config.embeddings_layers_to_save = embeddings_layers_to_save


    def forward_pass(tokens, attention_mask=None):
        model = BulkRNABert(config=config, name="bulk_bert")
        return model(tokens=tokens, attention_mask=attention_mask)

    # ✅ Wrap inside `hk.transform`
    forward_fn = hk.transform(forward_pass)


    parameters = joblib.load(checkpoint_path / "params.joblib")

    return parameters, forward_fn, tokenizer, config
