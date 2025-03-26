"""
Prediction script for RNA 3D structure prediction model.
"""

import pickle
from pathlib import Path
import json

import jax.numpy as jnp
import pandas as pd
import numpy as np

from src.model.model import RNAFoldingModel, ModelConfig
from src.preprocess.preprocess import encode_sequence, pad_sequences_jax
from src.utils.utils import log_message, check_jax_device


def load_model_and_params(model_dir):
    """Load model configuration and parameters."""
    model_dir = Path(model_dir)

    # Load configuration
    try:
        with open(model_dir / "config.json", "r") as f:
            config_dict = json.load(f)

        config = ModelConfig()
        for key, value in config_dict.items():
            setattr(config, key, value)

        log_message(f"Loaded model configuration from {model_dir / 'config.json'}")
    except Exception as e:
        log_message(f"Error loading model configuration: {e}", level="ERROR")
        raise

    # Load model parameters (try best model first, fall back to final model)
    try:
        params_path = model_dir / "best_model_params.pkl"
        if not params_path.exists():
            params_path = model_dir / "final_model_params.pkl"

        with open(params_path, "rb") as f:
            params = pickle.load(f)

        log_message(f"Loaded model parameters from {params_path}")
        return config, params
    except Exception as e:
        log_message(f"Error loading model parameters: {e}", level="ERROR")
        raise


def preprocess_sequences(sequences, max_seq_len):
    """Preprocess RNA sequences for model input."""
    # Encode sequences
    encoded_seqs = [encode_sequence(seq) for seq in sequences]

    # Pad sequences to maximum length
    padded_seqs = pad_sequences_jax(encoded_seqs, max_seq_len)

    # The model expects shape (batch_size, seq_len, num_features=2)
    # But we only have sequence information (1 feature), so we need to add
    # a second feature dimension with zeros to match the model's expected shape

    # First expand to add initial feature dimension
    padded_seqs = jnp.expand_dims(padded_seqs, axis=-1)

    # Create a second feature channel (zeros) and concatenate
    # This ensures we have 2 feature channels as the model expects
    batch_size, seq_len, _ = padded_seqs.shape
    zeros = jnp.zeros((batch_size, seq_len, 1), dtype=jnp.float32)
    padded_seqs = jnp.concatenate([padded_seqs, zeros], axis=-1)

    return padded_seqs


def predict_structures(model, params, sequences, batch_size=16):
    """Generate 3D structure predictions for RNA sequences."""
    n_samples = len(sequences)
    predictions = []

    # Process in batches to avoid memory issues
    for i in range(0, n_samples, batch_size):
        batch = sequences[i : i + batch_size]
        # Apply model in inference mode (training=False)
        batch_preds = model.apply(params, batch, training=False)
        predictions.append(batch_preds)

    # Concatenate all batch predictions
    all_predictions = jnp.concatenate(predictions, axis=0)
    return all_predictions


def save_predictions(predictions, target_ids, sequences, output_path):
    """Save predicted 3D coordinates to a detailed submission file."""
    output_path = Path(output_path)

    # Convert to numpy for easier handling
    predictions_np = np.array(predictions)

    # Save predictions as pickle for later use
    with open(output_path.with_suffix(".pkl"), "wb") as f:
        pickle.dump((predictions_np, target_ids), f)

    # Build detailed submission rows
    submission_rows = []
    for i, (target_id, seq) in enumerate(zip(target_ids, sequences)):
        # Get predicted coordinates (shape: [max_len, 3])
        pred_coords = predictions_np[i]

        # Determine actual sequence length
        seq_length = len(seq)

        # Only use coordinates for actual residues (not padding)
        pred_coords = pred_coords[:seq_length, :]

        # For each residue, create a row in the submission file
        for j in range(seq_length):
            coords = pred_coords[j, :]

            # Create row with replicated coordinates (5 times for each dimension)
            row_data = {
                "ID": f"{target_id}_{j+1}",
                "resname": seq[j],
                "resid": j + 1,
            }

            # Add x_1 through x_5 (all same value)
            for k in range(5):
                row_data[f"x_{k+1}"] = float(coords[0])
                row_data[f"y_{k+1}"] = float(coords[1])
                row_data[f"z_{k+1}"] = float(coords[2])

            submission_rows.append(row_data)

    # Save as CSV
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_path.with_suffix(".csv"), index=False)

    log_message(f"Created submission with {len(submission_rows)} entries")
    log_message(
        f"Saved predictions to {output_path.with_suffix('.csv')} and {output_path.with_suffix('.pkl')}"
    )

    # Print summary
    print("Submission DataFrame shape:", submission_df.shape)
    print(submission_df.head(5))


def main():
    """Main function for prediction."""
    # Define variables directly instead of parsing arguments
    input_file = "data/raw/test_sequences.csv"  # Path to input sequences CSV file
    model_dir = (
        "models/rna_transformer_20250326_165146"  # Directory containing trained model
    )
    output_file = "results/predictions.csv"  # Path to save predictions
    batch_size = 1  # Batch size for prediction

    # Check JAX device
    device = check_jax_device()
    log_message(f"Using device for prediction: {device}")

    # Load model configuration and parameters
    config, params = load_model_and_params(model_dir)

    # Create model instance
    model = RNAFoldingModel(config).model

    # Load input sequences
    try:
        input_df = pd.read_csv(input_file)
        log_message(f"Loaded {len(input_df)} sequences from {input_file}")
    except Exception as e:
        log_message(f"Error loading input sequences: {e}", level="ERROR")
        raise

    # Check required columns
    if "sequence" not in input_df.columns or "target_id" not in input_df.columns:
        log_message(
            "Input file must contain 'sequence' and 'target_id' columns", level="ERROR"
        )
        raise ValueError("Missing required columns in input file")

    # Preprocess sequences
    sequences = input_df["sequence"].tolist()
    target_ids = input_df["target_id"].tolist()

    log_message(f"Preprocessing {len(sequences)} sequences")
    processed_seqs = preprocess_sequences(sequences, config.max_seq_len)

    # Generate predictions
    log_message("Generating structure predictions")
    predictions = predict_structures(model, params, processed_seqs, batch_size)

    # Save predictions with the new format
    log_message("Saving predictions")
    save_predictions(predictions, target_ids, sequences, output_file)

    log_message("Prediction completed successfully!")


if __name__ == "__main__":
    main()
