import unittest
import pandas as pd
import jax.numpy as jnp

from src.utils import log_message
from src.preprocess.preprocess import (
    encode_sequence,
    process_labels,
    create_dataset,
    pad_sequences_jax,
    pad_coordinates_jax,
)

class TestPreprocessing(unittest.TestCase):
    """Tests for RNA sequence and label preprocessing."""

    def test_encode_sequence(self):
        """test_encode_sequence"""
        sequence = "ACGUAC"
        expected_encoding = jnp.array([1, 2, 3, 4, 1, 2], dtype=jnp.int32)
        result = encode_sequence(sequence)
        self.assertTrue(jnp.array_equal(result, expected_encoding))
        log_message("encode_sequence test ... ✅", "PASS")

    def test_process_labels(self):
        """test_process_labels"""
        data = {
            "ID": ["seq1_1", "seq1_2", "seq2_1"],
            "x_1": [1.0, 2.0, 3.0],
            "y_1": [4.0, 5.0, 6.0],
            "z_1": [7.0, 8.0, 9.0],
        }
        df = pd.DataFrame(data)
        processed = process_labels(df)

        self.assertIn("seq1", processed)
        self.assertIn("seq2", processed)
        self.assertEqual(processed["seq1"].shape, (2, 3))
        self.assertEqual(processed["seq2"].shape, (1, 3))
        log_message("Processing test labels ... ✅", "PASS")

    def test_create_dataset(self):
        """test_create_dataset"""
        sequences_data = pd.DataFrame({
            "target_id": ["seq1", "seq2", "seq3"],
            "sequence": ["ACG", "UGC", "GAU"]
        })

        labels_dict = {
            "seq1": jnp.array([[1.0, 2.0, 3.0]]),
            "seq2": jnp.array([[4.0, 5.0, 6.0]])
        }

        X, y, target_ids = create_dataset(sequences_data, labels_dict)

        self.assertEqual(len(X), 2)  # seq3 is missing in labels_dict
        self.assertEqual(len(y), 2)
        self.assertEqual(target_ids, ["seq1", "seq2"])
        log_message("Creating dataset ... ✅", "PASS")

    def test_pad_sequences_jax(self):
        """test_pad_sequences_jax"""
        sequences = [jnp.array([1, 2, 3]), jnp.array([4, 5])]
        max_len = 4
        expected_output = jnp.array([
            [1, 2, 3, 0],
            [4, 5, 0, 0]
        ])
        padded_result = pad_sequences_jax(sequences, max_len)
        self.assertTrue(jnp.array_equal(padded_result, expected_output))
        log_message("Creating test pad sequences ... ✅", "PASS")

    def test_pad_coordinates_jax(self):
        """test_pad_coordinates_jax"""
        coords = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        max_len = 4
        padded_result = pad_coordinates_jax(coords, max_len)

        expected_output = jnp.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        self.assertTrue(jnp.array_equal(padded_result, expected_output))
        log_message("Creating test pad coordinates ... ✅", "PASS")

if __name__ == "__main__":
    unittest.main(verbosity=0)
