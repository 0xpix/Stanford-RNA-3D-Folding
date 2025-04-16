"""
MSA Generator for RNA 3D Structure Prediction

This module handles the generation of Multiple Sequence Alignments (MSAs) for RNA sequences,
which are used as inputs for RNA 3D structure prediction models.

Key features:
1. During training: Builds MSAs using all_sequences and falls back to BLAST only if needed
2. During inference: Uses precomputed MSA files without running BLAST
3. Computes MSA features: conservation scores, nucleotide frequencies, gap rates, Neff
4. Fallback to raw sequence features if insufficient sequences for MSA

Usage:
    # For training data with fallback to BLAST
    python -m src.data.generate_msa_with_homologs --mode training --input data/raw/train_sequences.csv

    # For inference without using BLAST
    python -m src.data.generate_msa_with_homologs --mode inference --input data/raw/test_sequences.csv
"""

import os
import io
import time
import logging
import argparse
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("msa_generator")

# Define constants
MIN_SEQ_LENGTH = 8
MIN_SEQ_LENGTH_RELAXED = 6  # Relaxed sequence length for BLAST retry
MIN_SEQ_LENGTH_VERY_RELAXED = 5  # Even more relaxed for desperate cases
MIN_SEQUENCES_BEFORE_BLAST = 5  # Minimum sequences before skipping BLAST
MAX_HOMOLOGS = 30
MAX_HOMOLOGS_RELAXED = 50  # Increased homolog count for BLAST retry
MAX_HOMOLOGS_VERY_RELAXED = 100  # Even more homologs for desperate cases
MAX_THREADS = 8
NCBI_DELAY = 1.5

# Minimum sequence threshold to trigger raw fallback vs. very relaxed BLAST
MIN_SEQUENCES_FOR_MSA = 2  # Absolute minimum for running Clustal
TARGET_MIN_SEQUENCES = 4  # Target minimum for reasonable MSA (try harder if below this)

# RNA nucleotide set for validation
RNA_NUCLEOTIDES = set("ACGU")


class Mode(Enum):
    """Mode of operation for MSA generation."""

    TRAINING = "training"
    INFERENCE = "inference"


class MSAGenerator:
    """
    Handles generation and processing of Multiple Sequence Alignments for RNA sequences.
    """

    def __init__(
        self,
        mode: Mode,
        input_csv: str,
        output_dir: str = "data/processed/msa",
        raw_msa_dir: str = "data/raw/msa",
        blast_cache_dir: str = "data/processed/blast_cache",
        max_threads: int = MAX_THREADS,
        min_sequences: int = MIN_SEQUENCES_BEFORE_BLAST,
        use_all_sequences: bool = True,
        keep_failed_aln: bool = False,
    ):
        """
        Initialize the MSA generator.

        Args:
            mode: Whether in training or inference mode
            input_csv: Path to CSV file with RNA sequences
            output_dir: Where to store processed MSA files and features
            raw_msa_dir: Where to find precomputed MSA files (for inference)
            blast_cache_dir: Where to cache BLAST results
            max_threads: Maximum number of parallel threads
            min_sequences: Minimum sequences needed before skipping BLAST
            use_all_sequences: Whether to use sequences from all_sequences column
            keep_failed_aln: Whether to keep .aln files even when Clustal fails
        """
        self.mode = mode
        self.input_csv = input_csv
        self.output_dir = Path(output_dir)
        self.raw_msa_dir = Path(raw_msa_dir)
        self.blast_cache_dir = Path(blast_cache_dir)
        self.max_threads = max_threads
        self.min_sequences = min_sequences
        self.use_all_sequences = use_all_sequences
        self.keep_failed_aln = keep_failed_aln

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.blast_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.df = pd.read_csv(input_csv)
        self.targets = self.df[
            ["target_id", "sequence", "all_sequences"]
        ].drop_duplicates()
        logger.info(f"Loaded {len(self.targets)} unique targets from {input_csv}")

        # Track statistics
        self.stats = {
            "total": 0,
            "precomputed_msa_used": 0,
            "all_sequences_used": 0,
            "blast_used": 0,
            "blast_used_relaxed": 0,  # New stat for relaxed BLAST usage
            "blast_failed": 0,
            "clustal_failed": 0,
            "clustal_timeout": 0,
            "insufficient_sequences": 0,
            "raw_features_fallback": 0,  # New stat for raw sequence fallback
            "success": 0,
        }

    def write_fasta(
        self, seq: str, name: str, path: Union[str, Path], mode: str = "w"
    ) -> None:
        """Write a sequence to a FASTA file."""
        with open(path, mode) as f:
            f.write(f">{name}\n{seq}\n")

    def count_sequences_in_fasta(self, fasta_path: Union[str, Path]) -> int:
        """
        Count the number of sequences in a FASTA file by counting '>' characters.

        Args:
            fasta_path: Path to the FASTA file

        Returns:
            Number of sequences in the file
        """
        try:
            with open(fasta_path, "r") as f:
                content = f.read()
                # Count the number of '>' characters at the beginning of lines
                return content.count(">")
        except Exception as e:
            logger.error(f"Error counting sequences in {fasta_path}: {str(e)}")
            return 0

    def is_valid_rna(self, seq: str, min_length: int = MIN_SEQ_LENGTH) -> bool:
        """Check if a sequence is a valid RNA sequence."""
        if len(seq) < min_length:
            return False
        return all(c in RNA_NUCLEOTIDES for c in seq)

    def parse_all_sequences_block(
        self, all_sequences_str: str
    ) -> List[Tuple[str, str]]:
        """
        Parse the all_sequences block from the CSV file, which is in FASTA format.

        Returns:
            List of (header, sequence) tuples for valid RNA sequences
        """
        sequences = []
        if pd.isna(all_sequences_str):
            return sequences

        for entry in all_sequences_str.split(">"):
            lines = entry.strip().split("\n")
            if len(lines) < 2:
                continue

            header = lines[0].strip()
            seq = "".join(lines[1:]).strip().replace(" ", "").replace("-", "").upper()

            if self.is_valid_rna(seq):
                sequences.append((header, seq))
            else:
                logger.debug(f"Skipped non-RNA or invalid sequence: {header}")

        return sequences

    def load_precomputed_msa(self, target_id: str) -> Optional[Path]:
        """
        Check for a precomputed MSA file for this target.

        Returns:
            Path to the MSA file if found, None otherwise
        """
        # Check different possible file naming patterns
        candidates = [
            self.raw_msa_dir / f"{target_id}.MSA.fasta",
            self.raw_msa_dir / f"{target_id}.a2m",
            self.raw_msa_dir / f"{target_id}.aln",
            self.raw_msa_dir / f"{target_id}.msa",
        ]

        for path in candidates:
            if path.exists():
                logger.info(f"Found precomputed MSA file: {path}")
                return path

        return None

    def run_blast(
        self,
        seq: str,
        target_id: str,
        max_retries: int = 3,
        delay: int = 10,
        relaxed: bool = False,
    ) -> Optional[str]:
        """
        Run BLAST to find homologous sequences.

        Args:
            seq: RNA sequence to search for homologs
            target_id: The RNA target identifier
            max_retries: Maximum number of retry attempts
            delay: Delay between retries in seconds
            relaxed: Whether to use relaxed settings (more homologs, shorter sequences)

        Returns:
            The raw XML BLAST result as a string, or None if BLAST failed
        """
        # Skip BLAST in inference mode
        if self.mode == Mode.INFERENCE:
            logger.info(f"Skipping BLAST for {target_id} (inference mode)")
            return None

        # Modify cache file name if using relaxed settings
        cache_filename = f"{target_id}{'.relaxed' if relaxed else ''}.xml"
        xml_path = self.blast_cache_dir / cache_filename

        # Check cache first
        if xml_path.exists():
            with open(xml_path) as f:
                logger.info(
                    f"Using cached BLAST results for {target_id}{' (relaxed settings)' if relaxed else ''}"
                )
                return f.read()

        # Set parameters based on relaxed flag
        hitlist_size = MAX_HOMOLOGS_RELAXED if relaxed else MAX_HOMOLOGS

        # Run BLAST with retries
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Running BLAST for {target_id} with {'relaxed' if relaxed else 'standard'} settings "
                    f"(attempt {attempt}/{max_retries})"
                )
                result_handle = NCBIWWW.qblast(
                    "blastn", "nt", seq, hitlist_size=hitlist_size
                )
                result = result_handle.read()

                # Cache the result
                with open(xml_path, "w") as f:
                    f.write(result)

                # Update statistics
                if relaxed:
                    self.stats["blast_used_relaxed"] += 1
                else:
                    self.stats["blast_used"] += 1
                return result

            except Exception as e:
                logger.warning(
                    f"BLAST failed for {target_id} (attempt {attempt}): {str(e)}"
                )
                if attempt < max_retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"BLAST permanently failed for {target_id} after {max_retries} attempts"
                    )
                    self.stats["blast_failed"] += 1
                    return None

    def extract_homologs(
        self, xml_data: str, relaxed: bool = False, very_relaxed: bool = False
    ) -> List[Tuple[str, str]]:
        """
        Extract homologous sequences from BLAST XML results.

        Args:
            xml_data: BLAST XML result data
            relaxed: Whether to use relaxed validation criteria (shorter sequences)
            very_relaxed: Whether to use very relaxed validation criteria (even shorter sequences)

        Returns:
            List of (header, sequence) tuples for valid RNA homologs
        """
        homologs = []
        # Select appropriate thresholds based on relaxation level
        if very_relaxed:
            min_length = MIN_SEQ_LENGTH_VERY_RELAXED
            max_homologs = MAX_HOMOLOGS_VERY_RELAXED
        elif relaxed:
            min_length = MIN_SEQ_LENGTH_RELAXED
            max_homologs = MAX_HOMOLOGS_RELAXED
        else:
            min_length = MIN_SEQ_LENGTH
            max_homologs = MAX_HOMOLOGS

        try:
            blast_record = NCBIXML.read(io.StringIO(xml_data))
            for alignment in blast_record.alignments:
                for hsp in alignment.hsps:
                    seq = hsp.sbjct.replace("-", "").replace(" ", "").upper()
                    # Check if sequence is valid RNA using appropriate length threshold
                    if len(seq) >= min_length and all(
                        c in RNA_NUCLEOTIDES for c in seq
                    ):
                        homologs.append((alignment.hit_id, seq))
                    else:
                        if len(seq) < min_length:
                            logger.debug(
                                f"Skipped homolog (too short): {alignment.hit_id} [len={len(seq)}]"
                            )
                        else:
                            logger.debug(f"Skipped non-RNA homolog: {alignment.hit_id}")

                    if len(homologs) >= max_homologs:
                        return homologs
        except Exception as e:
            logger.error(f"Error parsing BLAST XML: {str(e)}")

        return homologs

    def generate_raw_sequence_features(self, sequence: str) -> Dict:
        """
        Generate one-hot encoded features directly from a raw RNA sequence.
        Used as a fallback when MSA cannot be generated.

        Args:
            sequence: RNA sequence string

        Returns:
            Dictionary with the same feature keys as MSA features
        """
        seq_len = len(sequence)

        # Initialize one-hot encoded matrices
        freq_A = np.zeros(seq_len, dtype=np.float32)
        freq_C = np.zeros(seq_len, dtype=np.float32)
        freq_G = np.zeros(seq_len, dtype=np.float32)
        freq_U = np.zeros(seq_len, dtype=np.float32)

        # Fill one-hot vectors
        for i, nt in enumerate(sequence):
            if nt == "A":
                freq_A[i] = 1.0
            elif nt == "C":
                freq_C[i] = 1.0
            elif nt == "G":
                freq_G[i] = 1.0
            elif nt == "U":
                freq_U[i] = 1.0

        # Conservation is 1.0 for all positions (since there's only one sequence)
        conservation = np.ones(seq_len, dtype=np.float32)

        # Gap frequency is 0 for all positions
        gap_freq = np.zeros(seq_len, dtype=np.float32)

        # Track that we're using a fallback with neff=1
        neff = np.array([1], dtype=np.int32)

        logger.info(f"Generated raw sequence features for sequence of length {seq_len}")

        return {
            "conservation": conservation,
            "freq_A": freq_A,
            "freq_C": freq_C,
            "freq_G": freq_G,
            "freq_U": freq_U,
            "gap_freq": gap_freq,
            "neff": neff,
        }

    def save_features_to_npz(self, features: Dict, output_path: Path) -> bool:
        """
        Save features to an NPZ file.

        Args:
            features: Dictionary of feature arrays
            output_path: Path to save the NPZ file

        Returns:
            True if successful, False otherwise
        """
        try:
            np.savez(
                output_path,
                conservation=features["conservation"].astype(np.float32),
                freq_A=features["freq_A"].astype(np.float32),
                freq_C=features["freq_C"].astype(np.float32),
                freq_G=features["freq_G"].astype(np.float32),
                freq_U=features["freq_U"].astype(np.float32),
                gap_freq=features["gap_freq"].astype(np.float32),
                neff=features["neff"].astype(np.int32),
            )
            return True
        except Exception as e:
            logger.error(f"Error saving features to {output_path}: {str(e)}")
            return False

    def run_clustalo(self, input_file: Path, output_file: Path) -> bool:
        """
        Run Clustal Omega to align sequences.

        Returns:
            True if alignment succeeded, False otherwise
        """
        # Check if the input file contains enough sequences to align
        seq_count = self.count_sequences_in_fasta(input_file)
        logger.info(f"Found {seq_count} sequences in {input_file} for alignment")

        # Need at least 2 sequences to perform an alignment
        if seq_count < 2:
            logger.warning(
                f"Insufficient sequences for alignment: {input_file} has only {seq_count} sequence(s)"
            )
            self.stats["insufficient_sequences"] += 1
            return False

        cmd = [
            "clustalo",
            "-i",
            str(input_file),
            "-o",
            str(output_file),
            "--outfmt",
            "fasta",
            "--force",
            "--auto",
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,  # 5-minute timeout
            )

            if result.returncode != 0:
                logger.error(f"Clustal Omega failed: {result.stderr}")
                self.stats["clustal_failed"] += 1
                return False

            return True
        except subprocess.TimeoutExpired:
            logger.error("Clustal Omega timed out after 300 seconds")
            self.stats["clustal_timeout"] += 1

            # Check if output file was created despite timeout
            if output_file.exists():
                logger.warning(
                    f"Partial output file found at {output_file} after timeout"
                )
                if self.keep_failed_aln:
                    logger.info(
                        "Keeping partial MSA file due to --keep-aln-failed flag"
                    )
                    return False
                else:
                    # Don't delete the output file as it might have partial results
                    return False
            return False
        except Exception as e:
            logger.error(f"Error running Clustal Omega: {str(e)}")
            self.stats["clustal_failed"] += 1
            return False

    def extract_msa_features(self, msa_path: Path, feature_out_path: Path) -> bool:
        """
        Extract features from an MSA file and save them to NPZ format.

        Features include:
        - Conservation scores
        - Nucleotide frequencies
        - Gap frequencies
        - Number of effective sequences (Neff)

        Returns:
            True if extraction succeeded, False otherwise
        """
        try:
            # Read sequences from MSA file
            records = list(SeqIO.parse(msa_path, "fasta"))
            if not records:
                logger.warning(f"No sequences in MSA: {msa_path}")
                return False

            # Extract sequences and calculate frequencies
            sequences = [str(rec.seq).upper() for rec in records]
            n_seq = len(sequences)

            # Convert to numpy array for vectorized operations
            msa_mat = np.array([list(seq) for seq in sequences])

            # Calculate nucleotide frequencies
            freq_A = np.sum(msa_mat == "A", axis=0) / n_seq
            freq_C = np.sum(msa_mat == "C", axis=0) / n_seq
            freq_G = np.sum(msa_mat == "G", axis=0) / n_seq
            freq_U = np.sum(msa_mat == "U", axis=0) / n_seq
            freq_gap = np.sum(msa_mat == "-", axis=0) / n_seq

            # Calculate conservation (max frequency at each position)
            conservation = np.max(np.stack([freq_A, freq_C, freq_G, freq_U]), axis=0)

            # Save features to NPZ file
            np.savez(
                feature_out_path,
                conservation=conservation.astype(np.float32),
                freq_A=freq_A.astype(np.float32),
                freq_C=freq_C.astype(np.float32),
                freq_G=freq_G.astype(np.float32),
                freq_U=freq_U.astype(np.float32),
                gap_freq=freq_gap.astype(np.float32),
                neff=np.array([n_seq], dtype=np.int32),
            )

            logger.info(
                f"Saved MSA features for {n_seq} sequences to {feature_out_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return False

    def process_target(self, row) -> Dict:
        """
        Process a single RNA target: generate MSA and extract features.

        Returns:
            Dictionary with statistics about the processing
        """
        target_id = row.target_id
        sequence = row.sequence
        all_sequences_raw = row.all_sequences

        result = {
            "target_id": target_id,
            "status": "skipped",
            "sequences_count": 1,  # Start with query sequence
            "used_precomputed": False,
            "used_all_sequences": False,
            "used_blast": False,
            "used_blast_relaxed": False,  # Standard relaxed BLAST
            "used_blast_very_relaxed": False,  # Even more relaxed BLAST
            "used_raw_fallback": False,  # Raw sequence fallback
        }

        # Define output file paths
        msa_out = self.output_dir / f"{target_id}.aln"
        feature_out = self.output_dir / f"{target_id}_features.npz"

        # Skip if already processed
        if msa_out.exists() and feature_out.exists():
            result["status"] = "already_processed"
            return result

        logger.info(f"Processing target: {target_id}")
        self.stats["total"] += 1

        # Check for precomputed MSA
        precomputed_msa = self.load_precomputed_msa(target_id)
        if precomputed_msa:
            # Copy precomputed MSA to output dir if needed
            if not msa_out.exists() or os.path.getmtime(
                precomputed_msa
            ) > os.path.getmtime(msa_out):
                logger.info(f"Using precomputed MSA for {target_id}")
                self.stats["precomputed_msa_used"] += 1
                result["used_precomputed"] = True

                with open(precomputed_msa, "r") as src, open(msa_out, "w") as dst:
                    dst.write(src.read())

            # Extract features from the MSA
            if self.extract_msa_features(msa_out, feature_out):
                result["status"] = "success"
                self.stats["success"] += 1
                return result

        # Create temporary FASTA file with the query sequence
        with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as temp:
            temp_path = Path(temp.name)
            self.write_fasta(sequence, f"{target_id}_query", temp_path)

        try:
            # Parse additional sequences from all_sequences if enabled
            additional_seqs = []
            if self.use_all_sequences:
                additional_seqs = self.parse_all_sequences_block(all_sequences_raw)
                if additional_seqs:
                    logger.info(
                        f"Found {len(additional_seqs)} valid RNA sequences in all_sequences for {target_id}"
                    )
                    self.stats["all_sequences_used"] += 1
                    result["used_all_sequences"] = True
                    result["sequences_count"] += len(additional_seqs)

                    # Append to temporary FASTA - fix: use append mode
                    with open(temp_path, "a") as f:
                        for header, seq in additional_seqs:
                            f.write(f">{header}\n{seq}\n")

                    # Verify sequences were written properly
                    seq_count = self.count_sequences_in_fasta(temp_path)
                    logger.info(
                        f"After adding all_sequences: {seq_count} sequences in FASTA for {target_id}"
                    )

            # Run BLAST only if in training mode
            if self.mode == Mode.TRAINING:

                # ===== First BLAST attempt with standard settings =====
                xml_data = self.run_blast(sequence, target_id)
                homologs = []

                if xml_data:
                    homologs = self.extract_homologs(xml_data)
                    if homologs:
                        logger.info(
                            f"Found {len(homologs)} homologs via BLAST for {target_id}"
                        )
                        result["used_blast"] = True
                        result["sequences_count"] += len(homologs)

                        # Append homologs to temporary FASTA
                        with open(temp_path, "a") as f:
                            for i, (hit_id, hseq) in enumerate(homologs):
                                f.write(f">homolog_{i}\n{hseq}\n")

                        # Verify sequences were written properly after BLAST
                        seq_count = self.count_sequences_in_fasta(temp_path)
                        logger.info(
                            f"After adding BLAST homologs: {seq_count} sequences in FASTA for {target_id}"
                        )

                # Check if we have enough sequences yet
                seq_count = self.count_sequences_in_fasta(temp_path)

                # ===== Second BLAST attempt with relaxed settings =====
                # Try relaxed settings if we still need more sequences
                if seq_count < TARGET_MIN_SEQUENCES:
                    logger.info(
                        f"Trying BLAST with relaxed settings for {target_id} (current sequences: {seq_count})"
                    )
                    xml_data_relaxed = self.run_blast(sequence, target_id, relaxed=True)

                    if xml_data_relaxed:
                        homologs_relaxed = self.extract_homologs(
                            xml_data_relaxed, relaxed=True
                        )
                        if homologs_relaxed:
                            logger.info(
                                f"Found {len(homologs_relaxed)} homologs via relaxed BLAST for {target_id}"
                            )
                            result["used_blast_relaxed"] = True
                            result["sequences_count"] += len(homologs_relaxed)

                            # Append relaxed homologs to temporary FASTA
                            with open(temp_path, "a") as f:
                                for i, (hit_id, hseq) in enumerate(homologs_relaxed):
                                    f.write(f">homolog_relaxed_{i}\n{hseq}\n")

                            # Get updated sequence count
                            seq_count = self.count_sequences_in_fasta(temp_path)
                            logger.info(
                                f"After adding relaxed BLAST homologs: {seq_count} sequences in FASTA for {target_id}"
                            )

                # ===== Third BLAST attempt with very relaxed settings =====
                # Try with very relaxed settings if we still don't have enough sequences
                if seq_count < MIN_SEQUENCES_FOR_MSA:
                    logger.warning(
                        f"Still insufficient sequences ({seq_count}). Trying BLAST with VERY relaxed settings for {target_id}"
                    )

                    # Define cache filename for very relaxed BLAST
                    cache_filename = f"{target_id}.very_relaxed.xml"
                    xml_path = self.blast_cache_dir / cache_filename

                    # Check cache first
                    xml_data_very_relaxed = None
                    if xml_path.exists():
                        with open(xml_path) as f:
                            logger.info(
                                f"Using cached very relaxed BLAST results for {target_id}"
                            )
                            xml_data_very_relaxed = f.read()
                    else:
                        # Run with increased hitlist size
                        try:
                            logger.info(
                                f"Running BLAST for {target_id} with very relaxed settings"
                            )
                            result_handle = NCBIWWW.qblast(
                                "blastn",
                                "nt",
                                sequence,
                                hitlist_size=MAX_HOMOLOGS_VERY_RELAXED,
                                expect=100.0,  # Very high e-value threshold
                            )
                            xml_data_very_relaxed = result_handle.read()

                            # Cache the result
                            with open(xml_path, "w") as f:
                                f.write(xml_data_very_relaxed)

                            # Track stats
                            self.stats["blast_used_very_relaxed"] = (
                                self.stats.get("blast_used_very_relaxed", 0) + 1
                            )

                        except Exception as e:
                            logger.error(
                                f"Very relaxed BLAST failed for {target_id}: {str(e)}"
                            )
                            xml_data_very_relaxed = None

                    # Extract homologs with very relaxed criteria if we got results
                    if xml_data_very_relaxed:
                        homologs_very_relaxed = self.extract_homologs(
                            xml_data_very_relaxed, very_relaxed=True
                        )
                        if homologs_very_relaxed:
                            logger.info(
                                f"Found {len(homologs_very_relaxed)} homologs via VERY relaxed BLAST for {target_id}"
                            )
                            result["used_blast_very_relaxed"] = True
                            result["sequences_count"] += len(homologs_very_relaxed)

                            # Append very relaxed homologs to temporary FASTA
                            with open(temp_path, "a") as f:
                                for i, (hit_id, hseq) in enumerate(
                                    homologs_very_relaxed
                                ):
                                    f.write(f">homolog_very_relaxed_{i}\n{hseq}\n")

                            seq_count = self.count_sequences_in_fasta(temp_path)
                            logger.info(
                                f"After adding VERY relaxed BLAST homologs: {seq_count} sequences in FASTA for {target_id}"
                            )

            # Final check of sequence count before alignment
            seq_count = self.count_sequences_in_fasta(temp_path)
            logger.info(
                f"Final check before alignment: {seq_count} sequences for {target_id}"
            )

            # Fallback to raw sequence features if we have insufficient sequences for MSA
            if seq_count < MIN_SEQUENCES_FOR_MSA:
                logger.warning(
                    f"Insufficient sequences for MSA: {target_id} has only {seq_count} sequence(s). Using raw sequence features fallback."
                )
                self.stats["insufficient_sequences"] += 1
                self.stats["raw_features_fallback"] += 1
                result["used_raw_fallback"] = True

                # Generate raw sequence features
                features = self.generate_raw_sequence_features(sequence)

                # Save features to NPZ file
                if self.save_features_to_npz(features, feature_out):
                    result["status"] = "success_raw_fallback"
                    self.stats["success"] += 1
                    logger.info(
                        f"Successfully generated raw sequence features for {target_id}"
                    )
                else:
                    result["status"] = "raw_feature_generation_failed"

                # We don't need the MSA output file for raw sequence fallback
                if msa_out.exists():
                    os.remove(msa_out)

                return result

            # Run Clustal Omega alignment
            clustal_success = self.run_clustalo(temp_path, msa_out)

            if clustal_success:
                # Extract features from the alignment
                if self.extract_msa_features(msa_out, feature_out):
                    result["status"] = "success"
                    self.stats["success"] += 1
                else:
                    result["status"] = "feature_extraction_failed"

                    # Fallback to raw features if MSA feature extraction fails
                    logger.warning(
                        f"MSA feature extraction failed for {target_id}. Falling back to raw sequence features."
                    )
                    features = self.generate_raw_sequence_features(sequence)

                    if self.save_features_to_npz(features, feature_out):
                        result["status"] = "success_raw_fallback"
                        self.stats["raw_features_fallback"] += 1
                        result["used_raw_fallback"] = True
                        self.stats["success"] += 1
                    else:
                        result["status"] = "all_feature_generation_failed"
            else:
                # Check if the output file exists despite clustal failure (e.g., timeout)
                if msa_out.exists() and self.keep_failed_aln:
                    logger.info(
                        f"Keeping failed alignment file: {msa_out} due to --keep-aln-failed flag"
                    )
                    if seq_count < MIN_SEQUENCES_FOR_MSA:
                        result["status"] = "insufficient_sequences"
                    else:
                        # Check if it was a timeout or other failure
                        result["status"] = (
                            "clustal_timeout"
                            if self.stats["clustal_timeout"]
                            > self.stats["clustal_failed"]
                            else "clustal_failed"
                        )
                elif seq_count < MIN_SEQUENCES_FOR_MSA:
                    result["status"] = "insufficient_sequences"
                else:
                    # Could be either timeout or general failure
                    result["status"] = (
                        "clustal_timeout"
                        if self.stats["clustal_timeout"] > self.stats["clustal_failed"]
                        else "clustal_failed"
                    )

                    # Remove the output file if it exists and we don't want to keep it
                    if msa_out.exists() and not self.keep_failed_aln:
                        logger.info(f"Removing failed alignment file: {msa_out}")
                        msa_out.unlink()

                # FALLBACK: Generate raw sequence features if alignment failed
                logger.warning(
                    f"MSA generation failed for {target_id}. Falling back to raw sequence features."
                )
                features = self.generate_raw_sequence_features(sequence)

                if self.save_features_to_npz(features, feature_out):
                    result["status"] = "success_raw_fallback"
                    self.stats["raw_features_fallback"] += 1
                    result["used_raw_fallback"] = True
                    self.stats["success"] += 1
                    logger.info(
                        f"Successfully generated raw sequence features for {target_id}"
                    )
                else:
                    result["status"] = "all_feature_generation_failed"

        finally:
            # Clean up temporary file
            if temp_path.exists():
                os.remove(temp_path)

        # Add polite delay when using BLAST
        if (
            result["used_blast"]
            or result["used_blast_relaxed"]
            or result["used_blast_very_relaxed"]
        ):
            time.sleep(NCBI_DELAY)

        return result

    def generate_all(self, batch_size: int = None) -> Dict:
        """
        Generate MSAs for all targets in parallel.

        Args:
            batch_size: Optional batch size to process targets in chunks

        Returns:
            Statistics dictionary
        """
        logger.info(
            f"Starting MSA generation in {self.mode.value} mode with {self.max_threads} threads"
        )
        logger.info(f"Keep failed alignment files: {self.keep_failed_aln}")
        logger.info(
            "Multi-level BLAST fallbacks enabled (standard → relaxed → very relaxed)"
        )
        logger.info("Raw sequence fallback enabled for insufficient sequences")

        all_targets = list(self.targets.itertuples(index=False))
        results = []

        # Process all targets in parallel
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # If batch size specified, process in batches to avoid memory issues
            if batch_size:
                for i in range(0, len(all_targets), batch_size):
                    batch = all_targets[i : i + batch_size]
                    logger.info(
                        f"Processing batch {i//batch_size + 1}/{(len(all_targets) + batch_size - 1) // batch_size}"
                    )
                    batch_results = list(
                        tqdm(
                            executor.map(self.process_target, batch),
                            total=len(batch),
                            desc=f"Batch {i//batch_size + 1}",
                        )
                    )
                    results.extend(batch_results)
            else:
                # Process all at once
                results = list(
                    tqdm(
                        executor.map(self.process_target, all_targets),
                        total=len(all_targets),
                        desc="Generating MSAs",
                    )
                )

        # Summarize results
        status_counts = {}
        for result in results:
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        logger.info("=== MSA Generation Summary ===")
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count} targets")

        logger.info(f"Total targets processed: {self.stats['total']}")
        logger.info(f"Precomputed MSAs used: {self.stats['precomputed_msa_used']}")
        logger.info(f"All_sequences data used: {self.stats['all_sequences_used']}")
        logger.info(f"BLAST searches performed: {self.stats['blast_used']}")
        logger.info(
            f"BLAST relaxed searches performed: {self.stats['blast_used_relaxed']}"
        )

        # Report on very relaxed BLAST usage if applicable
        blast_very_relaxed = self.stats.get("blast_used_very_relaxed", 0)
        if blast_very_relaxed > 0:
            logger.info(f"BLAST very relaxed searches performed: {blast_very_relaxed}")

        logger.info(f"BLAST failures: {self.stats['blast_failed']}")
        logger.info(
            f"Insufficient sequences to align: {self.stats['insufficient_sequences']}"
        )
        logger.info(
            f"Raw sequence feature fallbacks: {self.stats['raw_features_fallback']}"
        )
        logger.info(f"Clustal failures: {self.stats['clustal_failed']}")
        logger.info(f"Clustal timeouts: {self.stats['clustal_timeout']}")
        logger.info(f"Successfully processed: {self.stats['success']}")

        return self.stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate MSAs for RNA 3D structure prediction"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["training", "inference"],
        default="training",
        help="Mode of operation (training allows BLAST, inference does not)",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with RNA sequences",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/msa",
        help="Directory to store processed MSA files and features",
    )

    parser.add_argument(
        "--raw-msa-dir",
        type=str,
        default="data/raw/msa",
        help="Directory where precomputed MSA files can be found",
    )

    parser.add_argument(
        "--blast-cache-dir",
        type=str,
        default="data/processed/blast_cache",
        help="Directory to cache BLAST results",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=MAX_THREADS,
        help="Maximum number of parallel threads",
    )

    parser.add_argument(
        "--min-sequences",
        type=int,
        default=MIN_SEQUENCES_BEFORE_BLAST,
        help="Minimum sequences needed before skipping BLAST",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process targets in batches of this size",
    )

    parser.add_argument(
        "--no-all-sequences",
        action="store_true",
        help="Don't use sequences from all_sequences column",
    )

    parser.add_argument(
        "--keep-aln-failed",
        action="store_true",
        help="Keep .aln files even when Clustal fails (for debugging)",
    )

    parser.add_argument(
        "--no-raw-fallback",
        action="store_true",
        help="Disable falling back to raw sequence features when MSA generation fails",
    )

    # New fallback control parameters
    parser.add_argument(
        "--target-min-sequences",
        type=int,
        default=TARGET_MIN_SEQUENCES,
        help="Target minimum number of sequences for a good MSA. Will try relaxed BLAST if below this threshold",
    )

    parser.add_argument(
        "--min-sequences-for-msa",
        type=int,
        default=MIN_SEQUENCES_FOR_MSA,
        help="Absolute minimum number of sequences needed to attempt alignment. Will use raw sequence fallback if below",
    )

    parser.add_argument(
        "--no-relaxed-blast", action="store_true", help="Disable relaxed BLAST fallback"
    )

    parser.add_argument(
        "--no-very-relaxed-blast",
        action="store_true",
        help="Disable very relaxed BLAST fallback",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Initialize and run MSA generator
    generator = MSAGenerator(
        mode=Mode(args.mode),
        input_csv=args.input,
        output_dir=args.output_dir,
        raw_msa_dir=args.raw_msa_dir,
        blast_cache_dir=args.blast_cache_dir,
        max_threads=args.threads,
        min_sequences=args.min_sequences,
        use_all_sequences=not args.no_all_sequences,
        keep_failed_aln=args.keep_aln_failed,
    )

    # Override default constants if specified through command line
    if (
        hasattr(args, "target_min_sequences")
        and args.target_min_sequences != TARGET_MIN_SEQUENCES
    ):
        TARGET_MIN_SEQUENCES = args.target_min_sequences
        logger.info(
            f"Setting TARGET_MIN_SEQUENCES to {TARGET_MIN_SEQUENCES} (from command line)"
        )

    if (
        hasattr(args, "min_sequences_for_msa")
        and args.min_sequences_for_msa != MIN_SEQUENCES_FOR_MSA
    ):
        MIN_SEQUENCES_FOR_MSA = args.min_sequences_for_msa
        logger.info(
            f"Setting MIN_SEQUENCES_FOR_MSA to {MIN_SEQUENCES_FOR_MSA} (from command line)"
        )

    # Check for disabled fallbacks
    if hasattr(args, "no_relaxed_blast") and args.no_relaxed_blast:
        logger.info("Relaxed BLAST fallback disabled from command line")

    if hasattr(args, "no_very_relaxed_blast") and args.no_very_relaxed_blast:
        logger.info("Very relaxed BLAST fallback disabled from command line")

    if hasattr(args, "no_raw_fallback") and args.no_raw_fallback:
        logger.warning(
            "Raw sequence fallback disabled. Targets with insufficient sequences will fail!"
        )

    # Generate MSAs
    stats = generator.generate_all(batch_size=args.batch_size)

    # Calculate success rate
    success_rate = (
        (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    )

    logger.info(
        f"✅ MSA generation complete! Success rate: {success_rate:.2f}% ({stats['success']}/{stats['total']} targets)"
    )
