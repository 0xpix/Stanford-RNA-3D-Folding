"""
Step 1: Read the csv file and get the target_id, sequence, and all_sequences columns.
    This is the input phase.
        You're reading your main dataset that contains:
        Unique RNA targets (target_id)
        The primary RNA sequence to align (sequence)
        Optional known homologs (all_sequences) — some of which may be RNA or protein.
    This sets the stage for all further processing — everything works per target_id.

Step 2: Builds a collection of sequences for each target to align.
    For each RNA target:
        Start with the main sequence from sequence.
        Parse all_sequences into individual entries.
        Keep only those that contain valid RNA characters (A, C, G, U).
        Discard proteins or malformed sequences that contain things like M, K, Q, etc.
    This gives you a clean group of RNA sequences ready to align.

Step 3: For each target, run BLAST to find homologous sequences.
    You send your target RNA to BLAST (using blastn remotely):
        BLAST returns similar sequences from other organisms.
        Some of these are RNA, others might be noise or protein.
        You apply the same RNA-only filter to keep only usable homologs.
    Result: You get evolutionary information for your RNA — more sequences = better alignment.

Step 4: Write the sequences to a temporary FASTA file.
    This temporary file is the input for alignment. It contains:
        The original target sequence
        Any good sequences from all_sequences
        Any good homologs from BLAST
    This is the unified set of sequences you want to align.

Step 5: Calculate Features from the MSA
    You align all sequences using Clustal Omega to produce an MSA (.aln file).
        From the aligned result, you compute:
        Conservation: How similar each column (position) is across sequences
        Nucleotide frequency: Percent of A, C, G, U at each position
        Gap frequency: How often a gap (-) appears in each column
        Neff: Number of sequences in the MSA
    These features capture the evolutionary and structural context of the RNA.

Step 6: Save the features to a .npz file.
    Result: .npz file will be used later by machine learning model, structure predictor, or analysis script.
"""

import os
import io
import time
import subprocess
from tqdm import tqdm
from pathlib import Path

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

# === CONFIGURATION ===
INPUT_CSV = "data/raw/train_sequences.csv"
MSA_OUTPUT_DIR = Path("data/processed/msa")
BLAST_CACHE_DIR = Path("data/processed/blast_cache")
MAX_HOMOLOGS = 30
INCLUDE_ALL_SEQUENCES = True
MAX_THREADS = 8
NCBI_DELAY = 1.5  # polite delay between BLAST calls

# === MAKE DIRECTORIES ===
MSA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BLAST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD CSV ===
df = pd.read_csv(INPUT_CSV)
targets = df[["target_id", "sequence", "all_sequences"]].drop_duplicates()


# === HELPERS ===
def write_fasta(seq, name, path, mode="w"):
    with open(path, mode) as f:
        f.write(f">{name}\n{seq}\n")


def run_blast(seq, target_id, max_retries=3, delay=10):
    xml_path = BLAST_CACHE_DIR / f"{target_id}.xml"
    if xml_path.exists():
        with open(xml_path) as f:
            return f.read()

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[BLAST] Querying {target_id} (Attempt {attempt})...")
            result_handle = NCBIWWW.qblast(
                "blastn", "nt", seq, hitlist_size=MAX_HOMOLOGS
            )
            result = result_handle.read()
            with open(xml_path, "w") as f:
                f.write(result)
            return result
        except Exception as e:
            print(f"[!] BLAST failed for {target_id} (Attempt {attempt}): {e}")
            if attempt < max_retries:
                print(f"    ⏳ Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(
                    f"[✗] BLAST permanently failed for {target_id} after {max_retries} attempts."
                )
                return None


def extract_homologs(xml_data):
    homologs = []
    try:
        blast_record = NCBIXML.read(io.StringIO(xml_data))
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                seq = hsp.sbjct.replace("-", "").replace(" ", "").upper()
                if all(c in "ACGU" for c in seq):
                    homologs.append((alignment.hit_id, seq))
                else:
                    print(f"[!] Skipped non-RNA homolog: {alignment.hit_id}")
                if len(homologs) >= MAX_HOMOLOGS:
                    return homologs
    except Exception as e:
        print(f"[!] Error parsing BLAST XML: {e}")
    return homologs


def parse_all_sequences_block(all_sequences_str):
    sequences = []
    if pd.isna(all_sequences_str):
        return sequences
    for entry in all_sequences_str.split(">"):
        lines = entry.strip().split("\n")
        if len(lines) < 2:
            continue
        header = lines[0].strip()
        seq = "".join(lines[1:]).strip().replace(" ", "").replace("-", "").upper()
        if all(c in "ACGU" for c in seq) and len(seq) >= 8:
            sequences.append((header, seq))
        else:
            print(f"[!] Skipped non-RNA or invalid sequence: {header}")
    return sequences


def extract_msa_features(msa_path, feature_out_path):
    try:
        records = list(SeqIO.parse(msa_path, "fasta"))
        if not records:
            print(f"[!] No sequences in MSA: {msa_path}")
            return

        sequences = [str(rec.seq).upper() for rec in records]
        n_seq = len(sequences)

        msa_mat = np.array([list(seq) for seq in sequences])

        freq_A = np.sum(msa_mat == "A", axis=0) / n_seq
        freq_C = np.sum(msa_mat == "C", axis=0) / n_seq
        freq_G = np.sum(msa_mat == "G", axis=0) / n_seq
        freq_U = np.sum(msa_mat == "U", axis=0) / n_seq
        freq_gap = np.sum(msa_mat == "-", axis=0) / n_seq

        conservation = np.max(np.stack([freq_A, freq_C, freq_G, freq_U]), axis=0)

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
    except Exception as e:
        print(f"[!] Feature extraction failed for {msa_path}: {e}")


# === MAIN PROCESSING FUNCTION ===
def process_target(row):
    target_id = row.target_id
    sequence = row.sequence
    all_sequences_raw = row.all_sequences

    msa_out = MSA_OUTPUT_DIR / f"{target_id}.aln"
    feature_out = MSA_OUTPUT_DIR / f"{target_id}_features.npz"

    if msa_out.exists() and feature_out.exists():
        return  # Already done

    temp_fasta = MSA_OUTPUT_DIR / f"{target_id}_temp.fasta"
    write_fasta(sequence, f"{target_id}_query", temp_fasta)

    if INCLUDE_ALL_SEQUENCES:
        additional = parse_all_sequences_block(all_sequences_raw)
        with open(temp_fasta, "a") as f:
            for header, seq in additional:
                f.write(f">{header}\n{seq}\n")

    xml_data = run_blast(sequence, target_id)
    if xml_data:
        homologs = extract_homologs(xml_data)
        with open(temp_fasta, "a") as f:
            for i, (hit_id, hseq) in enumerate(homologs):
                f.write(f">homolog_{i}\n{hseq}\n")

    # Align with Clustal Omega
    cmd = [
        "clustalo",
        "-i",
        str(temp_fasta),
        "-o",
        str(msa_out),
        "--outfmt",
        "fasta",
        "--force",
        "--auto",
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print(f"[✗] Clustal failed for {target_id}:\n{result.stderr}")
    elif not msa_out.exists():
        print(f"[!] MSA output not found for {target_id}")
    else:
        extract_msa_features(msa_out, feature_out)

    os.remove(temp_fasta)
    time.sleep(NCBI_DELAY)


# === RUN PARALLEL EXECUTION ===
print(f"🚀 Starting with {MAX_THREADS} threads...")
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    list(
        tqdm(
            executor.map(process_target, targets.itertuples(index=False)),
            total=len(targets),
        )
    )

print("✅ All MSAs and features generated.")
