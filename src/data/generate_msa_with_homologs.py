import os
import io
import time
import subprocess
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

# === CONFIGURATION ===
INPUT_CSV = "data/raw/train_sequences.csv"
MSA_OUTPUT_DIR = Path("data/raw/msa")
BLAST_CACHE_DIR = Path("data/raw/blast_cache")
MAX_HOMOLOGS = 30
INCLUDE_ALL_SEQUENCES = True
MAX_THREADS = 8
NCBI_DELAY = 1.5

MSA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BLAST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV)
targets = df[["target_id", "sequence", "all_sequences"]].drop_duplicates()


# === HELPERS ===
def write_fasta(seq, name, path, mode="w"):
    with open(path, mode) as f:
        f.write(f">{name}\n{seq}\n")


def run_blast(seq, target_id):
    xml_path = BLAST_CACHE_DIR / f"{target_id}.xml"
    if xml_path.exists():
        with open(xml_path) as f:
            return f.read()
    try:
        result_handle = NCBIWWW.qblast("blastn", "nt", seq, hitlist_size=MAX_HOMOLOGS)
        result = result_handle.read()
        with open(xml_path, "w") as f:
            f.write(result)
        return result
    except Exception as e:
        print(f"[!] BLAST failed for {target_id}: {e}")
        return None


def extract_homologs(xml_data):
    homologs = []
    try:
        blast_record = NCBIXML.read(io.StringIO(xml_data))
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                homologs.append((alignment.hit_id, hsp.sbjct))
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
        seq = "".join(lines[1:]).strip().replace(" ", "").replace("-", "")
        if seq:
            sequences.append((header, seq))
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
        return  # Already processed

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
                seq_clean = hseq.replace("-", "").replace(" ", "").upper()
                f.write(f">homolog_{i}\n{seq_clean}\n")

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


# === RUN IN PARALLEL ===
print(f"🚀 Starting with {MAX_THREADS} threads...")
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    list(
        tqdm(
            executor.map(process_target, targets.itertuples(index=False)),
            total=len(targets),
        )
    )

print("✅ All MSAs and features generated.")
