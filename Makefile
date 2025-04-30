# Just to test
handle:
	@python -m src.data.data_handle

#---------------------------------------------------
# Targets to run the model pipeline
#---------------------------------------------------

# create Requirements.txt
requirements:
	@pipreqs . --force

# conda environment (This is just for my pc - you can ignore this)
conda:
	@conda activate AI

# Run the Unitest
test:
# @PYTHONPATH=. python tests/test_download.py
	@python -m unittest discover tests

# Download the data
download:
	@python -m src.data.download

# Preprocess the data
gen_msa-train:
	@python -m src.data.generate_msa_with_homologs --mode training --input data/raw/train_sequences.csv --target-min-sequences 6 --min-sequences-for-msa 3

gen_msa-valid:
	@python -m src.data.generate_msa_with_homologs --mode training --input data/raw/validation_sequences.csv --target-min-sequences 6 --min-sequences-for-msa 3

gen_msa-inference:
	@python -m src.data.generate_msa_with_homologs --mode inference --input data/raw/test_sequences.csv

bppms:
	@python -m src.extract_and_generate_bppms --use_raw_fallback

bppms-full:
	@python -m src.extract_and_generate_bppms --use_raw_fallback --msa_dir data/processed/MSA

bppms-all:
	@python -m src.extract_all_sequences_and_generate_bppms --msa_dir data/processed/MSA

bppms-all-limit:
	@python -m src.extract_all_sequences_and_generate_bppms --msa_dir data/processed/msa --limit 10

# Generate and pad BPPMs from MSA files
bppms-padded:
	@python -m src.generate_and_pad_bppms_from_msa

# Generate and pad BPPMs with custom max length
bppms-padded-custom:
	@python -m src.generate_and_pad_bppms_from_msa --max_len 2048 --min_pad 1024

preprocess-rna:
	@python -m src.preprocess.preprocess

# Train the model
train:
	@python3 -m src.model.train
# Train the model
train-model-1:
	@python -m src.model.model1.train --epochs 1 --batch_size 4 --model_dim 128 --depth 2 --max_seq_len 512

# Make predictions on the test data
predict:
	@python -m src.model.predict

# Evaluate performance
evaluate:
	@python -m src.evaluate.evaluate

# Produce visualizations
visualize:
	@python -m src.visualization.visualize

# Run all: RUNS ALL SCRIPTS - DEFAULT
all: download preprocess train predict evaluate visualize

#---------------------------------------------------
# SSH into Kaggle
#---------------------------------------------------

zrok-access:
	@echo "Zrok Access"
# @zrok disable
	@zrok enable "sTi4BOxak4Ox"
	@zrok access private lyj5ps95om2p

ssh-kaggle:
	@echo "SSH into Kaggle"
	@ssh-keygen -f "/home/pix/.ssh/known_hosts" -R "[127.0.0.1]:9191"
	@rsync -avz -e "ssh -p 9191 -i ~/.ssh/kaggle_rsa" . root@127.0.0.1:/kaggle/working/Stanford-RNA-3D-Folding/
# @ssh -p 9191 -i ~/.ssh/kaggle_rsa root@127.0.0.1
	@ssh Kaggle

#---------------------------------------------------
# Cleaning folders
#---------------------------------------------------

## Delete all compiled Python files
clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete

# Delete all data
clean-data:
	@rm -rf data/raw/*
	@rm -rf data/processed/*

# Delete all models, metrics, and visualizations
clean-results:
	@rm -rf models/*
	@rm -rf results/*
	@rm -rf reports/figures/*

# Delete everything
clean-all: clean clean-data clean-results
