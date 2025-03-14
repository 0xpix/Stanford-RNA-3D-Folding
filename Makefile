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
preprocess:
	@python -m src.preprocess.preprocess

# Train the model
train:
	@python -m src.model.train

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
	@zrok access private fqo3t0n4tgx1

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
