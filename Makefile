#---------------------------------------------------
# Targets to run the model pipeline
#---------------------------------------------------

# create Requirements.txt
requirements:
	@pip freeze > requirements.txt

# conda environment (This is just for my pc - you can ignore this)
conda:
	@conda activate AI

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
test:
	@python -m src.model.predict

# Evaluate performance
evaluate:
	@python -m src.evaluate.evaluate

# Produce visualizations
visualize:
	@python -m src.visualization.visualize

# Run all: RUNS ALL SCRIPTS - DEFAULT
all: download preprocess train test evaluate visualize

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
