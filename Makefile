.PHONY: clean, setup

deepvifi:
	chmod +x *.sh && ./setup.sh

create_data:
	python src/create_data.py

train_transformer:
	python src/train_transformer.py

train_rf:
	python src/train_rf.py

test_pipeline:
	python src/test_pipeline.py