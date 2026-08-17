PY := /opt/homebrew/bin/python3.10
export PYTHONPATH := .

.PHONY: help setup papers data verify test lab table clean-nb kernel judge judge-data judge-test judge-publish

help:            ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

setup: kernel    ## install the few packages the system python is missing
	$(PY) -m pip install --quiet --upgrade polars huggingface_hub datasets seaborn \
	    statsmodels pytest tabulate nbstripout
	$(PY) -m pip install --quiet --upgrade -r judge/requirements-judge.txt

kernel:          ## register the Jupyter kernel this repo's notebooks expect
	$(PY) -m ipykernel install --user --name ads-ml-lab --display-name "ads-ml-lab"

papers:          ## download the reading spine into weekNN/papers/
	$(PY) tools/fetch_papers.py

data:            ## download the datasets into data/raw/ (~4.2 GB)
	$(PY) tools/fetch_datasets.py

verify:          ## end-to-end check: data loads, splits hold, harness imports
	$(PY) tools/verify_setup.py

test:            ## run the harness contract tests
	$(PY) -m pytest tests -q

lab:             ## start JupyterLab
	$(PY) -m jupyterlab

table:           ## print the full results table
	$(PY) -c "from adslab import registry; print(registry.to_markdown())"

clean-nb:        ## strip outputs from every notebook (do this before committing)
	$(PY) -m nbstripout week*/*.ipynb

judge-data:      ## build week 1's competition files (train/test/solution) from data/raw
	$(PY) -m judge.prepare_data --week 1

judge:           ## run the competition server at http://localhost:8000
	$(PY) -m uvicorn judge.app:app --host 127.0.0.1 --port 8000 --reload

judge-test:      ## end-to-end smoke test against a running judge
	$(PY) -m judge.smoke_test

judge-publish:   ## push week 1's data to the Hub (needs a WRITE token; --dry-run first)
	$(PY) tools/publish_competition.py --week 1
