.PHONY: setup data run report test lint notebook clean

setup:  ## install the locked environment
	uv sync --locked

data:  ## download LFW once (~230 MB, cached in data/raw/)
	uv run cnn-xai data

run:  ## all experiments -> results/ (about 1 h on a laptop CPU; embeddings cached in data/interim/)
	uv run cnn-xai run

report:  ## figures -> docs/figures/, HTML report -> site/index.html
	uv run cnn-xai report

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

notebook:  ## re-execute the walkthrough notebook in place
	uv run --group notebook jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb

clean:  ## drop cached embeddings (forces a full re-run)
	rm -rf data/interim
