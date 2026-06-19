.PHONY: install test train explain app api all

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

train:
	python -m src.train

explain:
	python -m src.explain

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api:app --reload

all: test train explain test
