# Semantic Manifolds in Sentence Embeddings

This repository is associated to the paper <a href="">*Semantic Manifolds in Sentence Embeddings*</a> by Matthieu and Andreas. 


## Set-up

First, ensure you have [Poetry](https://python-poetry.org/docs/#installation) installed on your system. Then, navigate to the location where you want to clone this repository, clone it, and enter the repository directory:

## Reproducing the results

The Jupyter notebooks are the easiest way to reproduce the code and explore the project. The underlying machinery can be found in the `src` folder and imported in the notebook.

You can run the code with different models by changing the model name in the `config.yaml` file.

## Notebooks

- `1_existence.ipynb`: for testing the existence of the manifold and the boundary
- `2_structure.ipynb`: for testing manifold combinations

## Files

This directory contains the following files:
* `data/generate.py`: for generating hyponyms and creating sentences for them.
* `data/embed.py`: for obtaining sentence embeddings using a pre-trained model.
* `src/dimensionality_reduction`: for dimensionality reduction using PCA, manifold learning and autoencoders.
* `src/parametric_models/fischer_bingham.py`: for fitting the Fisher-Bingham model and evaluating it.

