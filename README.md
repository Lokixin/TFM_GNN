# Master Thesis: Alzheimer detection in EEGs with Graph Neural Networks

This repository contains all the source code of Dimas Ávila Martínez master thesis. The main goal is to detect alzheimer and dementia in electroencephalograms using graph neural networks as subject classifiers. The code contains several GNN models, a set of tools for building graphs from EEGs and all the code needed to train and evaluate the models. 


## Structure of the repository

### 1. Models

The models can be found in the module: **entities.models.models**. They can be created used the ModelFactory class defined in **entities.models.factory**. 

### 2. Graph building

The graphs are constructed using one of the GraphBuilder subclasses. The graph builders are composed by one **edge extractor** and one **node extractor**. Each one of this objects is in charge of pre-processing and extract the features from the raw EEG signal to create the edges and the nodes of the graphs. The graphs are built as PyG Data objects. 

### 3. Model training and evaluating

The full procedure of creating a custom dataset, creating a model, training and evaluating it can be followed in the **main.ipynb** file. 