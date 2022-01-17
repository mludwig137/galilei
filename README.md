# README

## Galilei: Problem Statement

Create a graph representation model of latent semantic attributes and relations of scientific papers for anomaly detection and research topic recommendation through the detection of structural holes.

## Galilei: Executive Summary

Aim to create a graph representation using structured and unstructured data from ArXiv scientific repository of scientific literatute. Utilize graph attention networks and related methods for anomaly detection. Anomalies can represent entities which are bad actors, highly influential, or structural gaps.

Create weighted edges based on conceptual similarity. Conceptual similarity being a concept/context vector created through topic modeling: LDA, Dynamic LDA, and other approachs.

Summurization of text within a node's neighborhood or between nodes.

Graph representation learning to be used for node prediction and edge prediction

#### Data

ArXiv's bulk data access options can be viewed here:

https://arxiv.org/help/bulk_data

The ArXiv snapshots data was utilized throughout:

https://www.kaggle.com/Cornell-University/arxiv

#### Topic Model

All relevant features are full rank. 

* Abstract and title were the only features considered for topic modelling. In future iterations sampling with take into account explicit category labels and limit observations to a time window.

Text data was preprocessed using regex and spacy.

* Documents were tokenized and special characters, custom stop words, LaTex, Markdown, and other special symbols were all removed.
* Documents were also lemmatized and pipelined through spaCys language model to add part of speech tags. These tags were not utilized in this iteration, but the option is retain as a potentially important feature.
* The data was again tokenized using a gensim module(that is effectively RegexpTokenizer)
* A dictionary of these tokens is created with id:token pairs as key:value pairs. These ids are generated sequentially.
* A corpus is created matching document tokens to tokens in the dictionary and generating a list of ids and word frequecies.
* These are the inputs for the LDA model.

LDA model created on corpus and dictionary

* The LDA model has several hyperparameters: alpha, (b)eta, number of topics, passes, and others.
* Special care is given to number of topics as this best determines performance.
* Perplexity and coherence scores are taken over a range of 2 - 100 topic models(all other hyperparameters held constant). Coherence can inform a neighborhood for manual search though are poor metrics to evaluate performance.
* Human intuition is necessary to evaulate the validity and effectiveness of topic and document pairs.

The topic model and visualizations were deployed using streamlit:

https://share.streamlit.io/mludwig137/galilei-streamlit/galilei.py

A unigram and n-gram can be loaded or a user can create their own with a choice of hyperparamters including number of topics(from provided data).

#### Graph Preprocessing

All relevant features are full rank.

* The authors feature was used as relation/edge to experiment with graph representations. Authors parsed was manipulated and cleaned of trailing entries and empty strings. Additionally names were reformatted as first_last.

Relations were generated from papers with the greatest number of shared collaborators. As a demonstration of clusters possibly representing in-groups/social groups whether from geography, prestige, or field.

Random walks were run on these graphs to generate sentences/paths and embedded in word2vec. These represent structural similarities between neighborhoods(variation is minimal in the subgraph queried).

#### Abstractive Text Summarization with Attention

All relevant features are full rank.

Abstractive text summarization using attention is built generalizing from instructions for an encoder-decoder network("tutored" seq2seq).

The training component of this model is complete up to input shape to be determined.

Paper abstracts are inputs to the encoder, while titles are inputs to the decoder. On a later component of the model summarizations are generated from a holdout set of abstracts.

Preprocessing includes the standard stopwords and tokenization, but adds a further step of encoding and padding. Padding is necessary to homogenize abstract and title lengths.