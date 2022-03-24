# Towards Fairness in NLP: Neural Methods for Flavor Extraction and Bias Mitigation

## Requirements
Requirement | Comment
:--- | :---
python 3.10 | 
virtualenv | <ul style="padding-left:0em;"><li>install by running `sudo apt-get install virtualenv` (Linux) or `pip install virtualenv`</li><li>create new environment by running `virtualenv -p python3.10 .venv`</li><li>activate by running `source .venv/bin/activate` (Linux) or `source .venv/Scripts/activate` (Windows)</li><li>deactivate by running `deactivate`</li></ul>
pip-tools | <ul style="padding-left:0em;"><li>compiles requirements.in to requirements.txt</li><li>install by running `pip install pip-tools`</li><li>run `pip-compile requirements.in` to generate or refresh requirements.txt</li></ul>
requirements.txt | run `pip install -r requirements.txt`
[GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings | [download](https://nlp.stanford.edu/data/glove.6B.zip) and extract the `.zip` file and save `glove.6B.50d.txt` to [data](./data)

## Notebooks
Notebook | Content
:--- | :---
[embeddings.ipynb](./embeddings.ipynb) | Playground for GloVe and BERT embeddings

## Hypotheses
Hypothesis | Approach | Validated | Comment | Notebook | References
:--- | :--- | :---: | :--- | :--- | :---
Gender bias in GloVe embeddings can be mitigated using a de-biasing algorithm. | Load pre-trained GloVe word embeddings and de-bias them using the algorithm proposed by [Bolukbasi et al.](https://arxiv.org/abs/1607.06520) | yes | Drawback: Gender-intrinsic word pairs to be equalized have to be hand-picked. | [embeddings.ipynb](./embeddings.ipynb) | <ul style="padding-left:0em;"></ul>
The de-biasing algorithm works the same for BERT. | Load a pre-trained BERT model and try to de-bias it using the algorithm proposed by [Bolukbasi et al.](https://arxiv.org/abs/1607.06520) | no | The de-biasing algorithm cannot be applied to contextual language models. | [embeddings.ipynb](./embeddings.ipynb) | <ul style="padding-left:0em;"></ul>
The extraction, insertion, and removal of text flavor can be modeled as a Neural Machine Translation (NMT) task. | Set-up a "translation into simplified English" NMT model and configure it to insert and remove flavor.  | in progress |  |  | <ul style="padding-left:0em;"></ul>

## References
Reference | Content
:--- | :---
[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) | GloVe homepage containing the paper as well as pre-trained word embeddings
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) | BERT paper
[BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/) | Tutorial for configuring BERT
[Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520) | De-biasing algorithm for word embeddings
[A complete tutorial on masked language modelling using BERT](https://analyticsindiamag.com/a-complete-tutorial-on-masked-language-modelling-using-bert/) | Tutorial on how to set up a masked language model using BERT
[Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer](https://paperswithcode.com/paper/delete-retrieve-generate-a-simple-approach-to) | Model for text attribute transfer. Approach to alter a specific attribute (e.g., sentiment) while preserving its attribute-independent content (e.g., changing "screen is just the right size" to "screen is too small"). Training data contains only sentences labeled with their attribute, no parallel data. Disentanglement of attributes from attribute-independent content is learnt in an unsupervised way. Generates grammatical and appropriate responses, e.g., altering sentiment of reviews on Amazon and altering image captions to be more romantic or humorous.
[Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach](https://paperswithcode.com/paper/unpaired-sentiment-to-sentiment-translation-a) | Model for sentiment-to-sentiment translation. Changes the underlying sentiment of a sentence while keeping its content. Cycled reinforcement learning is used to cope with the lack of parallel data.
[Towards Controlled Transformation of Sentiment in Sentences](https://paperswithcode.com/paper/towards-controlled-transformation-of) | Method to transform the sentiment of sentences in order to limit the work necessary to generate training data. Transforms a sentence to an opposite sentiment sentence. The pipeline consists of a sentiment classifier with an attention mechanism to highlight the short phrases that determine the sentiment of a sentence. Then, these phrases are changed to phrases of the opposite sentiment using a baseline model and an autoencoder approach. Success rate of 54.7% on sentiment change.
