# Towards Fairness in NLP: Neural Methods for Flavor Detection and Bias Mitigation
This repository contains the source code for my Master's thesis [Towards Fairness in NLP: Neural Methods for Flavor Detection and Bias Mitigation](https://www.dropbox.com/s/57hbw8n1i6c8x8z/master_thesis.pdf?dl=0).

## Evaluation
This folder contains the evaluation results described in _Chapter 5: Results and Evaluation_. The human evaluation forms and the analysis sheet are located [here](https://drive.google.com/drive/folders/1JzKLK7a0VTdmGSNNTdqMW4oy8R1lKEAg?usp=sharing).
The Perspective API evaluation results and scripts are located [here](evaluation/perspective_api/). You need an [API key](https://developers.perspectiveapi.com/s/docs-enable-the-api) to run the scripts.

## Neutralizing Bias
This folder contains __MODULAR__ and __CONCURRENT__ described in _Chapter 4: Experimental Setup_ based on the paper [Automatically Neutralizing Subjective Bias in Text](https://arxiv.org/abs/1911.09709). You can find the original repository [here](https://github.com/rpryzant/neutralizing-bias).

### Requirements
1. Set up your environment:
```
$ virtualenv -p python3 .venv-nb
$ source .venv-nb/bin/activate
$ pip install -r req-nb.txt
$ python
>> import nltk; nltk.download("punkt")
```
2. Download the Wiki Neutrality Corpus (WNC) data [here](http://nlp.stanford.edu/projects/bias/bias_data.zip). Extract it to the [data folder](neutralizing-bias/data/).

3. Download the __MODULAR__ checkpoint [here](https://nlp.stanford.edu/projects/bias/model.ckpt) and save it to the [model folder](neutralizing-bias/models/) or train your own model using [this script](neutralizing-bias/joint/train_modular.sh). Please contact me if you need a checkpoint for __CONCURRENT__. You can train your own model using [this script](neutralizing-bias/seq2seq/train_concurrent.sh).

4. Use the [model interface](neutralizing-bias/interface.ipynb) or the [inference scripts](neutralizing-bias/inference/) for inference.

## Style Transfer Paraphrase
This folder contains __STRAP__ described in _Chapter 4: Experimental Setup_ based on the paper [Reformulating Unsupervised Style Transfer as Paraphrase Generation](https://arxiv.org/abs/2010.05700). You can find the original repository [here](https://github.com/martiansideofthemoon/style-transfer-paraphrase).

### Requirements
1. Set up your environment:
```
$ virtualenv -p python3 .venv-stp
$ source .venv-stp/bin/activate
$ pip install transformers
$ pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install -r req-stp.txt
```

2. Please contact me if you need the preprocessed WNC data.

3. You can download the __Diverse Paraphraser__ (paraphraser_gpt2_large) [here](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing). Save it to the [models folder](style-transfer-paraphrase/models/). Please contact me if you need a checkpoint for the __Inverse Paraphraser__ trained on WNC. If you want to train it yourself, please follow the steps described [here](https://github.com/martiansideofthemoon/style-transfer-paraphrase#custom-datasets).

4. You can use the command line-based [interface](style-transfer-paraphrase/interface.py) to interact with the model. It is documented [here](https://github.com/martiansideofthemoon/style-transfer-paraphrase/blob/master/README_terminal_demo.md#paraphrase-model-demo).

5. Evaluation requires the SIM model ([Wieting et al., 2019](https://www.aclweb.org/anthology/P19-1427/)). You can download it [here](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing) and save it to [this folder](style-transfer-paraphrase/style_paraphrase/evaluation/similarity/).


## Playground
### Requirements
1. Set up your environment:
```
$ virtualenv -p python3.10 .venv-pg
$ source .venv-pg/bin/activate
$ pip install -r req-pg.txt
```

2. Download pretrained [GloVe](https://nlp.stanford.edu/data/glove.6B.zip) word embeddings and extract the `.zip` file. Save `glove.6B.50d.txt` to the [data folder](playground/data/).

3. Follow the descriptions in the [notebooks](playground).
