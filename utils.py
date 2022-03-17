import os
import numpy as np
import torch
import tensorflow as tf
from tqdm import tqdm
from tensorboard.plugins import projector
from transformers import BertTokenizer, BertModel


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map


def visualize_embedding(word_to_vec_map: dict, dir_name: str):
    """
    Creates TensorBoard files to visualize a given word_to_vec_map.

    Arguments:
    word_to_vec_map -- dictionary mapping words to vectors
    dir_name -- name of the directory to be created in TensorBoard logs
    """
    # Create logging directory
    log_dir = os.path.join(os.getcwd(), "logs", dir_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Write word-index dictionary
    with open(os.path.join(log_dir, "vectors.tsv"), "w") as vectors, open(os.path.join(log_dir, "metadata.tsv"), "w") as metadata:
        for word, vec in word_to_vec_map.items():
            vectors.write("\t".join([str(x) for x in vec]) + "\n")
            metadata.write(word + "\n")

    with open(os.path.join(log_dir, "vectors.tsv"), "r") as vectors:
        feature_vectors = np.loadtxt(vectors, delimiter="\t")

    # Create checkpoint from embeddings
    weights = tf.Variable(feature_vectors)
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Configure set-up
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = os.path.join(log_dir, "metadata.tsv")
    projector.visualize_embeddings(log_dir, config)


def create_word_to_vec_map(vocab: list) -> dict({str: list[float]}):
    bert = {}

    # Load pre-trained model weights
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

    # Set evaluation mode (feed-forward operation without dropout regularization)
    model.eval()

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for word in tqdm(vocab):
        text = "[CLS] " + word + " [SEP]"

        # Add tags ([CLS] in the beginning, [SEP] at the end or as separator between two sentences)
        marked_text = "[CLS] " + text + " [SEP]"

        # Tokenize the sentence with the BERT tokenizer
        tokenized_text = tokenizer.tokenize(marked_text)

        # Map the token strings to their vocab indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Set segment IDs (0 for the first sentence, 1 for the second sentence or a single sentence)
        segments_ids = [1] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Forward pass without constructing compute graph (only needed for backprop) to reduce memory
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)

        # Third item of outputs contains hidden states from all layers
        hidden_states = outputs[2]

        # Concatenate tensors for all layers
        token_embeddings = torch.stack(hidden_states, dim=0)

        # Remove "batches" dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Switch "layers" and "tokens" dimension to result in [tokens, layers, dimensions]
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # Average the second to last hidden layer with shape [768]
        token_vecs = hidden_states[-2][0]
        embedding = torch.mean(token_vecs, dim=0)

        bert[word] = embedding.tolist()

    return bert