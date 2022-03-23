
import os
import numpy as np
import torch
import tensorflow as tf
from tqdm.notebook import tqdm
from tensorboard.plugins import projector
from transformers import BertTokenizer, BertModel

###########
## GLOVE ##
###########
def read_glove_vecs(glove_file: str) -> dict[str, list[np.float64]]:
    """
    Reads GloVe vectors from a downloaded file and creates a word-to-vec dictionary.

    Arguments:
    glove_file -- path to the .txt file containing the vectors
    """
    with open(glove_file, "r", encoding="utf-8") as f:
        words = set()
        glove_map = {}
        
        for line in tqdm(f):
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            glove_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return glove_map


##########
## BERT ##
##########
def create_bert_map(vocab: list) -> dict[str, list[np.float64]]:
    """
    Create a dictionary mapping from words of a given vocab to the respective word"s embedding.

    Arguments:
    vocab -- list of words

    Returns:
    bert -- Dictionary containing the words with their respective word embeddings
    """
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

        bert[word] = embedding.numpy()

    return bert


####################################
## COSINE SIMILARITY AND DISTANCE ##
####################################
def cosine_similarity(u: np.float64, v: np.float64) -> float:
    """
    Returns the cosine similarity between two vectors u and v.
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    # Special case. Consider the case u = [0, 0], v=[0, 0]
    if np.all(u == v):
        return 1
    
    # Compute the dot product between u and v 
    dot = np.dot(u, v)
    
    # Compute the L2 norm of u 
    norm_u = np.sqrt(np.sum(u**2))
    
    # Compute the L2 norm of v 
    norm_v = np.sqrt(np.sum(v**2))
    
    # Avoid division by 0
    if np.isclose(norm_u * norm_v, 0, atol=1e-32):
        return 0
    
    # Compute the cosine similarity defined by formula (1) 
    cosine_similarity = dot / (norm_u * norm_v)
    
    return cosine_similarity


###############
## ANALOGIES ##
###############
def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task: a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    
    # Convert words to lowercase
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    # Get the word embeddings e_a, e_b and e_c
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None                   # Initialize best_word with None, it will help keep track of the word to output
    
    # Loop over the whole word vector set
    for w in tqdm(words):
        # To avoid best_word being one of the input words, skip the input word_c
        # Skip word_c from query
        if w == word_c:
            continue
        
        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w"s vector representation) - e_c)  
        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)
        
        # If the cosine_sim is more than the max_cosine_sim seen so far,
            # Then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        
    return best_word


###############
## DEBIASING ##
###############
def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
    This function ensures that gender neutral words are zero in the gender subspace.
    
    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.
    
    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """
    
    # Select word vector representation of "word". Use word_to_vec_map. 
    e = word_to_vec_map[word]
    
    # Compute e_biascomponent using the formula given above. 
    e_biascomponent = np.dot(np.dot(e, g) / np.sum(g ** 2), g)
 
    # Neutralize e by subtracting e_biascomponent from it.
    # e_debiased should be equal to its orthogonal projection.
    e_debiased = e - e_biascomponent
    
    return e_debiased


def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debiases gender-specific words by following the equalize method described above.
    
    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors
    
    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    
    # Step 1: Select word vector representation of "word". Use word_to_vec_map
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    
    # Step 2: Compute the mean of e_w1 and e_w2
    mu = (e_w1 + e_w2) / 2

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis
    mu_B = np.dot(np.dot(mu, bias_axis) / np.sum(bias_axis ** 2), bias_axis)
    mu_orth = mu - mu_B

    # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B 
    e_w1B = np.dot(np.dot(e_w1, bias_axis) / np.sum(bias_axis ** 2), bias_axis)
    e_w2B = np.dot(np.dot(e_w2, bias_axis) / np.sum(bias_axis ** 2), bias_axis)
        
    # Step 5: Adjust the bias part of e_w1B and e_w2B using the formulas (9) and (10) given above 
    corrected_e_w1B = np.dot(np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))), (e_w1B - mu_B) / np.linalg.norm(e_w1 - mu_orth - mu_B))
    corrected_e_w2B = np.dot(np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))), (e_w2B - mu_B) / np.linalg.norm(e_w2 - mu_orth - mu_B))

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections 
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth
    
    return e1, e2


#################
## TENSORBOARD ##
#################
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
