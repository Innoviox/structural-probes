import torch
from transformers import AutoModel, AutoTokenizer
import h5py
import numpy as np
from tqdm import tqdm
import argparse
import os

def parse_conllx(file_path):
    """Parse a CoNLL-X formatted file into sentences."""
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:  # Ensure there are at least ID and FORM fields
                    # Add the word form (usually the 2nd column in CoNLL-X)
                    current_sentence.append(parts[1])
    
    # Add the last sentence if it wasn't followed by an empty line
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def align_subword_embeddings(original_tokens, tokenizer, all_hidden_states):
    """
    Align subword token embeddings to original tokens in the CoNLL-X file.
    
    Args:
        original_tokens: List of original tokens from CoNLL-X
        tokenizer: The tokenizer used
        all_hidden_states: List of hidden states tensors for all layers
        
    Returns:
        List of numpy arrays, each with shape (len(original_tokens), hidden_dim)
    """
    # Get the tokenization of the whole sentence
    text = ' '.join(original_tokens)
    tokenized_text = tokenizer.tokenize(text)
    
    # Get the token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    
    # Handle special tokens
    start_idx = 1  # Skip first special token (CLS, BOS, etc.)
    end_idx = len(token_ids) - 1  # Skip last special token (SEP, EOS, etc.)
    
    # Create a mapping from original tokens to subword tokens
    subword_to_token_mapping = []
    token_idx = 0
    
    # For debugging
    # print(f"Original tokens: {original_tokens}")
    # print(f"Tokenized text: {tokenized_text}")
    
    # Create alignment by tokenizing each original token separately
    # and tracking how many subwords it creates
    alignment = []
    current_idx = start_idx
    
    for orig_token in original_tokens:
        # Tokenize this single token (with space prefix if needed)
        if token_idx == 0:
            subwords = tokenizer.tokenize(orig_token)
        else:
            # Add space before token for most tokenizers
            subwords = tokenizer.tokenize(' ' + orig_token)
        
        # Record the span of subword indices for this token
        token_subwords = len(subwords)
        if token_subwords > 0:
            alignment.append((current_idx, current_idx + token_subwords - 1))
            current_idx += token_subwords
        else:
            # Handle case where token produces no subwords (rare)
            # Just assign it the current position
            alignment.append((current_idx, current_idx))
        
        token_idx += 1
    
    # For debugging
    # print(f"Alignment: {alignment}")
    
    # Verify alignment doesn't exceed token_ids length
    max_alignment = max([end for _, end in alignment]) if alignment else 0
    if max_alignment >= len(token_ids):
        # print(f"Warning: Alignment exceeds token length. Max alignment: {max_alignment}, Token length: {len(token_ids)}")
        # Adjust alignment to fit within token bounds
        alignment = [(start, min(end, len(token_ids)-1)) for start, end in alignment]
    
    # Now average the embeddings for each original token
    aligned_embeddings = []
    
    for layer_idx, hidden_state in enumerate(all_hidden_states):
        # Convert to numpy
        hidden_np = hidden_state.cpu().numpy()[0]  # [0] because batch size is 1
        
        # Create aligned embeddings for this layer
        layer_aligned = []
        
        for start_subword, end_subword in alignment:
            # Ensure indices are within bounds
            start_subword = max(0, min(start_subword, hidden_np.shape[0]-1))
            end_subword = max(start_subword, min(end_subword, hidden_np.shape[0]-1))
            
            # Average embeddings for this token's subwords
            token_embedding = np.mean(hidden_np[start_subword:end_subword+1], axis=0)
            layer_aligned.append(token_embedding)
        
        # Check if we have the right number of embeddings
        if len(layer_aligned) != len(original_tokens):
            print(f"Warning: Misaligned tokens. Got {len(layer_aligned)}, expected {len(original_tokens)}")
            # Pad or truncate to match expected length
            if len(layer_aligned) < len(original_tokens):
                # Pad with zeros
                pad_count = len(original_tokens) - len(layer_aligned)
                padding = [np.zeros_like(layer_aligned[0]) for _ in range(pad_count)]
                layer_aligned.extend(padding)
            else:
                # Truncate
                layer_aligned = layer_aligned[:len(original_tokens)]
        
        aligned_embeddings.append(np.array(layer_aligned))
    
    return aligned_embeddings

def get_model_embeddings(model_name, sentences, max_length=512):
    """Extract embeddings from a model for given sentences with alignment to original tokens."""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Process each sentence
    all_aligned_embeddings = []
    
    for sentence in tqdm(sentences, desc='Extracting aligned embeddings'):
        if not sentence:
            continue
            
        # Join tokens and tokenize
        text = ' '.join(sentence)
        inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Forward pass through model
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get hidden states from all layers
        hidden_states = outputs.hidden_states
        
        # Align subword embeddings to original tokens
        aligned_embeddings = align_subword_embeddings(sentence, tokenizer, hidden_states)
        
        # Convert to numpy array of shape (layers, tokens, hidden_dim)
        all_aligned_embeddings.append(np.array(aligned_embeddings))
    
    return all_aligned_embeddings, tokenizer

def create_hdf5_file(embeddings, output_path):
    """Create an HDF5 file with embeddings for each sentence."""
    with h5py.File(output_path, 'w') as f:
        for idx, sentence_embeddings in tqdm(enumerate(embeddings), desc='Writing to HDF5', total=len(embeddings)):
            # Store the full embedding tensor for each sentence
            f.create_dataset(
                str(idx), 
                data=sentence_embeddings,
                compression="gzip"
            )

def main():
    parser = argparse.ArgumentParser(description='Generate HDF5 files from transformer models')
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name or path')
    parser.add_argument('--conllx', type=str, required=True, help='Path to CoNLL-X file')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file path')
    parser.add_argument('--verbose', action='store_true', help='Print verbose alignment information')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Parse CoNLL-X file
    print(f"Parsing CoNLL-X file: {args.conllx}")
    sentences = parse_conllx(args.conllx)
    print(f"Found {len(sentences)} sentences")
    
    # Get embeddings with alignment
    print(f"Extracting and aligning embeddings using {args.model}")
    aligned_embeddings, _ = get_model_embeddings(args.model, sentences)
    
    # Create HDF5 file
    print(f"Writing aligned embeddings to {args.output}")
    create_hdf5_file(aligned_embeddings, args.output)
    
    print("Done!")

if __name__ == '__main__':
    main()