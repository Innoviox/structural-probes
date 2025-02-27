import argparse
import os
import random
from tqdm import tqdm

def read_conllx_file(file_path):
    """Read a CoNLL-X file and return a list of sentences."""
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # Empty line indicates end of sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
    
    # Add the last sentence if the file doesn't end with an empty line
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def write_conllx_file(sentences, file_path):
    """Write a list of sentences to a CoNLL-X file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for line in sentence:
                f.write(line + '\n')
            f.write('\n')  # Empty line between sentences

def split_conllx(input_file, output_dir, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, seed=42):
    """Split a CoNLL-X file into train, dev, and test sets."""
    # Validate ratios
    total_ratio = train_ratio + dev_ratio + test_ratio
    if not (0.99 < total_ratio < 1.01):  # Allow for small floating point errors
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Read the input file
    print(f"Reading input file: {input_file}")
    sentences = read_conllx_file(input_file)
    total_sentences = len(sentences)
    print(f"Total sentences: {total_sentences}")
    
    # Shuffle sentences with a fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(sentences)
    
    # Calculate split sizes
    train_size = int(total_sentences * train_ratio)
    dev_size = int(total_sentences * dev_ratio)
    test_size = total_sentences - train_size - dev_size
    
    # Split the sentences
    train_sentences = sentences[:train_size]
    dev_sentences = sentences[train_size:train_size + dev_size]
    test_sentences = sentences[train_size + dev_size:]
    
    # Create output paths
    train_path = os.path.join(output_dir, 'train.conllx')
    dev_path = os.path.join(output_dir, 'dev.conllx')
    test_path = os.path.join(output_dir, 'test.conllx')
    
    # Write split files
    print(f"Writing train set ({len(train_sentences)} sentences) to {train_path}")
    write_conllx_file(train_sentences, train_path)
    
    print(f"Writing dev set ({len(dev_sentences)} sentences) to {dev_path}")
    write_conllx_file(dev_sentences, dev_path)
    
    print(f"Writing test set ({len(test_sentences)} sentences) to {test_path}")
    write_conllx_file(test_sentences, test_path)
    
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description='Split a CoNLL-X file into train/dev/test sets')
    parser.add_argument('--input', required=True, help='Input CoNLL-X file')
    parser.add_argument('--output', required=True, help='Output directory for split files')
    parser.add_argument('--train', type=float, default=0.8, help='Train set ratio (default: 0.8)')
    parser.add_argument('--dev', type=float, default=0.1, help='Dev set ratio (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.1, help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    
    split_conllx(
        args.input, 
        args.output, 
        train_ratio=args.train, 
        dev_ratio=args.dev, 
        test_ratio=args.test, 
        seed=args.seed
    )

if __name__ == '__main__':
    main()