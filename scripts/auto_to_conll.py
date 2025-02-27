import os
import re
import argparse
from collections import defaultdict
from tqdm import tqdm

def parse_auto_file(auto_file):
    """Parse a CCGbank .auto file to extract tokens and dependencies."""
    try:
        with open(auto_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(auto_file, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {auto_file}: {e}")
            return []
    
    # Split into sentences
    sentences = content.strip().split('\n\n')
    parsed_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        # Extract tokens
        tokens = []
        token_pattern = r'<L (.*?)>'
        token_matches = re.findall(token_pattern, sentence)
        
        for match in token_matches:
            parts = match.split()
            if len(parts) >= 5:
                # CCG format is typically: N N/N N NN word
                pos_tag = parts[2]  # POS tag
                word = parts[4]     # actual word
                tokens.append((word, pos_tag))
        
        if not tokens:
            continue
            
        # For dependencies, we'll need to extract them from the CCG derivation
        # This is a simplified approach - actual CCG to dependency conversion is more complex
        dependencies = []
        
        # For now, we'll create a simple dependency structure where most tokens
        # depend on the first token (as a basic tree structure)
        root_idx = 0
        for i in range(len(tokens)):
            if i == root_idx:
                # Root token has head 0
                dependencies.append((0, i+1))
            else:
                # Other tokens depend on root
                dependencies.append((root_idx+1, i+1))
        
        parsed_sentences.append((tokens, dependencies))
    
    return parsed_sentences

def write_conllx(sentences, output_file):
    """Write sentences in CoNLL-X format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence_idx, (tokens, dependencies) in enumerate(sentences):
            for i, (word, pos) in enumerate(tokens, 1):
                # Find head for this token
                head = 0  # Default to root
                deprel = "root"
                
                for dep_head, dep_dependent in dependencies:
                    if dep_dependent == i:
                        head = dep_head
                        deprel = "dep" if head != 0 else "root"  # Using a generic dependency relation
                
                # CoNLL-X format: ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL
                f.write(f"{i}\t{word}\t{word.lower()}\t{pos}\t{pos}\t_\t{head}\t{deprel}\t_\t_\n")
            
            # Empty line between sentences
            f.write("\n")

def find_auto_files(directory):
    """Find all .auto files in a directory tree."""
    auto_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.auto'):
                auto_files.append(os.path.join(root, file))
    return auto_files

def main():
    parser = argparse.ArgumentParser(description='Convert CCGbank .auto files to CoNLL-X format')
    parser.add_argument('--input', required=True, help='Input directory containing .auto files')
    parser.add_argument('--output', required=True, help='Output CoNLL-X file')
    parser.add_argument('--split', action='store_true', help='Split output into multiple files (one per input file)')
    args = parser.parse_args()
    print(args.input)
    if not os.path.isdir(args.input):
        print("Error: Input must be a directory")
        return
        
    # Find all .auto files in the directory tree
    auto_files = find_auto_files(args.input)
    print(f"Found {len(auto_files)} .auto files")
    
    if args.split:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Process each file separately
        for auto_file in tqdm(auto_files, desc="Processing files"):
            sentences = parse_auto_file(auto_file)
            if not sentences:
                continue
                
            # Create output filename based on input filename
            rel_path = os.path.relpath(auto_file, args.input)
            output_file = os.path.join(args.output, rel_path.replace('.auto', '.conllx'))
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            write_conllx(sentences, output_file)
    else:
        # Process all files and write to a single output
        all_sentences = []
        for auto_file in tqdm(auto_files, desc="Processing files"):
            sentences = parse_auto_file(auto_file)
            all_sentences.extend(sentences)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        write_conllx(all_sentences, args.output)
        print(f"Converted {len(all_sentences)} sentences to CoNLL-X format")

if __name__ == '__main__':
    main()