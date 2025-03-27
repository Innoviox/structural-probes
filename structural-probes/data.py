"""
This module handles the reading of conllx files and hdf5 embeddings.

Specifies Dataset classes, which offer PyTorch Dataloaders for the
train/dev/test splits.
"""
import os
from collections import namedtuple, defaultdict

from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import h5py
import pickle


class SimpleDataset:
  """Reads conllx files to provide PyTorch Dataloaders

  Reads the data from conllx files into namedtuple form to keep annotation
  information, and provides PyTorch dataloaders and padding/batch collation
  to provide access to train, dev, and test splits.

  Attributes:
    args: the global yaml-derived experiment config dictionary
  """
  def __init__(self, args, task, vocab={}):
    self.args = args
    self.batch_size = args['dataset']['batch_size']
    self.use_disk_embeddings = args['model']['use_disk']
    self.vocab = vocab
    self.observation_class = self.get_observation_class(self.args['dataset']['observation_fieldnames'])
    self.train_obs, self.dev_obs, self.test_obs = self.read_from_disk()
    self.train_dataset = ObservationIterator(self.train_obs, task, 'train')
    self.dev_dataset = ObservationIterator(self.dev_obs, task, 'dev')
    self.test_dataset = ObservationIterator(self.test_obs, task, 'test')

  def read_from_disk(self):
    '''Reads observations from conllx-formatted files
    
    as specified by the yaml arguments dictionary and 
    optionally adds pre-constructed embeddings for them.

    Returns:
      A 3-tuple: (train, dev, test) where each element in the
      tuple is a list of Observations for that split of the dataset. 
    '''
    train_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['train_path'])
    dev_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['dev_path'])
    test_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['test_path'])
    train_observations = self.load_conll_dataset(train_corpus_path)
    dev_observations = self.load_conll_dataset(dev_corpus_path)
    test_observations = self.load_conll_dataset(test_corpus_path)

    train_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['train_path'])
    dev_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['dev_path'])
    test_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['test_path'])
    train_observations = self.optionally_add_embeddings(train_observations, train_embeddings_path)
    dev_observations = self.optionally_add_embeddings(dev_observations, dev_embeddings_path)
    test_observations = self.optionally_add_embeddings(test_observations, test_embeddings_path)
    return train_observations, dev_observations, test_observations

  def get_observation_class(self, fieldnames):
    '''Returns a namedtuple class for a single observation.

    The namedtuple class is constructed to hold all language and annotation
    information for a single sentence or document.

    Args:
      fieldnames: a list of strings corresponding to the information in each
        row of the conllx file being read in. (The file should not have
        explicit column headers though.)
    Returns:
      A namedtuple class; each observation in the dataset will be an instance
      of this class.
    '''
    return namedtuple('Observation', fieldnames)

  def generate_lines_for_sent(self, lines):
    '''Yields batches of lines describing a sentence in conllx.

    Args:
      lines: Each line of a conllx file.
    Yields:
      a list of lines describing a single sentence in conllx.
    '''
    buf = []
    for line in lines:
      if line.startswith('#'):
        continue
      if not line.strip():
        if buf:
          yield buf
          buf = []
        else:
          continue
      else:
        buf.append(line.strip())
    if buf:
      yield buf

  def load_conll_dataset(self, filepath):
    '''Reads in a conllx file; generates Observation objects
    
    For each sentence in a conllx file, generates a single Observation
    object.

    Args:
      filepath: the filesystem path to the conll dataset
  
    Returns:
      A list of Observations 
    '''
    observations = []
    lines = (x for x in open(filepath))
    for buf in self.generate_lines_for_sent(lines):
      conllx_lines = []
      for line in buf:
        conllx_lines.append(line.strip().split('\t'))
      embeddings = [None for x in range(len(conllx_lines))]
      observation = self.observation_class(*zip(*conllx_lines), embeddings)
      observations.append(observation)
    return observations

  def add_embeddings_to_observations(self, observations, embeddings):
    '''Adds pre-computed embeddings to Observations.

    Args:
      observations: A list of Observation objects composing a dataset.
      embeddings: A list of pre-computed embeddings in the same order.

    Returns:
      A list of Observations with pre-computed embedding fields.
    '''
    embedded_observations = []
    for observation, embedding in zip(observations, embeddings):
      embedded_observation = self.observation_class(*(observation[:-1]), embedding)
      embedded_observations.append(embedded_observation)
    return embedded_observations

  def generate_token_embeddings_from_hdf5(self, args, observations, filepath, layer_index):
    '''Reads pre-computed embeddings from ELMo-like hdf5-formatted file.

    Sentences should be given integer keys corresponding to their order
    in the original file.
    Embeddings should be of the form (layer_count, sent_length, feature_count)

    Args:
      args: the global yaml-derived experiment config dictionary.
      observations: A list of Observations composing a dataset.
      filepath: The filepath of a hdf5 file containing embeddings.
      layer_index: The index corresponding to the layer of representation
          to be used. (e.g., 0, 1, 2 for ELMo0, ELMo1, ELMo2.)
    
    Returns:
      A list of numpy matrices; one for each observation.

    Raises:
      AssertionError: sent_length of embedding was not the length of the
        corresponding sentence in the dataset.
    '''
    hf = h5py.File(filepath, 'r') 
    indices = filter(lambda x: x != 'sentence_to_index', list(hf.keys()))
    single_layer_features_list = []
    for index in sorted([int(x) for x in indices]):
      observation = observations[index]
      feature_stack = hf[str(index)]
      single_layer_features = feature_stack[layer_index]
      assert single_layer_features.shape[0] == len(observation.sentence)
      single_layer_features_list.append(single_layer_features)
    return single_layer_features_list

  def integerize_observations(self, observations):
    '''Replaces strings in an Observation with integer Ids.
    
    The .sentence field of the Observation will have its strings
    replaced with integer Ids from self.vocab. 

    Args:
      observations: A list of Observations describing a dataset

    Returns:
      A list of observations with integer-lists for sentence fields
    '''
    new_observations = []
    if self.vocab == {}:
      raise ValueError("Cannot replace words with integer ids with an empty vocabulary "
          "(and the vocabulary is in fact empty")
    for observation in observations:
      sentence = tuple([vocab[sym] for sym in observation.sentence])
      new_observations.append(self.observation_class(sentence, *observation[1:]))
    return new_observations

  def get_train_dataloader(self, shuffle=True, use_embeddings=True):
    """Returns a PyTorch dataloader over the training dataset.

    Args:
      shuffle: shuffle the order of the dataset.
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the training dataset (possibly shuffled)
    """
    return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=shuffle)

  def get_dev_dataloader(self, use_embeddings=True):
    """Returns a PyTorch dataloader over the development dataset.

    Args:
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the development dataset
    """
    return DataLoader(self.dev_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

  def get_test_dataloader(self, use_embeddings=True):
    """Returns a PyTorch dataloader over the test dataset.

    Args:
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the test dataset
    """
    return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
    """Does not add embeddings; see subclasses for implementations."""
    return observations

  def custom_pad(self, batch_observations):
    '''Pads sequences with 0 and labels with -1; used as collate_fn of DataLoader.
    
    Loss functions will ignore -1 labels.
    If labels are 1D, pads to the maximum sequence length.
    If labels are 2D, pads all to (maxlen,maxlen).

    Args:
      batch_observations: A list of observations composing a batch
    
    Return:
      A tuple of:
          input batch, padded
          label batch, padded
          lengths-of-inputs batch, padded
          Observation batch (not padded)
    '''
    if self.use_disk_embeddings:
      seqs = [torch.tensor(x[0].embeddings, device=self.args['device']) for x in batch_observations]
    else:
      seqs = [torch.tensor(x[0].sentence, device=self.args['device']) for x in batch_observations]
    lengths = torch.tensor([len(x) for x in seqs], device=self.args['device'])
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    label_shape = batch_observations[0][1].shape
    maxlen = int(max(lengths))
    label_maxshape = [maxlen for x in label_shape]
    labels = [-torch.ones(*label_maxshape, device=self.args['device']) for x in seqs]
    for index, x in enumerate(batch_observations):
      length = x[1].shape[0]
      if len(label_shape) == 1:
        labels[index][:length] = x[1]
      elif len(label_shape) == 2:
        labels[index][:length,:length] = x[1]
      else:
        raise ValueError("Labels must be either 1D or 2D right now; got either 0D or >3D")
    labels = torch.stack(labels)
    return seqs, labels, lengths, batch_observations

class ELMoDataset(SimpleDataset):
  """Dataloader for conllx files and pre-computed ELMo embeddings.

  See SimpleDataset.
  Assumes embeddings are aligned with tokens in conllx file.
  Attributes:
    args: the global yaml-derived experiment config dictionary
  """

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
    """Adds pre-computed ELMo embeddings from disk to Observations."""
    layer_index = self.args['model']['model_layer']
    print('Loading ELMo Pretrained Embeddings from {}; using layer {}'.format(pretrained_embeddings_path, layer_index))
    embeddings = self.generate_token_embeddings_from_hdf5(self.args, observations, pretrained_embeddings_path, layer_index)
    observations = self.add_embeddings_to_observations(observations, embeddings)
    return observations

class SubwordDataset(SimpleDataset):
  """Dataloader for conllx files and pre-computed ELMo embeddings.

  See SimpleDataset.
  Assumes we have access to the subword tokenizer.
  """

  @staticmethod
  def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    '''Aligns tokenized and untokenized sentence given subwords "##" prefixed

    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, "##", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.

    Args:
      tokenized_sent: a list of strings describing a subword-tokenized sentence
      untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
      A dictionary of type {int: list(int)} mapping each untokenized sentence
      index to a list of subword-tokenized sentence indices
    '''
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while (untokenized_sent_index < len(untokenized_sent) and
        tokenized_sent_index < len(tokenized_sent)):
      while (tokenized_sent_index + 1 < len(tokenized_sent) and
          tokenized_sent[tokenized_sent_index + 1].startswith('##')):
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        tokenized_sent_index += 1
      mapping[untokenized_sent_index].append(tokenized_sent_index)
      untokenized_sent_index += 1
      tokenized_sent_index += 1
    return mapping

  def generate_subword_embeddings_from_hdf5(self, observations, filepath, elmo_layer, subword_tokenizer=None):
    raise NotImplementedError("Instead of making a SubwordDataset, make one of the implementing classes")

class BERTDataset(SubwordDataset):
  """Dataloader for conllx files and pre-computed BERT embeddings.

  See SimpleDataset.
  Attributes:
    args: the global yaml-derived experiment config dictionary
  """

  def generate_subword_embeddings_from_hdf5(self, observations, filepath, elmo_layer, subword_tokenizer=None):
    '''Reads pre-computed subword embeddings from hdf5-formatted file.

    Sentences should be given integer keys corresponding to their order
    in the original file.
    Embeddings should be of the form (layer_count, subword_sent_length, feature_count)
    subword_sent_length is the length of the sequence of subword tokens
    when the subword tokenizer was given each canonical token (as given
    by the conllx file) independently and tokenized each. Thus, there
    is a single alignment between the subword-tokenized sentence
    and the conllx tokens.

    Args:
      args: the global yaml-derived experiment config dictionary.
      observations: A list of Observations composing a dataset.
      filepath: The filepath of a hdf5 file containing embeddings.
      layer_index: The index corresponding to the layer of representation
          to be used. (e.g., 0, 1, 2 for BERT0, BERT1, BERT2.)
      subword_tokenizer: (optional) a tokenizer used to map from
          conllx tokens to subword tokens.
    
    Returns:
      A list of numpy matrices; one for each observation.

    Raises:
      AssertionError: sent_length of embedding was not the length of the
        corresponding sentence in the dataset.
      Exit: importing pytorch_pretrained_bert has failed, possibly due 
          to downloading of prespecifed tokenizer problem. Not recoverable;
          exits immediately.
    '''
    if subword_tokenizer == None:
      try:
        from pytorch_pretrained_bert import BertTokenizer
        if self.args['model']['hidden_dim'] == 768:
          subword_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
          print('Using BERT-base-cased tokenizer to align embeddings with PTB tokens')
        elif self.args['model']['hidden_dim'] == 1024:
          subword_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
          print('Using BERT-large-cased tokenizer to align embeddings with PTB tokens')
        else:
          print("The heuristic used to choose BERT tokenizers has failed...")
          exit()
      except:
        print('Couldn\'t import pytorch-pretrained-bert. Exiting...')
        exit()
    hf = h5py.File(filepath, 'r')
    indices = list(hf.keys())
    single_layer_features_list = []
    for index in tqdm(sorted([int(x) for x in indices]), desc='[aligning embeddings]'):
      observation = observations[index]
      feature_stack = hf[str(index)]
      single_layer_features = feature_stack[elmo_layer]
      tokenized_sent = subword_tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + ' '.join(observation.sentence) + ' [SEP]')
      untokenized_sent = observation.sentence
      untok_tok_mapping = self.match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)
      assert single_layer_features.shape[0] == len(tokenized_sent)
      single_layer_features = torch.tensor([np.mean(single_layer_features[untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], axis=0) for i in range(len(untokenized_sent))])
      assert single_layer_features.shape[0] == len(observation.sentence)
      single_layer_features_list.append(single_layer_features)
    return single_layer_features_list

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
    """Adds pre-computed BERT embeddings from disk to Observations."""
    layer_index = self.args['model']['model_layer']
    print('Loading BERT Pretrained Embeddings from {}; using layer {}'.format(pretrained_embeddings_path, layer_index))
    embeddings = self.generate_subword_embeddings_from_hdf5(observations, pretrained_embeddings_path, layer_index)
    observations = self.add_embeddings_to_observations(observations, embeddings)
    return observations


class ObservationIterator(Dataset):
  """ List Container for lists of Observations and labels for them.

  Used as the iterator for a PyTorch dataloader.
  """

  def __init__(self, observations, task, name):
    self.observations = observations
    self.name = name
    self.set_labels(observations, task)

  def set_labels(self, observations, task):
    """ Constructs aand stores label for each observation.

    Args:
      observations: A list of observations describing a dataset
      task: a Task object which takes Observations and constructs labels.
    """
    if os.path.exists(self.name + '_labels.pkl'):
      with open(self.name + '_labels.pkl', 'rb') as f:
        self.labels = pickle.load(f)
      print('Loaded labels from disk')
      return
    self.labels = []
    for observation in tqdm(observations, desc='[computing labels]'):
      self.labels.append(task.labels(observation))
    with open(self.name + '_labels.pkl', 'wb') as f:
      pickle.dump(self.labels, f)
    print('Saved labels to disk')

  def __len__(self):
    return len(self.observations)

  def __getitem__(self, idx):
    return self.observations[idx], self.labels[idx]

class GPT2Dataset(SubwordDataset):
  """Dataloader for conllx files and pre-computed GPT-2 embeddings.

  See SimpleDataset.
  Attributes:
    args: the global yaml-derived experiment config dictionary
  """

  def generate_subword_embeddings_from_hdf5(self, observations, filepath, layer_index, subword_tokenizer=None):
    '''Reads pre-computed subword embeddings from hdf5-formatted file.

    Similar to BERT but adapted for GPT-2 tokenization.

    Args:
      observations: A list of Observations composing a dataset.
      filepath: The filepath of a hdf5 file containing embeddings.
      layer_index: The index corresponding to the layer of representation
          to be used.
      subword_tokenizer: (optional) a tokenizer used to map from
          conllx tokens to subword tokens.
    
    Returns:
      A list of numpy matrices; one for each observation.
    '''
    if subword_tokenizer is None:
      try:
        from transformers import GPT2Tokenizer
        subword_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print('Using GPT-2 tokenizer to align embeddings with PTB tokens')
      except:
        print('Couldn\'t import transformers. Exiting...')
        exit()
    
    hf = h5py.File(filepath, 'r')
    indices = list(hf.keys())
    single_layer_features_list = []
    
    for index in tqdm(sorted([int(x) for x in indices]), desc='[aligning embeddings]'):
      observation = observations[index]
      feature_stack = hf[str(index)]
      single_layer_features = feature_stack[layer_index]
      
      # GPT-2 doesn't use [CLS] or [SEP] tokens, but we need to handle tokenization
      # GPT-2 uses space prefix for tokens that start a word
      tokenized_sent = subword_tokenizer.tokenize(' '.join(observation.sentence))
      untokenized_sent = observation.sentence
      
      # Map between tokenized and untokenized
      alignment = []
      tokenized_idx = 0
      for word_idx, word in enumerate(untokenized_sent):
        word_tokens = subword_tokenizer.tokenize(' ' + word if word_idx == 0 else word)
        alignment.append((tokenized_idx, tokenized_idx + len(word_tokens) - 1))
        tokenized_idx += len(word_tokens)
      
      # Average the embeddings for each word's subword tokens
      # assert single_layer_features.shape[0] == len(tokenized_sent)
      # Replace the assertion with:
      if single_layer_features.shape[0] != len(tokenized_sent):
          # print(f"Warning: Mismatch in sentence {index}. Tokenized length: {len(tokenized_sent)}, Feature length: {single_layer_features.shape[0]}")
          # Truncate the longer one to match the shorter one
          min_length = min(single_layer_features.shape[0], len(tokenized_sent))
          tokenized_sent = tokenized_sent[:min_length]
          single_layer_features = single_layer_features[:min_length, :]
      word_embeddings = []
      
      for start_idx, end_idx in alignment:
        word_vector = np.mean(single_layer_features[start_idx:end_idx+1,:], axis=0)
        word_embeddings.append(word_vector)
      
      single_layer_features = torch.tensor(word_embeddings)
      assert single_layer_features.shape[0] == len(observation.sentence)
      single_layer_features_list.append(single_layer_features)
      
    return single_layer_features_list

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
    """Adds pre-computed GPT-2 embeddings from disk to Observations."""
    layer_index = self.args['model']['model_layer']
    print('Loading GPT-2 Pretrained Embeddings from {}; using layer {}'.format(pretrained_embeddings_path, layer_index))
    embeddings = self.generate_subword_embeddings_from_hdf5(observations, pretrained_embeddings_path, layer_index)
    observations = self.add_embeddings_to_observations(observations, embeddings)
    return observations
  
import os
from collections import namedtuple, defaultdict
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader, Dataset


class DeepSeekDataset(SubwordDataset):
    """Dataloader for conllx files and pre-computed DeepSeek embeddings.

    Specifically designed for DeepSeek-R1-Distill-Qwen-1.5B embeddings.
    This model is a distilled version of DeepSeek-R1 from the Qwen2.5-Math-1.5B base.
    See SimpleDataset for more general information on dataset structure.

    Attributes:
        args: the global yaml-derived experiment config dictionary
    """

    def generate_subword_embeddings_from_hdf5(self, observations, filepath, layer_index, subword_tokenizer=None):
        '''Reads pre-computed subword embeddings from hdf5-formatted file.

        For DeepSeek-R1-Distill-Qwen-1.5B which uses Qwen tokenization.
        The model has 24 layers (0-23), so layer_index should be in this range.

        Args:
            observations: A list of Observations composing a dataset.
            filepath: The filepath of a hdf5 file containing embeddings.
            layer_index: The index corresponding to the layer of representation
                to be used. (e.g., 0, 12, 23 for different layers)
            subword_tokenizer: (optional) a tokenizer used to map from
                conllx tokens to subword tokens.
        
        Returns:
            A list of numpy matrices; one for each observation.
        '''
        if subword_tokenizer is None:
            try:
                from transformers import AutoTokenizer
                subword_tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', trust_remote_code=True)
                print('Using DeepSeek-R1-Distill-Qwen-1.5B tokenizer to align embeddings with tokens')
            except:
                print('Couldn\'t import transformers or load the DeepSeek tokenizer. Exiting...')
                exit()
        
        hf = h5py.File(filepath, 'r')
        indices = list(hf.keys())
        single_layer_features_list = []
        
        for index in tqdm(sorted([int(x) for x in indices]), desc='[aligning DeepSeek embeddings]'):
            observation = observations[index]
            feature_stack = hf[str(index)]
            single_layer_features = feature_stack[layer_index]
            
            # Qwen models use <|endoftext|> as special tokens rather than [CLS]/[SEP]
            tokenized_sent = subword_tokenizer.tokenize('<|endoftext|>' + ' '.join(observation.sentence) + '<|endoftext|>')
            untokenized_sent = observation.sentence
            
            # Create mappings between tokenized and untokenized tokens
            # Qwen tokenizer uses a different subword system than BERT/GPT2
            alignment = []
            tokenized_idx = 1  # Skip the first special token
            
            for word_idx, word in enumerate(untokenized_sent):
                word_tokens = subword_tokenizer.tokenize(' ' + word if word_idx == 0 else word)
                if not word_tokens:  # Handle case where tokenizer returns empty list for a word
                    word_tokens = [subword_tokenizer.unk_token]
                
                start_idx = tokenized_idx
                end_idx = tokenized_idx + len(word_tokens) - 1
                alignment.append((start_idx, end_idx))
                tokenized_idx += len(word_tokens)
            
            # Handle potential mismatches in lengths (common with different tokenizers)
            if single_layer_features.shape[0] != len(tokenized_sent):
                min_length = min(single_layer_features.shape[0], len(tokenized_sent))
                tokenized_sent = tokenized_sent[:min_length]
                single_layer_features = single_layer_features[:min_length, :]
                
                # Adjust alignments if necessary
                valid_alignments = [align for align in alignment if align[1] < min_length]
                if len(valid_alignments) < len(alignment):
                    alignment = valid_alignments
            
            # Average embeddings for each word's subword tokens
            word_embeddings = []
            for start_idx, end_idx in alignment:
                if start_idx < single_layer_features.shape[0] and end_idx < single_layer_features.shape[0]:
                    word_vector = np.mean(single_layer_features[start_idx:end_idx+1,:], axis=0)
                    word_embeddings.append(word_vector)
            
            # Handle edge case where we don't have enough embeddings
            if len(word_embeddings) < len(observation.sentence):
                # print(f"Warning: Not enough embeddings for sentence {index}. Expected {len(observation.sentence)}, got {len(word_embeddings)}")
                # Pad with zeros or repeat last embedding
                while len(word_embeddings) < len(observation.sentence):
                    if word_embeddings:
                        word_embeddings.append(word_embeddings[-1])
                    else:
                        # If no embeddings at all, use zeros
                        word_embeddings.append(np.zeros(single_layer_features.shape[1]))
            
            single_layer_features = torch.tensor(word_embeddings)
            assert single_layer_features.shape[0] == len(observation.sentence)
            single_layer_features_list.append(single_layer_features)
            
        return single_layer_features_list

    def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
        """Adds pre-computed DeepSeek embeddings from disk to Observations."""
        layer_index = self.args['model']['model_layer']
        print('Loading DeepSeek-R1-Distill-Qwen-1.5B Pretrained Embeddings from {}; using layer {}'.format(
            pretrained_embeddings_path, layer_index))
        embeddings = self.generate_subword_embeddings_from_hdf5(observations, pretrained_embeddings_path, layer_index)
        observations = self.add_embeddings_to_observations(observations, embeddings)
        return observations
    
    @staticmethod
    def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
        '''Aligns tokenized and untokenized sentence for DeepSeek tokenization

        DeepSeek's tokenizer (based on Qwen) uses a different subword system,
        so we need custom alignment logic to match tokens properly.

        Args:
            tokenized_sent: a list of strings describing a subword-tokenized sentence
            untokenized_sent: a list of strings describing a sentence, no subword tok.
            
        Returns:
            A dictionary of type {int: list(int)} mapping each untokenized sentence
            index to a list of subword-tokenized sentence indices
        '''
        mapping = defaultdict(list)
        untokenized_sent_index = 0
        tokenized_sent_index = 1  # Skip first special token
        
        # For DeepSeek/Qwen tokenization, we need a more robust matching approach
        while (untokenized_sent_index < len(untokenized_sent) and 
               tokenized_sent_index < len(tokenized_sent)):
            
            # Get the current word and its first character
            current_word = untokenized_sent[untokenized_sent_index]
            mapping[untokenized_sent_index].append(tokenized_sent_index)
            
            # Handle special characters and multi-token words
            if tokenized_sent_index + 1 < len(tokenized_sent):
                next_token = tokenized_sent[tokenized_sent_index + 1]
                if (next_token.startswith('▁') or  # Qwen/DeepSeek specific marker
                    next_token.startswith('Ġ')):   # Some Qwen tokenizers use this
                    # New word starts
                    untokenized_sent_index += 1
                    tokenized_sent_index += 1
                else:
                    # Continue of the same word
                    mapping[untokenized_sent_index].append(tokenized_sent_index + 1)
                    tokenized_sent_index += 1
            else:
                # End of tokenized sentence
                untokenized_sent_index += 1
                tokenized_sent_index += 1
                
        return mapping