"""
This file is part of the DeepTMHMM project.

For license information, please see the README.txt file in the root directory.
"""

import sys
import glob
import pickle
import time
from typing import Tuple
from pathlib import Path


import subprocess
import shlex
from hashlib import md5
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from deeptmhmm import openprotein
from experiments.tmhmm3.tm_util import batch_sizes_to_mask, \
    initialize_crf_parameters, decode
from experiments.tmhmm3.tm_util import is_topologies_equal, \
    allowed_start_states, allowed_end_states, allowed_state_transitions, crf_states, max_signal_length, max_alpha_membrane_length, max_beta_membrane_length
from experiments.tmhmm3.tm_util import original_labels_to_fasta

from experiments.tmhmm3.tm_util import distributed_data_path, hash_aa_string, calculate_acc

from pytorchcrf.torchcrf import CRF
from deeptmhmm.util import write_out, get_experiment_id
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained

# seed random generator for reproducibility
torch.manual_seed(1)


class TMHMM3(openprotein.BaseModel):
    def __init__(self,
                 embedding,
                 hidden_size,
                 use_gpu,
                 use_marg_prob,
                 type_predictor_model,
                 profile_path,
                 esm_embeddings_dir=None):
        super(TMHMM3, self).__init__(embedding, use_gpu)

        # initialize model variables
        num_labels = 6 # I + O + P + S + M + B
        num_tags = len(crf_states)
        self.hidden_size = hidden_size
        self.use_gpu = use_gpu
        self.use_marg_prob = use_marg_prob
        self.embedding = embedding
        if self.embedding == "PYTORCH":
            self.pytorch_embedding = nn.Embedding(self.get_embedding_size(),
                                                  self.get_embedding_size())
        
        self.esm_embeddings_dir = esm_embeddings_dir
        
        self.profile_path = profile_path
        self.bi_lstm = nn.LSTM(self.get_embedding_size(),
                               self.hidden_size,
                               num_layers=1,
                               bidirectional=True)
        self.hidden_to_labels = nn.Linear(self.hidden_size * 2, num_labels)  # * 2 for bidirectional
        self.hidden_layer = None
        self.dropout_p = 0.4
        self.dropout_layer = nn.Dropout(p=self.dropout_p)
        crf_start_mask = torch.ones(num_tags, dtype=torch.uint8) == 1
        crf_end_mask = torch.ones(num_tags, dtype=torch.uint8) == 1

        allowed_transitions = allowed_state_transitions

        for i in allowed_start_states:
            crf_start_mask[i] = 0
        for i in allowed_end_states:
            crf_end_mask[i] = 0


        self.allowed_transitions = allowed_transitions
        self.crf_model = CRF(num_tags)
        self.type_classifier = type_predictor_model
        if self.type_classifier:
            # type prediction is only used for evaluation
            self.type_classifier.eval()
        self.type_tm_classier = None
        self.type_sp_classier = None
        crf_transitions_mask = torch.ones((num_tags, num_tags), dtype=torch.uint8) == 1

        self.type_01loss_values = []
        self.topology_01loss_values = []

        # if on GPU, move state to GPU memory
        if self.use_gpu:
            self.crf_model = self.crf_model.cuda()
            self.bi_lstm = self.bi_lstm.cuda()
            self.dropout_layer = self.dropout_layer.cuda()
            self.hidden_to_labels = self.hidden_to_labels.cuda()
            crf_transitions_mask = crf_transitions_mask.cuda()
            crf_start_mask = crf_start_mask.cuda()
            crf_end_mask = crf_end_mask.cuda()

        # compute mask matrix from allow transitions list
        for i in range(num_tags):
            for k in range(num_tags):
                if (i, k) in self.allowed_transitions:
                    crf_transitions_mask[i][k] = 0

        # generate masked transition parameters
        crf_start_transitions, crf_end_transitions, crf_transitions = \
            generate_masked_crf_transitions(
                self.crf_model, (crf_start_mask, crf_transitions_mask, crf_end_mask)
            )

        # initialize CRF
        initialize_crf_parameters(self.crf_model,
                                  start_transitions=crf_start_transitions,
                                  end_transitions=crf_end_transitions,
                                  transitions=crf_transitions)

        # generate crf model use for marginal probability decoding
        self.marginal_crf_model = CRF(num_tags)
        start_transitions = torch.ones(num_tags) * -100000000
        end_transitions = torch.ones(num_tags) * -100000000

        transitions = torch.ones((num_tags, num_tags)) * -100000000
        for i in allowed_start_states:
            start_transitions[i] = 0
        for i in allowed_end_states:
            end_transitions[i] = 0

        for i in range(num_tags):
            for k in range(num_tags):
                if (i, k) in allowed_transitions:
                    transitions[i][k] = 0

        if self.use_gpu:
            self.marginal_crf_model = self.marginal_crf_model.cuda()
            start_transitions = start_transitions.cuda()
            transitions = transitions.cuda()
            end_transitions = end_transitions.cuda()

        initialize_crf_parameters(self.marginal_crf_model,
                                  start_transitions=start_transitions,
                                  transitions=transitions,
                                  end_transitions=end_transitions)


    def get_embedding_size(self):
        if self.embedding == "BLOSUM62":
            return 24  # bloom matrix has size 24
        elif self.embedding == "ONEHOT":
            return 24
        elif self.embedding == "ESM":
            return 1280
        elif self.embedding == "PYTORCH":
            return 24  # map an index (0-23) to 24 floats
        elif self.embedding == "PROFILE":
            return 51  # protein profiles have size 51

    def flatten_parameters(self):
        self.bi_lstm.flatten_parameters()

    def encode_amino_acid(self, letter):
        if self.embedding == "BLOSUM62":
            # blosum encoding
            if not globals().get('blosum_encoder'):
                blosum = \
                    """4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0,-2,-1,0,-4
                    -1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3,-1,0,-1,-4
                    -2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3,3,0,-1,-4
                    -2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3,4,1,-1,-4
                    0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4
                    -1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2,0,3,-1,-4
                    -1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2,1,4,-1,-4
                    0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3,-1,-2,-1,-4
                    -2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3,0,0,-1,-4
                    -1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3,-3,-3,-1,-4
                    -1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1,-4,-3,-1,-4
                    -1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2,0,1,-1,-4
                    -1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1,-3,-1,-1,-4
                    -2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1,-3,-3,-1,-4
                    -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2,-2,-1,-2,-4
                    1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2,0,0,0,-4
                    0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0,-1,-1,0,-4
                    -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3,-4,-3,-2,-4
                    -2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1,-3,-2,-1,-4
                    0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4,-3,-2,-1,-4
                    -2,-1,3,4,-3,0,1,-1,0,-3,-4,0,-3,-3,-2,0,-1,-4,-3,-3,4,1,-1,-4
                    -1,0,0,1,-3,3,4,-2,0,-3,-3,1,-1,-3,-1,0,-1,-3,-2,-2,1,4,-1,-4
                    0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,0,0,-2,-1,-1,-1,-1,-1,-4
                    -4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,1""" \
                        .replace('\n', ',')
                blosum_matrix = np.fromstring(blosum, sep=",").reshape(24, 24)
                blosum_key = "A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,B,Z,X,U".split(",")
                key_map = {}
                for idx, value in enumerate(blosum_key):
                    key_map[value] = list([int(v) for v in blosum_matrix[idx].astype('int')])
                globals().__setitem__("blosum_encoder", key_map)
            return globals().get('blosum_encoder')[letter]
        elif self.embedding == "ONEHOT":
            # one hot encoding
            one_hot_key = "A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,B,Z,X,U".split(",")
            arr = []
            for idx, k in enumerate(one_hot_key):
                if k == letter:
                    arr.append(1)
                else:
                    arr.append(0)
            return arr
        elif self.embedding == "PYTORCH":
            key_id = "A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,B,Z,X,U".split(",")
            for idx, k in enumerate(key_id):
                if k == letter:
                    return idx

    def embed(self, prot_aa_list):

        if self.embedding == "ESM":
            return self.get_esm_embeddings(prot_aa_list)
        embed_list = []
        for aa_list in prot_aa_list:
            if self.embedding == "PYTORCH":
                tensor = list([self.encode_amino_acid(aa) for aa in aa_list])
                tensor = torch.LongTensor(tensor)
                if self.use_gpu:
                    tensor = tensor.cuda()
                tensor = self.pytorch_embedding(tensor)
            elif self.embedding == "PROFILE":
                if not globals().get('profile_encoder'):
                    print("Load profiles...")
                    files = glob.glob(self.profile_path.strip("/") + "/*")
                    profile_dict = {}
                    for profile_file in files:
                        profile = pickle.load(open(profile_file, "rb")).popitem()[1]
                        profile_dict[profile["seq"]] = torch.from_numpy(profile["profile"]).float()
                    globals().__setitem__("profile_encoder", profile_dict)
                    print("Loaded profiles")
                tensor = globals().get('profile_encoder')[aa_list]
            else:
                tensor = list([self.encode_amino_acid(aa) for aa in aa_list])
                tensor = torch.FloatTensor(tensor)
            if self.use_gpu:
                tensor = tensor.cuda()
            embed_list.append(tensor)
        input_sequences = list([x for x in embed_list])
        input_sequences_padded = torch.nn.utils.rnn.pad_sequence(input_sequences)
        return input_sequences_padded

    def init_hidden(self, minibatch_size):
        # number of layers (* 2 since bidirectional), minibatch_size, hidden size
        initial_hidden_state = torch.zeros(1 * 2, minibatch_size, self.hidden_size)
        initial_cell_state = torch.zeros(1 * 2, minibatch_size, self.hidden_size)
        if self.use_gpu:
            initial_hidden_state = initial_hidden_state.cuda()
            initial_cell_state = initial_cell_state.cuda()
        self.hidden_layer = (autograd.Variable(initial_hidden_state),
                             autograd.Variable(initial_cell_state))

    def _get_network_emissions(self, original_aa_string):
        input_sequences_padded = self.embed(original_aa_string)
        batch_sizes = torch.IntTensor(list([len(x) for x in original_aa_string]))
        if self.use_gpu:
            batch_sizes = batch_sizes.cuda()

        minibatch_size = input_sequences_padded.size(1)

        self.init_hidden(minibatch_size)
        bi_lstm_out, self.hidden_layer = self.bi_lstm(input_sequences_padded, self.hidden_layer)

        dropout = self.dropout_layer(bi_lstm_out)
        emissions = self.hidden_to_labels(dropout)
        ones = torch.ones(1, dtype=torch.long)

        if emissions.is_cuda:
            ones = ones.cuda()
        I = torch.index_select(emissions, 2, ones * 0)
        O = torch.index_select(emissions, 2, ones * 1)
        P = torch.index_select(emissions, 2, ones * 2)
        S = torch.index_select(emissions, 2, ones * 3)
        M = torch.index_select(emissions, 2, ones * 4)
        B = torch.index_select(emissions, 2, ones * 5)

        emissions = torch.cat((I.expand(-1, minibatch_size, 1),
                               O.expand(-1, minibatch_size, 2),
                               P.expand(-1, minibatch_size, 1),
                               S.expand(-1, minibatch_size, max_signal_length),
                               M.expand(-1, minibatch_size, max_alpha_membrane_length),
                               M.expand(-1, minibatch_size, max_alpha_membrane_length),
                               B.expand(-1, minibatch_size, max_beta_membrane_length),
                               B.expand(-1, minibatch_size, max_beta_membrane_length),
                               ), 2)
        return emissions, batch_sizes

    def compute_loss(self, training_minibatch):
        _, labels_list, remapped_labels_list_crf_hmm, _, _, _, names, original_aa_string, \
        _original_label_string = training_minibatch

        labels_to_use = remapped_labels_list_crf_hmm

        actual_labels = torch.nn.utils.rnn.pad_sequence([l for l in labels_to_use])
        emissions, batch_sizes = self._get_network_emissions(original_aa_string)
        mask = batch_sizes_to_mask(batch_sizes)
        
        loss = self.calculate_loss(emissions, actual_labels, mask=mask)
        if float(loss) > 100000:  # if loss is this large, an invalid tx must have been found
            for idx, batch_size in enumerate(batch_sizes):
                last_label = None
                write_out("protein issue:", names[idx])
                write_out("labels to use:", actual_labels[idx])
                write_out("original labels:", _original_label_string[idx])
                write_out("original aa string:", original_aa_string[idx])
                write_out("loss:", loss)
                for i in range(batch_size):
                    label = int(actual_labels[i][idx])
                    write_out(str(label) + ",", end='')
                    if last_label is not None and (last_label, label) \
                            not in self.allowed_transitions:
                        write_out("Error: invalid transition found")
                        write_out((last_label, label))
                        sys.exit(1)
                    last_label = label
                write_out(" ")
        return loss

    def calculate_loss(self, emissions, actual_labels, mask):
        loss = -1 * self.crf_model(emissions, actual_labels, mask=mask) / int(emissions.size()[1]) # crf loss / minibatch size
        return loss

    def get_emissions_for_decoding(self, original_aa_string, optional_actual_labels=None):
        emissions, batch_sizes = self._get_network_emissions(original_aa_string)

        mask = batch_sizes_to_mask(batch_sizes)
        if emissions.is_cuda:
            mask = mask.cuda()

        if optional_actual_labels is not None:
            loss = self.calculate_loss(emissions, optional_actual_labels, mask=mask)
        else:
            loss = None

        if self.use_marg_prob:
            return self.crf_model.compute_marginal_probabilities(emissions, mask), mask, loss
        else:
            return emissions, mask, loss


    def forward(self, original_aa_string, optional_actual_labels=None):
        emissions, mask, loss = self.get_emissions_for_decoding(original_aa_string, optional_actual_labels)
        predicted_labels, predicted_types, predicted_topologies, raw_crf_labels = \
            decode(emissions, mask, self.marginal_crf_model if self.use_marg_prob else self.crf_model)
        return predicted_labels, predicted_types, predicted_topologies, loss, emissions, mask, raw_crf_labels

    def evaluate_model(self, data_loader, distribute=False, writer=None, minibatches_proccesed=None):
        # toggle eval
        self.eval()
        experiment_id = get_experiment_id()
        if distribute:
            print("Distributing validation")
            torch.save(self, open(f'{distributed_data_path}/eval_model.pt', 'wb'))
            path_to_the_folder_containing_evaluate = Path(__file__).parent.parent.parent
            subprocess.run(shlex.split(f'python3 {path_to_the_folder_containing_evaluate}/evaluate.py --experiment_id {experiment_id}'))
            validation_loss_tracker = torch.load(open(f'{distributed_data_path}/val_loss_tracker.pt', 'rb'), map_location=torch.device('cpu'))
            confusion_matrix = torch.load(open(f'{distributed_data_path}/confusion_matrix.pt', 'rb'), map_location=torch.device('cpu'))
            protein_names = torch.load(open(f'{distributed_data_path}/protein_names.pt', 'rb'), map_location=torch.device('cpu'))
            protein_aa_strings = torch.load(open(f'{distributed_data_path}/protein_aa_strings.pt', 'rb'), map_location=torch.device('cpu'))
            protein_label_actual = torch.load(open(f'{distributed_data_path}/protein_label_actual.pt', 'rb'), map_location=torch.device('cpu'))
            protein_label_prediction = torch.load(open(f'{distributed_data_path}/protein_label_prediction.pt', 'rb'), map_location=torch.device('cpu'))
        else:
            validation_loss_tracker = []
            confusion_matrix = torch.zeros((6, 6), dtype=torch.int64)
            protein_names = []
            protein_aa_strings = []
            protein_label_actual = []
            protein_label_prediction = []
            with torch.no_grad():
                for _, minibatch in enumerate(data_loader, 0):

                    _, _, remapped_labels_list_crf_hmm, _, prot_type_list, prot_topology_list, \
                    prot_name_list, original_aa_string, original_label_string = minibatch
                    actual_labels = torch.nn.utils.rnn.pad_sequence([l for l in remapped_labels_list_crf_hmm])
                    predicted_labels, predicted_types, predicted_topologies, loss = self(original_aa_string, actual_labels)
                    validation_loss_tracker.append(loss.detach())
                    protein_names.extend(prot_name_list)
                    protein_aa_strings.extend(original_aa_string)
                    protein_label_actual.extend(original_label_string)

                    for idx, actual_type in enumerate(prot_type_list):

                        predicted_type = predicted_types[idx]
                        predicted_topology = predicted_topologies[idx]
                        predicted_labels_for_protein = predicted_labels[idx]

                        prediction_topology_match = is_topologies_equal(prot_topology_list[idx],
                                                                        predicted_topology, 5)

                        if actual_type == predicted_type:
                            # if we guessed the type right for SP+GLOB or GLOB,
                            # count the topology as correct
                            if actual_type == 2 or actual_type == 3 or prediction_topology_match:
                                confusion_matrix[actual_type][5] += 1
                            else:
                                confusion_matrix[actual_type][predicted_type] += 1
                                                            
                        else:
                            confusion_matrix[actual_type][predicted_type] += 1
                        
                        protein_label_prediction.append(predicted_labels_for_protein)

        write_out(confusion_matrix)
        _loss = float(torch.stack(validation_loss_tracker).mean())
        write_out("Validation error function loss: ", _loss)

        type_correct_ratio = \
            calculate_acc(confusion_matrix[0][0] + confusion_matrix[0][5], confusion_matrix[0].sum()) + \
            calculate_acc(confusion_matrix[1][1] + confusion_matrix[1][5], confusion_matrix[1].sum()) + \
            calculate_acc(confusion_matrix[2][2] + confusion_matrix[2][5], confusion_matrix[2].sum()) + \
            calculate_acc(confusion_matrix[3][3] + confusion_matrix[3][5], confusion_matrix[3].sum()) + \
            calculate_acc(confusion_matrix[4][4] + confusion_matrix[4][5], confusion_matrix[4].sum())
        type_accuracy = float((type_correct_ratio / 5).detach())

        tm_topology_acc = calculate_acc(confusion_matrix[0][5], confusion_matrix[0].sum())
        tm_sp_topology_acc = calculate_acc(confusion_matrix[1][5], confusion_matrix[1].sum())
        sp_topology_acc = calculate_acc(confusion_matrix[2][5], confusion_matrix[2].sum())
        glob_topology_acc = calculate_acc(confusion_matrix[3][5], confusion_matrix[3].sum())
        beta_topology_acc = calculate_acc(confusion_matrix[4][5], confusion_matrix[4].sum())
                
        topology_accuracy = float((tm_topology_acc + tm_sp_topology_acc + sp_topology_acc + glob_topology_acc + beta_topology_acc).detach() / 5)
        
        if writer:
            tm_type_acc = calculate_acc(confusion_matrix[0][0] + confusion_matrix[0][5], confusion_matrix[0].sum())
            tm_sp_type_acc = calculate_acc(confusion_matrix[1][1] + confusion_matrix[1][5], confusion_matrix[1].sum())
            sp_type_acc = calculate_acc(confusion_matrix[2][2] + confusion_matrix[2][5], confusion_matrix[2].sum())
            glob_type_acc = calculate_acc(confusion_matrix[3][3] + confusion_matrix[3][5], confusion_matrix[3].sum())
            beta_type_acc = calculate_acc(confusion_matrix[4][4] + confusion_matrix[4][5], confusion_matrix[4].sum())

            writer.add_scalar("Error/validation", _loss, minibatches_proccesed)

            writer.add_scalar("Validation Topology Acc/TM", tm_topology_acc, minibatches_proccesed)
            writer.add_scalar("Validation Topology Acc/TM+SP", tm_sp_topology_acc, minibatches_proccesed)
            writer.add_scalar("Validation Topology Acc/Beta", beta_topology_acc, minibatches_proccesed)

            writer.add_scalar("Validation Type Acc/TM", tm_type_acc, minibatches_proccesed)
            writer.add_scalar("Validation Type Acc/TM+SP", tm_sp_type_acc, minibatches_proccesed)
            writer.add_scalar("Validation Type Acc/SP", sp_type_acc, minibatches_proccesed)
            writer.add_scalar("Validation Type Acc/Glob", glob_type_acc, minibatches_proccesed)
            writer.add_scalar("Validation Type Acc/Beta", beta_type_acc, minibatches_proccesed)

 
        validation_loss = _loss
        validation_accuracy = topology_accuracy

        data = {}
        data['confusion_matrix'] = confusion_matrix
        data['validation_loss_tracker'] = validation_loss_tracker
        data['validation_accuracy'] = validation_accuracy
        
        # re-toggle train
        self.train()

        return validation_loss, data, (
            protein_names, protein_aa_strings, protein_label_actual, protein_label_prediction)                                          
                                                  
    def get_esm_embeddings(self, original_aa_string):
        embeddings_list = []
        masked_embeddings_list = []
        max_length = 0
        for i, aa_string in enumerate(original_aa_string):
            if self.use_gpu:
                embedding = torch.load(f'{self.esm_embeddings_dir}/{hash_aa_string(aa_string)}', map_location=f'cuda:{torch.cuda.current_device()}')
            else:
                embedding = torch.load(f'{self.esm_embeddings_dir}/{hash_aa_string(aa_string)}', map_location='cpu')
            embeddings_list.append(embedding)
            if embedding.size()[0] > max_length:
                max_length = embedding.size()[0]
        
        for embedding in embeddings_list:
            sequence_len = embedding.size()[0]
            if sequence_len < max_length:
                tensor_mask_to_append = torch.zeros(max_length - sequence_len, 1280)
                if self.use_gpu:
                    tensor_mask_to_append = tensor_mask_to_append.cuda()
    
                masked_embeddings_list.append(torch.cat((embedding, tensor_mask_to_append)))
                
            else:
                masked_embeddings_list.append(embedding)
        
        return torch.stack(masked_embeddings_list).transpose(0,1)


def post_process_prediction_data(prediction_data):
    data = []
    for (name, aa_string, actual, prediction) in zip(*prediction_data):
        data.append("\n".join([">" + name,
                               aa_string,
                               actual,
                               original_labels_to_fasta(prediction)]))
    return "\n".join(data)


def logsumexp(data, dim):
    return data.max(dim)[0] + torch.log(torch.sum(
        torch.exp(data - data.max(dim)[0].unsqueeze(dim)), dim))


def generate_masked_crf_transitions(crf_model, transition_mask):
    start_transitions_mask, transitions_mask, end_transition_mask = transition_mask
    start_transitions = crf_model.start_transitions.data.clone()
    end_transitions = crf_model.end_transitions.data.clone()
    transitions = crf_model.transitions.data.clone()
    if start_transitions_mask is not None:
        start_transitions.masked_fill_(start_transitions_mask, -100000000)
    if end_transition_mask is not None:
        end_transitions.masked_fill_(end_transition_mask, -100000000)
    if transitions_mask is not None:
        transitions.masked_fill_(transitions_mask, -100000000)
    return start_transitions, end_transitions, transitions
