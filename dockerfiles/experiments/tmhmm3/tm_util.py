"""
This file is part of the DeepTMHMM project.

For license information, please see the README.txt file in the root directory.
"""

import math
import pathlib
import random
import json

from typing import List

from hashlib import md5

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from pytorchcrf.torchcrf import CRF
from deeptmhmm.util import write_out, load_model_from_disk
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained

 
distributed_data_path = '/tmp/distributed_data'

crf_states = {}
crf_states["I"] = 0
crf_states["Om"] = 1
crf_states["Ob"] = 2
crf_states["Pb"] = 3
crf_next_state = 4

# Segment lengths
max_signal_length = 60
max_alpha_membrane_length = 30
max_beta_membrane_length = 15

minimum_segment_lengths = {
    'S': 10,   # Signal
    'Mio': 10, # Alpha TM
    'Moi': 10, # Alpha TM
    'Bpo': 5,  # Beta TM
    'Bop': 5,  # Beta TM
}


for tm_type in ["S", "Mio", "Moi", "Bpo", "Bop"]:
    if tm_type == 'S':
        max_segment_length = max_signal_length
    elif tm_type in ("Mio", "Moi"):
        max_segment_length = max_alpha_membrane_length
    else:
        max_segment_length = max_beta_membrane_length

    for i in range(max_segment_length):
        crf_states[tm_type+str(i)] = crf_next_state
        crf_next_state += 1
        

allowed_state_transitions = []
allowed_state_transitions.append((crf_states["I"], crf_states["I"]))
allowed_state_transitions.append((crf_states["Om"], crf_states["Om"]))
allowed_state_transitions.append((crf_states["Ob"], crf_states["Ob"]))
allowed_state_transitions.append((crf_states["Pb"], crf_states["Pb"]))

last_signal_state = crf_states["S"+str(max_signal_length-1)]
allowed_state_transitions.append((last_signal_state, crf_states["Pb"]))
allowed_state_transitions.append((last_signal_state, crf_states["Om"]))

allowed_state_transitions.append((crf_states["Pb"], crf_states["Bpo0"]))
allowed_state_transitions.append((crf_states["Om"], crf_states["Moi0"]))

allowed_state_transitions.append((crf_states["Bpo" + str(max_beta_membrane_length - 1)], crf_states["Ob"]))
allowed_state_transitions.append((crf_states["Moi" + str(max_alpha_membrane_length - 1)], crf_states["I"]))

allowed_state_transitions.append((crf_states["Ob"], crf_states["Bop0"]))
allowed_state_transitions.append((crf_states["I"], crf_states["Mio0"]))

allowed_state_transitions.append((crf_states["Bop" + str(max_beta_membrane_length - 1)], crf_states["Pb"]))
allowed_state_transitions.append((crf_states["Mio" + str(max_alpha_membrane_length - 1)], crf_states["Om"]))

for tm_type in ["S", "Mio", "Moi", "Bpo", "Bop"]:
    if tm_type == 'S':
        max_segment_length = max_signal_length
    elif tm_type in ("Mio", "Moi"):
        max_segment_length = max_alpha_membrane_length
    else:
        max_segment_length = max_beta_membrane_length

    for i in range(max_segment_length - 1): # no transition for last label
        allowed_state_transitions.append((crf_states[tm_type + str(i)], crf_states[tm_type + str(i + 1)]))
        start_segment_length = math.ceil(minimum_segment_lengths[tm_type] / 2)
        end_segment_length = math.floor(minimum_segment_lengths[tm_type] / 2)
        if i >= start_segment_length + 1 and i <= max_segment_length - end_segment_length:  # minimum segment size of start_segment_length + end_segment_length
            allowed_state_transitions.append((crf_states[tm_type + str(start_segment_length - 1)], crf_states[tm_type + str(i)]))

allowed_start_states = [
    crf_states["S0"],
    crf_states["Om"],
    crf_states["I"],
    crf_states["Pb"],
]

allowed_end_states = [
    crf_states["Pb"],
    crf_states["Ob"],
    crf_states["Om"],
    crf_states["I"],
]

def get_remapped_labels_from_topology(topology, protein_length):
    remapped_labels_crf_hmm = []

    labels = list(zip(*topology))[1]
    contains_B = 5 in labels
    topology_length = len(topology)
    for idx, (pos, label) in enumerate(topology):
        labels_to_add = (protein_length - pos) if idx == (topology_length - 1) else (topology[idx + 1][0] - pos)
        if label == 3:  # S
            s_beginning = math.ceil(minimum_segment_lengths["S"] / 2)
            for i in range(s_beginning):
                remapped_labels_crf_hmm.append(crf_states["S"+str(i)])
            for i in range(max_signal_length - (labels_to_add - s_beginning), max_signal_length):
                remapped_labels_crf_hmm.append(crf_states["S"+str(i)])

        elif label == 4:  # M
            mm_beginning = math.ceil(minimum_segment_lengths["Mio"] / 2)
            prior_label = topology[idx - 1][1]
            type_id = "Moi" if prior_label in (1, 2) else "Mio"
            for i in range(mm_beginning):
                remapped_labels_crf_hmm.append(crf_states[type_id+str(i)])
            for i in range(max_alpha_membrane_length - (labels_to_add - mm_beginning), max_alpha_membrane_length):
                remapped_labels_crf_hmm.append(crf_states[type_id + str(i)])
        elif label == 5:  # B
            mm_beginning = math.ceil(minimum_segment_lengths["Bop"] / 2)
            prior_label = topology[idx - 1][1]
            type_id = "Bpo" if prior_label == 2 else "Bop"
            for i in range(mm_beginning):
                remapped_labels_crf_hmm.append(crf_states[type_id+str(i)])
            for i in range(max_beta_membrane_length - (labels_to_add - mm_beginning), max_beta_membrane_length):
                try:
                    remapped_labels_crf_hmm.append(crf_states[type_id + str(i)])
                except:
                    print(range(max_beta_membrane_length - (labels_to_add - mm_beginning), max_beta_membrane_length))
                    print("Max length ", max_beta_membrane_length)
                    print("labels to add ", labels_to_add)
                    print("mm_beginning ", mm_beginning)
                    print("2nd term ", (labels_to_add - mm_beginning))
                    print(i)
        else:
            label_remapped = None
            if label == 0: # I
                label_remapped = "I"
            elif label == 1: # O
                label_remapped = "Ob" if contains_B else "Om"
            elif label == 2: # P
                label_remapped = "Pb" if contains_B else "Om"
            else:
                print("Unknown label:", label)
                exit(1)
            for i in range(labels_to_add):
                remapped_labels_crf_hmm.append(crf_states[label_remapped])

    return remapped_labels_crf_hmm

class TMDataset(Dataset):
    def __init__(self,
                 aa_list,
                 label_list,
                 remapped_labels_list_crf_hmm,
                 remapped_labels_list_crf_marg,
                 type_list,
                 topology_list,
                 prot_name_list,
                 original_aa_string_list,
                 original_label_string):
        assert len(aa_list) == len(label_list)
        assert len(aa_list) == len(type_list)
        assert len(aa_list) == len(topology_list)
        self.aa_list = aa_list
        self.label_list = label_list
        self.remapped_labels_list_crf_hmm = remapped_labels_list_crf_hmm
        self.remapped_labels_list_crf_marg = remapped_labels_list_crf_marg
        self.type_list = type_list
        self.topology_list = topology_list
        self.prot_name_list = prot_name_list
        self.original_aa_string_list = original_aa_string_list
        self.original_label_string = original_label_string

    @staticmethod
    def from_disk(dataset, use_gpu):
        print("Constructing data set from disk...")
        aa_list = []
        labels_list = []
        remapped_labels_list_crf_hmm = []
        remapped_labels_list_crf_marg = []
        prot_type_list = []
        prot_topology_list_all = []
        prot_aa_list_all = []
        prot_labels_list_all = []
        prot_name_list = []
        # sort according to length of aa sequence
        dataset.sort(key=lambda x: len(x[1]), reverse=True)
        for prot_name, prot_aa_list, prot_original_label_list, type_id, _cluster_id in dataset:
            prot_name_list.append(prot_name)
            prot_aa_list_all.append(prot_aa_list)
            prot_labels_list_all.append(prot_original_label_list)
            aa_tmp_list_tensor = []
            labels = None
            remapped_labels_crf_hmm = None
            last_non_membrane_position = None
            if prot_original_label_list is not None:
                labels = []
                for topology_label in prot_original_label_list:
                    if topology_label == "L":
                        topology_label = "I"
                    if topology_label == "I":
                        last_non_membrane_position = "I"
                        labels.append(0)
                    elif topology_label == "O":
                        last_non_membrane_position = "O"
                        labels.append(1)
                    elif topology_label == "P":
                        last_non_membrane_position = "P"
                        labels.append(2)
                    elif topology_label == "S":
                        last_non_membrane_position = "S"
                        labels.append(3)
                    elif topology_label == "M":
                        if last_non_membrane_position in ("I", "O", "P"):
                            labels.append(4)
                        else:
                            print("Error: unexpected label found in last_non_membrane_position:",
                                  topology_label, "for", prot_name)
                    elif topology_label == "B":
                        if last_non_membrane_position in ("I", "O", "P"):
                            labels.append(5)
                        else:
                            print("Error: unexpected label found in last_non_membrane_position:",
                                  topology_label, "for", prot_name)
                    else:
                        print("Error: unexpected label found:", topology_label, "for protein",
                              prot_name)
                labels = torch.LongTensor(labels)

                topology = label_list_to_topology(labels)

                remapped_labels_crf_hmm = torch.LongTensor(get_remapped_labels_from_topology(topology, len(labels)))

                # check that protein was properly parsed                
#                 if remapped_labels_crf_hmm.size() != labels.size():
#                     print('Remapped: ', remapped_labels_crf_hmm)
#                     print('Labels: ', labels)
#                     print('FASTA: ', original_labels_to_fasta(labels))
#                     if original_labels_to_fasta(labels) != 'IIIIIIIIIIIIIMMMMMMMMMMMMMMMMMMMMMMMOOOOMOOOOOOOOOOOOOOOOOO':
                assert remapped_labels_crf_hmm.size() == labels.size()

            if use_gpu:
                if labels is not None:
                    labels = labels.cuda()
                remapped_labels_crf_hmm = remapped_labels_crf_hmm.cuda()
            aa_list.append(aa_tmp_list_tensor)
            labels_list.append(labels)
            remapped_labels_list_crf_hmm.append(remapped_labels_crf_hmm)
            prot_type_list.append(type_id)
            prot_topology_list_all.append(label_list_to_topology(labels))
        return TMDataset(aa_list, labels_list, remapped_labels_list_crf_hmm,
                         remapped_labels_list_crf_marg,
                         prot_type_list, prot_topology_list_all, prot_name_list,
                         prot_aa_list_all, prot_labels_list_all)

    def __getitem__(self, index):
        return self.aa_list[index], \
               self.label_list[index], \
               self.remapped_labels_list_crf_hmm[index], \
               None, \
               self.type_list[index], \
               self.topology_list[index], \
               self.prot_name_list[index], \
               self.original_aa_string_list[index], \
               self.original_label_string[index]

    def __len__(self):
        return len(self.aa_list)


def merge_samples_to_minibatch(samples):
    samples_list = []
    for sample in samples:
        samples_list.append(sample)
    # sort according to length of aa sequence
    samples_list.sort(key=lambda x: len(x[7]), reverse=True)
    aa_list, labels_list, remapped_labels_list_crf_hmm, \
    remapped_labels_list_crf_marg, prot_type_list, prot_topology_list, \
    prot_name, original_aa_string, original_label_string = zip(
        *samples_list)
    write_out(prot_type_list)
    return aa_list, labels_list, remapped_labels_list_crf_hmm, remapped_labels_list_crf_marg, \
           prot_type_list, prot_topology_list, prot_name, original_aa_string, original_label_string

def tm_contruct_dataloader_from_disk(tm_dataset, minibatch_size, balance_classes=False):
    if balance_classes:
        batch_sampler = RandomBatchClassBalancedSequentialSampler(tm_dataset, minibatch_size)
    else:
        batch_sampler = RandomBatchSequentialSampler(tm_dataset, minibatch_size)
    return torch.utils.data.DataLoader(tm_dataset,
                                       batch_sampler=batch_sampler,
                                       collate_fn=merge_samples_to_minibatch)


class RandomBatchClassBalancedSequentialSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size):
        self.sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self):
        data_class_map = {}

        for idx in self.sampler:
            class_id = self.dataset[idx][4]
            if class_id not in data_class_map:
                data_class_map[class_id] = []
            data_class_map[class_id].append(idx)

        num_classes = len(data_class_map.keys())
        num_each_class = int(self.batch_size / num_classes)

        class_sizes = []
        for i in range(num_classes):
            class_sizes.append(len(data_class_map[i]))
        max_class_size = max(class_sizes)

        batch_num = int(max_class_size / num_each_class)
        if max_class_size % num_each_class != 0:
            batch_num += 1

        batch_relative_offset = (1.0 / float(batch_num)) / 2.0
        batches = []
        for _ in range(batch_num):
            batch = []
            for _class_id, data_rows in data_class_map.items():
                int_offset = int(batch_relative_offset * len(data_rows))
                batch.extend(sample_at_index(data_rows, int_offset, num_each_class))
            batch_relative_offset += 1.0 / float(batch_num)
            batches.append(batch)

        random.shuffle(batches)

        for batch in batches:
            write_out("Using minibatch from RandomBatchClassBalancedSequentialSampler")
            yield batch

    def __len__(self):
        length = 0
        for _ in self.sampler:
            length += 1
        return length


class RandomBatchSequentialSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size):
        self.sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        data = []
        for idx in self.sampler:
            data.append(idx)

        batch_num = int(len(data) / self.batch_size)
        if len(data) % self.batch_size != 0:
            batch_num += 1

        batch_order = list(range(batch_num))
        random.shuffle(batch_order)

        batch = []
        for batch_id in batch_order:
            write_out("Accessing minibatch #" + str(batch_id))
            for i in range(self.batch_size):
                if i + (batch_id * self.batch_size) < len(data):
                    batch.append(data[i + (batch_id * self.batch_size)])
            yield batch
            batch = []

    def __len__(self):
        length = 0
        for _ in self.sampler:
            length += 1
        return length


def sample_at_index(rows, offset, sample_num):
    assert sample_num < len(rows)
    sample_half = int(sample_num / 2)
    if offset - sample_half <= 0:
        # sample start has to be 0
        samples = rows[:sample_num]
    elif offset + sample_half + (sample_num % 2) > len(rows):
        # sample end has to be an end
        samples = rows[-(sample_num + 1):-1]
    else:
        samples = rows[offset - sample_half:offset + sample_half + (sample_num % 2)]
    assert len(samples) == sample_num
    return samples

def label_list_to_topology(labels):

    if isinstance(labels, list):
        labels = torch.LongTensor(labels)

    if isinstance(labels, torch.Tensor):
        zero_tensor = torch.LongTensor([0])
        if labels.is_cuda:
            zero_tensor = zero_tensor.cuda()

        unique, count = torch.unique_consecutive(labels, return_counts=True)
        top_list = [torch.cat((zero_tensor, labels[0:1]))]
        prev_count = 0
        i = 0
        for _ in unique.split(1):
            if i == 0:
                i += 1
                continue
            prev_count += count[i - 1]
            top_list.append(torch.cat((prev_count.view(1), unique[i].view(1))))
            i += 1
        return top_list



def remapped_labels_hmm_to_orginal_labels(labels):

    if isinstance(labels, list):
        labels = torch.LongTensor(labels)

    if isinstance(labels, (torch.LongTensor, torch.Tensor)):

        torch_ones = torch.ones(labels.size(), dtype=torch.long)

        if labels.is_cuda:
            torch_ones = torch_ones.cuda()

        labels = torch.where((labels == crf_states["I"]), torch_ones * 0, labels)
        labels = torch.where((labels == crf_states["Ob"]) | (labels == crf_states["Om"]), torch_ones * 1, labels)
        labels = torch.where((labels == crf_states["Pb"]), torch_ones * 2, labels)
        labels = torch.where((labels >= crf_states["S0"]) & (labels <= crf_states["S"+str(max_signal_length-1)]), torch_ones * 3, labels)
        labels = torch.where((labels >= crf_states["Mio0"]) & (labels <= crf_states["Moi"+str(max_alpha_membrane_length-1)]), torch_ones * 4, labels)
        labels = torch.where((labels >= crf_states["Bpo0"]) & (labels <= crf_states["Bop"+str(max_beta_membrane_length-1)]), torch_ones * 5, labels)

        return labels

def batch_sizes_to_mask(batch_sizes: torch.Tensor) -> torch.Tensor:
    arange = torch.arange(batch_sizes[0], dtype=torch.int32)
    if batch_sizes.is_cuda:
        arange = arange.cuda()
    res = (arange.expand(batch_sizes.size(0), batch_sizes[0])
           < batch_sizes.unsqueeze(1)).transpose(0, 1)
    return res

def original_labels_to_fasta(label_tensor):
    sequence_topology = []
    label_map = {
        0 : "I",
        1 : "O",
        2 : "P",
        3 : "S",
        4 : "M",
        5 : "B"
    }
    for label in label_tensor.tolist():
        sequence_topology.append(label_map[label])
    return "".join(sequence_topology)

def get_predicted_type_from_labels(labels):
    torch_zero = torch.zeros(1, dtype=torch.int64)

    if labels.is_cuda:
        torch_zero = torch_zero.cuda()

    # beta
    if (labels == 5).int().sum() > 0:
        return torch_zero + 4

    # tm
    is_tm = False
    if (labels == 4).int().sum() > 0:
        is_tm = True

    # sp
    if (labels == 3).int().sum() > 0:
        if is_tm:
            return torch_zero + 1
        else:
            return torch_zero + 2

    if is_tm:
        return torch_zero + 0
    else:
        return torch_zero + 3


def is_topologies_equal(topology_a, topology_b, minimum_seqment_overlap=5):

    if isinstance(topology_a[0], torch.Tensor):
        topology_a = list([a.cpu().numpy() for a in topology_a])
    if isinstance(topology_b[0], torch.Tensor):
        topology_b = list([b.cpu().numpy() for b in topology_b])
    if len(topology_a) != len(topology_b):
        return False
    for idx, (_position_a, label_a) in enumerate(topology_a):
        if label_a != topology_b[idx][1]:
            if (label_a in (1,2) and topology_b[idx][1] in (1,2)): # assume O == P
                continue
            else:
                return False
        if label_a in (3, 4, 5):
            overlap_segment_start = max(topology_a[idx][0], topology_b[idx][0])
            overlap_segment_end = min(topology_a[idx + 1][0], topology_b[idx + 1][0])
            if label_a == 5:
                # Set minimum segment overlap to 3 for Beta regions
                minimum_seqment_overlap = 3
            if overlap_segment_end - overlap_segment_start < minimum_seqment_overlap:
                return False
    return True


def parse_3line_format(lines):
    i = 0
    prot_list = []
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        prot_name_comment = lines[i]
        type_string = None
        cluster_id = None
        if prot_name_comment.__contains__(">"):
            i += 1
            splitted = prot_name_comment.split("|")
            prot_name = splitted[0].split(">")[1]
            if len(splitted) > 1:
                type_string = splitted[-1]
            #if len(splitted) > 2:
            #    cluster_id = int(splitted[2])
        else:
            # assume this is data
            prot_name = "> Unknown Protein Name"
        prot_aa_list = lines[i].upper()
        i += 1
        if len(prot_aa_list) > 6000:
            print("Discarding protein", prot_name, type_string, "as length larger than 1023:",
                  len(prot_aa_list))
            if i < len(lines) and not lines[i].__contains__(">"):
                i += 1
        else:
            if i < len(lines) and not lines[i].__contains__(">"):
                prot_topology_list = lines[i].upper()
                i += 1
                if prot_topology_list.__contains__("B"):
                    type_id = 4
                    assert type_string == "BETA"
                elif prot_topology_list.__contains__("S"):
                    if prot_topology_list.__contains__("M"):
                        type_id = 1
                        assert type_string == "SP+TM"
                    else:
                        type_id = 2
                        assert type_string == "SIGNAL"
                else:
                    if prot_topology_list.__contains__("M"):
                        type_id = 0
                        if type_string and not type_string == "TM":
                            print("NOT TM", prot_topology_list)
                            assert False
                    else:
                        type_id = 3
                        if type_string and not type_string == "GLOBULAR":
                            print("NOT GLOB", prot_topology_list)
                            print(lines[i-1])
                            assert False
            else:
                type_id = None
                prot_topology_list = None
            prot_list.append((prot_name, prot_aa_list, prot_topology_list,
                              type_id, cluster_id))

    # shuffle input data
    random.Random(2021).shuffle(prot_list)
    return prot_list


def parse_datafile_from_disk(file):
    lines = list([line.strip() for line in open(file)])
    return parse_3line_format(lines)


def calculate_partitions(partitions_count, cluster_partitions, types):
    partition_distribution = torch.ones((partitions_count,
                                         len(torch.unique(types))),
                                        dtype=torch.long)

    if all(v is None for v in cluster_partitions):
        cluster_partitions = torch.arange(types.shape[0]).long()
    else:
        cluster_partitions = torch.LongTensor(cluster_partitions)
    partition_assignments = torch.zeros(cluster_partitions.shape[0],
                                        dtype=torch.long)

    for i in torch.unique(cluster_partitions):
        cluster_positions = (cluster_partitions == i).nonzero()
        cluster_types = types[cluster_positions]
        unique_types_in_cluster, type_count = torch.unique(cluster_types, return_counts=True)
        tmp_distribution = partition_distribution.clone()
        tmp_distribution[:, unique_types_in_cluster] += type_count
        relative_distribution = partition_distribution.double() / tmp_distribution.double()
        min_relative_distribution_group = torch.argmin(torch.sum(relative_distribution, dim=1))
        partition_distribution[min_relative_distribution_group,
                               unique_types_in_cluster] += type_count
        partition_assignments[cluster_positions] = min_relative_distribution_group

    write_out("Loaded data into the following partitions")
    write_out("[[  TM  SP+TM  SP Glob]")
    write_out(partition_distribution - torch.ones(partition_distribution.shape,
                                                  dtype=torch.long))
    return partition_assignments


def load_data_from_disk(filename, partition_rotation=0):
    print("Loading data from disk...")
    data = parse_datafile_from_disk(filename)
    data_unzipped = list(zip(*data))
    partitions = calculate_partitions(
        cluster_partitions=np.array(data_unzipped[4]),
        types=torch.LongTensor(np.array(data_unzipped[3])),
        partitions_count=5)
    train_set = []
    val_set = []
    test_set = []
    for idx, sample in enumerate(data):
        partition = int(partitions[idx])  # in range 0-4
        rotated = (partition + partition_rotation) % 5
        if int(rotated) <= 2:
            train_set.append(sample)
        elif int(rotated) == 3:
            val_set.append(sample)
        else:
            test_set.append(sample)

    print("Data splited as:",
          len(train_set), "train set",
          len(val_set), "validation set",
          len(test_set), "test set")
    return train_set, val_set, test_set


def normalize_confusion_matrix(confusion_matrix):
    confusion_matrix = confusion_matrix.astype(np.float64)
    for i in range(4):
        accumulator = int(confusion_matrix[i].sum())
        if accumulator != 0:
            confusion_matrix[4][i] /= accumulator * 0.01  # 0.01 to convert to percentage
        for k in range(5):
            if accumulator != 0:
                confusion_matrix[i][k] /= accumulator * 0.01  # 0.01 to convert to percentage
            else:
                confusion_matrix[i][k] = math.nan
    return confusion_matrix.round(2)

def decode(emissions, mask, crf_model):

    labels_predicted = []
    for l in crf_model.decode(emissions, mask=mask):
        val = torch.tensor(l)
        if emissions.is_cuda:
            val = val.cuda()
        labels_predicted.append(val)


    predicted_labels = []
    for l in labels_predicted:
        predicted_labels.append(remapped_labels_hmm_to_orginal_labels(l))

    predicted_types_list = []
    for p_label in predicted_labels:
        predicted_types_list.append(get_predicted_type_from_labels(p_label))
    predicted_types = torch.cat(predicted_types_list)



    if emissions.is_cuda:
        predicted_types = predicted_types.cuda()

    # if all O's, change to all I's (by convention)
    torch_zero = torch.zeros(1, dtype=torch.long)
    if emissions.is_cuda:
        torch_zero = torch_zero.cuda()
    for idx, labels in enumerate(predicted_labels):
        predicted_labels[idx] = \
            labels - torch.where(torch.eq(labels, 4).min() == 1, torch_zero + 1, torch_zero)

    return predicted_labels, predicted_types, list(map(label_list_to_topology, predicted_labels)), labels_predicted


def batch_sizes_to_mask_numpy(seq_length, batch_sizes):
    l = []
    for i in batch_sizes:
        l.append(np.array([1] * i + [0] * (seq_length - i)))
    mask = np.array(l).T
    return mask

def decode_numpy(emissions, batch_sizes, start_transitions, transitions, end_transitions):
    mask = batch_sizes_to_mask_numpy(emissions.shape[0], batch_sizes)

    labels_predicted = []
    for l in numpy_viterbi_decode(emissions,
                                  mask=mask,
                                  start_transitions=start_transitions,
                                  transitions=transitions,
                                  end_transitions=end_transitions):
        val = np.expand_dims(np.array(l), 1)
        labels_predicted.append(val)


    predicted_labels = []
    for l in labels_predicted:
        predicted_labels.append(remapped_labels_hmm_to_orginal_labels(l))

    predicted_types_list = []
    for p_label in predicted_labels:
        predicted_types_list.append(get_predicted_type_from_labels(p_label))
    predicted_types = np.array(predicted_types_list).squeeze(axis=1)

    # if all O's, change to all I's (by convention)
    zero = np.zeros(1, dtype=np.long)

    for idx, labels in enumerate(predicted_labels):
        predicted_labels[idx] = \
            labels - np.where((labels == 4).min() == 1, zero + 1, zero)



    return predicted_labels, \
           predicted_types, \
           list(map(label_list_to_topology, predicted_labels))

def compute_marginal_probabilities(emissions,
                                   batch_sizes,
                                   start_transitions,
                                   transitions,
                                   end_transitions):
    mask = batch_sizes_to_mask_numpy(emissions.shape[0], batch_sizes)

    alpha = compute_log_alpha_numpy(emissions, mask=mask,
                                    start_transitions=start_transitions,
                                    transitions=transitions,
                                    end_transitions=end_transitions, run_backwards=False)
    beta = compute_log_alpha_numpy(emissions, mask=mask,
                                   start_transitions=start_transitions,
                                   transitions=transitions,
                                   end_transitions=end_transitions, run_backwards=True)
    z = log_sum_exp(alpha[alpha.shape[0]-1] + end_transitions, dim=1)
    prob = alpha + beta - z.reshape(1, -1, 1)
    marg_prob = np.exp(prob) # seq length, batch size, labels
    return marg_prob

def numpy_viterbi_decode(emissions,
                         mask,
                         start_transitions,
                         transitions,
                         end_transitions) -> List[List[int]]:
    # emissions: (seq_length, batch_size, num_tags)
    # mask: (seq_length, batch_size)
    assert len(emissions.shape) == 3
    #assert emissions.shape[:2] == mask.shape
    #assert emissions.size(2) == self.num_tags
    #assert mask[0].all()

    seq_length = emissions.shape[0]
    batch_size = emissions.shape[1]

    # Start transition and first emission
    # shape: (batch_size, num_tags)
    score = start_transitions + emissions[0]
    history = []

    # score is a tensor of size (batch_size, num_tags) where for every batch,
    # value at column j stores the score of the best tag sequence so far that ends
    # with tag j
    # history saves where the best tags candidate transitioned from; this is used
    # when we trace back the best tag sequence

    # Viterbi algorithm recursive case: we compute the score of the best tag sequence
    # for every possible next tag




    for i in range(1, seq_length):

        # Broadcast viterbi score for every possible next tag
        # shape: (batch_size, num_tags, 1)
        broadcast_score = np.expand_dims(score, 2)

        # Broadcast emission score for every possible current tag
        # shape: (batch_size, 1, num_tags)
        broadcast_emission = np.expand_dims(emissions[i], 1)

        # Compute the score tensor of size (batch_size, num_tags, num_tags) where
        # for each sample, entry at row i and column j stores the score of the best
        # tag sequence so far that ends with transitioning from tag i to tag j and emitting
        # shape: (batch_size, num_tags, num_tags)
        next_score = broadcast_score + transitions + broadcast_emission

        # Find the maximum score over all possible current tag
        # shape: (batch_size, num_tags)
        indices = next_score.argmax(axis=1)
        next_score = next_score.max(axis=1)


        # Set score to the next score if this timestep is valid (mask == 1)
        # and save the index that produces the next score
        # shape: (batch_size, num_tags)
        score = np.where(np.expand_dims(mask[i], 1), next_score, score) # pylint: disable=E1136
        history.append(indices)


    # End transition score
    # shape: (batch_size, num_tags)
    score += end_transitions

    # Now, compute the best path for each sample

    # shape: (batch_size,)
    seq_ends = np.sum(mask, axis=0) - 1
    best_tags_list = []

    for idx in range(batch_size):
        # Find the tag which maximizes the score at the last timestep; this is our best tag
        # for the last timestep
        best_last_tag = score[idx].argmax(axis=0)
        best_tags = [best_last_tag.item()]

        # We trace back where the best last tag comes from, append that to our best tag
        # sequence, and trace it back again, and so on
        for hist in reversed(history[:seq_ends[idx]]):
            best_last_tag = hist[idx][best_tags[-1]]
            best_tags.append(best_last_tag.item())

        # Reverse the order because we start from the last timestep
        best_tags.reverse()
        best_tags_list.append(best_tags)

    return best_tags_list


def compute_log_alpha_numpy(emissions, mask,
                            start_transitions,
                            transitions,
                            end_transitions,
                            run_backwards: bool):
    # emissions: (seq_length, batch_size, num_tags)
    # mask: (seq_length, batch_size)
    assert len(emissions.shape) == 3 and len(mask.shape) == 2
    assert emissions.shape[:2] == mask.shape
    assert all(mask[0].data)

    seq_length = emissions.shape[0]
    mask = mask.astype(float)
    broadcast_transitions = np.expand_dims(transitions, axis=0)  # (1, num_tags, num_tags)
    emissions_broadcast = np.expand_dims(emissions, axis=2)
    seq_iterator = range(1, seq_length)

    if run_backwards:
        # running backwards, so transpose
        broadcast_transitions = broadcast_transitions.transpose((0, 2, 1)) # (1, num_tags, num_tags)
        emissions_broadcast = emissions_broadcast.transpose((0, 1, 3, 2))

        # the starting probability is end_transitions if running backwards
        log_prob = [end_transitions.reshape(1, -1).repeat(emissions.shape[1], axis=0)]

        # iterate over the sequence backwards
        seq_iterator = reversed(seq_iterator)
    else:
        # Start transition score and first emission
        log_prob = [emissions[0] + start_transitions.reshape(1, -1)]

    for i in seq_iterator:
        # Broadcast log_prob over all possible next tags
        broadcast_log_prob = np.expand_dims(log_prob[-1], axis=2)  # (batch_size, num_tags, 1)
        # Sum current log probability, transition, and emission scores
        # (batch_size, num_tags, num_tags)
        score = broadcast_log_prob + broadcast_transitions + emissions_broadcast[i]
        # Sum over all possible current tags, but we're in log prob space, so a sum
        # becomes a log-sum-exp
        score = log_sum_exp(score, dim=1)
        # Set log_prob to the score if this timestep is valid (mask == 1), otherwise
        # copy the prior value
        log_prob.append(score * np.expand_dims(mask[i], axis=1) +
                        log_prob[-1] * np.expand_dims(1.-mask[i], axis=1))

    if run_backwards:
        log_prob.reverse()

    return np.array(log_prob)

def initialize_crf_parameters(crf_model,
                              start_transitions=None,
                              end_transitions=None,
                              transitions=None) -> None:
    """Initialize the transition parameters.

    The parameters will be initialized randomly from a uniform distribution
    between -0.1 and 0.1, unless given explicitly as an argument.
    """
    if start_transitions is None:
        torch.nn.init.uniform(crf_model.start_transitions, -0.1, 0.1)
    else:
        crf_model.start_transitions.data = start_transitions
    if end_transitions is None:
        torch.nn.init.uniform(crf_model.end_transitions, -0.1, 0.1)
    else:
        crf_model.end_transitions.data = end_transitions
    if transitions is None:
        torch.nn.init.uniform(crf_model.transitions, -0.1, 0.1)
    else:
        crf_model.transitions.data = transitions

def log_sum_exp(tensor, dim):
    # Find the max value along `dim`
    offset = tensor.max(axis=dim)
    # Make offset broadcastable
    broadcast_offset = np.expand_dims(offset, axis=dim)
    # Perform log-sum-exp safely
    safe_log_sum_exp = np.log(np.sum(np.exp(tensor - broadcast_offset), axis=dim))
    # Add offset back
    return offset + safe_log_sum_exp

def hash_aa_string(string):
    return md5(string.encode()).digest().hex()

def generate_esm_embeddings(dataset, esm_embeddings_dir, repr_layers=33):
    esm_model, esm_alphabet = pretrained.load_model_and_alphabet('data/raw/esm1b_model.pt')
    
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.set_device(1)
            esm_model = esm_model.cuda()

        batch_converter = esm_alphabet.get_batch_converter()
        
        print("Starting to generate embeddings")

        for split_idx, data_split in enumerate(dataset):
            print(f"Generating embeddings for data split #{split_idx}")
            
            for idx, seq in enumerate(data_split):
                
                _, _, remapped_labels_list_crf_hmm, _, prot_type_list, prot_topology_list, \
                            prot_name_list, original_aa_string, original_label_string = seq
                
                if os.path.isfile(f'{esm_embeddings_dir}/{hash_aa_string(original_aa_string)}'):
                    print("Already processed sequence")
                    continue
                                  
                print(f"Processing sequence nr. {idx} of split #{split_idx}")

                print(f"Sequence length: {len(original_aa_string)}")
                
                seqs = list([("seq", s) for s in [original_aa_string]])
                labels, strs, toks = batch_converter(seqs)
                #assert all(
                #    -(self.esm_model.num_layers + 1) <= i <= self.esm_model.num_layers for i in range(repr_layers)
                #)
                repr_layers_list = [
                    (i + esm_model.num_layers + 1) % (esm_model.num_layers + 1) for i in range(repr_layers)
                ]

                out = None

                write_out(
                    f"Processing ({toks.size(0)} sequences)"
                )
                if torch.cuda.is_available():
                    toks = toks.to(device="cuda", non_blocking=True)

                minibatch_max_length = toks.size(1)

                tokens_list = []
                end = 0
                while end <= minibatch_max_length:
                    start = end
                    end = start + 1022
                    if end <= minibatch_max_length:
                        # we are not on the last one, so make this shorter
                        end = end - 300
                    tokens = esm_model(toks[:, start:end], repr_layers=repr_layers_list, return_contacts=False)["representations"][repr_layers - 1]
                    tokens_list.append(tokens)

                out = torch.cat(tokens_list, dim=1)

                # set nan to zeros
                out[out!=out] = 0.0

                res = out.transpose(0,1)[1:-1] 
                seq_embedding = res[:,0]
                print(seq_embedding.size())

                output_file = open(f'{esm_embeddings_dir}/{hash_aa_string(original_aa_string)}', 'wb')
                torch.save(seq_embedding, output_file)
                output_file.close()

                print(f"Saved embedding to {esm_embeddings_dir}")


initial_experiment_data_json = {
    'validation': {
        '0': {},
        '1': {},
        '2': {},
        '3': {},
        '4': {},
    },
    'test': {
        '0': {},
        '1': {},
        '2': {},
        '3': {},
        '4': {},
    }
}


def write_experiment_validation_data(cv, accuracy, model_path, confusion_matrix, best_minibatch_time, experiment_tag):
    experiment_file_path = "output/experiments/" + experiment_tag + '/experiment.json'
    experiment_json = json.loads(open(experiment_file_path, 'r').read())
    experiment_json['validation'][cv] = {
        'accuracy': accuracy,
        'path': model_path,
        'confusion_matrix': confusion_matrix,
        'best_minibatch_time': best_minibatch_time
    }
    open(experiment_file_path, 'w').write(json.dumps(experiment_json))
    

def test_experiment_and_write_results_to_file(cv, experiment_file_path, test_loader):
    experiment_json = json.loads(open(experiment_file_path, 'r').read())

    model = load_model_from_disk(experiment_json['validation'][cv]['path'], force_cpu=False)    
    confusion_matrix = torch.zeros((6, 6), dtype=torch.int64)
    protein_names = []
    protein_aa_strings = []
    protein_label_actual = []
    protein_label_prediction = []

    with torch.no_grad():
        for _, minibatch in enumerate(test_loader, 0):
            _, _, remapped_labels_list_crf_hmm, _, prot_type_list, prot_topology_list, \
            prot_name_list, original_aa_strings, original_label_string = minibatch
            actual_labels = torch.nn.utils.rnn.pad_sequence([l for l in remapped_labels_list_crf_hmm])
            
            protein_names.extend(prot_name_list)
            protein_aa_strings.extend(original_aa_strings)
            protein_label_actual.extend(original_label_string)
                    
            # Make prediction with models
            predicted_labels, predicted_types, predicted_topologies, _ = predict.make_prediction(
                batch=original_aa_strings,
                model=model,
            )
            
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
    
    type_correct_ratio = \
    calculate_acc(confusion_matrix[0][0] + confusion_matrix[0][5], confusion_matrix[0].sum()) + \
    calculate_acc(confusion_matrix[1][1] + confusion_matrix[1][5], confusion_matrix[1].sum()) + \
    calculate_acc(confusion_matrix[2][2] + confusion_matrix[2][5], confusion_matrix[2].sum()) + \
    calculate_acc(confusion_matrix[3][3] + confusion_matrix[3][5], confusion_matrix[3].sum()) + \
    calculate_acc(confusion_matrix[4][4] + confusion_matrix[4][5], confusion_matrix[4].sum())
    type_accuracy = float((type_correct_ratio / 5).detach())

    tm_accuracy = float(calculate_acc(confusion_matrix[0][5], confusion_matrix[0].sum()).detach())
    sptm_accuracy = float(calculate_acc(confusion_matrix[1][5], confusion_matrix[1].sum()).detach())
    sp_accuracy = float(calculate_acc(confusion_matrix[2][5], confusion_matrix[2].sum()).detach())
    glob_accuracy = float(calculate_acc(confusion_matrix[3][5], confusion_matrix[3].sum()).detach())
    beta_accuracy = float(calculate_acc(confusion_matrix[4][5], confusion_matrix[4].sum()).detach())
    
    tm_type_acc = float(calculate_acc(confusion_matrix[0][0] + confusion_matrix[0][5], confusion_matrix[0].sum()).detach())
    tm_sp_type_acc = float(calculate_acc(confusion_matrix[1][1] + confusion_matrix[1][5], confusion_matrix[1].sum()).detach())
    sp_type_acc = float(calculate_acc(confusion_matrix[2][2] + confusion_matrix[2][5], confusion_matrix[2].sum()).detach())
    glob_type_acc = float(calculate_acc(confusion_matrix[3][3] + confusion_matrix[3][5], confusion_matrix[3].sum()).detach())
    beta_type_acc = float(calculate_acc(confusion_matrix[4][4] + confusion_matrix[4][5], confusion_matrix[4].sum()).detach())
    
    experiment_json['test'][cv]['confusion_matrix'] = confusion_matrix.tolist()
    experiment_json['test'][cv].update({
        'type': type_accuracy
    })
    
    # Topology 
    experiment_json['test'][cv].update({
        'tm': {
            'type': tm_type_acc,
            'topology': tm_accuracy
        }
    })
    
    experiment_json['test'][cv].update({
        'sptm': {
            'type': tm_sp_type_acc,
            'topology': sptm_accuracy
        }
    })
    
    experiment_json['test'][cv].update({
        'sp': {
            'type': sp_type_acc,
            'topology': sp_accuracy
        }
    })
    
    experiment_json['test'][cv].update({
        'glob': {
            'type': glob_type_acc,
            'topology': glob_accuracy
        }
    })
    
    experiment_json['test'][cv].update({
        'beta': {
            'type': beta_type_acc,
            'topology': beta_accuracy
        }
    })
    
    open(experiment_file_path, 'w').write(json.dumps(experiment_json))
    return (protein_names, protein_aa_strings, protein_label_actual, protein_label_prediction) 


def calculate_acc(correct, total):
    total = total.float()
    correct = correct.float()
    if total == 0.0:
        return 1
    return correct / total
