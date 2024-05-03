import sys
sys.path.append('.')
import torch
import numpy as np
from Bio import SeqIO
import os, argparse, math, random, csv
from tqdm import tqdm
from scripts.train_mlp import get_ec2occurance

def mutate(seq: str, position: int) -> str:
    seql = seq[ : position]
    seqr = seq[position+1 : ]
    seq = seql + '*' + seqr
    return seq

def mask_sequences(single_id, pid2seq, save_path) :
    single_id = set(single_id)
    outputs = []
    for pid in single_id:
        for j in range(10):
            seq = pid2seq[pid]
            mu, sigma = .10, .02
            s = np.random.normal(mu, sigma, 1)
            mut_rate = s[0]
            times = math.ceil(len(seq) * mut_rate)
            for k in range(times):
                position = random.randint(1 , len(seq) - 1)
                seq = mutate(seq, position)
            seq = seq.replace('*', '<mask>')
            outputs.append(f'>{pid}_{j}\n{seq}\n')
    with open(save_path, 'w') as f:
        f.writelines(outputs)
    print(f'Saved {len(outputs)} masked sequences to {save_path}')

def get_pid2seq(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    pid2seq = {record.id.split('|')[1]: str(record.seq) for record in records}
    
    return pid2seq

def get_args():
    parser = argparse.ArgumentParser(description='Mutate single ECs')
    parser.add_argument('--train_file', type=str, default='data/ec/ensemble/sprot_10_1022_esm1b_t33_ec_above_10_single_label_train_val.pt')
    parser.add_argument('--data_file', type=str, default='data/ec/ensemble/sprot_10_1022_esm1b_t33_ec_above_10_single_label_train_val.pt')
    parser.add_argument('--label_file', type=str, default='data/ec/swissprot_ec_list_above_10.json')
    parser.add_argument('--fasta_file', type=str, default='data/swissprot/uniprot_sprot_10_1022.fasta')
    parser.add_argument('--output', type=str)
    
    
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    train_data = torch.load(args.train_file)
    ec2occurance, _ = get_ec2occurance(args.data_file, args.label_file, 'ec', 4)
    pid2seq = get_pid2seq(args.fasta_file)
    single_pids = set()
    for pid in tqdm(train_data):
        if ec2occurance[train_data[pid]['ec'][0]] == 1:
            single_pids.add(pid)
    single_ecs = set()
    for ec, occurance in ec2occurance.items():
        if occurance == 1:
            single_ecs.add(ec)
    print(f'Number of single ECs: {len(single_ecs)}')
    print(f'Number of single-EC sequences: {len(single_pids)}')
    mask_sequences(single_pids, pid2seq, args.output)
    
    
if __name__ == '__main__':
    main()



