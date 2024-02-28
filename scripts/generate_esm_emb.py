import sys
sys.path.append('.')
import torch
import numpy as np
import esm
import os, argparse, logging
from Bio import SeqIO
from tqdm import tqdm

class ESMEmbedder:
    def __init__(self, model, device, logger=None) -> None:
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model)
        self.num_layers = int(model.split('_')[1][1:])
        self.device = device
        self.logger = logger if logger else logging.getLogger(__name__)
        self.model.eval()
        self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    def embed(self, fasta_file, output_file=None, batch_size=32, start=None, end=None):
        records = list(SeqIO.parse(fasta_file, "fasta"))
        if start is not None and end is not None:
            records = records[start: end]
        self.logger.info(f'Processing {len(records)} sequences from {fasta_file}: {start} to {end}')
        sequences = [str(record.seq) for record in records]
        pids = [record.id for record in records]
        num_batches = int(np.ceil(len(sequences) / batch_size))
        pid2embeddings = {}
        for i in tqdm(range(num_batches), desc='embedding', dynamic_ncols=True):
            try:
                batch_seqs = sequences[i*batch_size: (i+1)*batch_size]
                batch_pids = pids[i*batch_size: (i+1)*batch_size]
                batch = [(pid, seq) for pid, seq in zip(batch_pids, batch_seqs)]
                batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
                batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
                with torch.no_grad():
                    results = self.model(batch_tokens.to(self.device), repr_layers=[self.num_layers])
                token_representations = results["representations"][self.num_layers]
                for i, tokens_len in enumerate(batch_lens):
                    seq_repr = token_representations[i, 1 : tokens_len - 1].mean(0)
                    pid2embeddings[batch_pids[i]] = seq_repr.cpu()
            except:
                self.logger.error(f'Error in batch {i}')
                continue
        if output_file:
            torch.save(pid2embeddings, output_file)
            
        return pid2embeddings
    
def get_args():
    parser = argparse.ArgumentParser(description='Generate ESM embeddings')
    
    parser.add_argument('--input', type=str, required=True, help='Input fasta file')
    parser.add_argument('--output', type=str, required=True, help='Output file')
    parser.add_argument('--model', type=str, default='esm2_t33_650M_UR50D', help='ESM model')
    parser.add_argument('--batch_size', type=int, default=80, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--start', type=int, default=None, help='Start index')
    parser.add_argument('--end', type=int, default=None, help='End index')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    embedder = ESMEmbedder(args.model, args.device)
    embedder.embed(args.input, args.output, args.batch_size, args.start, args.end)
    
    
if __name__ == '__main__':
    main()