#! /usr/bin/env Python


'''
Translate a sequence of proteins into the corresponding Dayhoff encoding.

https://en.wikipedia.org/wiki/Margaret_Oakley_Dayhoff
'''


import argparse

import screed
from tqdm import tqdm


'''
The "default" Dayhoff encoding does not cover some of the letters in the 
"Extended IUPAC" protein encoding, see e.g.

https://biopython.org/DIST/docs/api/Bio.Alphabet.IUPAC.ExtendedIUPACProtein-class.html

> Extended uppercase IUPAC protein single letter alphabet including X etc.
In addition to the standard 20 single letter protein codes, this includes:

[ ] B = "Asx"; Aspartic acid (R) or Asparagine (N)
[ ] X = "Xxx"; Unknown or 'other' amino acid
[x] Z = "Glx"; Glutamic acid (E) or Glutamine (Q)
[x] J = "Xle"; Leucine (L) or Isoleucine (I), used in mass-spec (NMR)
[x] U = "Sec"; Seleno__cysteine__
[x] O = "Pyl"; Pyrro__lysine__

We simply add them, where they map to a unique Dayhoff character ([x]):
'''
dayhoff = {
    'C' + 'U': 'a', 
    'GSTAP': 'b',
    'DENQ' + 'Z': 'c',
    'RHK' + 'O': 'd',
    'LVMI' + 'J': 'e',
    'YFW': 'f',
    '*': '*',
    }


lu = {}  # lookup
for k, v in dayhoff.items():
    for i in k:
        lu[i] = v


def encode_dayhoff(letter, dayhoff):
    return ''.join([dayhoff.get(i, 'x') for i in letter])



'''
With this scheme, i.e. only excluding proteins that have letters X and B in
them, we can go through > 99.5 % of the Uniprot SPROT database (n=562,015):

fp = '/Users/phi/tmp/nighthoff/uniprot_sprot.fasta.gz'
translated, nottranslated = 0, 0

with screed.open(fp) as seqfile:
    for read in tqdm(seqfile):
        try:    
            seq = ''.join([encode(i, dayhoff) for i in read.sequence])
            translated += 1
        except IndexError:
            # E.g. "X" in sequence
            nottranslated += 1
            if not any([i in 'BX' for i in read.sequence]):
                print(read)
            continue
translated / (translated + nottranslated)
'''



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', help='Protein fasta')
parser.add_argument('-o', help='Dayhoff fasta')
parser.add_argument('--length', help='Truncate to this', default=None, type=int)
parser.add_argument('--skip-stop', help='Exclude stop codon *')
parser.add_argument('--skip-header', help='Skip fasta header', action='store_true')

args = parser.parse_args()



with screed.open(args.i) as seqfile, open(args.o, 'w+') as out:
    for read in tqdm(seqfile):
        try:    
            seq = encode_dayhoff(read.sequence, lu)
            if args.skip_stop:
                seq = ''.join([i for i in seq if i != '*'])
            if args.length:
                seq = seq[:args.length]
            if args.skip_header:
                out.write(f'{seq}\n')
            else:
                out.write(f'>{read.name}\n{seq}\n')
        except IndexError:
            continue




