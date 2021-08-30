import screed


def translate(nt):
    '''
    Translate a nucleotide sequence into protein. Don't stop at stop codons.
    '''
    codons = {
        'TTT':'F', 'TTC':'F', 'TTA':'L', 'TTG':'L',
        'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S',
        'TAT':'Y', 'TAC':'Y', 'TAA':'*', 'TAG':'*',
        'TGT':'C', 'TGC':'C', 'TGA':'*', 'TGG':'W',
        'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L',
        'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P',
        'CAT':'H', 'CAC':'H', 'CAA':'Q', 'CAG':'Q',
        'CGT':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R',
        'ATT':'I', 'ATC':'I', 'ATA':'I', 'ATG':'M',
        'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
        'AAT':'N', 'AAC':'N', 'AAA':'K', 'AAG':'K',
        'AGT':'S', 'AGC':'S', 'AGA':'R', 'AGG':'R',
        'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V',
        'GCT':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
        'GAT':'D', 'GAC':'D', 'GAA':'E', 'GAG':'E',
        'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G',
        }

    result = []
    for i in range(0, len(nt), 3):
        try:
            codon = codons[nt[i:i+3]]
            result.append(codon)
        except KeyError:
            pass
    return ''.join(result)


def dayhoff(seq):
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
    alphabet = {
        'C' + 'U': 'a', 
        'GSTAP': 'b',
        'DENQ' + 'Z': 'c',
        'RHK' + 'O': 'd',
        'LVMI' + 'J': 'e',
        'YFW': 'f',
        '*': '*'}
    '''
    With this scheme, i.e. only excluding proteins that have letters X and B in
    them, we can go through > 99.5 % of the Uniprot SPROT database (n=562,015):
    '''
    translation = ''
    for letter in seq:
        for k in alphabet.keys():
            if letter in k:
                translation += alphabet[k]
    assert len(translation) == len(seq)
    return translation


def frames(seq):
    '''
    6-frame translate sequence
    '''
    fwd1 = seq
    fwd2 = seq[1:]
    fwd3 = seq[2:]

    rc = screed.rc(seq)
    rev1 = rc
    rev2 = rc[1:]
    rev3 = rc[2:]

    return (fwd1, fwd2, fwd3, rev1, rev2, rev3)


def windows(iterable, length=2, overlap=0, truncate=False):
    '''
    Returns a generator of windows of <length> and with an <overlap>.

    Shamelessly stolen from: Python cookbook 2nd edition, chapter 19

    Usage:

    list(windows(seq, 100, 50, False))
    '''
    import itertools

    it = iter(iterable)
    results = list(itertools.islice(it, length))
    
    while len(results) == length:
        yield ''.join(results)
        results = results[length-overlap:]
        results.extend(itertools.islice(it, length-overlap))
    
    if truncate:
        if results and len(results) == length:
            yield ''.join(results)
    else:
        if results:
            yield ''.join(results)