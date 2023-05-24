from fairseq.data import Dictionary

ALPHABET_FOR_INIT = '#SGTYVALRDKQWINFEPCMH'

class AbgenDictionary(Dictionary):
    def __init__(self):
        self.symbols = [
            '#', 'S', 'G', 'T', 'Y', 'V', 'A', 'L', 'R', 'D', 'K', 'Q', 'W', 'I', 'N', 'F', 'E', 'P', 'C', 'M', 'H'
        ]
        self.count = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.indices = {
            '#': 0,  'S': 1, 'G': 2, 'T': 3, 'Y': 4, 'V': 5, 'A': 6, 'L': 7, 'R': 8, 'D': 9, 'K': 10, 
            'Q': 11, 'W': 12, 'I': 13, 'N': 14, 'F': 15, 'E': 16, 'P': 17, 'C': 18, 'M': 19, 'H': 20
        }
        self.nspecial = len(self.symbols) 


class AbbertDictionary(Dictionary):
    def __init__(self):
        bos="<s>"
        pad="<pad>"
        eos="</s>"
        unk="<unk>"
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = [
            '<s>', '<pad>', '</s>', '<unk>', 'S', 'G', 'T', 'Y', 'V', 'A', 'L', 'R', 'D', 'K', 'Q', 'W', 'I', 'N', 'F', 'E', 'P', 'C', 'M', 'H', 
            'X', '*', 'madeupword0000', 'madeupword0001', 'madeupword0002', 'madeupword0003', 'madeupword0004', 'madeupword0005'
        ]
        self.count = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.indices = {
            '<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, 'S': 4, 'G': 5, 'T': 6, 'Y': 7, 'V': 8, 'A': 9, 'L': 10, 'R': 11, 'D': 12, 
            'K': 13, 'Q': 14, 'W': 15, 'I': 16, 'N': 17, 'F': 18, 'E': 19, 'P': 20, 'C': 21, 'M': 22, 'H': 23, 'X': 24, '*': 25, 
            'madeupword0000': 26, 'madeupword0001': 27, 'madeupword0002': 28, 'madeupword0003': 29, 'madeupword0004': 30, 'madeupword0005': 31}
        self.bos_index = self.indices[bos]
        self.pad_index = self.indices[pad]
        self.eos_index = self.indices[eos]
        self.unk_index = self.indices[unk]

        self.nspecial = len(self.symbols) 

class TagDictionary(Dictionary):
    def __init__(self):
        bos="<s>"
        pad="<pad>"
        eos="</s>"
        unk="<unk>"
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = [
            '<s>', '<pad>', '</s>', '<unk>', '0', '3', '1', '2'
        ]
        self.count = [1, 1, 1, 1, 0, 0, 0, 0]
        self.indices = {
            '<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '0': 4, '3': 5, '1': 6, '2': 7
        }
        self.bos_index = self.indices[bos]
        self.pad_index = self.indices[pad]
        self.eos_index = self.indices[eos]
        self.unk_index = self.indices[unk]

        self.nspecial = len(self.symbols) 


ALPHABET = AbgenDictionary()
ALPHABET_FULL = AbbertDictionary()
TAG_FULL = TagDictionary()

RESTYPE_1to3 = {
     "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN","E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

# ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
ATOM_TYPES = [
    '', 'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

RES_ATOM14 = [
    ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''],
]

UNK_LIST = [
    '6u8k',
    '6db8',
    '5e08',
    '5gkr',
    '4xwo',
    '4kzd',
    '6x5n',
    '6db9',
    '1keg',
    '2ok0',
    '2fr4',
    '6b3k',
    '6b14',
    '2r8s',
    '1xf2',
    '1i8m',
    '6mwn',
    '6x5m',
    '1cbv',
    '4kze',
    '1ehl',
    '3ivk',
    '6u8d'
]
