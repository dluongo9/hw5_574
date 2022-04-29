"""Module for a Vocabulary object, and associated helpers.

Heavily inspired by torchtext's Vocab:
https://pytorch.org/text/stable/_modules/torchtext/vocab.html#Vocab
"""

from collections import Counter
from csv import DictReader, DictWriter
from typing import Iterable


class Vocabulary:
    """Stores a bidirectional mapping of tokens to integer indices.

    Attributes:
    """

    UNK = "<unk>"
    # for reading / writing to files
    column_names = ("index", "token", "frequency")
    delimiter = "\t"

    def __init__(
        self,
        frequencies: Counter,
        special_tokens: tuple[str] = ("<unk>",),
        special_first: bool = True,
        max_size: int = None,
        min_freq: int = 1,
    ) -> None:
        """Builds a new Vocabulary object from a counter of token frequencies.

        Args:
            frequencies: the Counter
            special_tokens: list of special tokens (unk, pad, etc)
            special_first: whether to give the special tokens the lowest indices
            max_size: maximum vocabulary size, or None; the first # of tokens read up to this size will be included
            min_freq: minimum frequency to include an item in the vocab; 1 will include all
        """
        self.frequencies = frequencies

        # index --> token is a list instead of a dict, since list indices can serve as the keys
        self.index_to_token = []
        if special_first:
            self.index_to_token.extend(list(special_tokens))
            if max_size:
                max_size += len(special_tokens)

        # make sure frequences of special tokens aren't considered
        for token in special_tokens:
            del frequencies[token]

        # add tokens from the frequency counter
        for token in frequencies:
            if len(self.index_to_token) == max_size:
                break
            if frequencies[token] < min_freq:
                continue
            self.index_to_token.append(token)

        if not special_first:
            self.index_to_token.extend(list(special_tokens))

        if Vocabulary.UNK in special_tokens:
            unk_index = special_tokens.index(Vocabulary.UNK)
            self._unk_index = (
                unk_index
                if special_first
                else len(self.index_to_token) - len(special_tokens) + unk_index
            )

        # generate the reverse token --> index mapping
        self.token_to_index = {
            token: index for index, token in enumerate(self.index_to_token)
        }

    def __len__(self) -> int:
        """Get length of the vocab. """
        return len(self.index_to_token)

    def __getitem__(self, token: str) -> int:
        """Get the index of a token.
        Returns the index of <unk> if there is an unk token and the token is not in vocab.
        Raises a ValueError if there is no unk token and the token is not in vocab. """
        if token in self.index_to_token:
            return self.token_to_index[token]
        elif Vocabulary.UNK in self.index_to_token:
            return self._unk_index
        else:
            raise ValueError(f"Token {token} not in vocab.")

    def tokens_to_indices(self, tokens: Iterable[str]) -> list[int]:
        """Get all indices for a list of tokens. """
        return [self.__getitem__(token) for token in tokens]

    def indices_to_tokens(self, indices: Iterable[int]) -> list[str]:
        """Get all tokens for a list of integer indices. """
        return [self.index_to_token[index] for index in indices]

    def save_to_file(self, filename: str) -> None:
        """Write the vocab to a file, including frequencies.

        Args:
            filename: name of file to save to.
        """
        with open(filename, "w") as f:
            f.write(f"{Vocabulary.delimiter.join(Vocabulary.column_names)}\n")
            for index, token in enumerate(self.index_to_token):
                row_string = Vocabulary.delimiter.join(
                    [str(index), token, str(self.frequencies.get(token, 0))]
                )
                f.write(f"{row_string}\n")

    @classmethod
    def load_from_file(cls, filename: str, **kwargs):
        """Load a Vocabulary object from a saved vocab file.

        Args:
            filename: file with vocab, assumed output from `save_to_file`
        """
        frequencies: Counter = Counter()
        with open(filename, "r") as f:
            reader = DictReader(f, delimiter=Vocabulary.delimiter)
            for row in reader:
                frequencies[row["token"]] = int(row["frequency"])
        return cls(frequencies, **kwargs)

    @classmethod
    def from_text_files(cls, texts: Iterable[str], **kwargs):
        """Initializes a Vocabulary object from a list of text files.

        Args:
            texts: list of file names containing text.
                The text in each file is assumed to be white-space tokenized,
                so do any additional pre-processing before this.

        Returns:
            Vocabulary object
        """
        counter: Counter = Counter()
        for filename in texts:
            with open(filename, "r") as f:
                for line in f:
                    tokens = line.strip("\n").split(" ")
                    counter.update(tokens)
        return cls(counter, **kwargs)