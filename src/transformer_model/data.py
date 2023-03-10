import torch
from torch.nn.utils.rnn import pad_sequence

# CONST
CTC_OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'

TRANSFORMER_PAD_TOKEN = '<PAD>'
TRANSFORMER_SOS_TOKEN = '<SOS>'
TRANSFORMER_EOS_TOKEN = '<EOS>'
TRANSFORMER_OOV_TOKEN = '<OOV>'


def collate_fn(batch):
    """Collate function for PyTorch dataloader."""
    batch_merged = {key: [elem[key] for elem in batch] for key in batch[0].keys()}
    out_dict = {
        'image': torch.stack(batch_merged['image'], 0),
        'image_mask': torch.BoolTensor(batch_merged['image_mask']),
        'text': batch_merged['text'],
        'text_len': torch.LongTensor([len(txt) for txt in batch_merged['text']]),
        'img_path': batch_merged['img_path'],
        'scale_coeff': batch_merged['scale_coeff']
    }
    if 'enc_text_transformer' in batch_merged:
        out_dict['enc_text_transformer'] = pad_sequence(batch_merged['enc_text_transformer'], batch_first=True, padding_value=0)
    if 'enc_text_ctc' in batch_merged:
        out_dict['enc_text_ctc'] = pad_sequence(batch_merged['enc_text_ctc'], batch_first=True, padding_value=0)
    return out_dict


def get_char_map(alphabet, *special_symbols):
    """Make from string alphabet character2int dict.
    Add BLANK char for CTC loss and OOV char for out of vocabulary symbols."""
    char_map = {value: idx + len(special_symbols) for (idx, value) in enumerate(alphabet)}
    for i, symbol in enumerate(special_symbols):
        char_map[symbol] = i
    return char_map


class BaseTokenizer:
    def __init__(self, charset: str, *special_symbols):
        self.char_map = get_char_map(charset, *special_symbols)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    def get_num_chars(self):
        return len(self.char_map)


class CTCTokenizer(BaseTokenizer):
    def __init__(self, charset: str):
        super().__init__(charset, CTC_BLANK, CTC_OOV_TOKEN)

    def encode(self, word_list):
        """Returns a list of encoded words (int)."""
        enc_words = []
        for word in word_list:
            enc_words.append(
                [self.char_map[char] if char in self.char_map
                 else self.char_map[CTC_OOV_TOKEN]
                 for char in str(word)]
            )
        return enc_words

    def decode_ctc(self, decoder, logits):
        beam_results, beam_scores, timesteps, out_lens = (x[:, 0] for x in decoder.decode(logits))
        words = []
        words_ts = []
        for word_enc, timestep, word_len in zip(beam_results, timesteps, out_lens):
            word_len = word_len.item()
            word_enc = word_enc[:word_len]
            words_ts.append(timestep[:word_len].tolist())
            word = []
            for char in word_enc:
                char = char.item()
                word.append(self.rev_char_map[char])
            words.append(''.join(word))
        return words, words_ts

    def decode(self, logits):
        """Returns a list of words (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary token."""
        logits = logits.permute(1, 0, 2)
        enc_word_list = torch.argmax(logits, -1).numpy()
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                # skip if blank symbol, oov token or repeated characters
                if (
                        char_enc != self.char_map[CTC_OOV_TOKEN]
                        and char_enc != self.char_map[CTC_BLANK]
                        # idx > 0 to avoid selecting [-1] item
                        and not (idx > 0 and char_enc == word[idx - 1])
                ):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words


class TransformerTokenizer(BaseTokenizer):
    def __init__(self, charset: str):
        super().__init__(charset, TRANSFORMER_PAD_TOKEN, TRANSFORMER_SOS_TOKEN, TRANSFORMER_EOS_TOKEN, TRANSFORMER_OOV_TOKEN)

    def encode(self, word_list):
        """Returns a list of encoded words (int)."""
        enc_words = []
        for word in word_list:
            enc_word = []
            enc_word.append(self.char_map[TRANSFORMER_SOS_TOKEN])
            enc_word.extend([self.char_map[char] if char in self.char_map
                             else self.char_map[TRANSFORMER_OOV_TOKEN]
                             for char in str(word)])
            enc_word.append(self.char_map[TRANSFORMER_EOS_TOKEN])
            enc_words.append(enc_word)
        return enc_words

    def decode(self, sequences):
        words = []
        for sequence in sequences:
            word = []
            for char in sequence:
                char = char.item()
                if char == 2:
                    break
                word.append(self.rev_char_map[char])
            words.append(''.join(word))
        return words
