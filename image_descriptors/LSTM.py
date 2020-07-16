import numpy as np
import sys

sys.path.append("../")

from models.encoders import EncoderNormal
from models.decoders.LSTM import TextualHeadLSTM


def lstm(vocab_size):

    encoder = EncoderNormal(300)
    decoder = TextualHeadLSTM(300, 512, vocab_size, num_layers=1)

    return encoder, decoder
