import pickle

import torch
from torch.autograd import Variable

from utils.clean_sentence import clean_sentence
from utils.load_checkpoint import load_checkpoint
from utils.transforms import transform

from image_descriptors.LSTM import lstm
from image_descriptors.attentionLSTM import attention_lstm

from inference.LSTM_decoder import beam_search_lstm
from inference.Attention_decoder import beam_search_attention

vocab_file = "./vocab.pkl"
model_dir = "./save_model"

with open(vocab_file, "rb") as f:
    vocab = pickle.load(f)

device = torch.device("cpu")

encoder_lstm, decoder_lstm = lstm(len(vocab))
encoder_att, decoder_att = attention_lstm(len(vocab))

encoder_lstm, decoder_lstm = load_checkpoint(encoder_lstm, decoder_lstm, device, model_dir, "lstm")
encoder_att, decoder_att = load_checkpoint(encoder_att, decoder_att, device, model_dir, "attention")

def get_predictions(img, model, beam):
    if model == "lstm":
        encoder, decoder = encoder_lstm, decoder_lstm
    elif model == "attention":
        encoder, decoder = encoder_att, decoder_att
    else:
        return []

    img = transform(img).float().unsqueeze_(0)
    img = Variable(img)

    visual_features = encoder(img)

    if model == "lstm":
        output_sentences = beam_search_lstm(visual_features, decoder, vocab, device, beam_size=beam)
    elif model == "attention":
        output_sentences, alphas = beam_search_attention(visual_features, decoder, vocab, device, beam_size=beam)

    sentences = []
    for l in output_sentences:
        sentences.append(clean_sentence(l, vocab))
    
    return sentences