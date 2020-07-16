import pickle

import torch
from torch.autograd import Variable

from utils.clean_sentence import clean_sentence
from utils.load_checkpoint import load_checkpoint
from utils.transforms import transform_val

from image_descriptors.LSTM import lstm
from image_descriptors.attentionLSTM import attention_lstm

from inference.LSTM_decoder import beam_search_lstm, greedy_search_lstm
from inference.Attention_decoder import beam_search_attention

vocab_file = "./vocab.pkl"
model_dir = "./save_model"

with open(vocab_file, "rb") as f:
    vocab = pickle.load(f)

device = torch.device("cpu")

def get_predictions(img, model, beam):
    if model == "lstm":
        encoder, decoder = lstm(len(vocab))
    elif model == "attention":
        encoder, decoder = attention_lstm(len(vocab))

    encoder, decoder = load_checkpoint(encoder, decoder, device, model_dir, model)

    encoder.eval()
    decoder.eval()

    img = transform_val(img).float()
    img = img.unsqueeze_(0)
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