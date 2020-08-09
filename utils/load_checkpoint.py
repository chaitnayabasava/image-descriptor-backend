import os, glob
import torch


def load_checkpoint(encoder, decoder, device, model_dir, model):

    pkl_list = glob.glob(os.path.join(model_dir, model, "*.pkl"))
    if len(pkl_list) == 0:
        print("No checkpoints available...")
        return encoder, decoder

    checkpoint = torch.load(pkl_list[-1], map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    return encoder, decoder
