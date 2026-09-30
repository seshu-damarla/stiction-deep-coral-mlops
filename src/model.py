# after training the Deep CORAL, the CNN encoder (architecture and its learned weights) is saved.
'''
torch.save(model.encoder, "artifacts/encoder.pt")
'''

from pathlib import Path
import torch

def load_CNNencoder():
    # to find the absolute path of the directory two levels up from the file where the code is currently running.
    project_root = Path(__file__).resolve().parent.parent
    encoder_path = project_root / "artifacts" / "encoder.pt"

    # check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # load the entire encoder model not just its weights
    encoder = torch.jit.load(encoder_path, map_location=device) # , weights_only = False
    encoder = encoder.to(device) # move the model to the selected hardware

    # set encoder to inference mode
    encoder.eval()

    return encoder, device

