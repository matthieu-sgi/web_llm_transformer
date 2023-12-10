'''Export the last checkpoint to onnx format'''

import torch
import json
from model import GPT
import onnx


EXPORT_PATH = './docs/gpt.onnx'

# Load the last checkpoint

config = json.load(open('config.json'))
vocab_size = json.load(open('./vocab/vocab.json'))['vocab_size']


model = GPT(vocab_size,
            config['block_size'],
            config['n_layer'],
            config['n_head'],
            config['n_embd'],
            config['dropout'])



loaded_model = torch.load('./models/gptV1.pt', map_location=torch.device('cpu'))['model']
# print(loaded_model.keys())
model.to('cpu')


model.load_state_dict(loaded_model)
model.eval()

dummy_input = torch.rand(1, 50).type(torch.LongTensor)

torch.onnx.export(model,
                  dummy_input,
                  EXPORT_PATH,
                  input_names=['input'],
                  output_names=['output'],
                  do_constant_folding=True,
                  export_params=True,
                  dynamic_axes={'input': {0: 'batch_size', 1: 'seq_len'},
                                'output': {0: 'batch_size', 1: 'seq_len'}})

# Load the model from onnx format
onnx_model = onnx.load(EXPORT_PATH)
onnx.checker.check_model(onnx_model)