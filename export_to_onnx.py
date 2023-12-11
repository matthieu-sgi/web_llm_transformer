'''Export the last checkpoint to onnx format'''

from re import S
import token
import torch
import json
from modules.model import GPT
from modules.tokenizer import Tokenizer
import onnx
from onnxruntime.quantization import quantize_dynamic


EXPORT_PATH = './docs/gpt.onnx'

# Load the last checkpoint

config = json.load(open('config.json'))

model = GPT(config['vocab_size'],
            config['block_size'],
            config['n_layer'],
            config['n_head'],
            config['n_embd'],
            config['dropout'])



loaded_model = torch.load('./models/gtpV3Quantized.pt', map_location=torch.device('cpu'))['model']
# print(loaded_model.keys())
model.to('cpu')


model.load_state_dict(loaded_model)
model.eval()

dummy_input = torch.ones(size=(1, 50), dtype = torch.int32) 

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
quantize_dynamic(EXPORT_PATH, EXPORT_PATH + '.quantized.onnx',use_external_data_format=False)