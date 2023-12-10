
from calendar import c
from email.generator import DecodedGenerator
import tokenize
from modules.model import GPT
from modules.tokenizer import Tokenizer

import json
import torch

if __name__ == '__main__':
    # with_my_toeknizer()
    config = json.load(open('config.json'))
    # vocab_size = json.load(open('./vocab/vocab.json'))['vocab_size']
    tokenizer = Tokenizer(vocab_file='./vocab/vocab.json')

    model = GPT(tokenizer.vocab_size,
                config['block_size'],
                config['n_layer'],
                config['n_head'],
                config['n_embd'],
                config['dropout'])
    
    load_weights = torch.load('./models/gptV1.pt')
    model.load_state_dict(load_weights['model'])
    model.eval()


    try:
        while True:
            sentence = input('Enter a sentence: ')
            tokenized_sentence = tokenizer.encode(sentence)
            if sentence == 'exit':
                break
            else:
                # print(tokenized_sentence)
                sentence = torch.tensor(tokenized_sentence, dtype=torch.long).unsqueeze(0)
                output = model.generate(sentence, 100)
                decoded_output = tokenizer.decode(output[0].cpu().tolist())
                print(decoded_output)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    