from modules.tokenizer import Tokenizer

import json

if __name__ == '__main__':

    with open('dataset/Harry_Potter_all_books_preprocessed.txt', 'r') as f:
        train_data = f.readlines()
    enc = Tokenizer()
    enc.fit(train_data[0])

    print(enc.vocab_size, enc.char_to_id)

    target_json ={
        "vocab_size": enc.vocab_size,
        "char_to_id": enc.char_to_id,
        "id_to_char": enc.id_to_char
    }

    with open('./vocab/vocab.json', 'w') as f:
        json.dump(target_json, f, indent=4, sort_keys=True)