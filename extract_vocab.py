from modules.tokenizer import Tokenizer
import tiktoken
import json


def with_my_toeknizer() -> None:
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

def using_tiktoken() -> None:
    with open('dataset/Harry_Potter_all_books_preprocessed.txt', 'r') as f:
        train_data = f.readlines()
    encoding = tiktoken.get_encoding("p50k_base")
    encoding.encode(train_data[0])
    
    # print the encoding
    print(encoding.token_byte_values())
    # Dump the encoding to a file
    # encoding.save("encoding.json")

if __name__ == '__main__':
    # with_my_toeknizer()
    using_tiktoken()
