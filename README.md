# Web Large Language Model (Web-LLM), NanoGPT

This project is a re-implementation and extension of [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

## Overview

nanoGPT is a simplified version of GPT, designed for educational purposes and small-scale text generation tasks. The model has undergone three phases of training, each enhancing its capabilities and performance:

### 1. Phase 1: Character-based Tokenization on Harry Potter's Books

In the initial phase, the model was trained on the text from the Harry Potter book series using a basic character-based tokenizer. This phase served as a baseline for subsequent improvements.

### 2. Phase 2: Transition to BPE Tokenization with Tiktoken

The second phase introduced an upgrade in the tokenizer to Byte Pair Encoding (BPE) using the Tiktoken library. BPE tokenization helps the model capture more meaningful sub-word representations, improving the overall quality of generated text.

### 3. Phase 3: Training on French "Code Civil" with Tiktoken

In the final phase, the model was further fine-tuned on the French "Code Civil" using the Tiktoken library. This phase aimed to showcase the model's adaptability to different domains and languages, highlighting its versatility in generating contextually relevant text.



## Acknowledgements

This project builds upon the original work by Andrej Karpathy and leverages the Tiktoken library for improved tokenization. Special thanks to the open-source community for their contributions and support. Also warm thanks to [@Nicolas Stas](https://github.com/COLVERTYETY)

