
from calendar import c
from multiprocessing import context
import os
import time
from modules.model import GPT
from modules.tokenizer import Tokenizer
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import json
import math
import numpy as np
import torch
import tqdm

config = json.load(open('config.json'))

BLOCK_SIZE = config['block_size']
N_LAYER = config['n_layer']
N_HEAD = config['n_head']
N_EMBED = config['n_embd']
DROPOUT = config['dropout']
BATCH_SIZE = config['batch_size']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = config['learning_rate']
OUT_DIR = 'models'
MAX_ITERS = config['max_iters']

val_data = []
train_data = []

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    if DEVICE == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters = 200):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)

            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, warmup_iters = 2000, learning_rate = LR, lr_decay_iters = 600000, min_lr = 6e-5):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ =='__main__':


    with open('dataset/Harry_Potter_all_books_preprocessed.txt', 'r') as f:
        train_data = f.readlines()



    enc = Tokenizer()
    # enc.fit(train_data[0])
    print(len(enc))
    enc_train_data = [enc.encode(t) for t in train_data][0]
    np.random.seed(0)
    ratio = 0.8



    # Split the dataset into training and validation sets
    val_data = np.array(enc_train_data[int(ratio * len(enc_train_data)):])
    train_data = np.array(enc_train_data[:int(ratio * len(enc_train_data))])
    # print("Length of training dataset: ", len(train_data))
    # print("Length of validation dataset: ", len(val_data))


    # Calculate the singularity of the dataset
    # singularity = len(set(enc_train_data))
    # print("Singularity of the dataset: ", singularity)

    # # Calculate number of tokens in the dataset
    # num_tokens = len(enc_train_data)
    # print("Number of tokens in the dataset: ", num_tokens)
    
    model = GPT(vocab_size=enc.vocab_size,
                block_size=BLOCK_SIZE,
                n_layer=N_LAYER,
                n_head=N_HEAD,
                n_embd=N_EMBED,
                dropout=DROPOUT)
    
    model.to('cuda')

    writer = SummaryWriter()
    
    dummy_input = torch.zeros(1, BLOCK_SIZE, dtype=torch.long).cuda()
    output = model(dummy_input)
    # print(output)    
    # writer.add_graph(model = model, input_to_model=dummy_input)
    iter_num = 0
    best_val_loss = 1e9

    optimizer = model.configure_optimizers(0.1,
                                           LR,
                                           (0.9, 0.95),
                                           DEVICE)
    compile = True
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0


    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process

    print("Starting training...")

    try : 
        while True:

            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % 10 == 0:
                losses = estimate_loss(model)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                if losses['val'] < best_val_loss :
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'optimized_model': model.state_dict(),
                            'model' : unoptimized_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': (len(enc), BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBED, DROPOUT),  
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                        }
                        print(f"saving checkpoint to {OUT_DIR}")
                        torch.save(checkpoint, os.path.join(OUT_DIR, 'ckpt.pt'))
                else:
                    print("Saving the probably overfitting model")
                    checkpoint = {
                        'optimized_model': model.state_dict(),
                        'model' : unoptimized_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': (len(enc), BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBED, DROPOUT),  
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"saving checkpoint to {OUT_DIR}")
                    torch.save(checkpoint, os.path.join(OUT_DIR, 'overfitting_ckpt.pt'))

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            gradient_accumulation_steps = 5*8
            for micro_step in tqdm.tqdm(range(gradient_accumulation_steps)):
                
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                loss.backward()

            # step the optimizer and scaler if training in fp16
            optimizer.step()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            writer.add_scalar('Learning rate', lr, iter_num)
            writer.add_scalar('Loss/val', losses['val'], iter_num)
            writer.add_scalar('Loss/train', losses['train'], iter_num)    
            
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % 10 == 0 :
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > MAX_ITERS:
                break
    except KeyboardInterrupt:
        # Generate a sequence of tokens

        context = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
        out = model.generate(context, 100)
        writer.close()
        print(enc.decode(out[0].cpu().tolist()))



