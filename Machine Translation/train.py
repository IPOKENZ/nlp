import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

use_gpu = torch.cuda.is_available()

def validate_model(val_iter_bs1, encoder, decoder, criterion, DE, EN, logger=None, beam_search=False):
    ## Assumes that val_iter has batch_size=1
    encoder.eval()
    decoder.eval()
    for i, batch in enumerate(val_iter_bs1): 
        source = batch.src.cuda() if use_gpu else batch.src
        target = batch.trg.cuda() if use_gpu else batch.trg

        # Initialize LSTM states
        batch_size = source.size(1)
        num_layers, hidden_size = encoder.num_layers, encoder.hidden_size
        init = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        init = init.cuda() if use_gpu else init
        states = (init, init.clone())

        outputs, states = encoder(source, states) #these have a batch_size=1 problem which is kind of annoying.
        sentence = [] #begin with a start token.
        vocab_size = len(EN.vocab)
        word = Variable(torch.LongTensor([EN.vocab.stoi["<s>"]])).cuda() #begin with a start token.
        
        max_trg_len = batch.trg.size(0) #cap it to compute perplexity, will be uncapped at test time 
        probs = [] #store all our probabilities to compute perplexity
        losses = 0

        while word.data[0] != EN.vocab.stoi["</s>"] and len(sentence) < max_trg_len: ### TO FIX: YOU DONT WANT A CAP HERE
            sentence.append(word.data[0])
            translated, states = decoder(word.unsqueeze(1), states)
            probs.append(translated)
            
            # figure out what the next word should be -- either argmax or beam-search
            if beam_search:
                pass
            else:
                word = torch.max(translated, 2)[1][0]
                
        sentence.append(EN.vocab.stoi["</s>"])
        probs = torch.stack(probs)
        
        # Log information -- Doesn't work at the moment
#         print(probs.view(-1, vocab_size), target.view(-1))
#         loss = criterion(probs.view(-1, vocab_size), target.view(-1)) 
#         losses += loss.data[0]
        
        # Every now and then, output a sentence and its translation
        log_freq = 100
        if i % log_freq == 10:
            info = ''
            info = info + "Source: {}".format([DE.vocab.itos[x[0]] for x in source.data.cpu().numpy()])
            info = info + "Target: {}".format([EN.vocab.itos[x[0]] for x in target.data.cpu().numpy()])
            info = info + "Model: {}".format([EN.vocab.itos[x] for x in sentence])
            logger.log(info) if logger is not None else print(info)

def train_model(train_iter, val_iter_bs1, encoder, decoder, optimizer, criterion, DE, EN,
                max_norm=1.0, num_epochs=10, logger=None):  
    encoder.train()
    decoder.train()
    best_ppl = 1000
    for epoch in range(num_epochs):

        # Validate model
        validate_model(val_iter_bs1, encoder, decoder, criterion, DE, EN, logger=None, beam_search=False)

        # Train model
        losses = 0
        for i, batch in enumerate(train_iter): 
            source = batch.src.cuda() if use_gpu else batch.src
            target = batch.trg.cuda() if use_gpu else batch.trg
            
            # Initialize LSTM states
            batch_size = source.size(1)
            num_layers, hidden_size = encoder.num_layers, encoder.hidden_size
            init = Variable(torch.zeros(num_layers, batch_size, hidden_size))
            init = init.cuda() if use_gpu else init
            states = (init, init.clone())

            # Forward, backprop, optimizer
            optimizer.zero_grad()
            
            outputs, states = encoder(source, states)
            translated, states = decoder(target, states)
            
            vocab_size = translated.size(2)
            start_tokens = Variable(torch.zeros(batch_size, vocab_size))
            start_tokens = start_tokens.scatter_(1, torch.ones(batch_size).unsqueeze(1).long() * EN.vocab.stoi["<s>"], 1)
            start_tokens = start_tokens.unsqueeze(0).cuda()
            translated = torch.cat((start_tokens, translated[:-1]), 0)
            
            loss = criterion(translated.view(-1, vocab_size), target.view(-1)) 
            loss.backward()
            torch.nn.utils.clip_grad_norm(encoder.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm(decoder.parameters(), max_norm)
            optimizer.step()

#             # Zero hidden state with certain probability
#             if (torch.rand(1)[0] < 0.95):
#                 states = (init.clone(), init.clone())

            # Log information
            losses += loss.data[0]
            log_freq = 1000
            if i % log_freq == 10:
                losses_for_log = losses / (i)
                info = 'Epoch [{epochs}/{num_epochs}], Batch [{batch}/{num_batches}], Loss: {loss:.3f}, Sorta-Perplexity: {perplexity:.3f}'.format(
                    epochs=epoch+1, num_epochs=num_epochs, batch=i, num_batches=len(train_iter), loss=losses_for_log, perplexity=torch.exp(torch.FloatTensor([losses_for_log]))[0])
                logger.log(info) if logger is not None else print(info)
                torch.save(encoder.state_dict(), 'saves/encoder.pkl')
                torch.save(decoder.state_dict(), 'saves/decoder.pkl')