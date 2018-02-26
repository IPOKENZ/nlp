import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import moses_multi_bleu

use_gpu = torch.cuda.is_available()

def validate_model(val_iter_bs1, encoder, decoder, criterion, DE, EN, logger=None, beam_search=True):
    ## Assumes that val_iter_bs1 has batch_size=1
    encoder.eval()
    decoder.eval()
    for i, batch in enumerate(val_iter_bs1): 
        
        target_sentences = []
        output_sentences = []
        
        source = batch.src.cuda() if use_gpu else batch.src
        target = batch.trg.cuda() if use_gpu else batch.trg

        # Initialize LSTM states
        batch_size = source.size(1)
        num_layers, hidden_size = encoder.num_layers, encoder.hidden_size
        init = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        init = init.cuda() if use_gpu else init
        states = (init, init.clone())
        outputs, states = encoder(source, states) #these have a batch_size=1 problem which is kind of annoying.
        
        max_trg_len = batch.trg.size(0) #cap it to compute perplexity, will be uncapped at test time 
        if beam_search: #beam search
            k = 10 #store best k options
            ## CONVERT BEST_OPTIONS WITH USE_GPU = TRUE OR FALSE.
            best_options = [(Variable(torch.zeros(1)).cuda(), Variable(torch.LongTensor([EN.vocab.stoi["<s>"]])).cuda(), states)] 
            length = 0
            while length < max_trg_len:
                options = [] #same format as best_options
                for lprob, sentence, current_state in best_options:
                    last_word = sentence[-1]
                    if last_word.data[0] != EN.vocab.stoi["</s>"]:
                        probs, new_state = decoder(last_word.unsqueeze(1), current_state, outputs)
                        probs = probs.squeeze()
                        for index in torch.topk(probs, k)[1]: #only care about top k options in probs for next word.
                            options.append((torch.add(probs[index], lprob), torch.cat([sentence, index]), new_state))
                    else:
                        options.append((lprob, sentence, current_state))
                options.sort(key = lambda x: x[0].data[0], reverse=True)
                best_options = options[:k] #sorts by first element, which is lprob.
                length = length + 1
            best_options.sort(key = lambda x: x[0].data[0], reverse=True)
            best_choice = best_options[0] #best overall
            sentence = best_choice[1].data
            
        else: #regular argmax search (aka beam search with k=1)
            sentence = [] #begin with a start token.
            vocab_size = len(EN.vocab)
            word = Variable(torch.LongTensor([EN.vocab.stoi["<s>"]])).cuda() #begin with a start token.

            probs = [] #store all our probabilities to compute perplexity
            losses = 0

            while word.data[0] != EN.vocab.stoi["</s>"] and len(sentence) < max_trg_len: ### TO FIX: YOU DONT WANT A CAP HERE
                sentence.append(word.data[0])
                translated, states = decoder(word.unsqueeze(1), states, outputs)
                probs.append(translated)
                word = torch.max(translated, 2)[1][0]

            sentence.append(EN.vocab.stoi["</s>"])
        
        target_sentences.append(" ".join([EN.vocab.itos[x[0]] for x in target.data.cpu().numpy()[1:-1]]))
        output_sentences.append(" ".join([EN.vocab.itos[x] for x in sentence][1:-1]))
        
        # Every now and then, output a sentence and its translation
        log_freq = 100
        if i % log_freq == 10:
            print("Source: {}\n".format([DE.vocab.itos[x[0]] for x in source.data.cpu().numpy()]))
            print("Target: {}\n".format([EN.vocab.itos[x[0]] for x in target.data.cpu().numpy()]))
            print("Model: {}\n".format([EN.vocab.itos[x] for x in sentence]))

    # Predict the BLEU score
    print("BLEU Score: ", moses_multi_bleu(output_sentences, target_sentences))

def train_model(train_iter, val_iter_bs1, encoder, decoder, optimizer, criterion, DE, EN,
                max_norm=1.0, num_epochs=10, logger=None):  
    encoder.train()
    decoder.train()
    best_ppl = 1000
    for epoch in range(num_epochs):

        # Validate model
        if epoch % 4 == 3:
            validate_model(val_iter_bs1, encoder, decoder, criterion, DE, EN, logger=None, beam_search=True)

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
            translated, states = decoder(target, states, outputs)
            
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