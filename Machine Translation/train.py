import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import moses_multi_bleu
import torchtext

use_gpu = torch.cuda.is_available()

def validate_model(val_iter, val_iter_bs1, encoder, decoder, criterion, DE, EN, 
                   logger=None, beam_width=None, compute_bleu=False):
    ## Assumes that val_iter_bs1 has batch_size=1
    encoder.eval()
    decoder.eval()

    ## Do fake validation to compute perplexity (same thing as train basically)

    losses = 0
    for i, batch in enumerate(val_iter): 
        source = batch.src.cuda() if use_gpu else batch.src
        target = batch.trg.cuda() if use_gpu else batch.trg
                
        outputs, states = encoder(source)
        translated, states = decoder(target, states, outputs)
        
        vocab_size = translated.size(2)
        translated = translated.contiguous()[:-1].view(-1, vocab_size)
        target = target[1:].view(-1).contiguous()
        loss = criterion(translated, target)

        # Log information
        losses += loss.data[0]

    losses_for_log = losses / len(val_iter)
    info = 'Validation Loss: {loss:.3f}, Validation Perplexity: {perplexity:.3f}'.format(
        loss=losses_for_log, perplexity=torch.exp(torch.FloatTensor([losses_for_log]))[0])
    logger.log(info) if logger is not None else print(info)

    if compute_bleu:
        ## Do real validation with beam search

        target_sentences = []
        output_sentences = []

        for i, batch in enumerate(val_iter_bs1): 
            
            source = batch.src.cuda() if use_gpu else batch.src
            target = batch.trg.cuda() if use_gpu else batch.trg

            outputs, states = encoder(source) #these have a batch_size=1 problem which is kind of annoying.
            
            max_trg_len = 50 #cap it to compute perplexity, will be uncapped at test time 
            if beam_width: #beam search
                k = beam_width #store best k options
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
                print("Model (Beam): {}\n".format([EN.vocab.itos[x] for x in sentence]))

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

                print("Model (Regular): {}\n".format([EN.vocab.itos[x] for x in sentence]))

        # Predict the BLEU score
        info = "BLEU Score: {}".format(moses_multi_bleu(output_sentences, target_sentences))
        logger.log(info) if logger is not None else print(info)

def train_model(train_iter, val_iter, val_iter_bs1, encoder, decoder, optimizer, criterion, DE, EN,
                max_norm=1.0, num_epochs=10, logger=None, beam_width=None, bidirectional=True):  
    encoder.train()
    decoder.train()
    for epoch in range(num_epochs):

        # Validate model
        validate_model(val_iter, val_iter_bs1, encoder, decoder, criterion, DE, EN, logger=logger, beam_width=beam_width)

        # Train model
        losses = 0
        for i, batch in enumerate(train_iter): 
            source = batch.src.cuda() if use_gpu else batch.src
            target = batch.trg.cuda() if use_gpu else batch.trg
            
            # Forward, backprop, optimizer
            optimizer.zero_grad()
            
            outputs, states = encoder(source)
            translated, states = decoder(target, states, outputs)
            
            vocab_size = translated.size(2)
            translated = translated.contiguous()[:-1].view(-1, vocab_size)
            target = target[1:].view(-1).contiguous()
            loss = criterion(translated, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm(encoder.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm(decoder.parameters(), max_norm)
            optimizer.step()

            # Log information
            losses += loss.data[0]
            log_freq = 1000
            if i % log_freq == 750:
                losses_for_log = losses / (i)
                info = 'Epoch [{epochs}/{num_epochs}], Batch [{batch}/{num_batches}], Loss: {loss:.3f}, Sorta-Perplexity: {perplexity:.3f}'.format(
                    epochs=epoch+1, num_epochs=num_epochs, batch=i, num_batches=len(train_iter), loss=losses_for_log, perplexity=torch.exp(torch.FloatTensor([losses_for_log]))[0])
                logger.log(info) if logger is not None else print(info)
                torch.save(encoder.state_dict(), 'saves/encoder{}.pkl'.format(epoch))
                torch.save(decoder.state_dict(), 'saves/decoder{}.pkl'.format(epoch))

def predict(in_file, out_file, encoder, decoder, DE, EN):
    encoder.eval()
    decoder.eval()
    
    with open(in_file, 'r') as in_f, open(out_file, 'w') as out_f:
        print('id,word', file=out_f)
        for i, line in enumerate(in_f):
            example = torchtext.data.example.Example.fromlist([line], [('src', DE)]).src
            source = DE.process([example], -1, False)
            source = source.cuda() if use_gpu else source

            outputs, states = encoder(source) #these have a batch_size=1 problem which is kind of annoying.

            max_trg_len = 3 
            k = 100 
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
                        pass #we don't want it to end too short so we aren't going to allow len<3 submissions.
                options.sort(key = lambda x: x[0].data[0], reverse=True)
                best_options = options[:k] #sorts by first element, which is lprob.
                length = length + 1
                
            best_options.sort(key = lambda x: x[0].data[0], reverse=True)
            sentence_strs = []
            for option in best_options[:k]:
                sentence = option[1].data[1:]
                sentence = [EN.vocab.itos[x] for x in sentence]
                sentence = [word.replace("\"", "<quote>").replace(",", "<comma>") for word in sentence]
                sentence_str = '|'.join(sentence)
                sentence_strs.append(sentence_str)
            print(str(i+1) + ',' + ' '.join(sentence_strs), file=out_f)

            if i % 50 == 0:
                print("Finished first {i} predictions.".format(i))