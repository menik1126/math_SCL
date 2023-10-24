# coding: utf-8
from src.train_and_evaluate_prune import *
from src.models_prune import *
import time
import torch.optim

from src.expressions_transfer import *


from tqdm import tqdm

from src import config

import torch.nn.utils.prune as prune
from src.prune_method import *
import pytorch_warmup as warmup
import os

batch_size = config.batch_size
embedding_size = config.embedding_size
hidden_size = config.hidden_size

learning_rate = config.learning_rate
weight_decay = config.weight_decay
beam_size = config.beam_size
n_layers = config.n_layers

num_list_text = []
for d in range(config.quantity_num):
    num_list_text.append('NUM'+str(d))

if config.MODEL_NAME=='roberta':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("./src/chinese_roberta/vocab.txt")
    tokenizer.add_special_tokens({'additional_special_tokens':num_list_text})
elif config.MODEL_NAME=='roberta-large':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("./src/chinese_roberta_large/vocab.txt")
    tokenizer.add_special_tokens({'additional_special_tokens':num_list_text})
elif config.MODEL_NAME =='xml-roberta':
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    tokenizer.additional_special_tokens = tokenizer.additional_special_tokens + num_list_text
elif config.MODEL_NAME =='xml-roberta-base':
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    tokenizer.additional_special_tokens = tokenizer.additional_special_tokens + num_list_text




vocab_size = len(tokenizer)

data = load_raw_data("data/Math_23K.json")

pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []

for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs


fold_size = int(len(pairs) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])


fold_pairs.append(pairs[(fold_size * 4):])

best_acc_fold=[[0,0,2316], [0,0,2316], [0,0,2316], [0,0,2316], [0,0,2316]]
print(np.array(best_acc_fold).shape)
for fold in range(5):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]
    

    input_lang, output_lang, train_pairs, test_pairs = prepare_data_5fold(tokenizer, pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, tree=True)

    # Initialize models
    
    encoder = EncoderSeq(input_size=input_lang.n_words, vocab_size=vocab_size, hidden_size=hidden_size,
                    n_layers=n_layers)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                    input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

     # Prue
    if config.is_prune:
        parameters_to_prune = []
        count = 0
        for models in [encoder.named_modules(), predict.named_modules(), generate.named_modules(), merge.named_modules()] :#+ predict.named_modules() + generate.named_modules()+ merge.named_modules():
            
            for name, module in models:
                if hasattr(module, 'weight'):
                    parameters_to_prune.append((module, 'weight'))
                    
                    count+=1
                
                
        parameters_to_prune = tuple(parameters_to_prune)

        print("moduel count:{}".format(count))
        print("len of parameters_to_prune list:{}".format(len(parameters_to_prune)))
        print("prunePercent:{}".format(config.prunePercent))
        global_unstructured_flag(parameters_to_prune, True)
        
        customFromMask_list = prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,#FooBarPruningMethod,
        amount=config.prunePercent,
        )







    
    # the embedding layer is  only for generated number embeddings, operators, and paddings
    encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
    predict_optimizer = torch.optim.AdamW(predict.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
    generate_optimizer = torch.optim.AdamW(generate.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
    merge_optimizer = torch.optim.AdamW(merge.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
    
    encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[config.n_epochs//3], gamma=0.1)
    predict_scheduler = torch.optim.lr_scheduler.MultiStepLR(predict_optimizer, milestones=[config.n_epochs//3], gamma=0.1)
    generate_scheduler = torch.optim.lr_scheduler.MultiStepLR(generate_optimizer, milestones=[config.n_epochs//3], gamma=0.1)
    merge_scheduler = torch.optim.lr_scheduler.MultiStepLR(merge_optimizer, milestones=[config.n_epochs//3], gamma=0.1)

    if config.warm_up_stratege == "original":
        encoder_warmup_scheduler = warmup.UntunedLinearWarmup(encoder_optimizer)
        encoder_warmup_scheduler.last_step = -1 # initialize the step counter

        predict_warmup_scheduler = warmup.UntunedLinearWarmup(predict_optimizer)
        predict_warmup_scheduler.last_step = -1

        generate_warmup_scheduler = warmup.UntunedLinearWarmup(generate_optimizer)
        generate_warmup_scheduler.last_step = -1

        merge_warmup_scheduler = warmup.UntunedLinearWarmup(merge_optimizer)
        merge_warmup_scheduler.last_step = -1

    elif config.warm_up_stratege == "LinearWarmup":
        encoder_warmup_scheduler =  warmup.LinearWarmup(encoder_optimizer, warmup_period=config.warmup_period )#warmup.RAdamWarmup(encoder_optimizer)#warmup.UntunedLinearWarmup(encoder_optimizer)
        encoder_warmup_scheduler.last_step = -1 # initialize the step counter

        predict_warmup_scheduler = warmup.LinearWarmup(predict_optimizer, warmup_period=config.warmup_period )#warmup.RAdamWarmup(predict_optimizer)
        predict_warmup_scheduler.last_step = -1

        generate_warmup_scheduler = warmup.LinearWarmup(generate_optimizer, warmup_period=config.warmup_period)#warmup.RAdamWarmup(generate_optimizer)
        generate_warmup_scheduler.last_step = -1

        merge_warmup_scheduler = warmup.LinearWarmup(merge_optimizer, warmup_period=config.warmup_period )#warmup.RAdamWarmup(merge_optimizer)
        merge_warmup_scheduler.last_step = -1

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])
    
    last_acc=0.0
    for epoch in range(1, config.n_epochs+1):
        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
        print("fold:", fold + 1)
        print("epoch:", epoch + 1)
        start = time.time()
        """
           input_lengths:和多个长度的batch组成的list
        """
        if config.is_RDrop:
           kl_loss_total = 0
        
        kl_loss_total_1 = 0
        loss_total_no_prue = 0

        kl_loss_total_2 = 0
        loss_total_prue = 0
        for idx in tqdm(range(len(input_lengths))):
            encoder_scheduler.step(epoch-1)
            predict_scheduler.step(epoch-1)
            generate_scheduler.step(epoch-1)
            merge_scheduler.step(epoch-1)

            encoder_warmup_scheduler.dampen()
            predict_warmup_scheduler.dampen()
            generate_warmup_scheduler.dampen()
            merge_warmup_scheduler.dampen()

            loss_no_prue, kl_loss1, loss_prue, kl_loss2 = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], 
                generate_num_ids, encoder, predict, generate, merge,
                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, 
                output_lang, num_pos_batches[idx], is_train = True,
                parameters_to_prune = parameters_to_prune)
        
            loss_total_prue += loss_prue
            kl_loss_total_1 += kl_loss1

            loss_total_no_prue += loss_no_prue
            kl_loss_total_2 += kl_loss2

        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        
        L = len(input_lengths)
        
        print("loss_1:{} contra_loss_1:{} loss_2:{} contra_loss_2:{} loss type:{}".format(loss_total_prue / L, kl_loss_total_1 / L, loss_total_no_prue / L, kl_loss_total_2 / L, config.RDloss))
    


        print("training time", time_since(time.time() - start))
        print("--------------------------------")
        if epoch % config.test_interval == 0 or epoch > config.n_epochs - 5:
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()

            global_unstructured_flag(parameters_to_prune, config.is_prune2test)
            for test_batch in tqdm(test_pairs):
                test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                         merge, output_lang, test_batch[5], beam_size=beam_size)
                                         
                val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                eval_total += 1

            print(equation_ac, value_ac, eval_total)
            print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            print("testing time", time_since(time.time() - start))
            print("dropout:{} contra_weight:{} is_mask:{} is_em_dropout:{} is_prune2test:{} prunePercent:{} embedding_size:{} hidden_size:{} loss_no_mask:{} warm_up_strategy:{} quantity_num:{}".format(config.dropout, config.contra_weight, config.is_mask, config.is_em_dropout, config.is_prune2test, config.prunePercent, config.embedding_size, config.hidden_size, config.is_loss_no_mask, config.warm_up_stratege, config.quantity_num))
            print("------------------------------------------------------")
            curr_acc=round(float(value_ac)/eval_total,4)
            if curr_acc>=last_acc:
                # torch.save(encoder.state_dict(), "models/encoder")
                # torch.save(predict.state_dict(), "models/predict")
                # torch.save(generate.state_dict(), "models/generate")
                # torch.save(merge.state_dict(), "models/merge")
                
                past_epoch_out=[]
                if os.path.exists("models/epoch_num"+str(fold)):
                    file_epoch_out=open( "models/epoch_num"+str(fold)).readlines()
                    for line in file_epoch_out:
                        past_epoch_out.append(line)
                        print(line.strip())
                file_epoch_out=open( "models/epoch_num"+str(fold),"w")
                file_epoch_out.write(str(epoch)+" "+str(curr_acc)+"\n")
                for line in past_epoch_out:
                    file_epoch_out.write(line)
                last_acc=curr_acc
                file_epoch_out.close()
                best_acc_fold[fold][0]=equation_ac
                best_acc_fold[fold][1]=value_ac
                best_acc_fold[fold][2]=eval_total

            if epoch == config.n_epochs - 1:
                #best_acc_fold.append((equation_ac, value_ac, eval_total))
                a, b, c = 0, 0, 0
                for bl in range(len(best_acc_fold)):
                    print(round(best_acc_fold[bl][0] / float(best_acc_fold[bl][2]),4), round(best_acc_fold[bl][1] / float(best_acc_fold[bl][2]),4))
                    a += best_acc_fold[bl][0]
                    b += best_acc_fold[bl][1]
                    c += best_acc_fold[bl][2]
                print("best_acc_fold:")
                print(round(a / float(c),4),round( b / float(c),4))

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print(a / float(c), b / float(c))
