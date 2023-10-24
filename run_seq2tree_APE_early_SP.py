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



batch_size = 16
learning_rate = config.learning_rate
weight_decay = 1e-5
beam_size = 5
n_layers = 2

hidden_size =config.hidden_size# 512
embedding_size = config.embedding_size
# APE dataset
n_epochs = 50

num_list_text = []
for d in range(config.quantity_num_ape):              # 22个数字
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
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')#("./src/chinese_roberta/vocab.txt")#, additional_special_tokens = num_list_text )
    #tokenizer.
    #special_tokens_dict = {'additional_special_tokens': num_list_text}
    tokenizer.additional_special_tokens = tokenizer.additional_special_tokens + num_list_text
elif config.MODEL_NAME =='xml-roberta-base':
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')#("./src/chinese_roberta/vocab.txt")#, additional_special_tokens = num_list_text )
    #tokenizer.
    #special_tokens_dict = {'additional_special_tokens': num_list_text}
    tokenizer.additional_special_tokens = tokenizer.additional_special_tokens + num_list_text
    
vocab_size = len(tokenizer)
dataset = "APE"


valid_data = load_data('data/ape/valid.ape.json',1)
print(valid_data[0])
print(valid_data[1])
train_data = load_data('data/ape/train.ape.json',1)
test_data = load_data('data/ape/test.ape.json',1)

# train_dataset
pairs, generate_nums, copy_nums = transfer_num(train_data)
temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_trained = temp_pairs

# valid_dataset
pairs_from_test, _, _ = transfer_num(valid_data)
temp_pairs = []
for p in pairs_from_test:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_tested = temp_pairs

# test_dataset
pairs_from_valid, _, _ = transfer_num(test_data)
temp_pairs = []
for p in pairs_from_valid:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_validated = temp_pairs


input_lang, output_lang, train_pairs, (valid_pairs, test_pairs) = prepare_data(tokenizer, pairs_trained, [pairs_validated, pairs_tested], 5, generate_nums,
                                                            copy_nums, tree=True)
best_acc_fold=[]



print("##############################")
print("input_lang words"+str(input_lang.n_words))
print("output_lang words"+str(output_lang.n_words))
print("generate nums:")
print(generate_nums)
print("copy number max nums"+str(copy_nums))
print("dataset_size:")
print(len(pairs))
print(len(pairs_from_test))
print(len(pairs_from_valid))
print("dataset_after indexed size:")
print(len(train_pairs))
print(len(test_pairs))
print(len(valid_pairs))

def indexes_to_sentence(lang, index_list, tree=False):
    res = []
    for index in index_list:
        if index < lang.n_words:
            res.append(lang.index2word[index])
    return res

UNK= output_lang.word2index["UNK"]
temp_pairs = []
i=0
for p in train_pairs:
    if UNK not in p[2]:
        temp_pairs.append(p)
    else:
        i+=1
        if i<5:
            #print( " ".join(indexes_to_sentence(input_lang,p[0])))
            print( " ".join(indexes_to_sentence(output_lang,p[2])))

train_pairs=temp_pairs
temp_pairs = []
for p in test_pairs:
    if UNK not in p[2]:
        temp_pairs.append(p)
test_pairs=temp_pairs
temp_pairs = []
for p in valid_pairs:
    if UNK not in p[2]:
        temp_pairs.append(p)
valid_pairs=temp_pairs

print("##############################")
print("dataset_after erase UNK data:")
print(len(train_pairs))
print(len(test_pairs))
print(len(valid_pairs))


# Initialize models,here op_nums [PAD, +,- ,*,^,/]
encoder = EncoderSeq(input_size=input_lang.n_words, vocab_size=vocab_size, hidden_size=hidden_size,
                n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                    embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

# the embedding layer is  only for generated number embeddings, operators, and paddings
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


encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[n_epochs//3], gamma=0.1)
predict_scheduler = torch.optim.lr_scheduler.MultiStepLR(predict_optimizer, milestones=[n_epochs//3], gamma=0.1)
generate_scheduler = torch.optim.lr_scheduler.MultiStepLR(generate_optimizer, milestones=[n_epochs//3], gamma=0.1)
merge_scheduler = torch.optim.lr_scheduler.MultiStepLR(merge_optimizer, milestones=[n_epochs//3], gamma=0.1)

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

# 注意step_size变小一倍
# if dataset=="APE":
#     encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=10, gamma=0.5)
#     predict_scheduler = lr_scheduler.StepLR(predict_optimizer, step_size=10, gamma=0.5)
#     generate_scheduler = lr_scheduler.StepLR(generate_optimizer, step_size=10, gamma=0.5)
#     merge_scheduler = lr_scheduler.StepLR(merge_optimizer, step_size=10, gamma=0.5)
# else:
#     encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
#     predict_scheduler = lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
#     generate_scheduler = lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
#     merge_scheduler = lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

# 
fold=0
start_epoch=1
last_acc=0.0
best_acc_fold=[[0,0,2316]]


# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])


best_acc=0
last_best_acc=0
for epoch in range(start_epoch, n_epochs):
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)

    all_len   = len(input_lengths)
    range_len = range(all_len)
             
    kl_loss_total_1 = 0
    loss_total_no_prue = 0

    kl_loss_total_2 = 0
    loss_total_prue = 0

    start = time.time()
    for idx in tqdm(range_len):#range_len:
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



    L = len(input_lengths)    
    print("loss_1:{} contra_loss_1:{} loss_2:{} contra_loss_2:{} loss type:{}".format(loss_total_prue / L, kl_loss_total_1 / L, loss_total_no_prue / L, kl_loss_total_2 / L, config.RDloss))
    
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    if (epoch-1) % config.test_interval == 0 or (epoch-1) > n_epochs - 5:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        global_unstructured_flag(parameters_to_prune, config.is_prune2test)

        for test_batch in tqdm(test_pairs):
            
            # batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5]) 
            
            # g2t中加入的图的特征
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

        print("dropout:{} contra_weight:{} is_mask:{} is_em_dropout:{} is_prune2test:{} prunePercent:{} embedding_size:{} hidden_size:{} warm_up_strategy:{} model_name:{} batch_size:{} USE_APE_word:{} quantity_num_ape:{} learning_rate:{}".format(config.dropout, config.contra_weight, config.is_mask, config.is_em_dropout, config.is_prune2test, config.prunePercent, config.embedding_size, config.hidden_size,  config.warm_up_stratege, config.MODEL_NAME, batch_size, config.USE_APE_word, config.quantity_num_ape, config.learning_rate))
        print("------------------------------------------------------")
        curr_acc=round(float(value_ac)/eval_total,4)
        if curr_acc>best_acc:
            last_acc = best_acc
            best_acc = curr_acc
            
            torch.save(encoder.state_dict(), "models/no_p2t/encoder")
            torch.save(predict.state_dict(), "models/no_p2t/predict")
            torch.save(generate.state_dict(), "models/no_p2t/generate")
            torch.save(merge.state_dict(), "models/no_p2t/merge")
        else:
            print("break early stoping=================================")
            break

        if epoch == n_epochs - 1:
            best_acc_fold.append((equation_ac, value_ac, eval_total))


value_ac = 0
equation_ac = 0
eval_total = 0
start = time.time()


encoder.load_state_dict(torch.load("models/no_p2t/encoder"))
predict.load_state_dict(torch.load("models/no_p2t/predict"))
generate.load_state_dict(torch.load("models/no_p2t/generate"))
merge.load_state_dict(torch.load("models/no_p2t/merge"))



parameters_to_prune = []
count = 0
for models in [encoder.named_modules(), predict.named_modules(), generate.named_modules(), merge.named_modules()] :#+ predict.named_modules() + generate.named_modules()+ merge.named_modules():
    for name, module in models:
        if hasattr(module, 'weight'):
            parameters_to_prune.append((module, 'weight'))
            count+=1


global_unstructured_flag(parameters_to_prune, config.is_prune2test)#False)

for test_batch in valid_pairs:
    # batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5]) 
    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                     merge, output_lang, test_batch[5], beam_size=beam_size)
    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    if val_ac:
        value_ac += 1
    if equ_ac:
        equation_ac += 1
    eval_total += 1
print(equation_ac, value_ac, eval_total)
print("valid_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
print("valid time", time_since(time.time() - start))
print("------------------------------------------------------")
