from src.masked_cross_entropy_prune import *
from src.pre_data_position import *


from src.expressions_transfer import *
from src.models_prune import *
from src.prune_method import *

import math
import torch
import torch.optim
import torch.nn.functional as f
import time
import numpy as np
MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
                       english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in generate_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["["], word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["["] or decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["]"]:
                res += [word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["EOS"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"],
                                      word2index["*"], word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["["], word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"],
                        word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums +\
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    """
        top1: 上一步的预测输出
        num_start: 数字在词表中的起始位置
    """

   
    target_input = copy.deepcopy(target)
    # print("target:{} \n nums_stack_batch:{}".format(len(target), len(nums_stack_batch)))
    
    
    
    # 遍历所有的batch
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0

    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # BoolTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.BoolTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    # print("num_size:{}".format(num_size))
    return all_num.masked_fill_(masked_index, 0.0)


def train_attn(input_batch, input_length, target_batch, target_length, num_batch, nums_stack_batch, copy_nums,
               generate_nums, encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang, clip=0,
               use_teacher_forcing=1, beam_size=1, english=False):
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.BoolTensor(seq_mask)

    num_start = output_lang.n_words - copy_nums - 2
    unk = output_lang.word2index["UNK"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)

    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_length, None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)

    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk)
            target[t] = decoder_input
    else:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                                               num_start, copy_nums, generate_nums, english)
                if USE_CUDA:
                    rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output
            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], nums_stack_batch, num_start, unk)
    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()
    
    
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    )

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    if clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return return_loss


def evaluate_attn(input_seq, input_length, num_list, copy_nums, generate_nums, encoder, decoder, output_lang,
                  beam_size=1, english=False, max_length=MAX_OUTPUT_LENGTH):
    seq_mask = torch.BoolTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 2

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(1)
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, [input_length], None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]])  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    beam_list = list()
    score = 0
    beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

    # Run through decoder
    for di in range(max_length):
        temp_list = list()
        beam_len = len(beam_list)
        for xb in beam_list:
            if int(xb.input_var[0]) == output_lang.word2index["EOS"]:
                temp_list.append(xb)
                beam_len -= 1
        if beam_len == 0:
            return beam_list[0].all_output
        beam_scores = torch.zeros(decoder.output_size * beam_len)
        hidden_size_0 = decoder_hidden.size(0)
        hidden_size_2 = decoder_hidden.size(2)
        all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
        if USE_CUDA:
            beam_scores = beam_scores.cuda()
            all_hidden = all_hidden.cuda()
        all_outputs = []
        current_idx = -1

        for b_idx in range(len(beam_list)):
            decoder_input = beam_list[b_idx].input_var
            if int(decoder_input[0]) == output_lang.word2index["EOS"]:
                continue
            current_idx += 1
            decoder_hidden = beam_list[b_idx].hidden

            # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
            #                                1, num_start, copy_nums, generate_nums, english)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
            score = f.log_softmax(decoder_output, dim=1)
            score += beam_list[b_idx].score
            beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
            all_hidden[current_idx] = decoder_hidden
            all_outputs.append(beam_list[b_idx].all_output)
        topv, topi = beam_scores.topk(beam_size)

        for k in range(beam_size):
            word_n = int(topi[k])
            word_input = word_n % decoder.output_size
            temp_input = torch.LongTensor([word_input])
            indices = int(word_n / decoder.output_size)

            temp_hidden = all_hidden[indices]
            temp_output = all_outputs[indices]+[word_input]
            temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0].all_output


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal

def train_one_batch(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos, generate_nums, output_lang,
                                       encoder, predict, generate, merge):
     # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.BoolTensor(seq_mask)
    
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
 
    

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)
    
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    
    
    
    # if config.is_exposure_bias:
    #     classifier.train()
    #     classifier_optimizer.zero_grad()

    # Run words through encoder
    
    if config.is_mask:
       encoder_outputs, problem_output = encoder(input_var, input_length, mask = seq_mask)
    else:
       encoder_outputs, problem_output = encoder(input_var, input_length)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)
    
    # 用来逼近最后的argmax参数
    all_node_outputs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)
    
    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    target = torch.LongTensor(target_batch).transpose(0, 1)

   
    for t in range(max_target_length):
        """
           current_nums_embeddings :来自于encoder的数字的embedding和predict模块的输出拼接 B x O x hidden_size 这是返回的原文的(num_size + generate_size)
           node_stacks: 包含了上个操作符结点生成的左右孩子信息
           left_childs: 传递上一步左孩子的运算结果

           current_embeddings: 结点自己对自己当前结点的预测值
        """
        # 这个函数都是embedding的软操作包括注意力机制, 没有输入ground_truth
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
        
        # all_leafs.append(p_leaf)

        # 对操作符和数字一起预测
        outputs = torch.cat((op, num_score), 1)
        # print("outputs shape:{}".format(outputs.shape))

        all_node_outputs.append(outputs)
        # print("Target_batch[t] shape:{}".format(np.array(target[t]).shape))
        
        #print("nums_stack_batch:{}".format(nums_stack_batch))
        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)


        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()

        # 在这里生成左结点和右节点, 及当前结点的标签
        
        """
           generate_input是对target的拷贝: 是当前结点label的ground_truth
           表达式树中的每个节点n由三个主要组件组成:目标向量q、令牌ˆy和子树嵌入t of n的子树
           目标向量用来预测标签, 该标签用来决定当前结点是否要继续分解, 如果预测的令牌是一个数学运算符，目标将被该运算符分解为两个子目标
        """

        # 每次固定预测出左右孩子结点, 这里是预测出来的embedding, 不是ground_truth指定的embedding
        # 只有左右孩子才是生成的, node_label是查询出来的embedding
        if config.is_exposure_bias:
             # op: (B x OP)     
             #logit_op = classifier(op, noise=config.greed_gumbel_noise)
             y_tm1_model = op.max(-1)[1]

             #logit_num = classifier(num_score, noise=config.greed_gumbel_noise, is_num = True)
             y_tm1_model_num = num_score.max(-1)[1] #logit_num.max(-1)[1]
             
              #= schedule_sampling(idx, range_len, config.MODE)
             
             left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context, teacher_forcing_ratio = teacher_forcing_ratio, y_tm1_model = y_tm1_model, is_train = is_train)
             

        else:
             left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        """
           l_child：Batch_size x hidden_size
           r_child: Batch_size x hidden_size
           node_label: Batch_size x embedding_size  这个是操作符的groud_truth的embedding
        """
         
        # 后面预测的时候需要左孩子的信息 Batch_size * embedding_size
        # 左孩子的信息是每次固定更新
        left_childs = []

        # 遍历整个batch
        # embeddings_stacks保证了整个前缀计算的embedding顺序
        """
            node_stacks: 保存的结点的当前状态, 每次弹出一次表示访问到当前结点
            embeddings_stacks: 保存的结点的历史状态, 里面结点都已经被访问过了
        """
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):

            """
                node_stacks:
            """
            if len(node_stack) != 0:
                # 进行弹出堆栈的操作
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue
            
            # 此处是操作符
            if i < num_start:
                
                # 扩展子树
                """
                    l  : Batch_size x hidden_size
                    r  : Batch_size x hidden_size
                    这里会不停的更新左右孩子
                """
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))

                # 把当前操作符的embedding追进embeddings_stack
                """
                    TreeEmbedding: 保存有当前结点的embedding, 以及是否被标记为terminal

                """
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            
            # 此处是数字
            else:
                # current_num这个只来自于groud_truth提供的标签
                # 这个i也是ground_truth,  这个是数字的ground_truth
                # current_nums_embeddings是待查询的embedding层
                
                #print("这里的i:{} num_start的值是:{}".format(i, num_start))
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                if config.is_exposure_bias and is_train:
                   # oracle_embedding
                   y_tm1_oracle = current_nums_embeddings[idx, y_tm1_model_num[idx] - num_start].unsqueeze(0) 
                   # pick gold with the probability of ss_prob
                   with torch.no_grad():
                        _g = torch.bernoulli( teacher_forcing_ratio * torch.ones(1, requires_grad=False) )
                        if USE_CUDA:
                            _g = _g.cuda()
                        current_num = current_num * _g + y_tm1_oracle * (1. - _g)


                # 在此处弹出结点, 凑齐current_num 和 o[-1].terminal 
                while len(o) > 0 and o[-1].terminal:
                    # 这个不知道是左孩子还是右孩子
                    sub_stree = o.pop()
                    op = o.pop()
                    
                    # 根据ground_truth, 当前的结点由merge模块得出, 这里可以不段运算
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    

                o.append(TreeEmbedding(current_num, True))

            if len(o) > 0 and o[-1].terminal:
                # 全靠左孩子追加运算结果
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    return all_node_outputs, target

def train_prune(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos, generate_nums, output_lang,
                                encoder, predict, generate, merge,
                                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
                                parameters_to_prune,
                                pos = 0, prune_start = True):
    
    # print("len of input_batch:{}".format(len(input_batch)))
    # Zero gradients of both optimizers
    

    #calculate the grad for non-pruned network
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    if parameters_to_prune is not None:
       # 设置开启剪枝模式下进行运算
       global_unstructured_flag(parameters_to_prune, prune_start)
       with torch.no_grad():

            # branch with pruned network
            all_node_outputs2, target_noGrad = train_one_batch(input_batch[pos], input_length[pos], target_batch[pos], target_length[pos], nums_stack_batch[pos], num_size_batch[pos], num_pos[pos], generate_nums, output_lang,
                                            encoder, predict, generate, merge) 
            all_node_outputs2_noGrad = torch.stack(all_node_outputs2, dim=1).detach()
    
    
    global_unstructured_flag(parameters_to_prune, bool(1-prune_start))

    
    # 这个是总的loss
    #print("parameters_to_prune2:{}".format(parameters_to_prune[1]))
    all_node_outputs1, target = train_one_batch(input_batch[1-pos], input_length[1-pos], target_batch[1-pos], target_length[1-pos], nums_stack_batch[1-pos], num_size_batch[1-pos], num_pos[1-pos], generate_nums, output_lang,
                                       encoder, predict, generate, merge)
    all_node_outputs1 = torch.stack(all_node_outputs1, dim=1)  # B x S x N

    
    # 再转置回来
    target = target.transpose(0, 1).contiguous()
    #target_noGrad = target_noGrad.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs1 = all_node_outputs1.cuda()
        target = target.cuda()
        #target_noGrad = target_noGrad.cuda()

   
    loss, kl_loss = masked_cross_entropy(all_node_outputs1, target, target_length[1-pos], logits_noGrad= all_node_outputs2_noGrad, target_noGrad = target_noGrad, length_noGrad = target_length[pos], temperature = config.temperature)
    

    loss.backward()
    
    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()

    return loss, kl_loss



def pre_double_data(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos):
    return   (input_batch, input_batch), (input_length, input_length), (target_batch, target_batch), (target_length, target_length), (nums_stack_batch, copy.deepcopy(nums_stack_batch)), (num_size_batch, num_size_batch), (num_pos, num_pos)

def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, 
               encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, 
               output_lang, num_pos, 
               teacher_forcing_ratio = None, classifier = None, classifier_optimizer = None, englishis_train = False, is_train = False,
               parameters_to_prune = None): # 
    # print("classifier:{}".format(classifier))
    
    # print("nums_stack_batch:{}".format(nums_stack_batch))
    # 将这个batch的数据扩增一倍
    if config.is_RDrop and is_train:
       input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos= pre_double_data(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos)
       
    
    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    

    nums_stack_batch_copy = copy.deepcopy(nums_stack_batch)
    
    #已被交换顺序, 记得换回来 
    
    
    

    

    loss_no_prue, kl_loss1 = train_prune(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos, generate_nums, output_lang,
                                encoder, predict, generate, merge,
                                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
                                parameters_to_prune,
                                pos = 0, prune_start = True)

    loss_prue, kl_loss2 = train_prune(input_batch, input_length, target_batch, target_length, nums_stack_batch_copy, num_size_batch, num_pos, generate_nums, output_lang,
                                encoder, predict, generate, merge,
                                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
                                parameters_to_prune,
                                pos = 1, prune_start = False)
   
    
    
   
    return loss_no_prue.item(), kl_loss1.item() , loss_prue.item(), kl_loss2.item()     # loss_no_prue.item(), kl_loss1



def train_prune_SWS(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos, generate_nums, output_lang,
                                encoder, predict, generate, merge,
                                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
                                parameters_to_prune,
                                pos = 0, prune_start = True):
    
    # print("len of input_batch:{}".format(len(input_batch)))
    # Zero gradients of both optimizers
    

    #calculate the grad for non-pruned network
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    if parameters_to_prune is not None:
       # 设置开启剪枝模式下进行运算
       #global_unstructured_flag(parameters_to_prune, prune_start)
       with torch.no_grad():

            # branch with pruned network
            all_node_outputs2, target_noGrad = train_one_batch(input_batch[pos], input_length[pos], target_batch[pos], target_length[pos], nums_stack_batch[pos], num_size_batch[pos], num_pos[pos], generate_nums, output_lang,
                                            encoder, predict, generate, merge) 
            all_node_outputs2_noGrad = torch.stack(all_node_outputs2, dim=1).detach()
    
    
    #global_unstructured_flag(parameters_to_prune, prune_start)

    
    # 这个是总的loss
    #print("parameters_to_prune2:{}".format(parameters_to_prune[1]))
    all_node_outputs1, target = train_one_batch(input_batch[1-pos], input_length[1-pos], target_batch[1-pos], target_length[1-pos], nums_stack_batch[1-pos], num_size_batch[1-pos], num_pos[1-pos], generate_nums, output_lang,
                                       encoder, predict, generate, merge)
    all_node_outputs1 = torch.stack(all_node_outputs1, dim=1)  # B x S x N

    
    # 再转置回来
    target = target.transpose(0, 1).contiguous()
    #target_noGrad = target_noGrad.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs1 = all_node_outputs1.cuda()
        target = target.cuda()
        #target_noGrad = target_noGrad.cuda()

   
    loss, kl_loss = masked_cross_entropy(all_node_outputs1, target, target_length[1-pos], logits_noGrad= all_node_outputs2_noGrad, target_noGrad = target_noGrad, length_noGrad = target_length[pos], temperature = config.temperature)
    

    loss.backward()
    
    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()

    return loss, kl_loss

def train_prune_SWS_divide(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos, generate_nums, output_lang,
                                encoder, predict, generate, merge,
                                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,

                                encoder1, predict1, generate1, merge1,
                                
                                pos = 0, prune_start = True):
    
    # print("len of input_batch:{}".format(len(input_batch)))
    # Zero gradients of both optimizers
    

    #calculate the grad for non-pruned network
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
   
    with torch.no_grad():

        # branch with pruned network
        all_node_outputs2, target_noGrad = train_one_batch(input_batch[pos], input_length[pos], target_batch[pos], target_length[pos], nums_stack_batch[pos], num_size_batch[pos], num_pos[pos], generate_nums, output_lang,
                                        encoder1, predict1, generate1, merge1) 
        all_node_outputs2_noGrad = torch.stack(all_node_outputs2, dim=1).detach()
    
    
    #global_unstructured_flag(parameters_to_prune, prune_start)

    
    # 这个是总的loss
    #print("parameters_to_prune2:{}".format(parameters_to_prune[1]))
    all_node_outputs1, target = train_one_batch(input_batch[1-pos], input_length[1-pos], target_batch[1-pos], target_length[1-pos], nums_stack_batch[1-pos], num_size_batch[1-pos], num_pos[1-pos], generate_nums, output_lang,
                                       encoder, predict, generate, merge)
    all_node_outputs1 = torch.stack(all_node_outputs1, dim=1)  # B x S x N

    
    # 再转置回来
    target = target.transpose(0, 1).contiguous()
    #target_noGrad = target_noGrad.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs1 = all_node_outputs1.cuda()
        target = target.cuda()
        #target_noGrad = target_noGrad.cuda()

   
    loss, kl_loss = masked_cross_entropy(all_node_outputs1, target, target_length[1-pos], logits_noGrad= all_node_outputs2_noGrad, target_noGrad = target_noGrad, length_noGrad = target_length[pos], temperature = config.temperature)
    

    loss.backward()
    
    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()

    return loss, kl_loss



def train_tree_SWS_divide(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, 
               encoder1, predict1, generate1, merge1,
               encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, 
               encoder_optimizer1, predict_optimizer1, generate_optimizer1, merge_optimizer1,
               output_lang, num_pos, 
               teacher_forcing_ratio = None, classifier = None, classifier_optimizer = None, englishis_train = False, is_train = False,
               parameters_to_prune = None,
               alternate_flag = None): # 
    # print("classifier:{}".format(classifier))
    
    # print("nums_stack_batch:{}".format(nums_stack_batch))
    # 将这个batch的数据扩增一倍
    if config.is_RDrop and is_train:
       input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos= pre_double_data(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos)
       
    
    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    

    nums_stack_batch_copy = copy.deepcopy(nums_stack_batch)
    
    #已被交换顺序, 记得换回来 

    
    if alternate_flag == True:
        loss_no_prue, kl_loss1 = train_prune_SWS_divide(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos, generate_nums, output_lang,
                                    encoder, predict, generate, merge,
                                    encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
                                    encoder1, predict1, generate1, merge1,
                                    pos = 0, prune_start = False)

        loss_prue, kl_loss2 = train_prune_SWS_divide(input_batch, input_length, target_batch, target_length, nums_stack_batch_copy, num_size_batch, num_pos, generate_nums, output_lang,
                                    encoder1, predict1, generate1, merge1,
                                    encoder_optimizer1, predict_optimizer1, generate_optimizer1, merge_optimizer1,
                                    encoder, predict, generate, merge,
                                    pos = 1, prune_start = False)
    else:
        loss_prue, kl_loss2 = train_prune_SWS_divide(input_batch, input_length, target_batch, target_length, nums_stack_batch_copy, num_size_batch, num_pos, generate_nums, output_lang,
                                    encoder1, predict1, generate1, merge1,
                                    encoder_optimizer1, predict_optimizer1, generate_optimizer1, merge_optimizer1,
                                    encoder, predict, generate, merge,
                                    pos = 1, prune_start = False)
        loss_no_prue, kl_loss1 = train_prune_SWS_divide(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos, generate_nums, output_lang,
                                    encoder, predict, generate, merge,
                                    encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
                                    encoder1, predict1, generate1, merge1,
                                    pos = 0, prune_start = False)

   
    
    
   
    return loss_no_prue.item(), kl_loss1.item() , loss_prue.item(), kl_loss2.item() 


def train_tree_SWS(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, 
               encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, 
               output_lang, num_pos, 
               teacher_forcing_ratio = None, classifier = None, classifier_optimizer = None, englishis_train = False, is_train = False,
               parameters_to_prune = None): # 
    # print("classifier:{}".format(classifier))
    
    # print("nums_stack_batch:{}".format(nums_stack_batch))
    # 将这个batch的数据扩增一倍
    if config.is_RDrop and is_train:
       input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos= pre_double_data(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos)
       
    
    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    

    nums_stack_batch_copy = copy.deepcopy(nums_stack_batch)
    
    #已被交换顺序, 记得换回来 
    
    
    

    

    loss_no_prue, kl_loss1 = train_prune_SWS(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos, generate_nums, output_lang,
                                encoder, predict, generate, merge,
                                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
                                parameters_to_prune,
                                pos = 0, prune_start = False)

    loss_prue, kl_loss2 = train_prune_SWS(input_batch, input_length, target_batch, target_length, nums_stack_batch_copy, num_size_batch, num_pos, generate_nums, output_lang,
                                encoder, predict, generate, merge,
                                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
                                parameters_to_prune,
                                pos = 1, prune_start = False)
   
    
    
   
    return loss_no_prue.item(), kl_loss1.item() , loss_prue.item(), kl_loss2.item()     



def train_tree_comp(input_batch, input_length, target_batch, target_length, 
               nums_stack_batch, num_size_batch, 
               generate_nums, encoder, predict, generate, merge, 
               encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, 
               output_lang, num_pos, 
               teacher_forcing_ratio = None, classifier = None, classifier_optimizer = None, englishis_train = False, is_train = False): # 
    # print("classifier:{}".format(classifier))

    # 将这个batch的数据扩增一倍
    #print("nums_stack_batch:{}".format(nums_stack_batch))
    if config.is_RDrop :#and is_train:
       input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos= pre_double_data(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_pos)
       
    # print("After double input_batch:{}   nums_stack_batch:{}".format(len(input_batch), len(nums_stack_batch)))

    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.BoolTensor(seq_mask)
    
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
 
    

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)
    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    
    # if config.is_exposure_bias:
    #     classifier.train()
    #     classifier_optimizer.zero_grad()

    # Run words through encoder
    
    if config.is_mask:
       encoder_outputs, problem_output = encoder(input_var, input_length, mask = seq_mask)
    else:
       encoder_outputs, problem_output = encoder(input_var, input_length)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)
    
    # 用来逼近最后的argmax参数
    all_node_outputs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)
    
    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    target = torch.LongTensor(target_batch).transpose(0, 1)

   
    for t in range(max_target_length):
        """
           current_nums_embeddings :来自于encoder的数字的embedding和predict模块的输出拼接 B x O x hidden_size 这是返回的原文的(num_size + generate_size)
           node_stacks: 包含了上个操作符结点生成的左右孩子信息
           left_childs: 传递上一步左孩子的运算结果

           current_embeddings: 结点自己对自己当前结点的预测值
        """
        # 这个函数都是embedding的软操作包括注意力机制, 没有输入ground_truth
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
        
        # all_leafs.append(p_leaf)

        # 对操作符和数字一起预测
        outputs = torch.cat((op, num_score), 1)
        # print("outputs shape:{}".format(outputs.shape))

        all_node_outputs.append(outputs)
        # print("Target_batch[t] shape:{}".format(np.array(target[t]).shape))
        
        
        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)


        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()

        # 在这里生成左结点和右节点, 及当前结点的标签
        
        """
           generate_input是对target的拷贝: 是当前结点label的ground_truth
           表达式树中的每个节点n由三个主要组件组成:目标向量q、令牌ˆy和子树嵌入t of n的子树
           目标向量用来预测标签, 该标签用来决定当前结点是否要继续分解, 如果预测的令牌是一个数学运算符，目标将被该运算符分解为两个子目标
        """

        # 每次固定预测出左右孩子结点, 这里是预测出来的embedding, 不是ground_truth指定的embedding
        # 只有左右孩子才是生成的, node_label是查询出来的embedding
        if config.is_exposure_bias:
             # op: (B x OP)     
             #logit_op = classifier(op, noise=config.greed_gumbel_noise)
             y_tm1_model = op.max(-1)[1]

             #logit_num = classifier(num_score, noise=config.greed_gumbel_noise, is_num = True)
             y_tm1_model_num = num_score.max(-1)[1] #logit_num.max(-1)[1]
             
              #= schedule_sampling(idx, range_len, config.MODE)
             
             left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context, teacher_forcing_ratio = teacher_forcing_ratio, y_tm1_model = y_tm1_model, is_train = is_train)
             

        else:
             left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        """
           l_child：Batch_size x hidden_size
           r_child: Batch_size x hidden_size
           node_label: Batch_size x embedding_size  这个是操作符的groud_truth的embedding
        """
         
        # 后面预测的时候需要左孩子的信息 Batch_size * embedding_size
        # 左孩子的信息是每次固定更新
        left_childs = []

        # 遍历整个batch
        # embeddings_stacks保证了整个前缀计算的embedding顺序
        """
            node_stacks: 保存的结点的当前状态, 每次弹出一次表示访问到当前结点
            embeddings_stacks: 保存的结点的历史状态, 里面结点都已经被访问过了
        """
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):

            """
                node_stacks:
            """
            if len(node_stack) != 0:
                # 进行弹出堆栈的操作
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue
            
            # 此处是操作符
            if i < num_start:
                
                # 扩展子树
                """
                    l  : Batch_size x hidden_size
                    r  : Batch_size x hidden_size
                    这里会不停的更新左右孩子
                """
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))

                # 把当前操作符的embedding追进embeddings_stack
                """
                    TreeEmbedding: 保存有当前结点的embedding, 以及是否被标记为terminal

                """
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            
            # 此处是数字
            else:
                # current_num这个只来自于groud_truth提供的标签
                # 这个i也是ground_truth,  这个是数字的ground_truth
                # current_nums_embeddings是待查询的embedding层
                
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                if config.is_exposure_bias and is_train:
                   # oracle_embedding
                   y_tm1_oracle = current_nums_embeddings[idx, y_tm1_model_num[idx] - num_start].unsqueeze(0) 
                   # pick gold with the probability of ss_prob
                   with torch.no_grad():
                        _g = torch.bernoulli( teacher_forcing_ratio * torch.ones(1, requires_grad=False) )
                        if USE_CUDA:
                            _g = _g.cuda()
                        current_num = current_num * _g + y_tm1_oracle * (1. - _g)


                # 在此处弹出结点, 凑齐current_num 和 o[-1].terminal 
                while len(o) > 0 and o[-1].terminal:
                    # 这个不知道是左孩子还是右孩子
                    sub_stree = o.pop()
                    op = o.pop()
                    
                    # 根据ground_truth, 当前的结点由merge模块得出, 这里可以不段运算
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    

                o.append(TreeEmbedding(current_num, True))

            if len(o) > 0 and o[-1].terminal:
                # 全靠左孩子追加运算结果
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    # 这个是总的loss
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    # 在这个地方加入mask
    # kl_loss = compute_kl_loss(all_node_outputs, )
    
    # 再转置回来
    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    # print("decoder output shape:{}".format(all_node_outputs.shape))


    if config.is_RDrop:
       loss, kl_loss = masked_cross_entropy(all_node_outputs, target, target_length)
    #    print("loss:{} kl_loss:{}".format(loss, kl_loss))
    else:
       loss, _ = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    
  

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    
    # if config.is_exposure_bias:
    #    classifier_optimizer.step()

    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos,
                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.BoolTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.BoolTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder
    if config.is_mask:
        encoder_outputs, problem_output = encoder(input_var, [input_length], mask = seq_mask)
    else:
        encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out


def topdown_train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch,
                       generate_nums, encoder, predict, generate, encoder_optimizer, predict_optimizer,
                       generate_optimizer, output_lang, num_pos, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.BoolTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        for idx, l, r, node_stack, i in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                            node_stacks, target[t].tolist()):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def topdown_evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, output_lang, num_pos,
                          beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.BoolTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.BoolTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder
    if config.is_mask:
       encoder_outputs, problem_output = encoder(input_var, [input_length], mask = seq_mask)
    else:
       encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, embeddings_stacks, left_childs,
                                              current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out
