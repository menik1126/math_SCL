import torch
import torch.nn as nn
from transformers import AutoModel

from src import config
USE_CUDA = torch.cuda.is_available()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=config.dropout):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=config.dropout):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        """
            input_size: hidden_size * 2
            hidden_size: hidden_size
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        """
            hidden:          当前节点的隐状态embedding B x 1 x 2 * hidden_size 包含注意力信息
            num_embeddings:  Batch_size x [max(num_size_batch) + generate_num] x hidden_size
            num_mask:        Batch_size * [max(num_size_batch) + len(generate_nums)]
        """
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x 2* hidden_size

        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        # B x O x (Hidden_size +  2 * hidden_size) -> (B x O) x 3* hidden_size
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)

        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)  
        # B x O 返回mask后的分数   
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, n_layers=2, dropout=config.dropout):
        super(EncoderSeq, self).__init__()

        self.input_size = input_size
        #self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        # self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        if config.is_em_dropout:
           self.em_dropout = nn.Dropout(dropout)
        # self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.enc = AutoModel.from_pretrained(config.MODEL_NAME)
        self.enc.resize_token_embeddings(vocab_size)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(self.enc.config.hidden_size, self.hidden_size))
        # 还加了nn.ReLU和nn.Linear层

    def forward(self, input_seqs, input_lengths, hidden=None, mask=None):
        # # Note: we run this all at once (over multiple batches of multiple sequences)
        # embedded = self.embedding(input_seqs)  # S x B x E
        # embedded = self.em_dropout(embedded)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # pade_hidden = hidden
        # pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        # pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        # problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        # pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        # pade_outputs = self.enc(**input_seqs)[0]
        # problem_output = self.sum(pade_outputs)

        input_seqs = input_seqs.transpose(0, 1) # S x B -> B x S

        if mask == None:
            
           pade_outputs = self.enc(input_seqs)[0]  # B x S x E
           if config.is_em_dropout:
              self.em_dropout(pade_outputs)
        # 传入mask 
        else:
           
           float_mask = 1 - mask.float()
        #    print("float_mask:{}".format(float_mask))
           pade_outputs = self.enc(input_seqs, attention_mask = float_mask)[0]
           if config.is_em_dropout:
              self.em_dropout(pade_outputs)
           

        pade_outputs = self.out(pade_outputs)   # B x S x H

        pade_outputs, problem_output = pade_outputs, pade_outputs[:,0,:]
        #print("pade_outputs shape:{}".format(pade_outputs.shape) )

        pade_outputs = pade_outputs.transpose(0, 1)   # S x B x H
        return pade_outputs, problem_output


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=config.dropout):
        super(Prediction, self).__init__()
        """
             input_size: len(generate_nums) 
        """
        print("len(generate_nums):{}".format(input_size))
        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size

        # 操作符的数目
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)
        
        # 相当于generate_nums的embedding参数 input_size = 2
        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)
        
        # 这里有个attention模块
        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        """
            node_stacks: batch_size (TreeNode:里面包含一个embedding和是否有左孩子的标记)  保存各个待遍历的节点

            padding_hidden: 这个padding_hidden不知道是干啥的 树上的hidden_dim, 初始节点是
            torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

            left_childs: 传入batch_size个None节点
            all_nums_encoder_outputs: num_pades (batch_size * num_size * hidden_size)  num_size: 该batch中num_pos最大数目
            从encoder_outputs中选出的数字的embedding

            encoder_outputs: S x B x H
            mask_nums: Batch_size * [max(num_size_batch) + len(generate_nums)]
        """

        # 里面保存各个结点当前的embedding
        current_embeddings = []  
        
        # 遍历整个batch
        for st in node_stacks:
            if len(st) == 0:
                
                # 第一步运算的初始化
                current_embeddings.append(padding_hidden)
            else:

                #
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)
        
        current_node_temp = []

        # 遍历当前这个batch , 要考虑左孩子的信息
        # 相当于lstm里面遗忘门那些信息, 还没合并条件概率中的ground_truth信息
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)
        
        # Batch_size * padding_hidden_size
        current_node = torch.stack(current_node_temp)

        # 经过drop_out得到embedding
        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)

        # 得到上下文向量
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # The information to get the current quantity (有可能有节点遍历结束)
        batch_size = current_embeddings.size(0)

        # predict the output (this node corresponding to output(number or operator)) with PADE
        # 总共3个维度
        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size

        
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)       # 1 x input_size x hidden_size -> B x input_size x hidden_size
                                                                            # num_pades : batch_size * num_size * hidden_size
        """
            num_pades: 是encoder的embedding (batch_size * num_size * hidden_size)
        """
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N   O = (num_size + generate_num)
        
        
        embedding_weight_ = self.dropout(embedding_weight)

        """
            current_node:      Batch_size x 1 x padding_hidden_size
            current_context:   Batch_size x 1 x N
            mask_nums:         Batch_size * [max(num_size_batch) + len(generate_nums)]
            embedding_weight_: encoder输出和当前embedding_weight的并联
        """
        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        # 传入当前的叶子结点和所有的embedding和对score的mask, 然后得到B x O的分数分布
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)
        
        # hidden_size * 2 x op_nums
        op = self.ops(leaf_input)
        
        """
            num_score:        返回B x O的数字的分数分布
            op:               返回B x OP的操作符的分数分布
            current_node:     返回B x hidden_size
            current_context:  返回B x 1 x hidden_size
            embedding_weight: 返回B x O x hidden_size 这是返回的原文的(num_size + generate_size)  encoder_embedding的输出并联上当前的模块的参数
        """
        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=config.dropout):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        # 各个操作符的embedding
        self.embeddings = nn.Embedding(op_nums, embedding_size)
        print("In GenerateNode and the number of op is:{}".format(op_nums))

        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
    

    def sample_prev_y(self, node_label_, y_tm1_model=None, ss_prob=1.):

        batch_size = node_label_.size(0)
        # print("batch_size:{}".format(batch_size))
        
        # oracle_embedding
        y_tm1_oracle = self.embeddings(y_tm1_model)  # word-level oracle (w/o w/ noise)
           

        # pick gold with the probability of ss_prob
        with torch.no_grad():
            _g = torch.bernoulli( ss_prob * torch.ones(batch_size, 1, requires_grad=False) )
            if USE_CUDA:
                 _g = _g.cuda()
            y_tm1 = node_label_ * _g + y_tm1_oracle * (1. - _g)
        
        return y_tm1

    def forward(self, node_embedding, node_label, current_context, teacher_forcing_ratio = None, y_tm1_model = None, is_train = False):
        """
            node_embedding:  返回B x 1 x hidden_size   (当前结点的embedding)
            node_label:      ground_truth
                             返回B X 1 即输出target当前step的, 标记输入中所有的数字为0, 非数字为相应操作符的标记(即操作符)
                             这里数字的标签已经被处理过了全部变成了第一个embedding

            current_context: 返回B x 1 x hidden_size   当前的节点与输入节点计算得出的上下文
        """
        # print("node_label shape:{} node_label:{}".format(node_label.shape, node_label))

        # 这里直接用ground_truth预测左右孩子的信息
        if config.is_exposure_bias:

             node_label_ = self.embeddings(node_label)
             """
                 node_label_: ground_truth embedding
                 y_tm1_model: oracle embedding
             """
             if is_train:
                node_label_ = self.sample_prev_y(node_label_, y_tm1_model, ss_prob = teacher_forcing_ratio)

             
        else:
             node_label_ = self.embeddings(node_label)
        
        # print("node_label_ shape:{}".format(node_label_.shape))
        # print("current_context shape:{}".format(current_context.shape))
        
        node_label = self.em_dropout(node_label_)

        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))

        l_child = l_child * l_child_g
        r_child = r_child * r_child_g

        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=config.dropout):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        """
             op.embedding:            操作符的embedding

             sub_stree.embedding:     子树的embedding
             current_num:             筛选出当前的数字的embedding
        """
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)

        """
             node_embedding:          当前操作符
        """
        node_embedding = self.em_dropout(node_embedding)
        
        
        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))

        # print("sub_tree_g:{}".format(sub_tree_g))
        sub_tree = sub_tree * sub_tree_g
        """
             根据两个子树和运算符的embedding生成新的数字
        """
        return sub_tree
