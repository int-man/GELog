import math
from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionEncoding(nn.Embedding):#使用了可学习的权重来表示位置编码，而不是固定的编码方式。
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__(max_len, d_model)#创建了一个形状为 (max_len, d_model) 的Embedding层，该层的权重是可学习的位置编码。
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(0)
        #通过 self.weight.data 获取位置编码的权重，并在第一个维度上添加一个维度，使其形状变为 (1, max_len, d_model)，以便与输入张量 x 进行相加。
        x = x + weight
        return self.dropout(x)


class AugementationAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_emd, num_heads=8):
        super(AugementationAttention, self).__init__()
        assert dim_emd % num_heads == 0 and dim_emd % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_emd, dim_emd, bias=False)
        self.linear_k = nn.Linear(dim_emd, dim_emd, bias=False)
        self.linear_v = nn.Linear(dim_emd, dim_emd, bias=False)
        self._norm_fact = 1 / sqrt(dim_emd // num_heads)
        self.linear_for_params = nn.Linear(dim_emd, dim_emd*2)  # 参数编码层
        self.dim_emd = dim_emd


    def forward(self, x, Augementation_embedding):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_emd

        # 获取参数字符级嵌入
        # chair_embeddings = self.feature_extractor(parsed_params)
          # 参数编码

        nh = self.num_heads
        dk = self.dim_emd // nh  # dim_k of each head
        dv = self.dim_emd // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)
        encoded_params = self.linear_for_params(Augementation_embedding).reshape(batch, n, nh, dv*2).transpose(1, 2)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist+encoded_params, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_emd)  # batch, n, dim_v
        return att



class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=3072, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,**factory_kwargs)#创建了一个多头自注意力层。


        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.dropout1(src2)
        src = self.norm1(src + src2)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(src2)
        src = self.norm2(src + src2)

        return src

    # def activate_adapter(self):
    #     tune_layers = [self.adapter1, self.adapter2, self.norm1, self.norm2]
    #     for layer in tune_layers:
    #         for param in layer.parameters():
    #             param.requires_grad = True


class Model(nn.Module):
    def __init__(self, num_layers=4, dim=768, window_size=100, nhead=8, dim_feedforward=3072, dropout=0.1):
        super(Model, self).__init__()
        #encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward, dropout, batch_first=True)

        #------------------------------------------new------------------------------------------
        self.attention1 = nn.MultiheadAttention(dim,nhead)
        self.attention2 = nn.MultiheadAttention(dim,nhead)
        self.fc = nn.Linear(dim*2, dim)
        # ------------------------------------------new------------------------------------------
        encoder_layer = TransformerEncoderLayer(
                dim, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

        self.trans_encder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers)
        self.pos_encoder1 = PositionalEncoding(d_model=768)
        # self.pos_encoder2 = LearnedPositionEncoding(
        #    d_model=768, max_len=window_size)
        self.fc1 = nn.Linear(dim * window_size, 2)

    def forward(self, x, z):
        B, _, _ = x.size()
        # ------------------------------------------new------------------------------------------
        # print(x.shape)
        # print(z.shape)
        output1, _ = self.attention1(x.transpose(0, 1), z.transpose(0, 1), z.transpose(0, 1))
        output2, _ = self.attention2(z.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        #print(output1.shape)
        output1 = output1.transpose(0, 1)
        output2 = output2.transpose(0, 1)
        output1 = F.layer_norm(x + output1, normalized_shape=[output1.size(-1)])
        output2 = F.layer_norm(z + output2, normalized_shape=[output2.size(-1)])
        fused_embedding = torch.cat((output1, output2), dim=-1)
        fused_embedding = F.relu(self.fc(fused_embedding))
        # ------------------------------------------new------------------------------------------

        fused_embedding = self.trans_encder(fused_embedding)
        fused_embedding = fused_embedding.contiguous().view(B, -1)
        fused_embedding = self.fc1(fused_embedding)
        return fused_embedding


        # #x = self.pos_encoder1(x)
        # # x = self.pos_encoder2(x)
        # x = self.trans_encder(x)  # mask默认None
        #
        # x = x.contiguous().view(B, -1)#将编码器的输出进行形状变换，使其符合线性层的输入要求。
        # #x.contiguous() 的目的是确保张量 x 在进行形状变换之前是连续的，这是因为 view 操作要求被操作的张量是内存连续的。
        #
        # x = self.fc1(x)
        #
        # return x

    # def train_classifier(self):
    #     for param in self.parameters():
    #         param.requires_grad = False
    #
    #     for param in self.fc1.parameters():
    #         param.requires_grad = True



class Model2plot(nn.Module):
    def __init__(self, num_layers=4, dim=768, window_size=100, nhead=8, dim_feedforward=3072, dropout=0.1):
        super(Model2plot, self).__init__()
        #encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward, dropout, batch_first=True)

        #------------------------------------------new------------------------------------------
        self.attention1 = nn.MultiheadAttention(dim,nhead)
        self.attention2 = nn.MultiheadAttention(dim,nhead)
        self.fc = nn.Linear(dim*2, dim)
        # ------------------------------------------new------------------------------------------
        encoder_layer = TransformerEncoderLayer(
                dim, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

        self.trans_encder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers)
        self.pos_encoder1 = PositionalEncoding(d_model=768)
        # self.pos_encoder2 = LearnedPositionEncoding(
        #    d_model=768, max_len=window_size)
        self.fc1 = nn.Linear(dim * window_size, 2)

    def forward(self, x, z):
        B, _, _ = x.size()
        # ------------------------------------------new------------------------------------------
        # print(x.shape)
        # print(z.shape)
        output1, _ = self.attention1(x.transpose(0, 1), z.transpose(0, 1), z.transpose(0, 1))
        output2, _ = self.attention2(z.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        #print(output1.shape)
        output1 = output1.transpose(0, 1)
        output2 = output2.transpose(0, 1)
        output1 = F.layer_norm(x + output1, normalized_shape=[output1.size(-1)])
        output2 = F.layer_norm(z + output2, normalized_shape=[output2.size(-1)])
        fused_embedding = torch.cat((output1, output2), dim=-1)
        fused_embedding = F.relu(self.fc(fused_embedding))
        # ------------------------------------------new------------------------------------------


        fused_embedding = self.trans_encder(fused_embedding)
        fused_embedding = fused_embedding.contiguous().view(B, -1)
        return fused_embedding




class Model_noenchanced(nn.Module):
    def __init__(self, num_layers=4, dim=768, window_size=100, nhead=8, dim_feedforward=3072, dropout=0.1):
        super(Model_noenchanced, self).__init__()
        #encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward, dropout, batch_first=True)

        #------------------------------------------new------------------------------------------
        self.attention1 = nn.MultiheadAttention(dim,nhead)
        self.attention2 = nn.MultiheadAttention(dim,nhead)
        self.fc = nn.Linear(dim*2, dim)
        # ------------------------------------------new------------------------------------------
        encoder_layer = TransformerEncoderLayer(
                dim, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

        self.trans_encder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers)
        self.pos_encoder1 = PositionalEncoding(d_model=768)
        # self.pos_encoder2 = LearnedPositionEncoding(
        #    d_model=768, max_len=window_size)
        self.fc1 = nn.Linear(dim * window_size, 2)

    def forward(self, x, z):
        B, _, _ = x.size()
        # ------------------------------------------new------------------------------------------
        # print(x.shape)
        # print(z.shape)
        # output1, _ = self.attention1(x.transpose(0, 1), z.transpose(0, 1), z.transpose(0, 1))
        # output2, _ = self.attention2(z.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        # #print(output1.shape)
        # output1 = output1.transpose(0, 1)
        # output2 = output2.transpose(0, 1)
        # output1 = F.layer_norm(x + output1, normalized_shape=[output1.size(-1)])
        # output2 = F.layer_norm(z + output2, normalized_shape=[output2.size(-1)])
        fused_embedding = torch.cat((x, z), dim=-1)
        fused_embedding = F.relu(self.fc(fused_embedding))
        # ------------------------------------------new------------------------------------------


        fused_embedding = self.trans_encder(fused_embedding)
        fused_embedding = fused_embedding.contiguous().view(B, -1)
        fused_embedding = self.fc1(fused_embedding)
        return fused_embedding



class Model_xiaorong(nn.Module):
    def __init__(self, num_layers=4, dim=768, window_size=100, nhead=8, dim_feedforward=3072, dropout=0.1):
        super(Model_xiaorong, self).__init__()
        #encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward, dropout, batch_first=True)

        encoder_layer = TransformerEncoderLayer(
                dim, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

        self.trans_encder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers)
        self.pos_encoder1 = PositionalEncoding(d_model=768)
        # self.pos_encoder2 = LearnedPositionEncoding(
        #    d_model=768, max_len=window_size)
        self.fc1 = nn.Linear(dim * window_size, 2)

    def forward(self, x):
        B, _, _ = x.size()
        #x = self.pos_encoder1(x)
        # x = self.pos_encoder2(x)
        x = self.trans_encder(x)  # mask默认None

        x = x.contiguous().view(B, -1)#将编码器的输出进行形状变换，使其符合线性层的输入要求。
        #x.contiguous() 的目的是确保张量 x 在进行形状变换之前是连续的，这是因为 view 操作要求被操作的张量是内存连续的。

        x = self.fc1(x)

        return x

    # def train_classifier(self):
    #     for param in self.parameters():
    #         param.requires_grad = False
    #
    #     for param in self.fc1.parameters():
    #         param.requires_grad = True






class discriminator(nn.Module):
    def __init__(self,input_dim,dropout):
        super(discriminator, self).__init__()
        self.linear1 = nn.Linear(input_dim,64)
        self.linear2 = nn.Linear(64,2)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x1):
        hidden_output = self.linear1(x1)
        hidden_output = torch.relu(hidden_output)
        #print(hidden_output.shape)
        output = self.linear2(hidden_output)
        return output

class Domaintransformer(nn.Module):
    def __init__(self,emb_dim,output_dim,dropout):
        super(Domaintransformer, self).__init__()
        self.emb_dim = emb_dim

        self.output_dim = output_dim
        # self.num_layers = num_layers
        self.dropout = dropout
        self.encoder = nn.TransformerEncoderLayer(emb_dim, 4, output_dim, dropout, batch_first=True)
        self.trans_encder = nn.TransformerEncoder(
            encoder_layer=self.encoder, num_layers=1)

    def forward(self,input):
        output = self.trans_encder(input)
        return output



class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):#ctx是一个上下文对象，用于保存中间变量以供反向传播使用。x 是输入张量，alpha 是梯度反转的系数。
        # Store context for backprop
        ctx.alpha = alpha#将传入的 alpha 存储在上下文对象中，以便在反向传播时使用。

        # Forward pass is a no-op
        return x.view_as(x)#将传入的 alpha 存储在上下文对象中，以便在反向传播时使用。

    @staticmethod
    def backward(ctx, grad_output):#grad_output 是损失函数对前向传播输出的梯度。
        # Backward pass is just to -alpha the gradient
        output = grad_output.neg() * ctx.alpha#grad_output 是损失函数对前向传播输出的梯度。

        # Must return same number as inputs to forward()
        return output, None



if __name__ == '__main__':

    pass
