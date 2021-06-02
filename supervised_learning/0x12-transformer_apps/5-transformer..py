import tensorflow as tf
import numpy as np


class Transformer(tf.keras.Model):
    """ reate a transformer network:"""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
        """ Class Constructor"""
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate)
        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """initialize trasformer"""
        enc_output = self.encoder(inputs, training, encoder_mask)

        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)

        return final_output


class Decoder(tf.keras.layers.Layer):
    """create the decoder for a transformer:r"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """constructor"""
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """initialize decoder """
        seq_len = x.shape[1]
        embeddings = self.embedding(x)
        embeddings *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embeddings += self.positional_encoding[:seq_len, :]
        output = self.dropout(embeddings, training=training)
        for i in range(self.N):
            output = self.blocks[i](output, encoder_output, training,
                                    look_ahead_mask, padding_mask)

        return output


class Encoder(tf.keras.layers.Layer):
    """create the encoder for a transformer"""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """constructor"""
        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """initialize decoder """
        seq_len = x.shape[1]
        embeddings = self.embedding(x)
        embeddings *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embeddings += self.positional_encoding[:seq_len, :]
        output = self.dropout(embeddings, training=training)

        for i in range(self.N):
            output = self.blocks[i](output, training, mask)
        return output


class DecoderBlock(tf.keras.layers.Layer):
    """ to create an encoder
        block for a transformer:"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """constructor"""
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """"initialize decoder"""
        attention_output1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention_output1 = self.dropout1(attention_output1, training=training)
        attention_output1 = self.layernorm1(attention_output1 + x)
        attention_output2, _ = self.mha2(attention_output1, encoder_output,
                                         encoder_output, padding_mask)
        attention_output2 = self.dropout2(attention_output2, training=training)
        attention_output2 = self.layernorm2(attention_output2 +
                                            attention_output1)

        ffn_output = self.dense_hidden(attention_output2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        decoder_output = self.layernorm3(ffn_output + attention_output2)
        return decoder_output


class EncoderBlock(tf.keras.layers.Layer):
    """to create an encoder block for a transformer:"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        '''costructor '''
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """initialize ecoder """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.dense_hidden(out1)

        out2 = self.dense_output(ffn_output)
        ffn_output = self.dropout2(out2, training=training)
        output = self.layernorm2(out1 + ffn_output)
        return output


class MultiHeadAttention(tf.keras.layers.Layer):
    """ to perform multi head attention:"""
    def __init__(self, dm, h):
        """costructor  """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(dm // h)
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is
        (batch_size, num_heads, seq_len, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """multi-head attention"""
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))
        output = self.linear(concat_attention)
        return output, attention_weights


def sdp_attention(Q, K, V, mask=None):
    """that calculates the scaled dot product attention"""
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention += (mask * -1e9)
    weights = tf.nn.softmax(scaled_attention, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights


def positional_encoding(max_seq_len, dm):
    """ calculates the positional
        encoding for a transformer: """
    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :],
                            dm)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding[0]


def get_angles(pos, i, d_model):
    """calculated the angle  """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


class RNNDecoder(tf.keras.layers.Layer):
    """ class """
    def __init__(self, vocab, embedding, units, batch):
        """contrutor """
        super(RNNDecoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
             units=units,
             recurrent_initializer='glorot_uniform',
             return_sequences=True,
             return_state=True
             )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """decode"""
        encode = SelfAttention(self.units)
        c, prev_state = encode(s_prev, hidden_states)
        x = self.embedding(x)
        inputs = tf.concat([tf.expand_dims(c, 1), x], axis=-1)
        decode_output, s = self.gru(inputs)
        output = tf.reshape(decode_output, (-1, decode_output.shape[2]))


class SelfAttention(tf.keras.layers.Layer):
    """ to calculate the attention"""
    def __init__(self, units):
        """ constructor """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """ calculate self-atention """
        s_newdims = self.W(tf.expand_dims(s_prev, axis=1))
        u = self.U(hidden_states)
        e = self.V(tf.tanh(u+s_newdims))
        a = tf.nn.softmax(e, axis=1)
        c = tf.reduce_sum(a * hidden_states, axis=1)
        return c, a


class RNNEncoder(tf.keras.layers.Layer):
    """ class encode """
    def __init__(self, vocab, embedding, units, batch):
        """ constuctor """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
             units=self.units,
             recurrent_initializer='glorot_uniform',
             return_sequences=True,
             return_state=True)

    def initialize_hidden_state(self):
        """Returns: a tensor of shape
            (batch, units)containing the
            initialized hidden states"""
        return tf.zeros([self.batch, self.units])

    def call(self, x, initial):
        """crate and return encode output"""
        inp = self.embedding(x)
        ouput, status = self.gru(inp, initial_state=initial)
        return ouput, status

        y = self.F(output)
        return y, s
