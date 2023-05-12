import tensorflow as tf
import numpy as np
import config
from data_utils import UNK_ID

class ParagraphEncoder(tf.keras.layers.Layer):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_layers, dropout_rate, ans_embedding_size, ans_embedding_length):
        super().__init__()
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.ans_embedding = tf.keras.layers.Embedding(ans_embedding_size, ans_embedding_length)
        self.lstm_input_size = embedding_size + ans_embedding_length

        self.num_layers = num_layers
        if self.num_layers == 1:
            dropout_rate = 0.0
        
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, dropout=dropout_rate, return_sequences=True, return_state=True), num_layers=self.num_layers, 
            batch_input_shape=(None, None, self.lstm_input_size))
        
        self.linear_transition = tf.keras.layers.Dense(2 * hidden_size)                     # input -> 2 * hidden_size
        self.update_layer = tf.keras.layers.Dense(2 * hidden_size, use_bias=False)          # input -> 4 * hidden_size
        self.gate = tf.keras.layers.Dense(2 * hidden_size, use_bias=False)                  # input -> 4 * hidden_size

    def gated_self_attention(self, queries, memories, mask):
        # queries -> (batch, t, d)
        # memories -> (batch, t, d)
        # mask -> (batch, t)

        energies = queries @ tf.transpose(memories, perm=[0, 2, 1]) # (batch, d, t)
        mask = tf.expand_dims(mask, 1)
        energies = tf.cast(energies, dtype=tf.float32) - tf.cast(1e12, dtype=tf.float32) * tf.cast((1 - mask), dtype=tf.float32)                     # this is very questionable

        scores = tf.keras.activations.softmax(energies, axis=2)     # (batch, d, t)
        context = scores @ queries                                  # (batch, d, d) = (batch, d, t) @ (batch, t, d)  
        inputs = tf.concat([queries, context], axis=2)
        f_t = tf.keras.activations.tanh(self.update_layer(inputs))
        g_t = tf.keras.activations.sigmoid(self.gate(inputs))
        updated_output = g_t * f_t + (1 - g_t) * queries

        return updated_output
    
    def call(self, src_seq, src_len, ans_seq):
        src_embedding = self.embedding(src_seq)
        ans_embedding = self.ans_embedding(ans_seq)
        embedding = tf.concat([src_embedding, ans_embedding], -1)

        # print(embedding.shape)
        
        padded_embedding = embedding
        outputs, h, c = self.lstm(padded_embedding)

        # self attention
        mask = tf.math.sign(src_seq)
        memories = self.linear_transition(outputs)
        outputs = self.gated_self_attention(outputs, memories, mask)

        concat_states = (h, c)
        
        return outputs, concat_states
    

class ParagraphDecoder(tf.keras.layers.Layer):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)

        if num_layers == 1:
            dropout_rate = 0
        
        self.encoder_transition = tf.keras.layers.Dense(hidden_size)                        # hidden_size -> hidden_size
        self.reduce_layer = tf.keras.layers.Dense(embedding_size)                           # embedding_size + hidden_size -> embedding_size
        
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, dropout=dropout_rate, return_sequences=True, return_state=True), # num_layers=num_layers, 
            batch_input_shape=(None, None, embedding_size))
        
        self.concat_layer = tf.keras.layers.Dense(hidden_size)                              # 2 * hidden_size -> hidden_size
        self.logit_layer = tf.keras.layers.Dense(vocab_size)                                # hidden_size -> vocab_size
    
    @staticmethod

    def attention(query, memories, mask):
        # query : [b, 1, d]
        energy = tf.matmul(query, tf.transpose(memories, perm=[0, 2, 1]))  # [b, 1, t]
        energy = tf.squeeze(energy, axis=1)
        mask = tf.cast(mask, tf.float32)
        energy = tf.where(mask == 0, tf.fill(tf.shape(energy), -1e12), energy)
        attn_dist = tf.nn.softmax(energy, axis=1)  # [b, 1, t]
        attn_dist = tf.expand_dims(attn_dist, 1)
        context_vector = tf.matmul(attn_dist, memories)  # [b, 1, d]

        return context_vector, energy
    
    def get_encoder_features(self, encoder_outputs):
        return self.encoder_transition(encoder_outputs)
    
    def call(self, trg_seq, ext_src_seq, init_states, encoder_outputs, encoder_mask):
        # trg_seq : [b,t]
        # init_states : [2,b,d]
        # encoder_outputs : [b,t,d]
        # init_states : a tuple of [2, b, d]
        batch_size, max_len = trg_seq.shape[0], trg_seq.shape[1]
        
        hidden_size = encoder_outputs.shape[-1]
        memories = self.get_encoder_features(encoder_outputs)
        logits = []

        # init decoder hidden state and context vector
        prev_states = init_states
        prev_context = tf.zeros((batch_size, 1, hidden_size))
        
        for i in range(max_len):
            y_i = tf.expand_dims(trg_seq[:, i], axis=1)
            embedded = self.embedding(y_i)

            lstm_inputs = tf.concat([embedded, prev_context], axis=2)
            lstm_inputs = self.reduce_layer(lstm_inputs)
            output, h, c = self.lstm(lstm_inputs, initial_state=prev_states)
            states = (h, c)

            # encoder-decoder attention
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = tf.concat([output, context], axis=2)
            concat_input = tf.expand_dims(concat_input, axis=1)
            logit_input = tf.keras.activations.tanh(self.concat_layer(concat_input))
            logit = self.logit_layer(logit_input)

            # maxout pointer network
            if config.use_pointer:
                num_oov = max(tf.math.reduce_max(ext_src_seq - self.vocab_size + 1), 0)
                zeros = tf.zeros((batch_size, num_oov))
                extended_logit = tf.concat([logit, zeros], axis=1)
                out = tf.zeros_like(extended_logit) - 1e12
                out = tf.math.segment_max(energy, ext_src_seq)  
                out = tf.map_fn(fn=lambda x: 0 if x == -1e12 else x, elems=out)     
                logit = extended_logit + out
                logit = logit - 1e12 * (1 - logit)

            logits.append(logit[:, :, 0, 0])
            prev_states = states
            prev_context = context
        
        # print(logits[0].shape, len(logits))

        logits = tf.stack(logits, axis=1)
        # print(logits.shape)

        return logits
    
    def decode(self, y, ext_x, prev_states, prev_context, encoder_features, encoder_mask):
        # forward one step lstm
        # y : [b]
        embedded = self.embedding(tf.expand_dims(y, axis=1))
        lstm_inputs = self.reduce_layer(tf.concat([embedded, prev_context], axis=2))
        output, states = self.lstm(lstm_inputs, initial_state=prev_states)

        context, energy = self.attention(output, encoder_features, encoder_mask)
        concat_input = tf.expand_dims(tf.concat([output, context], axis=2), axis=1)
        logit_input = tf.keras.activations.tanh(self.concat_layer(concat_input))
        logit = self.logit_layer(logit_input)

        batch_size = y.shape[0]
        num_oov = max(tf.math.maximum(ext_x - self.vocab_size + 1), 0)
        zeros = tf.zeros((batch_size, num_oov))
        extended_logit = tf.concat([logit, zeros], axis=1)
        out = tf.zeros_like(extended_logit) - 1e12
        out = tf.math.segment_max(energy, ext_x)
        out = tf.map_fn(fn=lambda x: 0 if x == -1e12 else x, elems=out)
        logit = extended_logit + out
        logit = tf.map_fn(fn=lambda x: 0 if x == -1e12 else x, elems=logit)
        # forcing UNK prob 0
        logit[:, UNK_ID] = -1e12

        return logit, states, context


class Seq2Seq(tf.keras.layers.Layer):
    def __init__(self, embedding=None):
        super().__init__()
        self.encoder = ParagraphEncoder(embedding,
                                        config.vocab_size,
                                        config.embedding_size,
                                        config.hidden_size,
                                        config.num_layers,
                                        config.dropout,
                                        config.hidden_size,
                                        config.embedding_size)
        self.decoder = ParagraphDecoder(embedding, 
                                        config.vocab_size,
                                        config.embedding_size,
                                        2 * config.hidden_size,
                                        config.num_layers,
                                        config.dropout)
        
    def call(self, src_seq, tag_seq, ext_src_seq, trg_seq):
        enc_mask = tf.math.sign(src_seq)
        src_len = tf.reduce_sum(enc_mask, axis=1)
        enc_outputs, enc_states = self.encoder(src_seq, src_len, tag_seq)
        sos_trg = trg_seq[:, :-1]

        # print(enc_outputs.shape, enc_states[0].shape, enc_states[1].shape, sos_trg.shape)

        logits = self.decoder(sos_trg, ext_src_seq, enc_states, enc_outputs, enc_mask)
        # print(logits.shape)
        return logits



