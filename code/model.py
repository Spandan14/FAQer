import tensorflow as tf
import config

class ParagraphEncoder(tf.keras.layers.Layer):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_layers, dropout_rate, ans_embedding_size, ans_embedding_length):
        super().__init__()
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.ans_embedding = tf.keras.layers.Embedding(ans_embedding_size, ans_embedding_length)
        lstm_input_size = embedding_size + ans_embedding_length

        if embeddings is not None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size).from_pretrained(embeddings, freeze=config.freeze_embedding)

        self.num_layers = num_layers
        if self.num_layers == 1:
            dropout_rate = 0.0
        
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, dropout=dropout_rate, num_layers=num_layers, return_sequences=True), 
            batch_input_shape=(None, None, lstm_input_size))
        
        self.linear_transition = tf.keras.layers.Dense(2 * hidden_size)                     # input -> 2 * hidden_size
        self.update_layer = tf.keras.layers.Dense(2 * hidden_size, use_bias=False)          # input -> 4 * hidden_size
        self.gate = tf.keras.layers.Dense(2 * hidden_size, use_bias=False)                  # input -> 4 * hidden_size

    def gated_self_attention(self, queries, memories, mask):
        # queries -> (batch, t, d)
        # memories -> (batch, t, d)
        # mask -> (batch, t)

        energies = queries @ tf.transpose(memories, perm=[0, 2, 1]) # (batch, d, t)
        mask = tf.expand_dims(mask, 1)
        energies = energies - 1e12 * (1 - mask)                     # this is very questionable

        scores = tf.keras.activations.softmax(energies, axis=2)     # (batch, d, t)
        context = scores @ queries                                  # (batch, d, d) = (batch, d, t) @ (batch, t, d)  # TODO verify this
        inputs = tf.concat([queries, context], axis=2)
        f_t = tf.keras.activations.tanh(self.update_layer(inputs))
        g_t = tf.keras.activations.sigmoid(self.gate(inputs))
        updated_output = g_t * f_t + (1 - g_t) * queries

        return updated_output
    
    def call(self, src_seq, src_len, ans_seq):
        total_length = src_seq.shape[1]
        src_embedding = self.embedding(src_seq)
        ans_embedding = self.ans_embedding(ans_seq)
        embedding = tf.concat([src_embedding, ans_embedding], 2)
        padded_embedding = tf.keras.preprocess.sequence.pad_sequences(
            embedding, padding="post"
        )
        outputs, h, c = self.lstm(embedding)

        # self attention
        mask = tf.math.sign(src_seq)
        memories = self.linear_transition(outputs)
        outputs = self.gated_self_attention(outputs, memories, mask)

        b = h.shape[1]
        d = h.shape[2]
        h = tf.reshape(h, (2, 2, b, d))
        h = tf.concat([h[:, 0, :, :], h[:, 1, :, :]], axis=-1)
        
        c = tf.reshape(c, (2, 2, b, d))
        c = tf.concat([c[:, 0, :, :], c[:, 1, :, :]], axis=-1)
        concat_states = (h, c)
        
        return outputs, concat_states
    

class PassageDecoder(tf.keras.layers.Layer):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        
        if embeddings is not None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size).from_pretrained(embeddings, freeze=config.freeze_embedding)

        if num_layers == 1:
            dropout_rate = 0
        
        self.encoder_transition = tf.keras.layers.Dense(hidden_size)                        # hidden_size -> hidden_size
        self.reduce_layer = tf.keras.layers.Dense(embedding_size)                           # embedding_size + hidden_size -> embedding_size
        
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, dropout=dropout_rate, num_layers=num_layers, return_sequences=True), 
            batch_input_shape=(None, None, embedding_size))
        
        self.concat_layer = tf.keras.layers.Dense(hidden_size)                              # 2 * hidden_size -> hidden_size
        self.logit_layer = tf.keras.layers.Dense(vocab_size)                                # hidden_size -> vocab_size
    
    @staticmethod
    def attention(query, memories, mask):
        # query : [b, 1, d]
        energy = query @ tf.transpose(memories, perm=[0, 2, 1])
        energy = tf.expand_dims(energy, 1)
        energy = energy - 1e12 * (1 - mask)                     # very questionable
        attn_dist = tf.keras.activations.softmax(energy, axis=1)
        attn_dist = tf.expand_dims(attn_dist, 1)
        context_vector = attn_dist @ memories

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
            output, states = self.lstm(lstm_inputs, initial_state=prev_states)

            # encoder-decoder attention
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = tf.concat([output, context], axis=2)
            concat_input = tf.expand_dims(concat_input, axis=1)
            logit_input = tf.keras.activations.tanh(self.concat_layer(concat_input))
            logit = self.logit_layer(logit_input)

            # maxout pointer network
            num_oov = max(tf.math.maximum(ext_src_seq - self.vocab_size + 1), 0)
            zeros = tf.zeros((batch_size, num_oov))
            extended_logit = tf.concat([logit, zeros], axis=1)
            out = tf.zeros_like(extended_logit) - 1e12
            out = tf.math.segment_max(energy, ext_src_seq)
            out = tf.map_fn(fn=lambda x: 0 if x == -1e12 else x, elems=out)
            logit = extended_logit + out
            logit = logit - 1e12 * (1 - logit)

            logits.append(logit)
            prev_states = states
            prev_context = context

        logits = tf.stack(logits, dim=1)

        return logits
    
    def decode(self, y, ext_x, prev_states, prev_context, encoder_features, encoder_mask):
        # forward one step lstm
        # y : [b]
        embedded = self.embedding(tf.expand_dims(y, axis=1))
        



