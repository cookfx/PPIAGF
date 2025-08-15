import tensorflow as tf
import keras
from keras.layers import (Input, Dense, Dropout, Conv1D, MaxPooling1D, Embedding,
                         BatchNormalization, Add, MultiHeadAttention, 
                         LayerNormalization, GlobalAveragePooling1D, concatenate)
from keras.models import Model

def build_branch_model(seq_size, dim, hidden_dim, l2_reg,name_prefix,num_heads):
    input_tensor = Input(shape=(seq_size, dim))
    
    def multi_scale_conv(x, hidden_dim, l2_reg, name_prefix, i):
        conv3 = Conv1D(hidden_dim, 3, padding='same', kernel_regularizer=l2_reg,
                       activation='relu', name=f'{name_prefix}_conv3_{i}')(x)
        conv3 = Dropout(0.1)(conv3)
        conv5 = Conv1D(hidden_dim, 5, padding='same', kernel_regularizer=l2_reg,
                       activation='relu', name=f'{name_prefix}_conv5_{i}')(x)
        conv5 = Dropout(0.1)(conv5)
        conv7 = Conv1D(hidden_dim, 7, padding='same', kernel_regularizer=l2_reg,
                       activation='relu', name=f'{name_prefix}_conv7_{i}')(x)
        conv7 = Dropout(0.1)(conv7)
        return concatenate([conv3, conv5, conv7], axis=-1)

    x = multi_scale_conv(input_tensor, hidden_dim, l2_reg, "shared", 0)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3)(x)
    
    for i in range(1, 4):
        skip = x
        conv_out = multi_scale_conv(x, hidden_dim, l2_reg, name_prefix, i)
        conv_out = BatchNormalization()(conv_out)
        x = Add()([skip, conv_out])
        x = MaxPooling1D(pool_size=3)(x)
        x = LayerNormalization()(x)
        # gru_out=Bidirectional(GRU(hidden_dim, return_sequences=True))(x)
        att_out = MultiHeadAttention(num_heads=1, key_dim=hidden_dim)(x, x)
        att_out = Dropout(0.2)(att_out)
        x = Add()([x, att_out])
        x = LayerNormalization()(x)
    return Model(inputs=input_tensor, outputs=x, name="shared_branch_model")


def build_model_wte(max_length, vocab_size,embed_dim, hidden_dim, l2_reg,num_heads):

    shared_branch = build_branch_model(max_length, embed_dim, hidden_dim, l2_reg, 'shared',num_heads)
    wte = Embedding(input_dim=vocab_size, output_dim=embed_dim, name='wte')

    seq_input1 = Input(shape=(max_length, ), name='seq_input1')
    seq_input2 = Input(shape=(max_length, ), name='seq_input2')
    
    embed1 = wte(seq_input1)  # shape (batch, L, embed_dim)
    embed2 = wte(seq_input2)
    s1 = shared_branch(embed1)
    s2 = shared_branch(embed2)
    
    s1 = GlobalAveragePooling1D()(s1)
    s2 = GlobalAveragePooling1D()(s2)
    
    gate_dim = s1.shape[-1]
    gate = Dense(gate_dim, activation='sigmoid')(concatenate([s1, s2]))
    merged = gate * s1 + (1 - gate) * s2
    
    x = Dense(100, activation='linear')(merged)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(int((hidden_dim + 7) / 2), activation='linear')(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=[seq_input1, seq_input2], outputs=main_output)
    return model

def build_model(max_length,embed_dim, hidden_dim, l2_reg,num_heads):

    shared_branch = build_branch_model(max_length, embed_dim, hidden_dim, l2_reg, 'shared',num_heads)

    seq_input1 = Input(shape=(max_length,embed_dim), name='seq_input1')
    seq_input2 = Input(shape=(max_length,embed_dim), name='seq_input2')
    
    s1 = shared_branch(seq_input1)
    s2 = shared_branch(seq_input2)
    
    s1 = GlobalAveragePooling1D()(s1)
    s2 = GlobalAveragePooling1D()(s2)
    
    gate_dim = s1.shape[-1]
    gate = Dense(gate_dim, activation='sigmoid')(concatenate([s1, s2]))
    merged = gate * s1 + (1 - gate) * s2
    
    x = Dense(100, activation='linear')(merged)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(int((hidden_dim + 7) / 2), activation='linear')(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=[seq_input1, seq_input2], outputs=main_output)
    return model