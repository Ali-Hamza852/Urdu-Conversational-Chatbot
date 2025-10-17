import streamlit as st
import torch
import torch.nn as nn
import math
import collections
import json

# --- STEP 1: DEFINE ALL CLASSES FROM SCRATCH ---
# This section includes the final, working versions of your custom tokenizer
# and the Transformer model architecture.

class SimpleUnigramTokenizer:
    def __init__(self, vocab_size=15000):
        self.vocab_size = vocab_size
        self.vocab, self.token_to_id, self.id_to_token = {}, {}, {}
        self.special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]

    def _viterbi_segment(self, text):
        best_scores = [-math.inf] * (len(text) + 1)
        best_scores[0] = 0
        backpointers = [None] * (len(text) + 1)
        for i in range(1, len(text) + 1):
            for j in range(max(0, i - 20), i):
                subword = text[j:i]
                if subword in self.vocab:
                    score = best_scores[j] + self.vocab[subword]
                    if score > best_scores[i]:
                        best_scores[i], backpointers[i] = score, j
        if best_scores[-1] == -math.inf: return [text]
        segments, end = [], len(text)
        while end > 0:
            start = backpointers[end]
            if start is None: segments.append(text[:end]); break
            segments.append(text[start:end]); end = start
        return segments[::-1]

    def encode(self, text):
        return [self.token_to_id.get(token, self.token_to_id["[UNK]"]) for token in self._viterbi_segment(text)]

    def decode(self, ids):
        return "".join([self.id_to_token.get(str(id), "[UNK]") for id in ids])

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab, tokenizer.token_to_id = data['vocab'], data['token_to_id']
        tokenizer.id_to_token, tokenizer.special_tokens = data['id_to_token'], data['special_tokens']
        return tokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        encoding_matrix = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        encoding_matrix[:, 0::2] = torch.sin(position * div_term)
        encoding_matrix[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding_matrix', encoding_matrix.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.positional_encoding_matrix[:, :x.size(1), :])

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0
        self.head_dim = embedding_dim // num_heads
        self.num_heads = num_heads
        self.query_layer = nn.Linear(embedding_dim, embedding_dim)
        self.key_layer = nn.Linear(embedding_dim, embedding_dim)
        self.value_layer = nn.Linear(embedding_dim, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None: attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))
        return torch.matmul(attn_weights, V)
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.query_layer(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_layer(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_layer(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        context = self.scaled_dot_product_attention(Q, K, V, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.output_layer(context)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(emb_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, emb_dim)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x): return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(emb_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(emb_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, enc_output, src_mask, trg_mask):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, trg_mask)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, enc_output, enc_output, src_mask)))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_layers, num_heads, ff_dim, max_len, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoding = PositionalEncoding(emb_dim, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(emb_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
    def forward(self, x, mask):
        x = self.pos_encoding(self.embedding(x))
        for layer in self.layers: x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_layers, num_heads, ff_dim, max_len, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoding = PositionalEncoding(emb_dim, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(emb_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(emb_dim, vocab_size)
    def forward(self, x, enc_output, src_mask, trg_mask):
        x = self.pos_encoding(self.embedding(x))
        for layer in self.layers: x = layer(x, enc_output, src_mask, trg_mask)
        return self.output_layer(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, emb_dim, num_layers, num_heads, ff_dim, max_len, dropout, pad_idx):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab, emb_dim, num_layers, num_heads, ff_dim, max_len, dropout)
        self.decoder = Decoder(trg_vocab, emb_dim, num_layers, num_heads, ff_dim, max_len, dropout)
        self.pad_idx = pad_idx
    def make_source_mask(self, src): return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    def make_target_mask(self, trg):
        pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
        seq_mask = torch.tril(torch.ones((trg.shape[1], trg.shape[1]), device=trg.device)).bool()
        return pad_mask & seq_mask
    def forward(self, src, trg):
        src_mask = self.make_source_mask(src)
        trg_mask = self.make_target_mask(trg)
        enc_output = self.encoder(src, src_mask)
        return self.decoder(trg, enc_output, src_mask, trg_mask)

# --- STEP 2: HYPERPARAMETERS AND DEVICE CONFIGURATION ---
# **THE FIX IS HERE:** We use the same fixed vocab size as during training.
VOCAB_SIZE = 15000
EMBEDDING_DIM = 256
NUM_LAYERS = 2
NUM_HEADS = 2
FEEDFORWARD_DIM = 1024
DROPOUT = 0.3
MAX_SEQ_LEN = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- STEP 3: LOAD MODEL AND TOKENIZER (CACHED) ---
@st.cache_resource
def load_assets():
    tokenizer = SimpleUnigramTokenizer.load("my_unigram_tokenizer.json")
    pad_idx = tokenizer.token_to_id["[PAD]"]
    
    # Initialize model with the HARDCODED vocab size
    model = Transformer(VOCAB_SIZE, VOCAB_SIZE, EMBEDDING_DIM, NUM_LAYERS, NUM_HEADS,
                        FEEDFORWARD_DIM, MAX_SEQ_LEN, DROPOUT, pad_idx)
    
    model.load_state_dict(torch.load("transformer_chatbot_best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_assets()
SOS_IDX = tokenizer.token_to_id["[SOS]"]
EOS_IDX = tokenizer.token_to_id["[EOS]"]

# --- STEP 4: INFERENCE FUNCTION ---
def get_bot_response(prompt_text):
    model.eval()
    source_ids = [SOS_IDX] + tokenizer.encode(prompt_text) + [EOS_IDX]
    source_tensor = torch.tensor(source_ids).unsqueeze(0).to(device)
    
    target_ids = [SOS_IDX]
    for _ in range(MAX_SEQ_LEN):
        target_tensor = torch.tensor(target_ids).unsqueeze(0).to(device)
        with torch.no_grad():
            output_logits = model(source_tensor, target_tensor)
        
        predicted_id = output_logits.argmax(dim=-1)[0, -1].item()
        if predicted_id == EOS_IDX: break
        target_ids.append(predicted_id)
            
    return tokenizer.decode(target_ids[1:])

# --- STEP 5: STREAMLIT USER INTERFACE ---
st.set_page_config(layout="centered")
st.title("🤖 Urdu Conversational Chatbot")
st.markdown("""
<style>
    .stTextInput > div > div > input, .stChatMessage { direction: rtl; }
    .st-emotion-cache-16idsys p { text-align: right; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt := st.chat_input("یہاں اپنا سوال لکھیں"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("جواب سوچ رہا ہوں..."):
            response = get_bot_response(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})