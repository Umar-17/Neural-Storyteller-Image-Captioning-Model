import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import re
import os
from collections import Counter

# --- Styling & Configuration ---
st.set_page_config(page_title="Neural Storyteller", page_icon="📸", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0e1117;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    .stCard {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stCard:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 65, 108, 0.5);
    }
    
    .caption-box {
        background: linear-gradient(135deg, rgba(255, 75, 43, 0.1), rgba(255, 65, 108, 0.1));
        border-left: 5px solid #FF416C;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: 600;
        color: #fff;
        margin-top: 2rem;
    }
    
    .sample-img-container {
        cursor: pointer;
        border-radius: 10px;
        overflow: hidden;
        border: 2px solid transparent;
        transition: 0.2s;
    }
    
    .sample-img-container:hover {
        border-color: #FF416C;
    }

    /* Hide streamlit header and footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary class needs to be defined for pickle to load the vocab object
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4 
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [self.stoi[word] if word in self.stoi else self.stoi["<unk>"] for word in tokenized_text]

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(2048, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, features):
        return self.dropout(self.relu(self.fc(features)))

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        return self.linear(hiddens)

class Seq2Seq(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)


def greedy_search(model, encoded_features, vocab):
    caption = []
    word_idx = torch.tensor([vocab.stoi["<start>"]]).to(device)
    states = None
    for i in range(20):
        embeddings = model.decoder.embed(word_idx).unsqueeze(1)
        if i == 0:
            embeddings = torch.cat((encoded_features.unsqueeze(1), embeddings), dim=1)
        hiddens, states = model.decoder.lstm(embeddings, states)
        outputs = model.decoder.linear(hiddens[:, -1, :])
        predicted = outputs.argmax(1)
        word_idx = predicted
        word = vocab.itos[predicted.item()]
        if word == "<end>": break
        caption.append(word)
    return " ".join(caption)

def beam_search(model, encoded_features, vocab, beam_width=5):
    start_token = vocab.stoi["<start>"]
    beams = [(0.0, [start_token], None)]
    for step in range(20):
        all_candidates = []
        for score, seq, states in beams:
            if seq[-1] == vocab.stoi["<end>"]:
                all_candidates.append((score, seq, states))
                continue
            last_word = torch.tensor([seq[-1]]).to(device)
            embeddings = model.decoder.embed(last_word).unsqueeze(1)
            if step == 0:
                embeddings = torch.cat((encoded_features.unsqueeze(1), embeddings), dim=1)
            hiddens, next_states = model.decoder.lstm(embeddings, states)
            outputs = model.decoder.linear(hiddens[:, -1, :])
            log_probs = torch.log_softmax(outputs, dim=1)
            topk_probs, topk_indices = log_probs.topk(beam_width)
            for i in range(beam_width):
                all_candidates.append((score + topk_probs[0, i].item(), seq + [topk_indices[0, i].item()], next_states))
        beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        if all(b[1][-1] == vocab.stoi["<end>"] for b in beams): break
    return " ".join([vocab.itos[i] for i in beams[0][1] if i not in [vocab.stoi["<start>"], vocab.stoi["<end>"]]])


@st.cache_resource
def load_all():
    vocab_path = 'Model/vocab.pkl'
    weights_path = 'Model/model_weights.pth'
    
    if not os.path.exists(vocab_path) or not os.path.exists(weights_path):
        st.error("Model files not found! Please ensure 'Model/vocab.pkl' and 'Model/model_weights.pth' exist.")
        st.stop()
        
    with open(vocab_path, 'rb') as f: 
        vocab = pickle.load(f)
    model = Seq2Seq(512, 512, len(vocab), 1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    res = models.resnet50(weights='DEFAULT')
    res = nn.Sequential(*list(res.children())[:-1])
    return model.eval(), vocab, res.to(device).eval()

# --- App Layout ---

st.markdown('<div class="main-header">Neural Storyteller 📸</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Image Captioning with Deep Learning</div>', unsafe_allow_html=True)

model, vocab, resnet = load_all()

# Sidebar for configuration
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    method = st.radio("Generation Method:", ("Greedy Search", "Beam Search"), help="Beam search is more accurate but slower.")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("This app uses a ResNet-50 encoder and an LSTM decoder to generate descriptive captions for images.")

# --- State Management ---
if 'selected_img' not in st.session_state:
    st.session_state.selected_img = None

# Main content tabs
tab1, tab2 = st.tabs(["🖼️ Sample Gallery", "📤 Upload Custom Image"])

with tab1:
    st.markdown("### Select a sample image to caption:")
    SAMPLES_DIR = "samples"
    if os.path.exists(SAMPLES_DIR):
        sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if sample_files:
            # Create a grid
            cols = st.columns(3)
            for idx, sample in enumerate(sample_files):
                with cols[idx % 3]:
                    img_path = os.path.join(SAMPLES_DIR, sample)
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                    if st.button(f"Analyze Sample {idx+1}", key=f"btn_{sample}"):
                        st.session_state.selected_img = img
        else:
            st.warning("No sample images found in 'samples/' directory.")
    else:
        st.warning("Samples directory not found.")

with tab2:
    file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    if file:
        st.session_state.selected_img = Image.open(file).convert('RGB')

# Process selected image
if st.session_state.selected_img:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.image(st.session_state.selected_img, use_container_width=True, caption="Selected Image")
        if st.button("🗑️ Clear Selection"):
            st.session_state.selected_img = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🧠 AI Analysis")
        with st.spinner("Generating caption..."):
            tf = transforms.Compose([
                transforms.Resize((224,224)), 
                transforms.ToTensor(), 
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
            with torch.no_grad():
                img_input = tf(st.session_state.selected_img).unsqueeze(0).to(device)
                feat = resnet(img_input).view(1, -1)
                encoded = model.encoder(feat)
            
                if method == "Greedy Search":
                    result = greedy_search(model, encoded, vocab)
                else:
                    result = beam_search(model, encoded, vocab)
            
            st.markdown(f'<div class="caption-box">✨ {result}</div>', unsafe_allow_html=True)
            st.balloons()
else:
    st.info("👈 Choose a sample from the gallery or upload your own image to get started!")
