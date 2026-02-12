import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import re
from collections import Counter

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


st.set_page_config(page_title="Neural Storyteller", page_icon="📸")
st.title("📸 Neural Storyteller")
method = st.radio("Choose Generation Method:", ("Greedy Search", "Beam Search"))

@st.cache_resource
def load_all():
    with open('Model/vocab.pkl', 'rb') as f: 
        vocab = pickle.load(f)
    model = Seq2Seq(512, 512, len(vocab), 1).to(device)
    model.load_state_dict(torch.load('Model/model_weights.pth', map_location=device))
    res = models.resnet50(weights='DEFAULT')
    res = nn.Sequential(*list(res.children())[:-1])
    return model.eval(), vocab, res.to(device).eval()

model, vocab, resnet = load_all()
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, use_container_width=True)
    
    tf = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    with torch.no_grad():
        feat = resnet(tf(img).unsqueeze(0).to(device)).view(1, -1)
        encoded = model.encoder(feat)
    
        if method == "Greedy Search":
            result = greedy_search(model, encoded, vocab)
        else:
            result = beam_search(model, encoded, vocab)
    
    st.success(f"**Result:** {result}")
