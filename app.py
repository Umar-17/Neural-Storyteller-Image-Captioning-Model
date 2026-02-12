import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle


class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(2048, embed_size)
        self.relu = nn.ReLU()
    def forward(self, features):
        return self.relu(self.fc(features))

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def forward(self, features, captions):
        embeddings = self.embed(captions)
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
    word_idx = torch.tensor([vocab.stoi["<start>"]])
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
    for _ in range(20):
        all_candidates = []
        for score, seq, states in beams:
            if seq[-1] == vocab.stoi["<end>"]:
                all_candidates.append((score, seq, states))
                continue
            last_word = torch.tensor([seq[-1]])
            embeddings = model.decoder.embed(last_word).unsqueeze(1)
            if len(seq) == 1:
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


st.title("📸 Neural Storyteller")
method = st.radio("Choose Generation Method:", ("Greedy Search", "Beam Search"))

@st.cache_resource
def load_all():
    with open('Model/vocab.pkl', 'rb') as f: vocab = pickle.load(f)
    model = Seq2Seq(512, 512, len(vocab), 1)
    model.load_state_dict(torch.load('Model/model_weights.pth', map_location='cpu'))
    res = models.resnet50(weights='DEFAULT')
    res = nn.Sequential(*list(res.children())[:-1])
    return model.eval(), vocab, res.eval()

model, vocab, resnet = load_all()
file = st.file_uploader("Upload Image", type=["jpg", "png"])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, use_container_width=True)
    
    
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    feat = resnet(tf(img).unsqueeze(0)).view(1, -1)
    encoded = model.encoder(feat)
    
   
    if method == "Greedy Search":
        result = greedy_search(model, encoded, vocab)
    else:
        result = beam_search(model, encoded, vocab)
    
    st.success(f"**Result:** {result}")