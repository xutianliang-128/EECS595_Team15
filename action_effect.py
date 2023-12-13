import torch
import clip
from PIL import Image
from constant import prompt_template, action_effect_pairs
from tqdm import tqdm
import json
import os
import numpy as np
import os
import random

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed_everything(595)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data, label = [], []
prompts = []
i = 1
prompts.append("background")
for pair in action_effect_pairs:
    prompts.append(prompt_template.format(pair.split("+")[0], pair.split("+")[1]))
    path = "./action_effect_image_rs/{}/positive/".format(pair)
    for img in os.listdir(path):
        data.append(path + img)
        label.append(i)
    i += 1
    path = "./action_effect_image_rs/{}/negative/".format(pair)
    for img in os.listdir(path):
        data.append(path + img)
        label.append(0) 


text = clip.tokenize(prompts).to(device)
text_features = model.encode_text(text)

preds = []
probs = []
pred_firsts = []

for img in tqdm(data):
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        logits_per_image, logits_per_text = model(image, text)
        prob = logits_per_image.softmax(dim=-1).cpu().numpy()
        pred = int(prob.argmax())
        pred_firsts.append(pred)
        probs.append(prob.tolist()[0])
        topk = torch.topk(logits_per_image, k=20, dim=1, largest=True)[1].cpu().numpy()
        pred = topk[0].tolist()
        preds.append(pred)

with open("results_topk_no_bg.json", "w") as f:
    json.dump({"preds": preds, "label": label, "probs": probs}, f)