from fastai.vision.all import load_learner
import torch
import json

learn = load_learner("model/model.pkl")

torch.save(learn.model.state_dict(), "model/weights.pth")

labels = list(learn.dls.vocab)
config = {
    "labels": labels,
    "preprocess": {
        "size": 224,
        "divide_255": True,
        "normalize": False
    }
}
with open("model/labels.json", "w") as f:
    json.dump(config, f)

print("Export สำเร็จ!")
print("Labels:", labels[:5])