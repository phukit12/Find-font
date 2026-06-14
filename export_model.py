import sys
import types
import plum

# สร้าง fake plum._function
fake = types.ModuleType('plum._function')
fake.Function = plum.Function
sys.modules['plum._function'] = fake

# Patch torch unpickler โดยตรง
import torch.serialization as ts

original_load = ts._load

def patched_torch_load(zip_file, map_location, pickle_module, **kwargs):
    import io
    import pickle

    class PatchedUnpickler(pickle_module.Unpickler):
        def find_class(self, module, name):
            if module == 'plum._function':
                return getattr(plum, name, getattr(fake, name))
            return super().find_class(module, name)

    class PatchedPickleModule:
        Unpickler = PatchedUnpickler
        def loads(self, *args, **kwargs):
            return pickle_module.loads(*args, **kwargs)

    return original_load(zip_file, map_location, PatchedPickleModule(), **kwargs)

ts._load = patched_torch_load

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