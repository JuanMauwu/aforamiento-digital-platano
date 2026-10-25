import torch

path = "runs/testing-trainning-2/weights/last.pt"

ckpt = torch.load(
    path,
    map_location="cpu",
    weights_only=False
)

print("Epoca guardada:", ckpt.get("epoch"))
print("Optimizador guardado:", ckpt.get("optimizer") is not None)
print("Argumentos guardados:", ckpt.get("opt"))