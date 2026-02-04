import torch
ckpt = torch.load('exp/tmp_20260202_175107/checkpoints/epoch=21-step=99000.ckpt', map_location='cpu')
print("Searching for 'auxiliary' in checkpoint state_dict:")
found = False
for k in ckpt['state_dict'].keys():
    if 'auxiliary' in k:
        print(k)
        found = True
if not found:
    print("No keys containing 'auxiliary' found.")
