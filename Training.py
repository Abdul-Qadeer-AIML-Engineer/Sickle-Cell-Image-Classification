import torch
print(torch.__version__)  # Should show something like 2.2.0+cu121
print(torch.version.cuda)  # Should say 12.1
print(torch.cuda.is_available())  # Should say True
print(torch.cuda.get_device_name(0))  # Should show "Quadro RTX 3000"