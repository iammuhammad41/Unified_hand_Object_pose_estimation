import torch
import torchvision
import PIL
import pickle
import trimesh
import numpy
import argparse
import tqdm
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.get_device_properties("cuda:0"))
print(torch.cuda.get_device_name("cuda:0"))

print('torchvision Version: ', torchvision.__version__)
print('PIL Version: ',PIL.__version__)
print('Pickle Version: ',pickle.format_version)
print('trimesh Version: ', trimesh.__version__)
print('numpy Version: ', numpy.__version__)
print('argparse Version: ', argparse._VersionAction)

# # setting device on GPU if available, else CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print()
#
# #Additional Info when using cuda
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')