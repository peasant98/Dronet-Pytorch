import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import torch
import torchvision
import dronet_torch

# load model 
def get_alexnet_onnx():
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    model = torchvision.models.alexnet(pretrained=True).cuda()
    print(model)
    torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True)

def get_dronet():
    dronet = dronet_torch.DronetTorch((224,224), 3, 1)
    dronet.to(dronet.device)
    inputs = torch.randn(1,3,224, 224).cuda()
    torch.onnx.export(dronet, inputs, 'dronet.onnx', verbose=True,
                        output_names=['steer', 'coll'])
    print(dronet)

def load_model(path, shape):
    model = onnx.load(path)
    engine = backend.prepare(model, device='CUDA:0')
    input_data = np.random.random(size=shape).astype(np.float32)
    # return 
    output_data = engine.run(input_data)
    print(output_data['steer'])
    print(output_data)

# get_alexnet_onnx()
get_dronet()
load_model('dronet.onnx', (1,3,224,224))