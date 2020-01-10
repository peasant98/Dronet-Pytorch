import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import torch
import torchvision

# load model 
def get_alexnet_onnx():
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    model = torchvision.models.alexnet(pretrained=True).cuda()
    print(model)

def load_model(path):
    model = onnx.load(path)
    engine = backend.prepare(model, device='CUDA:1')
    input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)

    output_data = engine.run(input_data)[0]
    print(output_data)
    print(output_data.shape)

get_alexnet_onnx()