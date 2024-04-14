from many_netx.configs.backbones.cspnet import CSPStage
import torch
from many_netx.configs.backbones.cspnet import DarknetBottleneck
from many_netx.configs.backbones.res2net import Bottle2neck
model = CSPStage(Bottle2neck,64,64)
inputs = torch.rand(12,64,200,100)
level_outputs = model(inputs)
print(level_outputs.shape)
# from OpenPCDet_test.pcdet.models.csp_attention_model.cspdarknet1 import CSPStage
# import torch
# from OpenPCDet_test.pcdet.models.csp_attention_model.cspdarknet1 import DarknetBottleneck
# model = CSPStage(DarknetBottleneck,128,128)
# inputs = torch.rand(12,128,117,128)
# level_outputs = model(inputs)
# print(level_outputs.shape)