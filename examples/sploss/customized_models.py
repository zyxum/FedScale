from re import L
from typing import Any, Type, List, Union
from torch import Tensor
from torchvision.models import ShuffleNetV2, ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
import torch.nn as nn
import torch.nn.functional as F
import torch, logging

def similarity_preserving_loss(x: Tensor):
    x = torch.reshape(x, (x.shape[0], -1))
    x = torch.matmul(x, torch.t(x))
    x = F.normalize(x, 2, 1)
    return x



class BasicBlock_SPloss(BasicBlock):
    def forward(self, x: Tensor, isTrain: bool = True, sploss_list: list = []) -> Tensor:
        if isTrain:
            return super().forward(x)

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        sploss_list.append(similarity_preserving_loss(out))

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        sploss_list.append(similarity_preserving_loss(out))

        return out, isTrain, sploss_list

class Resnet34_SPloss(ResNet):
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck, BasicBlock_SPloss]], planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, isTrain: bool=True):
        if isTrain:
            return super().forward(x)
        
        sploss_list = []
        sploss_temp = None

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        sploss_list.append(similarity_preserving_loss(x))
        x = self.maxpool(x)

        for layer in self.layer1.children():
            x, _, sploss_list = layer(x, False, sploss_list)

        for layer in self.layer2.children():
            x, _, sploss_list = layer(x, False, sploss_list)

        for layer in self.layer3.children():
            x, _, sploss_list = layer(x, False, sploss_list)

        for layer in self.layer4.children():
            x, _, sploss_list = layer(x, False, sploss_list)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, sploss_list

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck, BasicBlock_SPloss]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = Resnet34_SPloss(block, layers, **kwargs)
    return model

def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet34', BasicBlock_SPloss, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


class Shufflenet_V2_X2_SPloss(ShuffleNetV2):
    def forward(self, x: Tensor, isTrain: bool=True):
        if isTrain:
            return super().forward(x)
        sploss_list = []
        x = self.conv1(x)
        sploss_list.append(similarity_preserving_loss(x))
        
        x = self.maxpool(x)
        x = self.stage2(x)
        sploss_list.append(similarity_preserving_loss(x))

        x = self.stage3(x)
        sploss_list.append(similarity_preserving_loss(x))
        
        x = self.stage4(x)
        sploss_list.append(similarity_preserving_loss(x))
        
        x = self.conv5(x)
        sploss_list.append(similarity_preserving_loss(x))
        
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x, sploss_list

def _shufflenetv2(arch: str, pretrained: bool, progress: bool, *args: Any, **kwargs: Any) -> ShuffleNetV2:
    model = Shufflenet_V2_X2_SPloss(*args, **kwargs)

    return model

def shufflenet_v2_x2_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)