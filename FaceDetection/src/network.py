import torch
from torchsummary import summary

class Yolo(torch.nn.Module):
    def __init__(self, num_classes = 1, 
                anchors = [(1.3221, 1.73145),
                    (3.19275, 4.00944),
                    (5.05587, 8.09892),
                    (9.47112, 4.84053),
                    (11.2364, 10.0071)]):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors

        self.conv_layer_c1 = torch.nn.Sequential(
                                torch.nn.Conv2d(3, 32, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(32),
                                torch.nn.LeakyReLU(0.1),
                                torch.nn.MaxPool2d((2, 2))
                            )
        self.conv_layer_c2 = torch.nn.Sequential(
                                torch.nn.Conv2d(32, 64, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(64),
                                torch.nn.LeakyReLU(0.1),
                                torch.nn.MaxPool2d((2, 2))
                            )
        self.conv_layer_c3 = torch.nn.Sequential(
                                torch.nn.Conv2d(64, 128, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(128),
                                torch.nn.LeakyReLU(0.1),
                            )
        self.conv_layer_c4 = torch.nn.Sequential(
                                torch.nn.Conv2d(128, 64, 1, 1, 0, bias = False),
                                torch.nn.BatchNorm2d(64),
                                torch.nn.LeakyReLU(0.1)
                            )
        self.conv_layer_c5 = torch.nn.Sequential(
                                torch.nn.Conv2d(64, 128, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(128),
                                torch.nn.LeakyReLU(0.1),
                                torch.nn.MaxPool2d((2, 2))
                            )
        self.conv_layer_c6 = torch.nn.Sequential(
                                torch.nn.Conv2d(128, 256, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(256),
                                torch.nn.LeakyReLU(0.1)
                            )
        self.conv_layer_c7 = torch.nn.Sequential(
                                torch.nn.Conv2d(256, 128, 1, 1, 0, bias = False),
                                torch.nn.BatchNorm2d(128),
                                torch.nn.LeakyReLU(0.1)
                            )
        self.conv_layer_c8 = torch.nn.Sequential(
                                torch.nn.Conv2d(128, 256, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(256),
                                torch.nn.LeakyReLU(0.1),
                                torch.nn.MaxPool2d((2, 2))
                            )
        self.conv_layer_c9 = torch.nn.Sequential(
                                torch.nn.Conv2d(256, 512, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(512),
                                torch.nn.LeakyReLU(0.1)
                            )
        self.conv_layer_c10 = torch.nn.Sequential(
                                torch.nn.Conv2d(512, 256, 1, 1, 0, bias = False),
                                torch.nn.BatchNorm2d(256),
                                torch.nn.LeakyReLU(0.1)
                            )
        self.conv_layer_c11 = torch.nn.Sequential(
                                torch.nn.Conv2d(256, 512, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(512),
                                torch.nn.LeakyReLU(0.1),
                            )
        self.conv_layer_c12 = torch.nn.Sequential(
                                torch.nn.Conv2d(512, 256, 1, 1, 0, bias = False),
                                torch.nn.BatchNorm2d(256),
                                torch.nn.LeakyReLU(0.1)
                            )
        self.conv_layer_c13_con = torch.nn.Sequential(
                                torch.nn.Conv2d(256, 512, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(512),
                                torch.nn.LeakyReLU(0.1),
                            )

        self.MaxPool2d_ = torch.nn.MaxPool2d((2, 2))

        self.conv_layer_c14 = torch.nn.Sequential(
                                torch.nn.Conv2d(512, 1024, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(1024),
                                torch.nn.LeakyReLU(0.1),
                            )
        self.conv_layer_c15 = torch.nn.Sequential(
                                torch.nn.Conv2d(1024, 512, 1, 1, 0, bias = False),
                                torch.nn.BatchNorm2d(512),
                                torch.nn.LeakyReLU(0,1)
                            )
        self.conv_layer_c16 = torch.nn.Sequential(
                                torch.nn.Conv2d(512, 1024, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(1024),
                                torch.nn.LeakyReLU(0.1),
                            )
        self.conv_layer_c17 = torch.nn.Sequential(
                                torch.nn.Conv2d(1024, 512, 1, 1, 0, bias = False),
                                torch.nn.BatchNorm2d(512),
                                torch.nn.LeakyReLU(0,1)
                            )
        self.conv_layer_c18 = torch.nn.Sequential(
                                torch.nn.Conv2d(512, 1024, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(1024),
                                torch.nn.LeakyReLU(0.1),
                            )
        self.conv_layer_d1 = torch.nn.Sequential(
                                torch.nn.Conv2d(1024, 1024, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(1024),
                                torch.nn.LeakyReLU(0.1)
                            )
        self.conv_layer_d2 = torch.nn.Sequential(
                                torch.nn.Conv2d(1024, 1024, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(1024),
                                torch.nn.LeakyReLU(0.1)
                            )
        self.connected_layer = torch.nn.Sequential(
                                torch.nn.Conv2d(512, 64, 3, 2, 1, bias = False),
                                torch.nn.BatchNorm2d(64),
                                torch.nn.LeakyReLU(0.1)
                            )
        self.conv_layer_d3 = torch.nn.Sequential(
                                torch.nn.Conv2d(64 + 1024, 1024, 3, 1, 1, bias = False),
                                torch.nn.BatchNorm2d(1024),
                                torch.nn.LeakyReLU(0.1)
                            )
        self.output_layer = torch.nn.Conv2d(1024, (self.num_classes + 5)* len(self.anchors), (1, 1), 1, 0, bias = False)
    
    def forward(self, x):
        
        output = self.conv_layer_c1(x)
        output = self.conv_layer_c2(output)
        output = self.conv_layer_c3(output)
        output = self.conv_layer_c4(output)
        output = self.conv_layer_c5(output)
        output = self.conv_layer_c6(output)
        output = self.conv_layer_c7(output)
        output = self.conv_layer_c8(output)
        output = self.conv_layer_c9(output)
        output = self.conv_layer_c10(output)
        output = self.conv_layer_c11(output)
        output = self.conv_layer_c12(output)
        output = self.conv_layer_c13_con(output)

        output_1 = self.MaxPool2d_(output)

        output_1 = self.conv_layer_c14(output_1)
        output_1 = self.conv_layer_c15(output_1)
        output_1 = self.conv_layer_c16(output_1)
        output_1 = self.conv_layer_c17(output_1)
        output_1 = self.conv_layer_c18(output_1)
        output_1 = self.conv_layer_d1(output_1)
        output_1 = self.conv_layer_d2(output_1)
        
        output_2 = self.connected_layer(output)                     # Passthrough layer
        output_3 = torch.cat((output_1, output_2), 1).contiguous()
        
        output_3 = self.conv_layer_d3(output_3)
        output_3 = self.output_layer(output_3)
        return output_3
