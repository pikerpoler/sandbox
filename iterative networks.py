import torch


class Cell(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                 kernel_variation=(range(-1, 2))):
        super(Cell, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

        self.conv_options = {kernel_size: self.conv}
        for i in kernel_variation:
            self.conv_options[kernel_size + i] = torch.nn.Conv2d(in_channels, out_channels, kernel_size + i,
                                                                 stride, padding, dilation, groups, bias)
        self.kwargs = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups, 'bias': bias, 'kernel_variation': kernel_variation}
        self.kernel_weights = torch.nn.Parameter(torch.randn(len(kernel_variation) + 1) / (len(kernel_variation) + 1))
        print(self.kernel_weights)
        print(self.kernel_weights.softmax(0))
    def forward(self, x):
        n_passed = 0
        for i, conv in enumerate(self.conv_options.values()):
            if self.kernel_weights.softmax(0)[i] > 0.2:
                x = conv(x) * self.kernel_weights[i]
                n_passed += 1
                x = self.conv(x)
                # x = self.bn(x)
                x = self.relu(x)
        if n_passed == len(self.conv_options):
            self.upscale()
        if n_passed <= 0.5 * len(self.conv_options):
            self.downscale()
        return x

    def upscale(self):
        print('upscale')
        pass

    def downscale(self):
        print('downscale')
        pass




def main():
    cell = Cell(3, 3, 3, 1, 1, 1, 1, True)
    x = torch.randn(1, 3, 32, 32)
    for i in range(100):
        y = cell(x)
        print(y.shape)
        if y.shape != x.shape:
            # pad y with random values
            y = y.view(8, -1)
        print(y.view(-1).shape)
        print(y.shape)

        loss = torch.nn.CrossEntropyLoss()
        loss = loss(y, x)
        loss.backward()
        optimizer = torch.optim.Adam(cell.parameters(), lr=0.001)
        optimizer.step()


if __name__ == '__main__':
    main()

