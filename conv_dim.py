# helper script for conv dimension calculation

Hin = 256
kernel_size = 3
stride = 1
padding = 1
dilation = 1

Hout = (Hin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

print(Hout)