# helper script for avg_pool dimension calculation

Hin = 224
kernel_size = 113
stride = 1
padding = 0

Hout = (Hin + 2 * padding - kernel_size) / stride + 1

print(Hout)