# helper script for avg_pool dimension calculation

Hin = 112
kernel_size = 2
stride = 2
padding = 0

Hout = (Hin + 2 * padding - kernel_size) / stride + 1

print(Hout)