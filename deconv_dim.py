# helper script for deconv dimension calculation

Hin = 128
kernel_size = 4
stride = 2
padding = 17
dilation = 1
output_padding = 0

Hout = (Hin - 1) * stride - 2 * padding + dilation * (
    kernel_size - 1) + output_padding + 1
print(Hout)