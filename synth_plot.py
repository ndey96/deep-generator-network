import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

arch = 'Sim'
#arch = 'Gen'

filename_row_1 = [
    'Mosque - {}_0668.png'.format(arch),
    'Lipstick - {}_0629.png'.format(arch),
    'Brambling - {}_0010.png'.format(arch),
    'LeafBeetle - {}_0304.png'.format(arch),
    'Badger - {}_0362.png'.format(arch),
    'Toaster - {}_0859.png'.format(arch),
    'TriumphalArch - {}_0873.png'.format(arch),
    'Cloak - {}_0501.png'.format(arch),
    'LawnMower - {}_0621.png'.format(arch)
]
titlename_row_1 = [
    'mosque',
    'lipstick',
    'brambling',
    'leaf beetle',
    'badger',
    'toaster',
    'triumphal arch',
    'cloak',
    'lawn mower'
]
filename_row_2 = [
    'Library - {}_0624.png'.format(arch),
    'CheeseBurger - {}_0933.png'.format(arch),
    'SwimmingTrunks - {}_0842.png'.format(arch),
    'Barn - {}_0425.png'.format(arch),
    'Candle - {}_0470.png'.format(arch),
    'TableLamp - {}_0846.png'.format(arch),
    'Sandbar - {}_0977.png'.format(arch),
    'FrenchLoaf - {}_0930.png'.format(arch),
    'Lemon - {}_0951.png'.format(arch)
]
titlename_row_2 = [
    'library',
    'cheeseburger',
    'swimming trunks',
    'barn',
    'candle',
    'table lamp',
    'sandbar',
    'French loaf',
    'lemon'
]
filename_row_3 = [
    'Chest - {}_0492.png'.format(arch),
    'RunningShoe - {}_0770.png'.format(arch),
    'WaterJug - {}_0899.png'.format(arch),
    'PoolTable - {}_0736.png'.format(arch),
    'Broom - {}_0462.png'.format(arch),
    'Cellphone - {}_0487.png'.format(arch),
    'AircraftCarrier - {}_0403.png'.format(arch),
    'EntertainmentCtr - {}_0548.png'.format(arch),
    'Jeans - {}_0608.png'.format(arch)
]
titlename_row_3 = [
    'chest',
    'running shoe',
    'water jug',
    'pool table',
    'broom',
    'cellphone',
    'aircraft carrier',
    'entertainment ctr',
    'jean'
]

row_1 = [filename_row_1, titlename_row_1]
row_2 = [filename_row_2, titlename_row_2]
row_3 = [filename_row_3, titlename_row_3]

fig, axs = plt.subplots(3, 9, figsize=(16, 3))
for r1, r2, r3, f1, f2, f3, n1, n2, n3 in zip(axs[0], axs[1], axs[2], row_1[0], row_2[0], row_3[0], row_1[1], row_2[1], row_3[1]):

    # Read in the images.
    img1 = mpimg.imread('./img/'+f1)
    img2 = mpimg.imread('./img/'+f2)
    img3 = mpimg.imread('./img/'+f3)

    # Plot them onto the axs.
    r1.imshow(img1, aspect='equal')
    r2.imshow(img2, aspect='equal')
    r3.imshow(img3, aspect='equal')

    # Give titles.
    r1.set_title(n1)
    r2.set_title(n2)
    r3.set_title(n3)
    
    # Do some formatting.
    r1.axis('off')
    r2.axis('off')
    r3.axis('off')

plt.show()
