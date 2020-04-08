def center_crop(x, current_size, desired_size):
    start = int((current_size - desired_size)/2)
    return x[:,:, start:(start + desired_size), start:(start + desired_size)]
