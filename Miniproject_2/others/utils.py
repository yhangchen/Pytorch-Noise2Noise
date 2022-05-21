from torch import mean, log10


def psnr(denoised, ground_truth):
    mse = mean((denoised - ground_truth)**2)
    return -10 * log10(mse + 10**-8)