import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from spikingjelly.clock_driven.neuron import MultiStepLIFNode #T = timesteps, B= batch size, C = channels
import math
from spikingjelly.clock_driven.neuron import MultiStepIFNode
import urllib.request
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms


"""This file currently replicates Figure 2 of the paper, showing the frequency response of the SNN"""

def load_image(T=16):
    """Downloads a polar bear image and prepares it for the SNN"""

    url = "https://upload.wikimedia.org/wikipedia/commons/2/20/Eisb%C3%A4r_%28Ursus_maritimus%29_im_Schnee.jpg"
    
    # Wikimedia requires a User-Agent header
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with open("polar_bear.jpg", "wb") as f:
            f.write(response.read())

    # grayscale it
    img = Image.open("polar_bear.jpg").convert("L")

    preprocess = transforms.Compose([transforms.Resize((532, 800)), transforms.ToTensor()])

    # Prepare shape from [B, C, H, W] to [1, 1, 32, 32]
    img_tensor = preprocess(img).unsqueeze(0)

    spiking_input = img_tensor.repeat(T, 1, 1, 1, 1)

    return img_tensor, spiking_input


def freq_analysis(tau_vals = [1.11, 2.0, 4.0], T=16):
    #tau_vals correspon to beta values [0.1, 0.5, 0.75]
    static_img,spiking_img = load_image(T)

    # use Laplacian edge-detector (a filter which is a specific weight matrix below) to strongly pick up high frequencies

    conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    conv.weight.data = torch.tensor([[[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]])

    fig, axes = plt.subplots(len(tau_vals) + 1, 3, figsize=(15, 4 * (len(tau_vals) + 1)))

    # ANN basline with ReLU ===============================
    relu = nn.ReLU()
    
    # pass 2d static image: conv -> relu
    relu_out = relu(conv(static_img))

    # 2d FFT on ReLU feature map. Using fftshift to put low freqs in the center

    relu_fft = torch.fft.fftshift(torch.fft.fft2(relu_out.squeeze()))

    # Log scale for viewing 2d freq. spectrums
    relu_amp = torch.log(torch.abs(relu_fft) + 1e-8)

    axes[0, 0].imshow(static_img.squeeze(), cmap="gray")
    axes[0, 0].set_title("Original Polar Bear")

    axes[0, 1].imshow(relu_out.squeeze().detach().numpy(), cmap='hot')
    axes[0, 1].set_title("ReLU output")

    axes[0, 2].imshow(relu_amp.detach().numpy(), cmap='viridis')
    axes[0, 2].set_title("ReLU FFT")

    


    # SNN pipeline (LIF) ==============================

    for i, tau in enumerate(tau_vals):
        lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")

        # pass spiking image: conv -> lif
        # first apply conv to every time step
        conv_out_spiking = conv(static_img).repeat(T, 1, 1, 1, 1)

        lif_out = lif(conv_out_spiking)
        lif.reset()

        # avg spikes over time to get final firing rate feature map

        lif_feature_map = lif_out.mean(0)

        # 2d FFT on LIF feature map
        lif_fft = torch.fft.fftshift(torch.fft.fft2(lif_feature_map.squeeze()))
        lif_amp = torch.log(torch.abs(lif_fft) + 1e-8)

        axes[i+1, 0].imshow(static_img.squeeze(), cmap="gray")
        axes[i+1, 0].set_title("Original Polar Bear")
        
        axes[i+1, 1].imshow(lif_feature_map.squeeze().detach().numpy(), cmap="hot")
        axes[i+1, 1].set_title(f"SNN Spatial Features (Tau={tau}, B={1-(1/tau)})")

        #feature cutoff
        axes[i+1, 2].imshow(lif_amp.detach().numpy(), cmap="viridis")
        axes[i+1, 2].set_title(f"SNN FFT (Tau={tau}, B={1-(1/tau)})")

    
    # SNN pipeline with IF =================

    #if_ = MultiStepIFNode()

    plt.tight_layout()
    plt.savefig("freq_analysis.png")
    plt.show()

if __name__ == "__main__":
    freq_analysis()