import matplotlib.pyplot as plt

def visualize_results(image, mask, pred):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0))
    ax[0].set_title("Original")
    ax[1].imshow(mask.squeeze(), cmap='gray')
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred.squeeze(), cmap='gray')
    ax[2].set_title("Prediction")
    plt.show()