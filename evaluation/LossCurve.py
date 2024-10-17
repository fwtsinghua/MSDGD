# prepare plot
import matplotlib.pyplot as plt
import numpy as np

def visualization_loss_curve(train_epoch_losses,savepath):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # add grid
    ax.grid(linestyle='dotted')
    # plot the training epochs vs. the epochs' classification error
    ax.plot(np.array(range(1, len(train_epoch_losses) + 1)), train_epoch_losses, label='epoch loss (blue)')
    # add axis legends
    ax.set_xlabel('[Training Epoch $e_i$]', fontsize=10)
    ax.set_ylabel('[MSE Error $\mathcal{L}^{MSE}$]', fontsize=10)
    # set plot legend
    plt.legend(loc='upper right', numpoints=1, fancybox=True)
    # add plot title
    plt.title('Training Epochs $e_i$ vs. MSE Error $L^{MSE}$', fontsize=10)
    # save plot
    plt.savefig(savepath)
    plt.show()

