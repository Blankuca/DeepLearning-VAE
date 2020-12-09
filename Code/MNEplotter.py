import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display, clear_output
class MNEPlotter:
    def __init__(self,CH_names,lableEncoding=[None]):
        if lableEncoding.all==[None]:
            self.decodeLable=False
        else:
            self.decodeLable=True
            self.decoding = lableEncoding


        self.CH_names=CH_names

    def plot(self,window,show=True):
        """
        Plots a window if given in out dataformat
        """
        data=window['X']
        y=np.arange(window['stepval'][0],window['stepval'][1])
        if self.decodeLable:
            lable=self.decoding[window['Y']==True]
        else:
            lable = window['Y']

        fig, axs=plt.subplots(len(self.CH_names),1,sharey=True,figsize=(6,20))
        for n,CH in enumerate(self.CH_names):
            axs[n].plot(y,data[n])
            axs[n].set_ylabel(CH)
        fig.suptitle(f"window betwen {window['tval'][0]} and {window['tval'][1]}S, lable {lable}")
        plt.show()

    def plot_raw(self,data,Y):
        """
        Plots a window if given in out dataformat
        """

        if self.decodeLable:
            lable=self.decoding[Y==True]
        else:
            lable = Y

        fig, axs=plt.subplots(len(self.CH_names),1,sharey=True,figsize=(6,20))
        for n,CH in enumerate(self.CH_names):
            axs[n].plot(data[n])
            axs[n].set_ylabel(CH)
        fig.suptitle(f"lable {lable}")
        plt.show()


def plot_AC(training_data, validation_data, tmp_img="tmp_vae_ac.png", show=True, figsize=(18, 6)):
    fig, axes = plt.subplots(1, 3, figsize=figsize, squeeze=False)

    # plot ELBO
    ax = axes[0, 0]
    ax.set_title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
    ax.plot(training_data['elbo'], label='Training')
    ax.plot(validation_data['elbo'], label='Validation')
    ax.legend()

    # plot KL
    ax = axes[0, 1]
    ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
    ax.plot(training_data['kl'], label='Training')
    ax.plot(validation_data['kl'], label='Validation')
    ax.legend()

    # plot NLL
    ax = axes[0, 2]
    ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
    ax.plot(training_data['log_px'], label='Training')
    ax.plot(validation_data['log_px'], label='Validation')
    ax.legend()

    # display
    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)
    if show:
        display(Image(filename=tmp_img))
        clear_output(wait=True)


def plot_2d_latents(ax, outputs, y, tmp_img="tmp_vae_latent_space.png", show=True):
    z = outputs['z']
    qz = outputs['qz']
    z = z.to('cpu')
    y = y.to('cpu')
    scale_factor = 2
    batch_size = z.shape[0]
    palette = sns.color_palette()
    colors = [palette[l] for l in y]

    # plot prior
    prior = plt.Circle((0, 0), scale_factor, color='gray', fill=True, alpha=0.1)
    ax.add_artist(prior)

    # plot data points
    mus, sigmas = qz.mu.to('cpu'), qz.sigma.to('cpu')
    mus = [mus[i].numpy().tolist() for i in range(batch_size)]
    sigmas = [sigmas[i].numpy().tolist() for i in range(batch_size)]

    posteriors = [
        plt.matplotlib.patches.Ellipse(mus[i], *(scale_factor * s for s in sigmas[i]), color=colors[i], fill=False,
                                       alpha=0.3) for i in range(batch_size)]
    for p in posteriors:
        ax.add_artist(p)

    ax.scatter(z[:, 0], z[:, 1], color=colors)

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_aspect('equal', 'box')

    # display
    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)
    if show:
        display(Image(filename=tmp_img))
        clear_output(wait=True)