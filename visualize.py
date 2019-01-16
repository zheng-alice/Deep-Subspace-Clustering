import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from IPython.display import HTML

def tSNE_2D(images, labels):
    tsne = TSNE()
    images_tsne = tsne.fit_transform(images)
    plt.scatter(images_tsne[:, 0], images_tsne[:, 1], c=labels)
    plt.show()
    
def tSNE_3D(images, labels):
    tsne = TSNE(n_components=3)
    images_tsne = tsne.fit_transform(images)
    fig = plt.figure()
    ax = Axes3D(fig)
    scatter = ax.scatter(images_tsne[:, 0], images_tsne[:, 1], images_tsne[:, 2], c=labels)
    
    def init():
        return (scatter,)
    def animate(angle):
        ax.view_init(30, 2*angle)
        return (scatter,)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=180, interval=40, blit=True)
    return HTML(anim.to_html5_video())