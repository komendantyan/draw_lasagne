import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


__all__ = ['collage', 'make_gif']


def collage(batch, collage_shape):
    nrow, ncol = collage_shape
    batch_size, channel_count, image_height, image_width = batch.shape
    if nrow * ncol != batch_size:
        raise ValueError("nrow * ncol should be equal batch_size. "
                         "Got %r * %r == %r" % (nrow, ncol, batch_size))

    collage_ = batch.swapaxes(1, 2).\
        swapaxes(2, 3).\
        reshape((nrow, ncol, image_height, image_width, channel_count)).\
        swapaxes(1, 2).\
        reshape(nrow * image_height, ncol * image_width, channel_count)

    if channel_count == 1:
        return collage_.reshape(collage_.shape[:2])
    elif channel_count in [3, 4]:
        return collage_
    else:
        raise ValueError("channel_count should be in [1, 3, 4]. Got %r" %
                         channel_count)


def make_gif(images, file='noname.gif', dpi=150, interval=200):
    assert file.endswith('.gif')

    fig, ax = plt.subplots()

    def update(i):
        im = ax.imshow(images[i], interpolation='none')
        return im, ax

    anim = FuncAnimation(fig, update, frames=len(images), interval=interval)
    anim.save(file, dpi=dpi, writer='imagemagick')
