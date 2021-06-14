import cv2
import numpy as np

def normal(x, width):
    """
    scale stroke parameter x ([0,1]) based on width of the canvas
    :param x: stroke parameter x ([0,1])
    :param width: width of canvas
    :return: scaled parameter
    """
    return (int)(x * (width - 1) + 0.5)

def draw(f, width=128, mask = False):
    """
    Draw the Brezier curve on empty canvas
    :param f: stroke parameters (x0, y0, x1, y1, x2, y2, z0, z2, w0, w2)
    :param width: width of the canvas
    :param mask: boolean on whether mask is required for the canvas
    :return: painted canvas is zero at stroke locations and one otherwise,
            stroke_mask (ones at stroke parameters and zero otherwise)
    """
    # read stroke parameters (10 positional parameters)
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)

    # initialize empty canvas
    canvas = np.zeros([width * 2, width * 2]).astype('float32')
    tmp = 1. / 100

    # Brezier curve is made of 100 smaller circles
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)

    # return mask if required
    if mask:
        stroke_mask = (canvas!=0).astype(np.int32)
        return 1 - cv2.resize(canvas, dsize=(width, width)), stroke_mask
    return 1 - cv2.resize(canvas, dsize=(width, width))
