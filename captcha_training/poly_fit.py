import json
import os.path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from skimage.morphology import skeletonize
from scipy.spatial import distance
from shapely.geometry import Polygon, Point, LineString, GeometryCollection, MultiPoint
from shapely import affinity
from scipy import interpolate
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from pandas_test.polynomial_cleaning import polynomial_fit

BASE_DIR = os.path.dirname(__file__)


def plot_show(image, title: str):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_save(image, title: str, filepath: str):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filepath)


def plot_construct_point(base_image, *, pts=None, x_v=None, y_v=None, inverted: bool = False):
    if inverted:
        new_image = np.zeros_like(base_image)
        point_color = 255
    else:
        new_image = 255 - np.zeros_like(base_image)
        point_color = 0
    if pts:
        for pt in pts:
            cv2.circle(new_image, pt, 1, point_color, -1)
        return new_image
    else:
        for x_point, y_point in zip(x_v, y_v):
            pts = (int(x_point), int(y_point))
            cv2.circle(new_image, pts, 1, point_color, -1)
        return new_image


def extendline(line,length):
  a=line[0]
  b=line[1]
  lenab= distance.euclidean(a,b)
  cx = b[0] + ((b[0]-a[0]) / lenab*length)
  cy = b[1] + ((b[1]-a[1]) / lenab*length)
  return [cx,cy]


def XYclean(x,y):
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    # make PCA object
    pca = PCA(2)
    # fit on data
    pca.fit(xy)
    #transform into pca space
    xypca = pca.transform(xy)
    newx = xypca[:,0]
    newy = xypca[:,1]
    #sort
    indexSort = np.argsort(x)
    newx = newx[indexSort]
    newy = newy[indexSort]

    #add some more points (optional)
    f = interpolate.interp1d(newx, newy, kind='linear')
    newX=np.linspace(np.min(newx), np.max(newx), 100)
    newY = f(newX)

    # #smooth with a filter (optional)
    # window = 43
    # newY = savgol_filter(newY, window, 2)

    #return back to old coordinates
    xyclean = pca.inverse_transform(np.concatenate((newX.reshape(-1,1), newY.reshape(-1,1)), axis=1) )
    xc=xyclean[:,0]
    yc = xyclean[:,1]
    return np.hstack((xc.reshape(-1,1),yc.reshape(-1,1))).astype(int)

def contour2skeleton(cnt):
  x,y,w,h = cv2.boundingRect(cnt)
  cnt_trans = cnt - [x,y]
  bim = np.zeros((h,w))
  bim = cv2.drawContours(bim, [cnt_trans], -1, color=255, thickness=cv2.FILLED) //255
  plt.imshow(bim)
  plt.title("BIM")
  plt.show()
  sk = skeletonize(bim >0)
  #####
  skeleton_yx = np.argwhere(sk > 0)
  skeleton_xy = np.flip(skeleton_yx, axis=None)
  xx,yy=skeleton_xy[:,0],skeleton_xy[:,1]
  skeleton_xy= XYclean(xx,yy)
  skeleton_xy = skeleton_xy +[x,y]
  return skeleton_xy


img_filenames = next(os.walk(os.path.join(os.path.dirname(__file__), "src_img")))[2]
# img_filenames = img_filenames[:100]

for img_fname in img_filenames:
    img_path = os.path.join(os.path.dirname(__file__), "src_img", img_fname)
    mm = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(mm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts ,_ = cv2.findContours(binary_image.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = cnts[0].reshape(-1, 2)

    contour = max(cnts, key=cv2.contourArea)

    # Ensure contour is in numpy array format
    if isinstance(contour, cv2.UMat):
        contour = contour.get()  # or np.array(contour.get()) for clarity

    # Flatten the array and process
    points = contour.reshape(-1, 2)  # Use reshape directly instead of squeeze

    # Group y-values by x-values
    x_coords = np.unique(points[:, 0])
    median_points = []

    for x in x_coords:
        y_vals = points[points[:, 0] == x, 1]
        median_y = (np.min(y_vals) + np.max(y_vals)) / 2
        median_points.append((x, int(median_y)))

    # Sort points by x coordinate to facilitate plotting
    median_points.sort()

    # Median Line on base plot
    median_image = np.zeros_like(binary_image)

    new_median_points = list(map(lambda pt: (int(pt[0]), int(pt[1])), median_points))

    x_val = []
    y_val = []

    for point in median_points:
        x_val.append(point[0])
        y_val.append(point[1])
        cv2.circle(median_image, point, 1, 255, -1)

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    result_image = 255 - median_image

    median_filepath = os.path.join(BASE_DIR, "median_points", img_fname)
    plot_save(result_image, "Median points", median_filepath)

    x, y = polynomial_fit(x_val[4:len(x_val)-5], y_val[4:len(x_val)-5], median_image)

    fitted_poly = plot_construct_point(median_image, x_v=x, y_v=y)
    poly_plot_path = os.path.join(BASE_DIR, "mask_fit", img_fname)

    combined_image = cv2.bitwise_and(mm, fitted_poly)
    plot_save(combined_image, "Fitted polynomial", poly_plot_path)

    result_image_mask = result_image
    combined_image = cv2.bitwise_and(mm, result_image_mask)

    mask_median_points = os.path.join(BASE_DIR, "mask_median", img_fname)
    plot_save(combined_image, "Mask", mask_median_points)

    print(f"Done with img {img_fname}")