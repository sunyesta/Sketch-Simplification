from IPython.display import display
import matplotlib.pyplot as plt
import json
import numpy as np
import shapely
from shapely import Point, LineString, MultiLineString, get_coordinates
from shapely.ops import transform
import pandas as pd
from matplotlib.collections import LineCollection
import random
from random import randint
import math
from statistics import mean
from shapely import MultiLineString as Mls
from numpy import array as vec
import sklearn
from sklearn.preprocessing import normalize as normalizeVec
from matplotlib.widgets import Slider
from pathlib import Path
from PIL import Image, ImageOps
from scipy.spatial import KDTree
import io
import cairo
import os
from torchvision import transforms

# helpers


def randFloat(lower_bound, upper_bound):
    return random.random() * (upper_bound - lower_bound) + lower_bound


# Classes


class Segments:

    def __init__(self, segMatrix):
        """creates a Segments inst

        Args:
            segMatrix (3d array): array of arrays of points where each inner array represents 1 segment [[pt1, pt2, ...], [p1, pt2,...], ...]
        """

        # generate points by flattening the outermost dimention of segments
        points = []
        segMap = []
        for i, seg in enumerate(segMatrix):
            for point in seg:
                points.append(point)
                segMap.append(i)

        # assign the properties
        self.points = np.array(points)
        self.seg_map = np.array(segMap)

    def segCount(self):
        """returns the amount of segments in this instance"""
        return max(self.seg_map) + 1

    def getSeg(self, point_i):
        """get the segment that corresponds to a point
        Args:
            point_i (int): index of point in self

        Returns:
            (int): seg number
        """
        return self.seg_map[point_i]

    def toMatrix(self, useNumpy=True):
        """turns the segment obj to a matrix where its shape is like (segment, point_x, point_y)

        Returns:
            list of np arrays: the matrix described above
        """
        segments = [[] for _ in range(self.segCount())]

        for i, point in enumerate(self.points):
            segments[self.seg_map[i]].append((point[0], point[1]))

        if useNumpy:
            segments = [np.array(seg) for seg in segments]
        return segments

    def equivalent(self, other):
        """check if instances contain the same number of points in each segment"""
        return np.array.equal(self.seg_map, other.seg_map)

    def copy(self):
        """copies self"""
        return Segments(self.toMatrix())

    def export(self, outfile):
        """exports the segments into a json file

        Args:
            outfile (Path): json file where you want to save the segments to
        """
        with open(outfile, "wt") as file:
            json.dump(self.toMatrix(useNumpy=False), file)

    @staticmethod
    def load(infile, fromBlender=False):
        """loads segment data from a json file (Note: use self.export() to save segment data)

        Args:
            infile (Path): json file that you want to read from

        Returns:
            (Segments): an instance of Segments
        """

        points_mat = json.load(open(infile, "r"))

        if fromBlender:
            points_mat = [
                [[point[0], 1 - point[1]] for point in seg] for seg in points_mat
            ]

        return Segments(points_mat)


class PixelEncoding:
    def __init__(self, pt1_index, lerp, delta):
        self.pt1_index = pt1_index  # first pt in line tbat is made up of 2 consecutive points (pt1,pt2) where pt2_index = pt1_index + 1
        self.lerp = lerp  # distance along line
        self.delta = delta  # distance from line

    def toPoint(self, segments):
        """turns self into a point

        Args:
            segments (Segments):  Segments that is the same or equivalent to the instance of Segments used to generate the encoding

        Returns:
            (x,y): the point
        """
        pt1, pt2 = segments.points[self.pt1_index], segments.points[self.pt1_index + 1]
        line = LineString([pt1, pt2])
        line_angle = math.atan2((pt2[1] - pt1[1]), (pt2[0] - pt1[0]))
        pt = line.interpolate(self.lerp, normalized=True)

        return np.array(
            [
                pt.x + self.delta * math.cos(line_angle + math.radians(90)),
                pt.y + self.delta * math.sin(line_angle + math.radians(90)),
            ]
        )


class PixelEncodingSet:
    def __init__(self, size, pixelEncodings=None):
        """creates a PixelEncodingSet

        Args:
            pixelEncodings (list): list of PixelEncoding instances
            size (int): size of the original image (size,size)
        """

        self.size = size

        if pixelEncodings:
            self.items = pixelEncodings
            assert (
                len(pixelEncodings) == size * size
            ), f"pixelEncodings must be able to be reshaped into a square of size ({size},{size})"
        else:
            self.items = [None for _ in range(size * size)]

    def toMatrix(self):

        return self.items.reshape(self.imgShape)

    def get_points(self, segments):
        """applies pixelEncoding.getPoint() to each point in the pixeSet

        Args:
            segments (Segments): an instance of segments that is equivalent to the one used to generate the pixeSet

        Returns:
            (np array of points): np array of points from the pixeSet
        """
        points = np.zeros((self.size * self.size, 2))

        for i, pixe in enumerate(self.items):
            if pixe:
                point = pixe.toPoint(segments)
                points[i][0] = point[0]
                points[i][1] = point[1]
        return points

    def __getitem__(self, index):
        x, y = index
        return self.items[y * self.size + x]

    def __setitem__(self, index, value):
        x, y = index
        self.items[y * self.size + x] = value

    def export(self, outfile=None):
        """exports the pixeSet data

        Args:
            outfile (Path, optional): output file

        Returns:
            (list): pixe data ready for export
        """
        # print(self.items)
        exportData = []
        exportData.append(self.size)
        exportData.append(
            [
                (
                    [int(item.pt1_index), float(item.lerp), float(item.delta)]
                    if item
                    else None
                )
                for item in self.items
            ]
        )

        if outfile:
            json.dump(exportData, open(outfile, "wt"))

        return exportData

    @staticmethod
    def load(infile_path):
        size, data = json.load(open(infile_path, "r"))
        pixes = [
            PixelEncoding(item[0], item[1], item[2]) if item else None for item in data
        ]
        return PixelEncodingSet(size, pixes)

    def img(self, segs):
        size = self.size
        img = Image.new("L", (size, size), 255)
        pixels = img.load()

        # itterate through all the pixels of the image and plot the pixel at it's correct position determined by the pixe
        for x in range(size):
            for y in range(size):
                pixe = self[x, y]
                if pixe:
                    point = pixe.toPoint(segs)
                    pixel = pointToPixel(point, size)
                    pixels[pixel[0], pixel[1]] = 0

        return img


# Other


def pixelToPoint(pixel_xy, imgsize):
    return pixel_xy / imgsize


def pointToPixel(point_xy, imgsize):
    return np.round(point_xy * imgsize).astype(int)


def generateSketchSegs(segments, t, seed=1):
    """
    Generates sketch segments based on a set of input segments.

    Args:
        segments (Segments)
        t (float):  A number between 0 and 1 representing the percentage
        seed (int):  Seed

    Returns:
        (Segments): segments representing the sketch
    """
    assert 0 <= t and t <= 1, f"t must be between 0 and 1. Right now it's {t}"
    # set the seed
    random.seed(seed)
    np.random.seed(seed)

    # modifier that dashes the segment
    def dash(segMat, dashLength, dashLengthRange=0):
        assert dashLength > 0, "dashLength must be > 0"

        dashLengthRange = dashLengthRange // 2

        orignalDashLength = dashLength

        def newDashLength():
            return orignalDashLength + random.randint(-dashLengthRange, dashLengthRange)

        newSegs = []

        for seg in segMat:
            start = 1
            while start < len(seg) - 1:
                end = min(len(seg) - 1, start + newDashLength())
                newSegs.append(seg[start - 1 : end + 1])
                start = end

        return [np.array(seg) for seg in newSegs]

    # rotates a set of points around their center
    def rotate_points(points, angle):
        # TODO review
        # Convert angle to radians
        angle_rad = np.radians(angle)

        # Get the center of the points
        center = np.mean(points, axis=0)

        # Shift the points to be centered around the origin
        shifted_points = points - center

        # Create the rotation matrix
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )

        # Rotate the points
        rotated_points = np.dot(shifted_points, rotation_matrix)

        # Shift the points back to their original position
        rotated_points += center

        return rotated_points

    # cuts segments and rotates them individually
    def cutAndRotate(segMat, cutLen, angleRange):
        angleRange = angleRange / 2
        segMat = dash(segMat, cutLen, 5)
        for i, seg in enumerate(segMat):
            segMat[i] = rotate_points(seg, randFloat(-angleRange, angleRange))

        return segMat

    segMat = segments.toMatrix()

    segMat = cutAndRotate(segMat, 10, 40 * t)
    segMat = cutAndRotate(segMat, 5, 20 * t)

    return Segments(segMat)


# --- Encodings --- #


def generatePixelEncodings(segs, img):
    """generate a PixelEncodingSet from an image and it's corresponding Segments instance

    Args:
        segments (Segments): segments that make up the image
        img (PIL image): image to use as reference when doing the pixel encoding

    Returns:
        (PixelEncodingSet): a set of pixel encodings
    """

    # create the image

    # get the image size
    assert img.size[0] == img.size[1], "image must be square"
    size = img.size[0]
    print("SIZE =", size)

    # get the image pixel data
    img = img.convert("L")
    pixels = img.load()

    # generate a spacial representation of the points
    spacialSegments = KDTree(segs.points)

    def findClosestPair(point, maxPoints=10):
        """finds the closest pair of points that are on the same line and next to eachother

        Args:
            point (2d tuple): base point
            maxPoints (int, optional): max set of closest points to look for. Defaults to 10.

        Returns:
            (int): returns the index for the first point found. The second point's index = first_i + 1
        """

        # itterate through all the closest points
        distances, indexes = spacialSegments.query(point, k=10)
        for i, first_i in enumerate(indexes):
            for j in range(i + 1, len(indexes)):
                second_i = indexes[j]

                # ensure that the points are on the same line and next to eachother
                if (
                    segs.getSeg(first_i) == segs.getSeg(second_i)
                    and abs(first_i - second_i) == 1
                ):
                    # only need to return first point in the pair
                    return min(first_i, second_i)

        # case if no valid points are present
        return None

    pixeSet = PixelEncodingSet(size)

    # itterate through every pixel in the image
    for x in range(size):
        for y in range(size):

            # if pixel value is white, don't encode it
            v = pixels[x, y]
            if v == 255:
                continue
            # find the pixel's corresponding points
            point = pixelToPoint(np.array([x, y]), size)
            closest = findClosestPair(point)
            if closest == None:
                print("Error closest point not found")

            # get point and line info
            point = Point(point)
            pt1, pt2 = segs.points[closest], segs.points[closest + 1]
            line = LineString([pt1, pt2])

            # get pixel's distance and lerp information
            lerp = line.project(point, normalized=True)

            # get point displacement from line
            delta = line.distance(point)

            # if point is above the line, give it negative distance
            pointOnLine = line.interpolate(lerp, normalized=True)
            ptAboveLine = (pointOnLine.y - point.y) > 0
            if ptAboveLine:
                delta *= -1

            # store the data
            pixeSet[x, y] = PixelEncoding(closest, lerp, delta)

    return pixeSet


# UNUSED
def imgFromPixelEncoding(segs, pixelEncodingSet):
    """generates an image from segments and a corresponding pixelEncodingSet

    Args:
        segments (Segments): segments
        pixelEncodingSet (PixelEncodingSet): pixel encoding set generated with segs or an equivalent to segs

    Returns:
        (PIL img): image generated from the pixel encoding set
    """

    size = pixelEncodingSet.size
    img = Image.new("1", (size, size), 1)
    pixels = img.load()

    # itterate through all the pixels of the image and plot the pixel at it's correct position determined by the pixe
    for x in range(size):
        for y in range(size):
            pixe = pixelEncodingSet[x, y]
            if pixe:
                point = pixe.toPoint(segs)
                pixel = pointToPixel(point, size)
                pixels[pixel[0], pixel[1]] = 0

    return img


# FOR DEBUGGING
def displayOverlayImages(img1, img2, alpha=0.3):
    # both images must be Mode = "1"

    img1 = img1.copy()

    # Set the alpha value for each pixel in the second image to 50% (128 out of 255)
    alpha_mask = Image.new("L", img2.size, 255 - int(alpha * 255))

    # Overlay the second image with the alpha mask on top of the first image
    img1.paste(img2, mask=alpha_mask)

    img1.show()


# ----- rendering images -----


def interactiveSegsPreview(segments, showPoints=False, randColors=False, seed=10):

    def plotStrokes(ax, segments, t, seed, showPoints=False, randColors=False):

        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect(1)
        ax.invert_yaxis()

        newSegments = generateSketchSegs(segments, t, seed)
        for seg_i, seg in enumerate(newSegments.toMatrix()):

            points = [
                [(seg[i][0], seg[i][1]), (seg[i + 1][0], seg[i + 1][1])]
                for i in range(len(seg) - 1)
            ]

            color = None
            if randColors:
                color = np.random.rand(3)
            else:
                color = "black"

            # ax.plot(points, color="black")
            lc = LineCollection(points, color=color)
            ax.add_collection(lc)

        if showPoints:
            ax.scatter(
                segments.points[:, 0], segments.points[:, 1], c="red", zorder=10, s=3
            )

    BRUSH_SIZE = 2
    FIGSIZE = 5

    # configure plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect(1)

    # add slider
    ax_slider = fig.add_axes([0.2, 0.0, 0.65, 0.03])
    eInit = 0.5
    slider = Slider(ax=ax_slider, label="x", valmin=0, valmax=1, valinit=eInit)

    def update(t):
        ax.clear()
        plotStrokes(ax, segments, t, seed, showPoints=showPoints, randColors=randColors)

    update(eInit)
    slider.on_changed(update)

    plt.show()


def interactivePixePreview(
    segs, steps, img_size=300, showPoints=False, randColors=False, seed=10
):

    BRUSH_SIZE = 2
    FIGSIZE = 5

    # configure plot
    fig, ax = plt.subplots()

    # ax.

    # plt.gca().invert_yaxis()
    # add slider
    ax_slider = fig.add_axes([0.2, 0.0, 0.65, 0.03])
    eInit = 0.5
    slider = Slider(
        ax=ax_slider, label="x", valmin=0, valmax=steps - 1, valinit=eInit, valfmt="%d"
    )

    # get pixe data

    img = renderSegsCario(segs, img_size)

    segs_baked = []
    for step in range(steps):
        segs_baked.append(generateSketchSegs(segs, step / steps))

    pixe_set = generatePixelEncodings(segs_baked[0], img)

    imgs = []
    for seg in segs_baked:
        imgs.append(np.array(pixe_set.img(seg)))

    print(imgs[0])

    # get timestep data

    def update(t):
        t = int(t)
        ax.clear()

        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.set_aspect(1)
        # ax.invert_yaxis()

        points = pixe_set.get_points(segs_baked[t])
        # ax.scatter(points[:, 0], points[:, 1], c="red", zorder=10, s=0.1)
        ax.imshow(imgs[t])

    update(eInit)
    slider.on_changed(update)

    plt.show()


def renderSegsCario(segments, size):

    tempSize = 5000
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, tempSize, tempSize)
    ctx = cairo.Context(surface)

    # White background
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, tempSize, tempSize)
    ctx.fill()

    # black pen
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(10)

    # render each stroke
    for seg_i, seg in enumerate(segments.toMatrix()):
        pixels = [pointToPixel(point, tempSize) for point in seg]

        ctx.move_to(pixels[0][0], pixels[0][1])

        for pixel in pixels:
            ctx.line_to(pixel[0], pixel[1])

        ctx.stroke()

    # convert to PIL image
    data = surface.get_data()
    img = Image.frombytes("RGBA", (tempSize, tempSize), data)
    img = img.resize((size, size))

    return img


# ----- getting offsets -----


def generateOffsets(pixeSet, segs1, segs2):

    size = pixeSet.size

    offsets = np.zeros([size, size, 2])

    for x in range(size):
        for y in range(size):

            pixe = pixeSet[x, y]
            if pixe:
                offsets[y, x] = pixe.toPoint(segs2) - pixe.toPoint(segs1)

    return offsets


def generateSteppedOffsets(segs, size=300, steps=10, seed=1):
    """generates a list of stepped offsets given a segMatrix

    Args:
        jsonPath (Path): path of the segMatrix json file
        size (int, optional): size of the output image (size,size). Defaults to 300.
        steps (int, optional): number of steps to generate. Defaults to 10.
        seed (int, optional): random seed. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # get the segments from the json file

    # turn the segments into a non-sketchy sketch, timestep = 0
    segsBaked0 = generateSketchSegs(segs, t=0, seed=seed)

    # generate a clean image from the segments
    orgImg = renderSegsCario(segments=segsBaked0, size=size)

    # generate the pixel encodings from the clean image and the original segments
    pixelEncodings = generatePixelEncodings(segsBaked0, orgImg)

    segsBakedLast = segsBaked0
    steppedOffsets = []
    for step in range(0, steps):
        segsBakedStep = generateSketchSegs(segs, t=step / steps, seed=seed)
        steppedOffsets.append(
            generateOffsets(pixelEncodings, segsBakedLast, segsBakedStep)
        )
        segsBakedLast = segsBakedStep

    print("points shape = ", segsBaked0.points.shape)
    return pixelEncodings.get_points(segsBaked0), steppedOffsets


def bakeOffsets(orgPoints, offsets_set):
    """applies a set of offsets to the pixels

    Args:
        orgPoints (point array): array of points
        offsets (point array): array of offsets which are represented by (x,y) vectors

    Returns:
        (point array list): a point array for each baked offset
    """
    bakedOffsets = [orgPoints]

    # itterate through each of the offsets
    for i in range(1, len(offsets_set)):
        offsets = offsets_set[i]

        # append last points + offset
        bakedOffsets.append(
            np.array(bakedOffsets[i - 1])
            + np.array(offsets).reshape((offsets.shape[0] * offsets.shape[1], 2))
        )

    return bakedOffsets


def bakeOffset(orgPoints, offsets_set):
    """cumulativley applies offsets to the org points

    Args:
        orgPoints (array of shape (n,2)): array of origional points
        offsets (array of shape (n,2)): array of offsets

    Returns:
        (array of shape (n,2)): array of points after offsets have been applied
    """
    lastPoints = orgPoints
    for i in range(len(offsets_set)):
        lastPoints = lastPoints + offsets_set[i]
    return lastPoints


def viewOffsets(orgPoints, offsets):
    steps = len(offsets)

    # configure plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect(1)

    # add slider
    ax_slider = fig.add_axes([0.2, 0.0, 0.65, 0.03])
    eInit = 0
    slider = Slider(
        ax=ax_slider, label="x", valmin=0, valmax=steps - 1, valinit=eInit, valfmt="%d"
    )

    bakedOffsets = bakeOffsets(orgPoints, offsets)

    def update(t):
        t = int(t)
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect(1)

        # # TODO do this
        bakedOffset = bakedOffsets[t]
        ax.scatter(bakedOffset[:, 0], bakedOffset[:, 1], c="red", zorder=10, s=0.1)

    update(eInit)
    slider.on_changed(update)

    plt.show()


def exportData(infile, outdir, img_size):
    """exports model ready data from an image point matrix

    Args:
        infile (Path): json file with image point matrix
        outdir (Path): dir to output the image data
        img_size (int): output image will be size (img_size,img_size)
    """
    segs = None
    segs = Segments.load(infile, fromBlender=True)

    segsBaked = generateSketchSegs(segs, t=0)

    img = renderSegsCario(segsBaked, img_size)

    pixe_set = generatePixelEncodings(segsBaked, img)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pixe_set.export(outdir / Path("pixe_set.json"))

    # points to 1 if pixel is black and 0 if pixel is white
    img.save(outdir / Path("base.png"))

    segs.export(out_dir / Path("segs.json"))


# ----- Main Code -----

print("testing2")

points_json_path = Path(
    f"/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/disposable/blenderGeneratedJSON/monkey.json"
)


out_dir = Path(
    f"/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/dataset"
) / Path("monkey")


exportData(points_json_path, out_dir, 64)

# segs = Segments.load(points_json_path, fromBlender=True)
# segs = Segments.load(
#     Path(
#         f"/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/dataset/monkey/segs.json"
#     ),
#     fromBlender=False,
# )
# img = renderSegsCario(segs, 1000)
# pixe_set = generatePixelEncodings(segs, img)

# data_transforms(pixe_set)

# pixe_set.export("./test.json")
# pixe_set = PixelEncodingSet.load("./test.json")
# img.show()
# pixe_set.img(segs).show()
# segsBaked1 = generateSketchSegs(segs, 0, 1)
# segsBaked2 = generateSketchSegs(segs, 1, 1)


# interactivePixePreview(segs, 10, img_size=500)

# pixe_set.img(segs).show()

# interactiveSegsPreview(segs)
