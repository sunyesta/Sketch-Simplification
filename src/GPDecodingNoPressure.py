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


# Classes


class Segments:
    # points can not be reassigned to other segments
    def __init__(self, segments):

        # generate points by flattening the outermost dimention of segments
        points = []
        segMap = []
        for i, seg in enumerate(segments):
            for point in seg:
                points.append(point)
                segMap.append(i)

        self.points = np.array(points)
        self._segMap = np.array(segMap)

    def segCount(self):
        return self._segMap[-1] + 1

    def getSeg(self, point_i):
        return self._segMap[point_i]

    def toList(self):
        segments = [[] for _ in range(self.segCount())]

        for i, point in enumerate(self.points):
            segments[self._segMap[i]].append(point)

        return segments

    def equivalent(self, other):
        return np.array.equal(self._segMap.other._segMap)


class PixelEncoding:
    def __init__(self, pt1_index, lerp, delta):
        self.pt1_index = pt1_index  # first pt in line tbat is made up of 2 consecutive points (pt1,pt2) where pt2_index = pt1_index + 1
        self.lerp = lerp  # distance along line
        self.delta = delta  # distance from line


# Other


def generateSketch(segments, t, seed):
    random.seed(seed)
    np.random.seed(seed)
    return segments


# --- Encodings --- #


# normalizes strokes and return aspect ratio
def JSONDecode(file_path):
    def normalizeMLS(multiLineString, padding=0.2):
        assert padding < 1
        xmin, ymin, xmax, ymax = multiLineString.bounds

        dimentions = Point(xmax - xmin, ymax - ymin)
        maxDim = max(dimentions.x, dimentions.y)

        # normalize strokes so that the points lie between 0 and 1
        def normalize(x, y):
            return (x - xmin) / (maxDim), (y - ymin) / (maxDim)

        multiLineString = transform(normalize, multiLineString)

        # center points around (0.5, 0.5)
        minx, miny, maxx, maxy = multiLineString.bounds
        boundsCenter = Point((maxx - minx) / 2, (maxy - miny) / 2)

        def center(x, y):
            return x - boundsCenter.x + 0.5, y - boundsCenter.y + 0.5

        multiLineString = transform(center, multiLineString)

        # scale image so that padding fits around it
        multiLineString = shapely.affinity.scale(
            multiLineString, 1 - padding, 1 - padding
        )

        return multiLineString

    def changeMLSResolution(strokes, resolution):
        newStrokes = []
        for stroke in strokes.geoms:
            pointCount = max(int(stroke.length * resolution), 2)

            newStrokes.append(
                [
                    stroke.interpolate(t / pointCount, normalized=True)
                    for t in range(pointCount)
                ]
            )

        newStrokes = MultiLineString(newStrokes)

        return newStrokes

    # round the points to 3 decimal places
    def roundPoints(x, y):
        return round(x, 4), round(y, 4)

    def multiLineStringToList(multiline):
        lines = []
        for line in multiline.geoms:
            line_coords = []
            for coord in line.coords:
                line_coords.append(list(coord))
            lines.append(line_coords)
        return lines

    # load stroke data
    RAWStrokeInfo = None
    with open(file_path, "r") as file:
        RAWStrokeInfo = json.load(file)

    strokes = MultiLineString(RAWStrokeInfo)
    strokes = normalizeMLS(strokes)

    strokes = transform(roundPoints, strokes)
    strokes = changeMLSResolution(strokes, 150)
    segments = Segments(multiLineStringToList(strokes))
    return segments


def JSONEncode(outfile, segments):
    with open(outfile, "wt") as file:
        json.dump(segments.toList(), file)


def getPixelEncoding(img, segments):
    width, height = img.size
    pixels = np.array(img)
    spacialSegments = KDTree(segments.points)

    def pixelPoint(x, y):
        return (x / width, y / height)

    def findClosestPair(point, maxPoints=10):
        distances, indexes = spacialSegments.query(point, k=10)
        for i, first_i in enumerate(indexes):
            for j in range(i + 1, len(indexes)):
                second_i = indexes[j]

                # ensure that the points are on the same line and next to eachother
                if (
                    segments.getSeg(first_i) == segments.getSeg(second_i)
                    and abs(first_i - second_i) == 1
                ):
                    # only need to return first point in the pair
                    return min(first_i, second_i)

        return None

    pixelEncodings = [[None for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):

            # if pixel value is white, don't encode it
            v = pixels[x, y]
            if v == 255:
                continue

            point = pixelPoint(x, y)
            # get 2 closest points within the same segment
            closest = findClosestPair(point)
            if closest == None:
                print("NONEEEE")
            point = Point(point)
            # get pixel's distance and lerp information
            pt1, pt2 = segments.points[closest], segments.points[closest + 1]
            line = LineString([pt1, pt2])

            lerp = line.project(point, normalized=True)

            # TODO  rename distance?s
            # get point displacement from line
            delta = line.distance(point)
            slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
            ptAboveLine = (pt1[1] - point.y) > 0
            if ptAboveLine and slope > 0:
                delta *= -1

            # store the data
            pixelEncodings[x][y] = PixelEncoding(closest, lerp, delta)

    # for each black pixel:
    # flood search to find the corresponding points
    # calculate the encoding of the pixel given the points

    return np.array(pixelEncodings)


def imgFromPixelEncoding(segments, pixelEncodings, imgSize):
    width, height = imgSize
    img = Image.new("1", imgSize, 1)
    pixels = img.load()

    def getPointFromEncoding(pixe):
        # print("index = ", pixe.pt1_index)
        pt1, pt2 = segments.points[pixe.pt1_index], segments.points[pixe.pt1_index + 1]
        line = LineString([pt1, pt2])
        line_angle = math.atan2((pt2[1] - pt1[1]), (pt2[0] - pt1[0]))

        pt = line.interpolate(pixe.lerp, normalized=True)

        return np.array(
            [
                pt.x + pixe.delta * abs(math.cos(line_angle)),
                pt.y + pixe.delta * math.sin(line_angle),
            ]
        )
        # return segments.points[pixe.pt1_index]

    for y in range(height):
        for x in range(width):
            pixe = pixelEncodings[x, y]
            if pixe:
                point = getPointFromEncoding(pixe)
                pixel = np.array([point[0] * width, point[1] * height])
                # print((x, y), (pixel[0], pixel[1]))
                pixels[pixel[0], pixel[1]] = 0

    img.show()


# image rendering


def plotStrokes(ax, segments, t, seed, showPoints=False, randColors=False):

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect(1)

    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect(1)

    newSegments = generateSketch(segments, t, seed)
    for seg_i, seg in enumerate(newSegments.toList()):

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


def previewImage(segments, showPoints=False, randColors=False, seed=10):

    BRUSH_SIZE = 2
    FIGSIZE = 5
    plt.figure(figsize=(FIGSIZE, FIGSIZE))
    ax = plt.axes()

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


generatedJSONDir = Path(
    "/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/generatedJSON"
)

segments = JSONDecode(
    f"/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/disposable/blenderGeneratedJSON/monkey.json",
)

img = Image.open(
    "/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/disposable/blenderGeneratedRenders/monkey.png"
)
img = ImageOps.grayscale(img)

# previewImage(segments=segments, showPoints=False, randColors=True)
pixelEncoding = getPixelEncoding(img, segments)

imgFromPixelEncoding(segments, pixelEncoding, img.size)
# JSONEncode(generatedJSONDir / "gear.json", strokes)
