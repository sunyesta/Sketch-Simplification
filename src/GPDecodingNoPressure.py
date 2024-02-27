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

    def to2DArr(self):
        segments = [[] for _ in range(self.segCount())]

        for i, point in enumerate(self.points):
            segments[self._segMap[i]].append(point)

        return segments

    def equivalent(self, other):
        return np.array.equal(self._segMap.other._segMap)


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

    strokes = MultiLineString(RAWStrokeInfo["xy"])
    strokes = normalizeMLS(strokes)

    strokes = transform(roundPoints, strokes)
    strokes = changeMLSResolution(strokes, 150)
    segments = Segments(multiLineStringToList(strokes))
    # TODO convert to numpy

    return segments


def JSONEncode(outfile, strokes):
    with open(outfile, "wt") as file:
        json.dump(multiLineStringToList(strokes), file)


def getPixelEncoding(img, strokes):
    width, height = img.size()

    def pixelPoint(x, y):
        return Point(x / width, y / height)

    def findClosestPoints(x, y):
        # convert strokes into numpy of points
        # get closest point
        # remove that point from the point list
        # get next closest point
        # if both points are part of the same mls,then return the points
        # else, remove the point and get the next closest point
        return None

    encoding = []

    # for each black pixel:
    # flood search to find the corresponding points
    # calculate the encoding of the pixel given the points

    return None


# image rendering


def saveImages(infile, outdir, steps=100, seed=1):

    def plotStrokes(t, seed):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect(1)

        random.seed(seed)
        np.random.seed(seed)

        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect(1)

        newStrokes = cleanToSketch(strokes, t)
        for stroke_i, stroke in enumerate(newStrokes.geoms):
            x, y = stroke.xy
            points = [[(x[j], y[j]), (x[j + 1], y[j + 1])] for j in range(len(x) - 1)]
            # ax.plot(points, color="black")
            lc = LineCollection(points, color="black")
            ax.add_collection(lc)

        return plt

    strokes = JSONDecode(infile)

    outdir.mkdir(parents=True, exist_ok=True)

    plt.axis(False)

    for t in range(steps):
        plot = plotStrokes(t / steps, 1)
        plot.axis(False)
        plot.savefig(outdir / Path(str(t) + ".png"))


def saveImagesAsOne(infile, outFile, steps=100, seed=1):
    def plotStrokes(ax, t):

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect(1)

        random.seed(seed)
        np.random.seed(seed)

        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect(1)

        newStrokes = cleanToSketch(strokes, t)
        for stroke_i, stroke in enumerate(newStrokes.geoms):
            x, y = stroke.xy
            points = [[(x[j], y[j]), (x[j + 1], y[j + 1])] for j in range(len(x) - 1)]
            # ax.plot(points, color="black")
            lc = LineCollection(points, color="black")
            ax.add_collection(lc)

        return plt

    strokes = JSONDecode(infile)

    fig, axes = plt.subplots(1, steps)
    for t in range(steps):
        plot = plotStrokes(axes[t], t / steps)
        axes[t].axis(False)
        axes[t].margins(x=0)

    plot.show()
    # plot.savefig(outFile)


def previewImage(strokes, showPoints=False, randColors=False, seed=10):

    BRUSH_SIZE = 2
    FIGSIZE = 5
    plt.figure(figsize=(FIGSIZE, FIGSIZE))
    ax = plt.axes()

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect(1)

    # add slider
    ax_slider = fig.add_axes([0.2, 0.0, 0.65, 0.03])
    eInit = 0.5
    slider = Slider(ax=ax_slider, label="x", valmin=0, valmax=1, valinit=eInit)

    minSize = 0.5

    def taperFunc(x):
        return -((2 * x - 1) ** 2) + 1 + minSize

    # print(generateTaper(5, taperFunc))

    def update(e):

        random.seed(seed)
        np.random.seed(seed)

        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect(1)

        newStrokes = cleanToSketch(strokes, e)
        for stroke_i, stroke in enumerate(newStrokes.geoms):
            x, y = stroke.xy
            points = [[(x[j], y[j]), (x[j + 1], y[j + 1])] for j in range(len(x) - 1)]
            if randColors:
                color = np.random.rand(3)
            else:
                color = "black"
            # print("linewidths ", taper)
            # plot lines
            lc = LineCollection(points, color=color)
            # lc.set_capstyle("round")
            ax.add_collection(lc)

            if showPoints:
                ax.scatter(x, y, c="red", zorder=10, s=3)

    update(eInit)
    slider.on_changed(update)

    plt.show()


strokes = JSONDecode(
    f"/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/testStrokeData/gear.json",
)

generatedJSONDir = Path(
    "/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/generatedJSON"
)
JSONEncode(generatedJSONDir / "gear.json", strokes)

# strokes = generateSketchFramework(strokes)
# strokes = cleanToSketch(strokes)

previewImage(strokes=strokes, showPoints=False, randColors=False)
