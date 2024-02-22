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

# --- Helpers --- #


def equivalentMls(mls1, mls2):
    if len(mls1.geoms) != len(mls2.geoms):
        return False

    for i in range(len(mls1.geoms)):
        line1, line2 = mls1.geoms[i], mls2.geoms[i]

        if len(line1.coords) != len(line2.coords):
            return False

    return True


def interpolateOut(line, t):
    if t <= 1 and t >= 0:
        return line.interpolate(t, normalized=True)

    else:
        lastPt, sndLastPt = None, None
        if t > 1:
            lastPt = vec(line.interpolate(1, normalized=True).xy)
            sndLastPt = vec(line.interpolate(0.999, normalized=True).xy)

        elif t < 0:
            lastPt = vec(line.interpolate(0, normalized=True).xy)
            sndLastPt = vec(line.interpolate(0.01, normalized=True).xy)

        direction = normalizeVec(lastPt - sndLastPt)

        return lastPt + direction * (abs(t) - 1)


def transformEach(transformation, mls):
    newMls = []
    for line in mls.geoms:
        newMls.append(transformation(line))
    return MultiLineString(newMls)


def randFloat(lower_bound, upper_bound):
    return random.random() * (upper_bound - lower_bound) + lower_bound


def simpleTransform(x, y):
    return x + 1, y + 1


# --- Deform Modifiers --- #


def addNoise(a):
    def addNoiseInterior(x, y):
        return x + randFloat(-a, a), y + randFloat(-a, a)

    return addNoiseInterior


# TODO fix padding
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
    multiLineString = shapely.affinity.scale(multiLineString, 1 - padding, 1 - padding)

    return multiLineString


def extendLines(mls, distance):
    newMls = []

    for line in mls.geoms:
        newLine = list(line.coords)

        newLine[-1] = interpolateOut(line, 1 + distance)
        newLine[0] = interpolateOut(line, -distance)
        newMls.append(newLine)

    return Mls(newMls)


# --- Generative Modifiers --- #


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


def segmentMLS(mls, segments):
    newMLS = []

    def merge_last_two(data):
        if len(data) < 2:
            return data

        last_two = data[-2:]

        merged_list = last_two[0] + last_two[1]

        # Combine the merged list with the remaining sublists.
        return data[:-2] + [merged_list]

    for line in mls.geoms:
        lineList = list(line.coords)

        pointsPerSegment = len(lineList) // segments

        offset = randint(0, pointsPerSegment // 2)

        if offset > 0:
            newMLS.append(lineList[0 : offset + 1])

        for seg in range(segments):
            segStart = seg * pointsPerSegment + offset
            newMLS.append(
                lineList[segStart : min(segStart + pointsPerSegment + 1, len(lineList))]
            )

            # newMLS.append([lineList[0:5]])

            if len(newMLS[-1]) < 2:
                newMLS = merge_last_two(newMLS)

    # print("lines = ", newMLS)
    return MultiLineString(newMLS)


def cleanToSketch(strokes, e=1):
    strokesList = [Mls(strokes) for _ in range(3)]
    # todo use guassian noise

    def rotateRand(range=20):
        def rotateRandHelper(stroke):
            return shapely.affinity.rotate(
                stroke, randFloat(-range / 2 * e, range / 2 * e), origin="center"
            )

        return rotateRandHelper

    def splitAndRotate(rotation):
        for i, strokes in enumerate(strokesList):
            strokesList[i] = segmentMLS(strokesList[i], 2)
            strokesList[i] = transformEach(rotateRand(rotation), strokesList[i])

    splitAndRotate(10)
    splitAndRotate(10)
    splitAndRotate(40)

    for i, strokes in enumerate(strokesList):
        pass
        # strokesList[i] = extendLines(strokesList[i], 0.01 * e)

    # union the strokesList
    unioned = strokesList[0]
    for i in range(1, len(strokesList)):
        unioned = unioned.union(strokesList[i])
    return unioned


# --- File Manipulation --- #


# normalizes strokes and return aspect ratio
def JSONDecode(file_path, square=False):

    # load stroke data
    RAWStrokeInfo = None
    with open(file_path, "r") as file:
        RAWStrokeInfo = json.load(file)

    strokes = MultiLineString(RAWStrokeInfo["xy"])
    strokes = normalizeMLS(strokes)

    # round the points to 3 decimal places
    def roundPoints(x, y):
        return round(x, 4), round(y, 4)

    strokes = transform(roundPoints, strokes)
    strokes = changeMLSResolution(strokes, 150)
    return strokes


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
    f"/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/testStrokeData/monkey.json",
    square=False,
)

# strokes = generateSketchFramework(strokes)
# strokes = cleanToSketch(strokes)

previewImage(strokes=strokes, showPoints=False, randColors=False)
