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


def resampleList(data, target_length):
    """
    Resamples an ordered list to a different length using linear interpolation.

    Args:
      data: The input list.
      target_length: The desired length of the resampled list.

    Returns:
      A new list with the resampled elements.
    """

    if not data:
        return []

    # Calculate the spacing between samples based on the original data length.
    step = (len(data) - 1) / (target_length - 1)

    # Iterate through the list and interpolate values at each step.
    resampled_data = []
    for i in range(target_length):
        index = i * step
        # Check if the index is within the list range (handle edge cases).
        if 0 <= index < len(data) - 1:
            # Perform linear interpolation between the two elements.
            weight1 = 1 - (index % 1)
            weight2 = index % 1
            value = data[int(index)] * weight1 + data[int(index + 1)] * weight2
            resampled_data.append(value)
        else:
            # If the index is out of range, use the last or first element.
            if index < 0:
                resampled_data.append(data[0])
            else:
                resampled_data.append(data[-1])

    return resampled_data


def merge_last_two(data):
    """Merges the last two sublists of a 2D list.

    Args:
      data: A 2D list.

    Returns:
      A new 2D list with the last two sublists merged.
    """

    if len(data) < 2:
        return data

    # Get the last two sublists.
    last_two = data[-2:]

    # Merge the last two sublists.
    merged_list = last_two[0] + last_two[1]

    # Combine the merged list with the remaining sublists.
    return data[:-2] + [merged_list]


def randFloat(lower_bound, upper_bound):
    return random.random() * (upper_bound - lower_bound) + lower_bound


def multiLineStringToList(multiline):
    lines = []
    for line in multiline.geoms:
        line_coords = []
        for coord in line.coords:
            line_coords.append(list(coord))
        lines.append(line_coords)
    return lines


class Strokes:
    def __init__(self, points: MultiLineString, pressure: list, alpha: list):
        assert type(points) == MultiLineString
        assert type(pressure) == list
        assert type(pressure[0]) == list
        assert type(alpha) == list
        assert type(alpha[0]) == list

        self.points = points
        self.pressure = pressure
        self.alpha = alpha

    def __len__(self):
        # returns number of strokes
        return len(self.points.geoms)

    def copy(self):

        # print([row[:] for row in self.pressure])

        # exit()
        return Strokes(
            MultiLineString(self.points),
            [row[:] for row in self.pressure],
            [row[:] for row in self.alpha],
        )

    def union(self, other):

        newPressure = self.pressure[:]
        newPressure.extend(other.pressure)

        newAlpha = self.alpha[:]
        newAlpha.extend(other.alpha)

        points1 = multiLineStringToList(self.points)
        points2 = multiLineStringToList(other.points)

        points1.extend(points2)
        unionedPoints = points1
        # unionedPoints = shapely.union(self.points, other.points)

        # print(
        #     "lengths = ",
        #     len(self.pressure),
        #     len(newPressure),
        #     len(self.pressure) + len(other.pressure),
        # )

        return Strokes(MultiLineString(unionedPoints), newPressure, newAlpha)

    def isValid(self):
        for i, strokePoints in enumerate(multiLineStringToList(self.points)):
            if len(strokePoints) != len(
                self.pressure[i] or len(strokePoints) != len(self.alpha[i])
            ):

                print(
                    "points:",
                    len(strokePoints),
                    "pressure: ",
                    len(self.pressure[i]),
                    "alpha:",
                    len(self.alpha[i]),
                )
                return False

        return True

    def cutSegments(self, cuts, taperedEdges=True):
        newPoints, newPressures, newAlphas = [], [], []

        # NOTE: for even cuts, segment points should already be evenly spaced apart
        for i in range(len(self)):
            stroke = self.points.geoms[i]
            pressures = self.pressure[i]
            alphas = self.alpha[i]

            strokeList = list(stroke.coords)

            # todo make length jitter
            pointsPerCut = max(2, len(strokeList) // cuts)

            # create new segments

            def segmentize(list):
                newList = []
                newList.extend(
                    [
                        list[i : i + pointsPerCut]
                        for i in range(0, len(list), pointsPerCut)
                    ]
                )
                if len(newList[-1]) <= 1:
                    newList = merge_last_two(newList)
                return newList

            newPoints.extend(segmentize(strokeList))
            newPressures.extend(segmentize(pressures))
            newAlphas.extend(segmentize(alphas))

        self.points = MultiLineString(newPoints)
        self.pressure = newPressures
        self.alpha = newAlphas


# TODO fix padding
def fitStrokes(multiLineString, padding=0.2):
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


def newStrokesResolution(strokes, resolution):

    newStrokes = strokes.copy()
    newPoints = []
    for i in range(len(newStrokes)):
        line = newStrokes.points.geoms[i]
        pointCount = max(int(line.length * resolution), 2)

        newStrokes.pressure[i] = resampleList(newStrokes.pressure[i], pointCount)
        newStrokes.alpha[i] = resampleList(newStrokes.alpha[i], pointCount)

        newPoints.append(
            [
                line.interpolate(t / pointCount, normalized=True)
                for t in range(pointCount)
            ]
        )

    print(newStrokes.pressure[1])

    newStrokes.points = MultiLineString(newPoints)

    return newStrokes


# normalizes strokes and return aspect ratio
def JSONDecode(file_path, square=False):

    # load stroke data
    RAWStrokeInfo = None
    with open(file_path, "r") as file:
        RAWStrokeInfo = json.load(file)

    strokesPoint = MultiLineString(RAWStrokeInfo["xy"])
    strokesPressure = RAWStrokeInfo["pressure"]
    strokesAlpha = RAWStrokeInfo["alpha"]

    strokesPoint = fitStrokes(strokesPoint)

    # round the points to 3 decimal places
    def roundPoints(x, y):
        return round(x, 4), round(y, 4)

    strokesPoint = transform(roundPoints, strokesPoint)

    strokes = Strokes(strokesPoint, strokesPressure, strokesAlpha)

    strokes = newStrokesResolution(strokes, 150)
    return strokes


def previewImage(strokes, showPoints=False, randColors=False):
    assert strokes.isValid(), "strokes is not valid"
    BRUSH_SIZE = 2
    FIGSIZE = 5
    plt.figure(figsize=(FIGSIZE, FIGSIZE))
    ax = plt.axes()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for stroke_i, stroke in enumerate(strokes.points.geoms):

        x, y = stroke.xy
        points = [[(x[j], y[j]), (x[j + 1], y[j + 1])] for j in range(len(x) - 1)]
        lineWidths = [pressure * BRUSH_SIZE for pressure in strokes.pressure[stroke_i]]

        if randColors:
            color = np.random.rand(3)
        else:
            color = "black"
        lc = LineCollection(points, linewidths=lineWidths, color=color)
        lc.set_capstyle("round")
        ax.add_collection(lc)

        if showPoints:
            ax.scatter(x, y, c="red", zorder=10, s=3)

    plt.show()


def JSONEncode(strokes, outfile):

    strokeObj = {
        "xy": multiLineStringToList(strokes.points),
        "pressure": strokes.pressure,
        "alpha": strokes.alpha,
    }
    # print(strokeObj["xy"])

    with open(outfile, "wt") as file:
        json.dump(strokeObj, file)


def cleanToSketch(strokes, e=1, seed=1):
    strokeCount = 3
    strokesList = [strokes.copy() for i in range(strokeCount)]

    # print(
    #     strokesList[0].isValid(),
    #     strokesList[1].isValid(),
    #     strokesList[2].isValid(),
    # )

    def addNoise(a):
        def addNoiseInterior(x, y):
            return x + randFloat(-a, a), y + randFloat(-a, a)

        return addNoiseInterior

    # strokesList[0].points = transform(addNoise(0.01), strokesList[0].points)
    # print(strokesList[1].pressure)

    def applyTransformationPerStroke(transformation, strokes):
        linesList = []
        for stroke in strokes.points.geoms:
            lineString = transformation(stroke)
            linesList.append(lineString)

        return MultiLineString(linesList)

    for cut in range(3):

        def rotateRand(stroke):
            return shapely.affinity.rotate(stroke, randFloat(-5 * cut, 5 * cut))

        for i, strokes in enumerate(strokesList):
            strokes.cutSegments(2)
            strokesList[i].points = applyTransformationPerStroke(rotateRand, strokes)
    # union all strokes into same strokes obj
    unioned = strokesList[0]
    for i in range(1, len(strokesList)):
        unioned = unioned.union(strokesList[i])

    return unioned


strokes = JSONDecode(
    f"/Users/mary/Documents/School/Sketch-Simplification/testStrokeData/monkey.json",
    square=False,
)

print("starting: ", strokes.isValid())

strokes = cleanToSketch(strokes)

print("after: ", strokes.isValid())


# JSONEncode(
#     strokes,
#     f"/Users/mary/Documents/School/Sketch-Simplification/generatedJSON/test.json",
# )

previewImage(strokes=strokes, showPoints=False, randColors=False)
