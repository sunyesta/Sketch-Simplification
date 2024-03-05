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
        self.segMap = np.array(segMap)
        self.segMap.flags.writeable = False

    def segCount(self):
        return max(self.segMap) + 1

    def getSeg(self, point_i):
        return self.segMap[point_i]

    def toList(self):
        segments = [[] for _ in range(self.segCount())]

        for i, point in enumerate(self.points):
            segments[self.segMap[i]].append((point[0], point[1]))

        return segments

    def equivalent(self, other):
        return np.array.equal(self.segMap.other.segMap)

    def copy(self):
        return Segments(self.toList())

    def dash(self, dashLength, dashLengthRange=0):
        assert dashLength > 0, "dashLength must be > 0"

        dashLengthRange = dashLengthRange // 2

        orignalDashLength = dashLength

        def newDashLength():
            return orignalDashLength + random.randint(-dashLengthRange, dashLengthRange)

        self = self.copy()
        segMap = self.segMap
        # unlock the segMap
        self.segMap.flags.writeable = True

        nextSeg = self.segCount()

        dashLength = newDashLength()
        runLength = 1  # since we start at 1, that means our runlength is already 1
        for i in range(1, len(segMap)):

            # if we reach a new seg, reset the runLength
            if segMap[i] != segMap[i - 1]:
                runLength = 0

            runLength += 1

            # once we hit the dashLength, cut the segment
            if runLength > dashLength:
                runLength = 1  # since we are reseting the current val, we need to change the run length
                orgSeg = segMap[i]
                for j in range(i, len(segMap)):
                    # when we reach the end of the origional segment, stop changing the segment values
                    if segMap[j] != orgSeg:
                        break

                    segMap[j] = nextSeg

                nextSeg += 1
                dashLength = newDashLength()

        # relock the segMap
        self.segMap.flags.writeable = False

        return self


class PixelEncoding:
    def __init__(self, pt1_index, lerp, delta):
        self.pt1_index = pt1_index  # first pt in line tbat is made up of 2 consecutive points (pt1,pt2) where pt2_index = pt1_index + 1
        self.lerp = lerp  # distance along line
        self.delta = delta  # distance from line


# Other


def pixelToPoint(pixelXY, width, height):
    return (pixelXY[0] / width, 1 - pixelXY[1] / height)


def pointToPixel(pointXY, width, height):
    return (round(pointXY[0] * width), round(height - pointXY[1] * height))


def generateSketch(segments, t, seed):
    random.seed(seed)
    np.random.seed(seed)

    for cut in range(1):
        segments = segments.dash(5, 5)
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

        # multiLineString = transform(normalize, multiLineString)

        # center points around (0.5, 0.5)
        minx, miny, maxx, maxy = multiLineString.bounds
        boundsCenter = Point((maxx - minx) / 2, (maxy - miny) / 2)

        def center(x, y):
            return x - boundsCenter.x + 0.5, y - boundsCenter.y + 0.5

        # multiLineString = transform(center, multiLineString)

        # scale image so that padding fits around it
        # multiLineString = shapely.affinity.scale(
        #     multiLineString, 1 - padding, 1 - padding
        # )

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

    # segments = MultiLineString(RAWStrokeInfo)
    # segments = changeMLSResolution(segments, 100)
    segments = Segments(RAWStrokeInfo)
    return segments


def JSONEncode(outfile, segments):
    print(segments.toList()[0])
    with open(outfile, "wt") as file:
        json.dump(segments.toList(), file)


def getPixelEncoding(img, segments):
    width, height = img.size
    pixels = img.load()
    spacialSegments = KDTree(segments.points)

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
            # print(v)
            if v == (255, 255, 255, 255):
                # print("skipped")
                continue

            point = pixelToPoint((x, y), width, height)
            # get 2 closest points within the same segment
            closest = findClosestPair(point)
            if closest == None:
                print("Error closest point not found")

            point = Point(point)
            # get pixel's distance and lerp information
            pt1, pt2 = segments.points[closest], segments.points[closest + 1]
            line = LineString([pt1, pt2])

            lerp = line.project(point, normalized=True)
            pointOnLine = line.interpolate(lerp, normalized=True)
            # get point displacement from line
            delta = line.distance(point)
            # slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
            ptAboveLine = (pointOnLine.y - point.y) > 0
            if ptAboveLine:
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
        pt1, pt2 = segments.points[pixe.pt1_index], segments.points[pixe.pt1_index + 1]
        line = LineString([pt1, pt2])
        line_angle = math.atan2((pt2[1] - pt1[1]), (pt2[0] - pt1[0]))

        pt = line.interpolate(pixe.lerp, normalized=True)

        # return (pt.x, pt.y)
        # return segments.points[pixe.pt1_index]
        return np.array(
            [
                pt.x + pixe.delta * math.cos(line_angle + math.radians(90)),
                pt.y + pixe.delta * math.sin(line_angle + math.radians(90)),
            ]
        )

    for y in range(height):
        for x in range(width):
            pixe = pixelEncodings[x, y]
            if pixe:
                point = getPointFromEncoding(pixe)
                pixel = pointToPixel(point, width, height)
                # print((x, y), (pixel[0], pixel[1]))
                pixels[pixel[0], pixel[1]] = 0

    return img


# both images must be Mode = "1"
def displayOverlayImages(img1, img2, alpha=0.3):
    img1 = img1.copy()

    # Set the alpha value for each pixel in the second image to 50% (128 out of 255)
    alpha_mask = Image.new("L", img2.size, 255 - int(alpha * 255))

    # Overlay the second image with the alpha mask on top of the first image
    img1.paste(img2, mask=alpha_mask)

    img1.show()


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


def interactivePreview(segments, showPoints=False, randColors=False, seed=10):

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


def renderStrokes(
    segments, showPoints=False, randColors=False, seed=10, size=(1000, 1000)
):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plotStrokes(
        ax, segments, t=0, seed=seed, showPoints=showPoints, randColors=randColors
    )

    plt.axis(False)

    buffer = io.BytesIO()
    plt.savefig(
        buffer,
        bbox_inches="tight",
    )
    buffer.seek(0)  # Rewind the buffer
    img = Image.open(buffer)
    img.resize(size)
    return img


def renderStrokesCario(segments, size):

    tempSize = 5000
    width, height = tempSize, tempSize
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    # White background
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, width, height)
    ctx.fill()

    # black pen
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(1)

    for seg_i, seg in enumerate(segments.toList()):
        pixels = [pointToPixel(point, tempSize, tempSize) for point in seg]
        # points = [
        #     [(seg[i][0], seg[i][1]), (seg[i + 1][0], seg[i + 1][1])]
        #     for i in range(len(seg) - 1)
        # ]

        # print("points", pixels[0])
        ctx.move_to(pixels[0][0], pixels[0][1])

        for pixel in pixels:
            ctx.line_to(pixel[0], pixel[1])

        ctx.stroke()

    data = surface.get_data()
    img = Image.frombytes("RGBA", (width, height), data)
    img = img.resize((size, size))
    # Save the image as "pycario.png"
    # img.save("pycario.png")

    return img


def renderStrokesMine(segments, size):
    pass


def segmentsOnImage(img, segments):
    width, height = img.size
    img = img.copy()
    img = img.convert("RGB")

    pixels = img.load()

    for point in segments.points:
        pixel = pointToPixel(point, width, height)
        pixels[pixel[0], pixel[1]] = (255, 0, 0)

    return img


generatedJSONDir = Path(
    "/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/disposable/codeGeneratedJSON"
)

# JSONEncode(generatedJSONDir / "monkey.json", segments)

# img = Image.open(
#     "/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/disposable/blenderGeneratedRenders/monkey.png"
# )
# img = ImageOps.grayscale(img)


segments = JSONDecode(
    f"/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/disposable/blenderGeneratedJSON/monkey.json",
)


segmentsBaked = generateSketch(segments, t=0, seed=1)

interactivePreview(segmentsBaked, randColors=True)
# renderedStrokesImg = renderStrokesCario(segments=segmentsBaked, size=1000)
# renderedStrokesImg.show()

# pixelEncoding = getPixelEncoding(renderedStrokesImg, segments)
# encodedImage = imgFromPixelEncoding(segments, pixelEncoding, renderedStrokesImg.size)


# displayOverlayImages(renderedStrokesImg, encodedImage)

# JSONEncode(generatedJSONDir / "gear.json", strokes)

# segmentsOnImage(img, segments).show()
