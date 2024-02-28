import bpy
from pathlib import Path
import json


gp_object = bpy.data.objects.get("mainGP")
gp = gp_object.data
gp_layer = None


def make3D(points):
    return [(point[0], 0, point[1]) for point in points]


def restartGP():
    gp.clear()

    global gp_layer
    gp_layer = gp.layers.new("lines")


def insertFrame(strokesPath, frame=None):

    # MAKE SURE YOU CALL restartGP() BEFORE FIRST CALL
    strokes = None
    with open(strokesPath, "r") as file:
        strokes = json.load(file)

    if not frame:
        frame = bpy.context.scene.frame_current
    gp_frame = gp_layer.frames.new(frame)

    for points in strokes:
        points = make3D(points)
        print("new stroke")
        gp_stroke = gp_frame.strokes.new()
        gp_stroke.line_width = 12

        gp_stroke.points.add(len(points))

        for i, point in enumerate(points):
            gp_stroke.points[i].co = point


path = Path(
    f"/Users/mary/Documents/School/Sketch Simplification/Sketch-Simplification/disposable/blenderGeneratedJSON/monkey.json"
)


restartGP()
insertFrame(path)
