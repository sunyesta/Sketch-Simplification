import json
import bpy
import subprocess

# Get the active object (Assuming it's a grease pencil object)
obj = bpy.context.active_object


if not (obj and obj.type == "GPENCIL"):
    print("Active object is not a grease pencil object.")
    quit()

segments = []
# Iterate through the layers of the grease pencil object
for layer in obj.data.layers:
    # Iterate through the frames of the layer
    for frame in layer.frames:
        # Iterate through the strokes of the frame
        for stroke in frame.strokes:
            points = []
            segments.append(points)
            # Iterate through the points of the stroke
            for point in stroke.points:
                points.append((point.co.x, point.co.z))


jsonData = json.dumps(segments)
#    print(json.dumps(strokes))
subprocess.run("pbcopy", text=True, input=jsonData)
