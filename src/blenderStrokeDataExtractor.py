import json
import bpy
import subprocess

# Get the active object (Assuming it's a grease pencil object)
obj = bpy.context.active_object


if not (obj and obj.type == "GPENCIL"):
    print("Active object is not a grease pencil object.")
    quit()

strokesInfo = {"xy": [], "pressure": [], "alpha": []}
# Iterate through the layers of the grease pencil object
for layer in obj.data.layers:
    # Iterate through the frames of the layer
    for frame in layer.frames:
        # Iterate through the strokes of the frame
        for stroke in frame.strokes:
            points_xy, points_pressure, points_alpha = [], [], []

            strokesInfo["xy"].append(points_xy)
            strokesInfo["pressure"].append(points_pressure)
            strokesInfo["alpha"].append(points_alpha)

            # Iterate through the points of the stroke
            for point in stroke.points:
                points_xy.append((point.co.x, point.co.z))
                points_pressure.append(point.pressure)
                points_alpha.append(point.strength)

jsonData = json.dumps(strokesInfo)
#    print(json.dumps(strokes))
subprocess.run("pbcopy", text=True, input=jsonData)
