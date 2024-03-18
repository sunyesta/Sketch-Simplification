from pathlib import Path
import random
import bpy
import math
import json

data_output_path = Path(
    f"/Users/mary/Documents/School/Sketch Simplification/sketch_simplification_2/disposable/blenderGeneratedJSON"
)

collection_name = "objs"

# init code

scene_collection = bpy.context.scene.collection

if collection_name not in bpy.data.collections:
    print("creating")
    bpy.data.collections.new(collection_name)
    collection = bpy.data.collections[collection_name]
    scene_collection.children.link(collection)

collection = bpy.data.collections[collection_name]


def generateToruses():

    # Set the number of toruses to generate
    num_toruses = 10

    minLocation = (0, 0, 0)
    maxLocation = (1, 1, 1)

    # Function to generate a random location
    def random_location():
        x = random.uniform(minLocation[0], maxLocation[0])  # Adjust range as needed
        y = random.uniform(minLocation[1], maxLocation[1])  # Adjust range as needed
        z = random.uniform(minLocation[2], maxLocation[2])  # Adjust range as needed
        return (x, y, z)

    # Function to generate a random rotation
    def random_rotation():
        rx = math.radians(random.uniform(-180, 180))
        ry = math.radians(random.uniform(-180, 180))
        rz = math.radians(random.uniform(-180, 180))
        return (rx, ry, rz)

    def random_radius():
        return random.uniform(0.01, 0.2)

    # ------- Main -------

    # Check if the collection exists, create it if not

    #  ------ Create the toruses -------
    for i in range(num_toruses):
        # Create torus mesh
        bpy.ops.mesh.primitive_torus_add(
            align="WORLD",
            location=random_location(),
            rotation=random_rotation(),
            major_radius=random_radius(),
            minor_radius=random_radius(),
            major_segments=100,
            minor_segments=28,
        )

        obj = bpy.context.object

        # Set object name
        obj.name = f"Torus_{i}"

        # Add object to the collection
        for old_collection in obj.users_collection:
            old_collection.objects.unlink(obj)

        collection.objects.link(obj)


# ------- extract data -------


def extractData():

    # Get the active object (Assuming it's a grease pencil object)
    gp_obj = bpy.data.objects["mainGP"]

    # add the modifier
    bpy.ops.object.gpencil_modifier_add(type="GP_LINEART")
    modifier = gp_obj.grease_pencil_modifiers.get("Line Art")
    modifier.source_collection = collection
    modifier.target_later = gp_obj.data.layers["lines"]
    modifier.target_material = 0
    modifier.apply()

    # get the segments
    segments = []
    # Iterate through the layers of the grease pencil object
    layer = next(iter(gp_obj.data.layers))
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

    bpy.ops.object.lineart_clear()
    return jsonData


bpy.ops.ed.undo_push(message="My Action")
