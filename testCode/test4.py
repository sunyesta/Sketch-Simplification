import svgwrite

# Define the points
points = [(50, 50), (100, 100), (150, 80), (200, 120), (250, 50)]

# Create an SVG drawing
dwg = svgwrite.Drawing("path_with_points.svg", profile="tiny")

# Create the path
path_string = f"M {points[0][0]},{points[0][1]}"

for point in points[1:]:
    path_string += f" L {point[0]},{point[1]}"

# Add the path to the SVG drawing
dwg.add(dwg.path(d=path_string, fill="none", stroke="black"))

# Add circles at each point for clarity
for point in points:
    dwg.add(dwg.circle(center=point, r=3, fill="red"))

# Save the SVG file
dwg.save()
