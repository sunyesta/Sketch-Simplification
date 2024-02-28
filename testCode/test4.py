from shapely import LineString, Point, line_locate_point

line = LineString([(0, 2), (0, 10)])
point = Point(4, 4)
output = line.project(point, normalized=True)
distance = line.distance(point)
print(output, distance)
