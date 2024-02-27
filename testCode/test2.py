from scipy.spatial import KDTree

# Define your list of points
points = [(1, 2), (3, 4), (5, 6)]

# Define the point you want to find the closest to (not in the list)
query_point = (7, 8)

# Build a KDTree from the list of points
tree = KDTree(points)

# Find the index and distance to the closest point
# distance, index = tree.query(query_point, k=1)  # k=1 for single nearest neighbor
distances, indexes = tree.query(query_point, k=3)  # k=1 for single nearest neighbor

# Access the closest point from the list using the index
closest_points = [points[i] for i in indexes]
# closest_points = [points[i] for i in indexes]

print(f"Closest point: {closest_points}")
print(f"Distance to closest point: {distances}")
