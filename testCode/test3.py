import numpy as np


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
        self._segMap = np.array(segMap)

    def segCount(self):
        return self._segMap[-1] + 1

    def to2DArr(self):
        segments = [[] for _ in range(self.segCount())]

        for i, point in enumerate(self.points):
            segments[self._segMap[i]].append(point)

        return segments


segments = [[(1, 2), (3, 4)], [(5, 6), (7, 8)], [(9, 10), (11, 12), (13, 14)]]
segments = Segments(segments)
print(segments.points)
print(segments._segMap)
print(segments.to2DArr())
