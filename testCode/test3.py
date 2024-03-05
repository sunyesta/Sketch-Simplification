import numpy as np
import random


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

        print(self.segMap)
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


segmentsRaw = [
    [(3, 2), (8, 5), (9, 7)],
    [(7, 4), (2, 10), (4, 6), (1, 8), (5, 9), (6, 3), (10, 2), (9, 1)],
]

segments = Segments(segmentsRaw)
segments = segments.dash(1, 3)
segments = segments.dash(1, 3)
segments = segments.dash(1, 3)
print(segments.segMap)
