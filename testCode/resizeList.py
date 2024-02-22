
def resizeListToLength(data, desired_length):

    def stretchList():
        numDupes = math.ceil(desired_length / len(data))

        newData = []
        for x in data:
            newData.extend([x for _ in range(numDupes)])

        newData = newData[:desired_length]
        return newData

    def scrunchList():
        newData = []
        crunchRange = math.ceil(desired_length / len(data))

        for i in range(0, len(data), crunchRange):
            print(" "i)
            newData.append(mean(data[i:crunchRange]))

        return newData

    if len(data) > desired_length:
        return stretchList()
    elif len(data) < desired_length:
        return scrunchList()
    else:
        return data
