import numpy as np


def resample_list(data, target_length):
    """
    Resamples an ordered list to a different length using linear interpolation.

    Args:
      data: The input list.
      target_length: The desired length of the resampled list.

    Returns:
      A new list with the resampled elements.
    """

    if not data:
        return []

    # Calculate the spacing between samples based on the original data length.
    step = (len(data) - 1) / (target_length - 1)

    # Iterate through the list and interpolate values at each step.
    resampled_data = []
    for i in range(target_length):
        index = i * step
        # Check if the index is within the list range (handle edge cases).
        if 0 <= index < len(data) - 1:
            # Perform linear interpolation between the two elements.
            weight1 = 1 - (index % 1)
            weight2 = index % 1
            value = data[int(index)] * weight1 + data[int(index + 1)] * weight2
            resampled_data.append(value)
        else:
            # If the index is out of range, use the last or first element.
            if index < 0:
                resampled_data.append(data[0])
            else:
                resampled_data.append(data[-1])

    return resampled_data


# Example usage:
data = [1, 2, 3, 4, 5]
resampled_data = resample_list(data, 10)
print(resampled_data)
