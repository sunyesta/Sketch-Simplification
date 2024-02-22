def merge_last_two(data):
    """Merges the last two sublists of a 2D list.

    Args:
      data: A 2D list.

    Returns:
      A new 2D list with the last two sublists merged.
    """

    if len(data) < 2:
        return data

    # Get the last two sublists.
    last_two = data[-2:]

    # Merge the last two sublists.
    merged_list = last_two[0] + last_two[1]

    # Combine the merged list with the remaining sublists.
    return data[:-2] + [merged_list]


# Example usage
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
merged_data = merge_last_two(data)
print(merged_data)
