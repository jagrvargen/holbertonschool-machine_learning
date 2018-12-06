#!/usr/bin/env python3

def add_arrays(arr1, arr2):
    """ Performs element-wise addition of two arrays of equal length. """
    if len(arr1) != len(arr2):
        return None

    summed_array = []
    for i, val in enumerate(arr1):
        summed_array.append(val + arr2[i])

    return summed_array
