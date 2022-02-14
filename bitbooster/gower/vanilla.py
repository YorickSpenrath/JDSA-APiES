import warnings

import numpy as np
import pandas as pd
from bitbooster.abstract.clusterable import Clusterable
from bitbooster.manhattan.vanilla import ManhattanVanillaObject
from bitbooster.preprocessing import normalize


class GowerVanillaObject(Clusterable):

    def __init__(self, data, index=None, column_names=None, categorical_columns=None):
        assert isinstance(data, pd.DataFrame) and categorical_columns is None
        self._is_cat = np.array([data[i].dtype == 'O' for i in data.columns])

        if self._is_cat.sum() == len(self._is_cat):
            man_data = data.loc[:, ~self._is_cat]
        else:
            man_data = normalize(data.loc[:, ~self._is_cat])

        super().__init__(data=data, index=index, column_names=column_names)

        # Split into manhattan and dice component
        self._manhattan_component = ManhattanVanillaObject(data=man_data, index=index)
        self._dice_component = DiceVanillaObject(data=data.loc[:, self._is_cat], index=index)

        # Remove copy of the data from this object, but retain its properties
        self._size = self._data.shape[0]
        self._unique_size = len(data.drop_duplicates())
        del self._data

    @property
    def unique_size(self):
        return self._unique_size

    @property
    def size(self):
        return self._size

    def get_sub_distance_matrix(self, index0, index1):
        mm = self._manhattan_component.get_sub_distance_matrix(index0, index1)
        mc = self._dice_component.get_sub_distance_matrix(index0, index1)
        return mm + mc

    def distance_matrix_to_other(self, other, index_self, index_other):
        assert isinstance(other, GowerVanillaObject)
        mm = self._manhattan_component.distance_matrix_to_other(other._manhattan_component, index_self, index_other)
        mc = self._dice_component.distance_matrix_to_other(other._dice_component, index_self, index_other)
        return mm + mc

    def get_index_with_lowest_sum(self, index0, index1):
        if index0 is None:
            index0 = range(self.size)
        if index1 is None:
            index1 = range(self.size)

        n_vertical = len(index0)
        n_horizontal = len(index1)

        lowest_sum = np.inf
        lowest_index = -1

        for i in range(n_vertical):
            i_sum = 0
            x = 0

            while i_sum < lowest_sum and x < n_horizontal:
                i_sum += self._manhattan_component.get_sub_distance_matrix([index0[i]], index1[x:x + 1000]).sum()
                i_sum += self._dice_component.get_sub_distance_matrix([index0[i]], index1[x:x + 1000]).sum()
                x += 1000

            if i_sum < lowest_sum:
                lowest_sum = i_sum
                lowest_index = i

        return lowest_index

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        raise NotImplementedError

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        raise NotImplementedError


class DiceVanillaObject(Clusterable):
    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        ret = np.empty((vertical_data.shape[0], horizontal_data.shape[0]), dtype=np.float32)
        n_dim = vertical_data.shape[1]
        for vi, v_data in enumerate(vertical_data):
            for hi, h_data in enumerate(horizontal_data):
                ret[vi, hi] = n_dim - (v_data == h_data).sum()
        return ret

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        raise NotImplementedError()
