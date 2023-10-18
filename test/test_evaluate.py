import unittest
from src.evaluate import is_same_mapping, is_same_mapping_3, evaluate


class TestPlotEvaluation(unittest.TestCase):

    def test_is_same_mapping(self):
        # Test basic string matching
        self.assertTrue(is_same_mapping(['x'], ['10'], ['x'], ['10']))
        self.assertFalse(is_same_mapping(['x'], ['10'], ['y'], ['10']))

        # Test case-insensitive string matching
        self.assertTrue(is_same_mapping(['X'], ['10'], ['x'], ['10']))

        # Test non-string keys
        self.assertTrue(is_same_mapping([1], ['10'], [1], ['10']))
        self.assertFalse(is_same_mapping([1], ['10'], [2], ['10']))

    def test_is_same_mapping_3(self):
        # Test basic match
        self.assertTrue(is_same_mapping_3(['lat1', 'lat2'], ['lon1', 'lon2'], ['z1', 'z2'],
                                          ['lat1', 'lat2'], ['lon1', 'lon2'], ['z1', 'z2']))
        self.assertFalse(is_same_mapping_3(['lat1', 'lat2'], ['lon1', 'lon2'], ['z1', 'z2'],
                                           ['lat1', 'lat3'], ['lon1', 'lon2'], ['z1', 'z2']))

    def test_evaluate(self):
        golden = {'type': 'scatter', 'x': ['x1', 'x2'], 'y': ['y1', 'y2'], 'orientation': 'v',
                  'xaxis': 'x', 'yaxis': 'y'}
        predicted = {'type': 'scatter', 'x': ['x1', 'x2'], 'y': ['y1', 'y2'], 'orientation': 'v',
                     'xaxis': 'x', 'yaxis': 'y'}
        self.assertTrue(evaluate(golden, predicted))

        # Test mismatched types
        predicted_diff_type = predicted.copy()
        predicted_diff_type['type'] = 'bar'
        self.assertFalse(evaluate(golden, predicted_diff_type))

        # Test missing fields
        predicted_missing_field = predicted.copy()
        del predicted_missing_field['orientation']
        self.assertFalse(evaluate(golden, predicted_missing_field))

        # Test densitymapbox type
        golden_density = {'type': 'densitymapbox', 'lat': ['lat1', 'lat2'], 'lon': ['lon1', 'lon2'],
                          'z': ['z1', 'z2'], 'subplot': 'p1', 'type': 'densitymapbox'}
        predicted_density = {'type': 'densitymapbox', 'lat': ['lat1', 'lat2'],
                             'lon': ['lon1', 'lon2'], 'z': ['z1', 'z2'], 'subplot': 'p1',
                             'type': 'densitymapbox'}
        self.assertTrue(evaluate(golden_density, predicted_density))


if __name__ == '__main__':
    unittest.main()
