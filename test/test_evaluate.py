import unittest
from src.evaluate import is_same_mapping, is_same_mapping_3, evaluate, eq, items_equal


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
        golden = {'type': 'scatter', 'x': ['x1', 'x2'], 'y': ['y1', 'y2'],
                  'xaxis': 'x', 'yaxis': 'y'}
        predicted = {'type': 'scatter', 'x': ['x1', 'x2'], 'y': ['y1', 'y2'],
                     'xaxis': 'x', 'yaxis': 'y'}
        self.assertTrue(evaluate(golden, predicted))

        # Test mismatched types
        predicted_diff_type = predicted.copy()
        predicted_diff_type['type'] = 'bar'
        self.assertFalse(evaluate(golden, predicted_diff_type))

        # Test missing fields
        predicted_missing_field = predicted.copy()
        del predicted_missing_field['x']
        self.assertFalse(evaluate(golden, predicted_missing_field))

        # Test densitymapbox type
        golden_density = {'type': 'densitymapbox', 'lat': ['lat1', 'lat2'], 'lon': ['lon1', 'lon2'],
                          'z': ['z1', 'z2'], 'subplot': 'p1', 'type': 'densitymapbox'}
        predicted_density = {'type': 'densitymapbox', 'lat': ['lat1', 'lat2'],
                             'lon': ['lon1', 'lon2'], 'z': ['z1', 'z2'], 'subplot': 'p1',
                             'type': 'densitymapbox'}
        self.assertTrue(evaluate(golden_density, predicted_density))

    def test_eq(self):
        self.assertTrue(eq(25.09310344827585, 25.09310344827586))
        self.assertTrue(eq("hello", "HELLO"))
        self.assertTrue(eq("Hello", "hello"))
        self.assertFalse(eq(5, "5"))
        self.assertTrue(eq(5, 5))
        self.assertFalse(eq(5, 6))
        self.assertTrue(eq(5.123456, 5.123457))
        self.assertFalse(eq(5.123456, 5.123459))
        self.assertTrue(eq("HELLO", "hello"))
        self.assertFalse(eq("HELLO", "world"))
        self.assertTrue(eq(1 + 2j, 1 + 2j))
        self.assertFalse(eq(1 + 2j, 2 + 2j))

    def test_items_equal(self):
        dict1 = {'a': 25.09310344827585, 'b': "hello"}
        dict2 = {'a': 25.09310344827586, 'b': "HELLO"}
        self.assertTrue(items_equal(dict1, dict2))

        dict1 = {'a': 5, 'b': "hello"}
        dict2 = {'a': 6, 'b': "HELLO"}
        self.assertFalse(items_equal(dict1, dict2))

        dict1 = {'a': 5, 'b': "hello"}
        dict2 = {'a': 5, 'b': "world"}
        self.assertFalse(items_equal(dict1, dict2))

        dict1 = {'a': 5, 'b': "hello"}
        dict2 = {'a': 5}
        self.assertFalse(items_equal(dict1, dict2))

        dict1 = {'a': 5}
        dict2 = {'a': 5, 'b': "hello"}
        self.assertFalse(items_equal(dict1, dict2))


if __name__ == '__main__':
    unittest.main()
