import unittest
import rlkit.pythonplusplus as ppp


class TestDictionaryMethods(unittest.TestCase):
    def test_dict_list_to_list_dict(self):
        x = {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
        result_list = ppp.dict_of_list__to__list_of_dicts(x, 3)
        expected_list = [
            {'foo': 3, 'bar': 1},
            {'foo': 4, 'bar': 2},
            {'foo': 5, 'bar': 3},
        ]
        for result, expected in zip(result_list, expected_list):
            self.assertDictEqual(result, expected)

    def test_merge_recursive_dicts(self):
        a = {'foo': 1, 'bar': {'baz': 3}}
        b = {'foo2': 2, 'bar': {'baz2': 4}}
        result = ppp.merge_recursive_dicts(a, b)
        expected = {
            'foo': 1,
            'foo2': 2,
            'bar': {
                'baz': 3,
                'baz2': 4,
            }
        }
        self.assertDictEqual(result, expected)

    def test_dotmap_dict_to_nested_dict(self):
        a = {'foo': 1, 'bar.baz': 3}
        result = ppp.dot_map_dict_to_nested_dict(a)
        expected = {'foo': 1, 'bar': {'baz': 3}}
        self.assertDictEqual(result, expected)

    def test_nested_dict_to_dotmap_dict(self):
        a = {'foo': 1, 'bar': {'baz': 3}}
        result = ppp.nested_dict_to_dot_map_dict(a)
        expected = {'foo': 1, 'bar.baz': 3}
        self.assertDictEqual(result, expected)

    def test_recursive_items(self):
        a = {'foo': 1, 'bar': {'baz': 3}}
        full_items_dict = dict(ppp.recursive_items(a))
        expected = {
            'foo': 1,
            'bar': {
                'baz': 3
            },
            'baz': 3,
        }
        self.assertDictEqual(full_items_dict, expected)

if __name__ == '__main__':
    unittest.main()