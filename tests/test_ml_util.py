import unittest

from railrl.misc.ml_util import LossFollowingIntSchedule, \
    LossInverseFollowingIntSchedule


class TestLossFollowingIntSchedule(unittest.TestCase):

    def test_value_changes_average_1(self):
        schedule = LossFollowingIntSchedule(
            0,
            (-1, 1),
            1,
        )
        values = []
        for loss in [0, 0, 2, 2, -2]:
            schedule.update(loss)
            values.append(schedule.get_value(0))

        expected = [0, 0, 1, 2, 1]
        self.assertEqual(values, expected)

    def test_value_changes_average_3(self):
        schedule = LossFollowingIntSchedule(
            0,
            (-1, 1),
            3,
        )
        values = []
        for loss in [0, 0, 2, 2, -2, -2, -2]:
            schedule.update(loss)
            values.append(schedule.get_value(0))

        expected = [0, 0, 0, 1, 1, 1, 0]
        self.assertEqual(values, expected)

    def test_value_changes_average_1_inverse(self):
        schedule = LossInverseFollowingIntSchedule(
            0,
            (-1, 1),
            1,
        )
        values = []
        for loss in [0, 0, 2, 2, -2]:
            schedule.update(loss)
            values.append(schedule.get_value(0))

        expected = [0, 0, -1, -2, -1]
        self.assertEqual(values, expected)

    def test_value_changes_average_3_inverse(self):
        schedule = LossInverseFollowingIntSchedule(
            0,
            (-1, 1),
            3,
        )
        values = []
        for loss in [0, 0, 2, 2, -2, -2, -2]:
            schedule.update(loss)
            values.append(schedule.get_value(0))

        expected = [0, 0, 0, -1, -1, -1, 0]
        self.assertEqual(values, expected)


if __name__ == '__main__':
    unittest.main()