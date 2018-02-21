"""
Basic usage:
```

v_wall = VerticalWall(width, x_pos, bottom_y, top_y)
ball = Ball(radius=width)

init_xy = ...
new_xy = init_xy + velocity

if v_wall.collides_with(init_xy, new_xy):
    new_xy = v_wall.handle_collision(init_xy, new_xy)
```
"""
import abc


class Wall(object, metaclass=abc.ABCMeta):
    def __init__(self, segments):
        self.segments = segments

    def collides_with(self, start_point, end_point):
        trajectory_segment = (
            start_point[0],
            start_point[1],
            end_point[0],
            end_point[1],
        )
        for segment in self.segments:
            if segment.intersects_with(trajectory_segment):
                return True
        return False

    @abc.abstractmethod
    def handle_collision(self, start_point, end_point):
        pass


class Segment(object):
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def intersects_with(self, s2):
        left = max(min(self.x0, self.x1), min(s2[0], s2[2]))
        right = min(max(self.x0, self.x1), max(s2[0], s2[2]))
        top = max(min(self.y0, self.y1), min(s2[1], s2[3]))
        bottom = min(max(self.y0, self.y1), max(s2[1], s2[3]))

        if top > bottom or left > right:
            return False

        return True


class VerticalWall(Wall):
    def __init__(self, width, x_pos, bottom_y, top_y):
        self.left_segment = Segment(
            x_pos - width / 2,
            bottom_y,
            x_pos - width / 2,
            top_y,
        )
        self.right_segment = Segment(
            x_pos + width / 2,
            bottom_y,
            x_pos + width / 2,
            top_y,
        )
        self.top_segment = Segment(
            x_pos - width / 2,
            top_y,
            x_pos + width / 2,
            top_y,
        )
        self.bottom_segment = Segment(
            x_pos - width / 2,
            bottom_y,
            x_pos + width / 2,
            bottom_y,
        )
        super().__init__([
            self.left_segment,
            self.right_segment,
            self.top_segment,
            self.bottom_segment
        ])
        self.x_pos = x_pos
        self.width = width

        self.endpoint1 = (x_pos, bottom_y)
        self.endpoint2 = (x_pos, top_y)

    def handle_collision(self, start_point, end_point):
        if start_point[0] < end_point[0]:
            end_point[0] = self.x_pos - self.width
        elif start_point[0] > end_point[0]:
            end_point[0] = self.x_pos + self.width
        else:  # Ball is coming from above/below
            end_point[1] = start_point[1]
        return end_point


class HorizontalWall(Wall):
    def __init__(self, width, y_pos, left_x, right_x):
        self.top_segment = Segment(
            left_x,
            y_pos + width / 2,
            right_x,
            y_pos + width / 2,
        )
        self.bottom_segment = Segment(
            left_x,
            y_pos - width / 2,
            right_x,
            y_pos - width / 2,
        )
        self.left_segment = Segment(
            left_x,
            y_pos - width / 2,
            left_x,
            y_pos + width / 2,
        )
        self.right_segment = Segment(
            right_x,
            y_pos - width / 2,
            right_x,
            y_pos + width / 2,
        )
        super().__init__([
            self.top_segment,
            self.bottom_segment,
            self.left_segment,
            self.right_segment
        ])
        self.width = width
        self.y_pos = y_pos

        self.endpoint1 = (left_x, y_pos)
        self.endpoint2 = (right_x, y_pos)

    def handle_collision(self, start_point, end_point):
        if start_point[1] > end_point[1]:  # going -y direction
            end_point[1] = self.y_pos + self.width
        elif start_point[1] < end_point[1]:  # going +y direction
            end_point[1] = self.y_pos - self.width
        else:  # Ball is coming from the side
            end_point[0] = start_point[0]
        return end_point