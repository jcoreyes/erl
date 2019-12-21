import numpy as np

objects = """puck_black
puck_blue
puck_blue1
puck_gold
puck_green
puck_purple
puck_purple1
puck_red
puck_red1
puck_white
bear
cat
dog
jeans
star
towel_brown
towel_purple
towel_red
lego_blue
lego_red
lego_green
lego_yellow"""

objects = objects.split("\n")

for i in range(10):
    print(i)
    print(np.random.choice(objects), np.random.randint(-90, 90))

    position = np.zeros((18, ))
    i = np.random.randint(18)
    position[i] = 1
    print(position.reshape((3, 6)))
