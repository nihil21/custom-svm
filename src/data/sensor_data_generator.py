import math
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as m_patches


def create_10_rooms():
    # x, y (in meters) dimension of the plant of the building, each floor is 3 meters height
    max_x = 12
    max_y = 18
    floor_number = 2

    if max_x % 3 != 0 or max_y % 3 != 0:
        print("max_x and max_y must be multiple of 3")
        exit(1)

    # supposing each room is at least 3x3 meters
    max_x /= 3
    max_y /= 3
    max_x = int(max_x)
    max_y = int(max_y)

    # WATCH OUT FOR DIMENSIONS! ADJUST PREVIOUS VALUES BEFORE MODIFICATION
    rooms_list = [
        # floor 0
        [
            [0, 0, 1, 1, 1, 1],
            [0, 0, 2, 2, 1, 1],
            [0, 0, 2, 2, 3, 3],
            [4, 4, 4, 4, 3, 3],
        ],
        # floor 1
        [
            [5, 5, 5, 5, 6, 6],
            [7, 7, 5, 5, 6, 6],
            [7, 7, 8, 8, 9, 9],
            [7, 7, 8, 8, 9, 9],
        ],
    ]

    rooms = np.array(rooms_list, dtype=int)
    name = 11  # number of color in palette + 1

    return rooms, name, max_x, max_y, floor_number


def create_2_rooms():
    # x, y (in meters) dimension of the plant of the building, each floor is 3 meters height
    max_x = 12
    max_y = 18
    floor_number = 1

    if max_x % 3 != 0 or max_y % 3 != 0:
        print("max_x and max_y must be multiple of 3")
        exit(1)

    # supposing each room is at least 3x3 meters
    max_x /= 3
    max_y /= 3
    max_x = int(max_x)
    max_y = int(max_y)

    # WATCH OUT FOR DIMENSIONS! ADJUST PREVIOUS VALUES BEFORE MODIFICATION
    rooms_list = [
        # floor 0
        [
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
        ],
    ]

    rooms = np.array(rooms_list, dtype=int)
    name = 11  # number of color in palette + 1

    return rooms, name, max_x, max_y, floor_number


def from_hex_color(color_name):
    raw_col = colors.get_named_colors_mapping()["xkcd:" + color_name].lstrip('#')
    color = tuple(int(raw_col[i:i + 2], 16) for i in (0, 2, 4))
    return color[0] / 256.0, color[1] / 256.0, color[2] / 256.0


def fill_color(colors_np, colors_str):
    for index, string in enumerate(colors_str):
        colors_np[index] = from_hex_color(string)


def create_colors():
    # less color, better visualization! At MOST 10 rooms per floor whit this palette
    color_patches = []
    colors_str = ['blue', 'red', 'yellow', 'green', 'orange', 'purple',
                  'brown', 'pink', 'cyan', 'olive']
    colors_np = np.empty([len(colors_str), 3], dtype=float)
    fill_color(colors_np, colors_str)

    for k, color_str in enumerate(colors_str):
        color_patches.append(m_patches.Patch(color=color_str, label='Room' + str(k)))

    return colors_np, color_patches


def create_sensors_10_rooms(m_x, m_y, floor_number):
    m_z = floor_number * 3  # 3 meters per floor
    m_x *= 3  # in meters
    m_y *= 3  # in meters

    # floor 0
    sensors_0 = [(m_x * 0.3, m_y * 0.1, m_z * 0.2), (m_x * 0.2, m_y * 0.8, m_z * 0.4),
                 (m_x * 0.3, m_y * 0.5, m_z * 0.2), (m_x * 0.4, m_y * 0.2, m_z * 0.4),
                 (m_x * 0.5, m_y * 0.8, m_z * 0.2), (m_x * 0.6, m_y * 0.4, m_z * 0.4),
                 (m_x * 0.7, m_y * 0.2, m_z * 0.2), (m_x * 0.8, m_y * 0.6, m_z * 0.4),
                 (m_x * 0.9, m_y * 0.8, m_z * 0.2), (m_x * 0.95, m_y * 0.2, m_z * 0.4), ]
    # floor 1
    sensors_1 = [(m_x * 0.3, m_y * 0.1, m_z * 0.6), (m_x * 0.8, m_y * 0.2, m_z * 0.8),
                 (m_x * 0.5, m_y * 0.3, m_z * 0.6), (m_x * 0.2, m_y * 0.4, m_z * 0.8),
                 (m_x * 0.8, m_y * 0.5, m_z * 0.6), (m_x * 0.4, m_y * 0.6, m_z * 0.8),
                 (m_x * 0.2, m_y * 0.7, m_z * 0.6), (m_x * 0.6, m_y * 0.8, m_z * 0.8),
                 (m_x * 0.8, m_y * 0.9, m_z * 0.6), (m_x * 0.2, m_y * 0.95, m_z * 0.8), ]

    sensors_per_floor = []
    sensors_per_floor.append(sensors_0)
    sensors_per_floor.append(sensors_1)

    sensors_list = sensors_0 + sensors_1

    return np.array(sensors_list, dtype=float), sensors_per_floor


def create_sensors_2_rooms(m_x, m_y, floor_number):
    m_z = floor_number * 3  # 3 meters per floor
    m_x *= 3  # in meters
    m_y *= 3  # in meters

    # floor 0
    sensors_0 = [(m_x * 0.8, m_y * 0.2, m_z * 0.5), (m_x * 0.3, m_y * 0.3, m_z * 0.25),
                 (m_x * 0.5, m_y * 0.5, m_z * 0.5), (m_x * 0.7, m_y * 0.7, m_z * 0.75),
                 (m_x * 0.2, m_y * 0.8, m_z * 0.5), ]

    sensors_per_floor = []
    sensors_per_floor.append(sensors_0)

    return np.array(sensors_0, dtype=float), sensors_per_floor


def score_distance(coord, s):
    return 500 / ((coord[0] - s[0]) ** 2 + (coord[1] - s[1]) ** 2 + (coord[2] - s[2]) ** 2) ** 2


def generate_dataset(n, max_x, max_y, max_z, rooms, sensors):
    max_x *= 3  # in meters
    max_y *= 3  # in meters
    max_z *= 3  # in meters

    data = []
    data_coords = []
    for i in range(n):
        coord = (rnd.random() * max_x, rnd.random() * max_y, rnd.random() * max_z)
        data_coords.append(coord)

        label = rooms[math.floor(coord[2] / 3.0)][math.floor(coord[0] / 3.0)][math.floor(coord[1] / 3.0)]
        # storage of the intensity values measured by each sensor for each data point
        # simulate error in measurement process
        error_bias = 0.1
        entry = [score_distance(coord, s) + error_bias * rnd.random()
                 if rnd.random() < 0.5 else
                 score_distance(coord, s) - error_bias * rnd.random()
                 for s in sensors]

        # remove invalid negative values
        entry = [e if e > 0 else 0.001 * rnd.random() for e in entry]
        # smooth high values
        entry = [e if e < 50 else 50 for e in entry]

        # add class label (room)
        entry.append(label)

        data.append(entry)

    return data, data_coords


def get_sensors_labels(all_sensors, rooms):
    sensors_labels = []
    for s in all_sensors:
        sensors_labels.append(rooms[math.floor(s[2] / 3.0)][math.floor(s[0] / 3.0)][math.floor(s[1] / 3.0)])
    return sensors_labels


def sensor_input_from_spatial_position(x, y, z, sensors):
    # the input values x, y and z are supposed in meters!

    coord = (x, y, z)
    error_bias = 1
    sensors_input = [score_distance(coord, s) + error_bias * rnd.random()
                     if rnd.random() < 0.5 else
                     score_distance(coord, s) - error_bias * rnd.random()
                     for s in sensors]

    # remove invalid negative values
    sensors_input = [e if e > 0 else 0.01 * rnd.random() for e in sensors_input]
    # smooth high values
    sensors_input = [e if e < 50 else 50 for e in sensors_input]

    return sensors_input


# Hypothesis: max_x, max_y (in meters) dimension of the plant of the building, each floor is 3 meters height
def random_rooms(max_x: int,
                 max_y: int,
                 floor_number: int,
                 rooms_temperature: float):

    # It is reasonable to set as room minimum dimension 3 meters square
    if max_x % 3 != 0 or max_y % 3 != 0:
        print("max_x and max_y must be multiple of 3")
        exit(1)

    # conversion: max_x is max number of room along x axis, max_y along y axis
    max_x = int(max_x / 3)
    max_y = int(max_y / 3)

    # rooms_temperature in [0,1]: the higher the value, the higher the room number
    if rooms_temperature < 0 or rooms_temperature > 1:
        print("rooms_temperature in [0,1]")
        exit(2)

    rooms = np.empty([floor_number, max_x, max_y], dtype=int)

    # rooms initialization
    name = 0
    for floor in range(floor_number):
        for x in range(max_x):
            for y in range(max_y):
                temperature = rnd.random()
                # new room
                if temperature < rooms_temperature:
                    name += 1
                    rooms[floor, x, y] = name
                # already existing room
                else:
                    if rnd.random() < 0.5:
                        if x - 1 >= 0:
                            tmp_name = rooms[floor, x - 1, y]
                        elif y - 1 >= 0:
                            tmp_name = rooms[floor, x, y - 1]
                        else:
                            name += 1
                            tmp_name = name
                    else:
                        if y - 1 >= 0:
                            tmp_name = rooms[floor, x, y - 1]
                        elif x - 1 >= 0:
                            tmp_name = rooms[floor, x - 1, y]
                        else:
                            name += 1
                            tmp_name = name
                    rooms[floor, x, y] = tmp_name
        # different floor, different room
        name += 1

    return rooms, name, max_x, max_y


def random_palette(palette_dimension: int):

    color_array_1 = np.arange(0, palette_dimension, 1, dtype=float)
    np.random.shuffle(color_array_1)
    color_array_2 = np.arange(0, palette_dimension, 1, dtype=float)
    np.random.shuffle(color_array_2)
    color_array_3 = np.arange(0, palette_dimension, 1, dtype=float)
    np.random.shuffle(color_array_3)

    colors_np = np.empty([palette_dimension, 3], dtype=float)
    for i in range(0, palette_dimension):
        colors_np[i] = np.array(
            [color_array_1[i] / palette_dimension,
             color_array_2[i] / palette_dimension,
             color_array_3[i] / palette_dimension])

    return colors_np


if __name__ == "__main__":

    # rooms, name, max_x, max_y, floor_number = create_2_rooms()
    rooms, name, max_x, max_y, floor_number = create_10_rooms()

    # all_sensors, sensors_per_floor = create_sensors_2_rooms(max_x, max_y, floor_number)
    all_sensors, sensors_per_floor = create_sensors_10_rooms(max_x, max_y, floor_number)

    n = 1000
    dataset, data_coords = generate_dataset(n, max_x, max_y, floor_number, rooms, all_sensors)

    colors_np, patches = create_colors()

    cmap = colors.ListedColormap(colors_np)
    bounds = np.arange(0, name, 1).tolist()
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # uncomment to write dataset to file
#    with open("data.txt", "w") as file:
#        for data in dataset:
#            for k, sensor_data in enumerate(data):
#                file.write(str(sensor_data))
#                if k < len(data) - 1:
#                    file.write(',')
#            file.write('\n')

    ################################################
    # the following code is only for visualization
    ################################################

    for index, room in enumerate(rooms):
        fig, ax = plt.subplots()
        ax.imshow(room, cmap=cmap, norm=norm)

        # draw grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        plt.yticks(np.arange(-.5, max_x, 1).tolist(),
                   [int((x + 0.5) * 3) for x in np.arange(-.5, max_x, 1)])
        plt.xticks(np.arange(-.5, max_y, 1).tolist(),
                   [int((x + 0.5) * 3) for x in np.arange(-.5, max_y, 1)])

        plt.title("floor " + str(index))

        # sensors
        y_val = [x[0] / 3.0 - .5 for x in sensors_per_floor[index]]
        x_val = [x[1] / 3.0 - .5 for x in sensors_per_floor[index]]

        plt.plot(x_val, y_val, 'or', color="black", markersize=20)
        plt.plot(x_val, y_val, 'or', color="white", markersize=18)

        for i in range(0, len(x_val)):
            s_name = str(i + index * len(sensors_per_floor[index]))
            # plot sensors' names
            if int(s_name) < 10:
                plt.text(x_val[i] - 0.1, y_val[i] + 0.1, s_name, fontsize=12)
            else:
                plt.text(x_val[i] - 0.15, y_val[i] + 0.1, s_name, fontsize=12)

        # code for drawing samples into rooms:

        # for k, data in enumerate(dataset[:10]):
        #    if index * 10 <= data[-1] < index * 10 + 10:
        #        plt.plot([data_coords[k][1] / 3.0 - .5], [data_coords[k][0] / 3.0 - .5],
        #                 "*", color="white", markersize=18)
        #        plt.plot([data_coords[k][1] / 3.0 - .5], [data_coords[k][0] / 3.0 - .5],
        #                "*", color="black", markersize=7)

        # hard coded: 1 floor if create_2_rooms, 2 floors if create_10_rooms
        if len(rooms) == 1:
            plt.legend(handles=patches[:2], loc='upper center', bbox_to_anchor=(1.2, 1))
        else:
            if index == 0:
                plt.legend(handles=patches[:5], loc='upper center', bbox_to_anchor=(1.2, 1))
            else:
                plt.legend(handles=patches[5:], loc='upper center', bbox_to_anchor=(1.2, 1))

        plt.show()

    # bar plot of the sensors' intensities
    # dataset[n][-1] is class label! Must be removed in plotting
    for data in dataset[:5]:
        plt.title("Collected in room: " + str(data[-1]))
        plt.bar([n for n in range(0, len(all_sensors))], data[:-1])
        plt.xticks(np.arange(0, len(all_sensors), 1).tolist())
        plt.show()
