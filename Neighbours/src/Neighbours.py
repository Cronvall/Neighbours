import copy
import math
import random
from typing import List
from enum import Enum, auto

import pygame as pg


#  Program to simulate segregation.
#  See : http:#nifty.stanford.edu/2014/mccown-schelling-model-segregation/
#

# Enumeration type for the Actors
class Actor(Enum):
    BLUE = auto()
    RED = auto()
    NONE = auto()  # NONE used for empty locations

# Enumeration type for the state of an Actor
class State(Enum):
    UNSATISFIED = auto()
    SATISFIED = auto()
    NA = auto()  # Not applicable (NA), used for NONEs

World = List[List[Actor]]  # Type alias
SIZE = 64

def neighbours():
    pg.init()
    model = NeighboursModel(SIZE)
    _view = NeighboursView(model)
    model.run()

class NeighboursModel:
    # Tune these numbers to test different distributions or update speeds
    FRAME_RATE = 20  # Increase number to speed simulation up
    DIST = [0.4, 0.4, 0.2]  # % of RED, BLUE, and NONE
    THRESHOLD = 0.7  # % of surrounding neighbours that should be like me for satisfaction

    available_positions = [] #All empty Actors in the world (2D-matrix)

    # ########### These following two methods are what you're supposed to implement  ###########
    # In this method you should generate a new world
    # using randomization according to the given arguments.
    @staticmethod
    def __create_world(size) -> World:
        new_world = create_world(size, Actor.NONE)

        populate_new_world(new_world, NeighboursModel.DIST)

        return new_world

    # This is the method called by the timer to update the world
    # (i.e move unsatisfied) each "frame".
    def __update_world(self):

        #sets the list of coordinates for all empty actors
        self.create_available_list()

        #the 2D-matrix containing satisfaction-state for each cell
        sat_world = create_world(len(self.world), State.NA)
        #Makes copy of world so we don't edit the already read matrix
        copy_of_world = copy.deepcopy(self.world)

        #Read each cell in 2D-matrix
        for row in range(len(self.world)):
            for col in range(len(self.world[row])):
                #Writes satisfaction for current cell in matrix sat_world
                set_actor_satisfaction(self.world, row, col, sat_world)

                #If a cell is unsatisfied move it
                if sat_world[row][col] == State.UNSATISFIED:
                    self.move_unsatisfied(copy_of_world, row, col)

        #Update the printed world to our now modified world "copy_of_world"
        self.world = copy_of_world


    ###USED IN  __update_world()
    def move_unsatisfied(self, copy_of_world,row, col):
        #Random coordinate of empty cell, used to set destination for unsatisfied colored cells
        random_index = random.randint(0, len(self.available_positions) - 1)
        new_coordinates: List = self.available_positions[random_index]

        unsatisfied_color = self.get_unsatisfied_color(row, col, copy_of_world)
        self.move_actor(copy_of_world, new_coordinates, row, col, unsatisfied_color)

        self.available_positions.pop(random_index)
        self.available_positions.append([row, col])

    ###USED IN  move()
    #Purpose is to move the Colored Unsatisfied Actor to a pre-selected cell
    def move_actor(self, copy_of_world, new_coordinates, row,col, actor : Actor):
        new_x = new_coordinates[0]
        new_y = new_coordinates[1]
        copy_of_world[new_x][new_y] = actor
        copy_of_world[row][col] = Actor.NONE

    #Returns the unsatisfied Actors color (ex. Actor.BLUE)
    def get_unsatisfied_color(self, row,col,copy_of_world):
        return copy_of_world[row][col]

    # ########### the rest of this class is already defined, to handle the simulation clock  ###########
    def __init__(self, size):
        self.world: World = self.__create_world(size)
        self.observers = []  # for enabling discoupled updating of the view, ignore

    def run(self):
        clock = pg.time.Clock()
        running = True
        while running:
            running = self.__on_clock_tick(clock)
        # stop running
        print("Goodbye!")
        pg.quit()

    def __on_clock_tick(self, clock):
        clock.tick(self.FRAME_RATE)  # update no faster than FRAME_RATE times per second
        self.__update_and_notify()
        return self.__check_for_exit()

    # What to do each frame
    def __update_and_notify(self):
        self.__update_world()
        self.__notify_all()

    @staticmethod
    def __check_for_exit() -> bool:
        keep_going = True
        for event in pg.event.get():
            # Did the user click the window close button?
            if event.type == pg.QUIT:
                keep_going = False
        return keep_going

    # Use an Observer pattern for views
    def add_observer(self, observer):
        self.observers.append(observer)

    def __notify_all(self):
        for observer in self.observers:
            observer.on_world_update()

    def create_available_list(self):
        #Amount of empty cells in world (matrix)
        None_count = SIZE * SIZE * NeighboursModel.DIST[2]

        if len(self.available_positions) < None_count:
            for i in range(len(self.world)):
                for j in range(len(self.world[i])):
                    if self.world[i][j] == Actor.NONE:
                        self.available_positions.append([i, j])


# ---------------- Helper methods ---------------------
def create_world(size: int, default_type):
    new_world = [[default_type] * size for i in range(size)]
    return new_world

# Shuffles a list
def shuffle_list(_list: list):
    random.shuffle(_list)

# This method turns 1D lists into a 2D matrix
# More specific for this project the "World"
def list_to_matrix_converter(Actors_list: list, world: list):
    row = 0
    col = 0
    for index in range(len(Actors_list)):
        world[row][col] = Actors_list[index]
        print("row: ", row, "col: ", col)
        col += 1
        if col > len(world) - 1:
            row += 1
            col = 0

    return world


def populate_new_world(world: list, distribution: list):
    n = SIZE * SIZE
    blue_amt = int(distribution[0] * n)
    red_amt = int(distribution[1] * n)
    actors_list1D = [] #1D version of 2D-matrix used for shuffle an initializing

    #Fill 1D-list with all wanted Actors
    for cell in range(n):
        if cell < blue_amt:
            actors_list1D.append(Actor.BLUE)

        elif cell >= blue_amt and cell < (blue_amt + red_amt):
            actors_list1D.append(Actor.RED)

    #Shuffles the newly created 1D-list
    shuffle_list(actors_list1D)

    #Turn 1D-list to 2D matrix
    new_matrix2D = list_to_matrix_converter(actors_list1D, world)
    return new_matrix2D


# Counts specified Actor around given coordinate
def count_specified_actor(world: list, row: int, col: int, wanted_color: Actor):

    n_neighbours = set_start_count(world,row,col,wanted_color)

    for k in range(row - 1, row + 2):
        for l in range(col - 1, col + 2):
            n_neighbours += count_if_valid_orientation(world, k, l, wanted_color)

    return n_neighbours

# Make sure to not count the Actor at the specified coordinate
def set_start_count(world, row, col, wanted_color):
    n_neighbours = 0
    if world[row][col] == wanted_color:
        n_neighbours = -1

    return n_neighbours

def count_occupied_actors(populated_world: list, row: int, col: int):
    n_red = count_specified_actor(populated_world, row, col, Actor.RED)
    n_blue = count_specified_actor(populated_world, row, col, Actor.BLUE)
    total_n = n_red + n_blue
    return total_n


def set_actor_satisfaction(actors_world: list, row: int, col: int, satisfaction_world: list):
    n_wanted: int = count_specified_actor(actors_world, row, col, actors_world[row][col])
    n_neighbours = count_occupied_actors(actors_world, row, col)

    if actors_world[row][col] == Actor.NONE:
        return
    elif n_neighbours == 0:
        satisfaction_world[row][col] = State.UNSATISFIED
    elif n_wanted / n_neighbours < NeighboursModel.THRESHOLD:
        satisfaction_world[row][col] = State.UNSATISFIED
    elif n_wanted / n_neighbours >= NeighboursModel.THRESHOLD:
        satisfaction_world[row][col] = State.SATISFIED


# Makes sure that given coordinate is within list & adds if so
def count_if_valid_orientation(world: list, row: int, col: int, choice: Actor):
    is_orientation_valid = is_valid_location(len(world), row, col)

    if is_orientation_valid and world[row][col] == choice:
        return 1
        # Returning 'None' isn't wished. 0 is more like it
    else:
        return 0

# Check if inside world
def is_valid_location(size: int, row: int, col: int):
    return 0 <= row < size and 0 <= col < size


# ------- Testing -------------------------------------

# Here you run your tests i.e. call your logic methods
# to see that they really work
def test():
    # A small hard coded world for testing
    unsatisfied_world = [
        [State.NA, State.NA, State.NA, State.NA],
        [State.NA, State.NA, State.NA, State.NA],
        [State.NA, State.NA, State.NA, State.NA],
        [State.NA, State.NA, State.NA, State.NA],
    ]
    test_world = [
        [Actor.RED, Actor.RED, Actor.NONE, Actor.RED],
        [Actor.NONE, Actor.BLUE, Actor.NONE, Actor.RED],
        [Actor.RED, Actor.NONE, Actor.RED, Actor.RED],
        [Actor.RED, Actor.NONE, Actor.BLUE, Actor.RED]
    ]
    print(Actor.BLUE)
    th = 0.5  # Simpler threshold used for testing

    size = len(test_world)
    print(is_valid_location(size, 0, 0))
    print(not is_valid_location(size, -1, 0))
    print(is_valid_location(size, 0, 3))
    print(is_valid_location(size, 2, 2))

    # TODO More tests

    # Counts the "NONE" Actors around position 1,1. Should be 4 as seen above
    print(count_specified_actor(test_world, 1, 1, Actor.NONE) == 4)
    # Counts the "RED" Actors around position 1,0. Should be 3 as seen above
    print(count_specified_actor(test_world, 1, 0, Actor.RED) == 3)
    # Counts the Red Actors in the list. This value should be 3
    print(count(test_world, Actor.RED) == 9)
    print(count(test_world, Actor.BLUE) == 2)

    # Counts "living" neighbors around
    print(count_occupied_actors(test_world, 2, 2) == 5)
    print(count_occupied_actors(test_world, 1, 1) == 4)

    # Counts the neighbors of specified color
    print(count_specified_actor(test_world, 2, 2, Actor.RED) == 3)

    # Counts the colour of choice around neighbor
    print(count_specified_actor(test_world, 0, 0, Actor.RED) == 1)

    # Will neighbors become unsatisfied/satisfied in the test world?
    for i in range(len(test_world)):
        for j in range(len(test_world)):
            set_actor_satisfaction(test_world, i, j, unsatisfied_world)
    print(unsatisfied_world)

    exit(0)


# Helper method for testing
def count(a_list, to_find):
    the_count = 0
    for i in range(len(a_list)):
        for j in range(len(a_list[i])):
            if a_list[i][j] == to_find:
                the_count += 1
    return the_count


############  NOTHING to do below this row, it's pygame display stuff  ###########
# ... but by all means have a look at it, it's fun
class NeighboursView:
    # static class variables
    WIDTH = 400  # Size for window
    HEIGHT = 400
    MARGIN = 50

    WHITE = (200, 150, 100)
    RED = (255, 255, 255)
    BLUE = (0, 0, 0)

    # Instance methods

    def __init__(self, model: NeighboursModel):
        pg.init()  # initialize pygame, in case not already done
        self.dot_size = self.__calculate_dot_size(len(model.world))
        self.screen = pg.display.set_mode([self.WIDTH, self.HEIGHT])
        self.model = model
        self.model.add_observer(self)

    def render_world(self):
        # # Render the state of the world to the screen
        self.__draw_background()
        self.__draw_all_actors()
        self.__update_screen()

    # Needed for observer pattern
    # What do we do every time we're told the model had been updated?
    def on_world_update(self):
        self.render_world()

    # private helper methods
    def __calculate_dot_size(self, size):
        return max((self.WIDTH - 2 * self.MARGIN) / size, 2)

    @staticmethod
    def __update_screen():
        pg.display.flip()

    def __draw_background(self):
        self.screen.fill(NeighboursView.WHITE)

    def __draw_all_actors(self):
        for row in range(len(self.model.world)):
            for col in range(len(self.model.world[row])):
                self.__draw_actor_at(col, row)

    def __draw_actor_at(self, col, row):
        color = self.__get_color(self.model.world[row][col])
        xy = self.__calculate_coordinates(col, row)
        pg.draw.circle(self.screen, color, xy, self.dot_size / 2)

    # This method showcases how to nicely emulate 'switch'-statements in python
    @staticmethod
    def __get_color(actor):
        return {
            Actor.RED: NeighboursView.RED,
            Actor.BLUE: NeighboursView.BLUE
        }.get(actor, NeighboursView.WHITE)

    def __calculate_coordinates(self, col, row):
        x = self.__calculate_coordinate(col)
        y = self.__calculate_coordinate(row)
        return x, y

    def __calculate_coordinate(self, offset):
        x: float = self.dot_size * offset + self.MARGIN
        return x


if __name__ == "__main__":
    neighbours()
    #test()
