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

SIZE = 30


# Reads n elements in a row starting at start_index.
# Ex.(12, 3) reads the elements 12,13,14
def read_row(row_index: int, col_index, n: int):
    row = []
    for x in range(n):
        row.append(World[row_index[col_index + x]])
        print(row)
    return row


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

    # ########### These following two methods are what you're supposed to implement  ###########
    # In this method you should generate a new world
    # using randomization according to the given arguments.
    @staticmethod
    def __create_world(size) -> World:
        # TODO Create and populate world according to self.DIST distribution parameters
        brave_new_world = create_world(30)
        populate_world(brave_new_world, NeighboursModel.DIST, int(30 * 30))
        print(count(brave_new_world,Actor.BLUE))
        print(count(brave_new_world,Actor.RED))
        return brave_new_world

    # This is the method called by the timer to update the world
    # (i.e move unsatisfied) each "frame".
    def __update_world(self):
        # TODO Update logical state of world based on self.THRESHOLD satisfaction parameter
        pass

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


# ---------------- Helper methods ---------------------
def create_world(size: int):
    new_world = [[Actor.NONE] * size for i in range(size)]
    return new_world


def randomize_location(tot_spawns: int):
    world_size = int(math.sqrt(tot_spawns))
    position = random.randint(0, (world_size - 1))
    return position

#The purpose of this function is to set the actor for each cell in "world"
#This is made using a shuffled 1D list that is later turned into the 2D board "world"
def populate_individual_cell(world: list, dist: list):
    n = SIZE * SIZE
    blue_amt = int(dist[0] * n)
    red_amt = int(dist[1] * n)
    straight_world_list = []

    for cell in range(n):
        if cell < blue_amt:
            straight_world_list.append(Actor.BLUE)

        elif cell >= blue_amt and cell < (blue_amt + red_amt):
            straight_world_list.append(Actor.RED)

        elif cell >= (blue_amt + red_amt) and cell < n:
            straight_world_list.append(Actor.NONE)

    shuffle_list(straight_world_list)

    return  make_matrix(straight_world_list, world)

#Shuffles a list
def shuffle_list(_list :list):
    random.shuffle(_list)

#This method turns 1D lists into a 2D matrix
#More specific for this project the "World"
def make_matrix(Actors_list: list, world: list):
    row = 0
    col = 0
    for  index in range(len(Actors_list)):
        world[row][col] = Actors_list[index]
        print("row: ", row, "col: ", col)
        col += 1
        if col >29:
            row += 1
            col = 0

    return world


def populate_world(world: list, distribution: list, tot_spawns: int):
    populate_individual_cell(world, distribution)


# Counts specified Actor around given coordinate
def count_neighbors(world: list, i: int, j: int, choice: Actor):
    n_neighbors: int
    # Make sure to not count the Actor at the specified coordinate
    if world[i][j] == choice:
        n_neighbors = -1
    else:
        n_neighbors = 0

    for k in range(i - 1, i + 2):
        for l in range(j - 1, j + 2):
            n_neighbors += control_coordinate(world, k, l, choice)

    return n_neighbors


# OBS! Väldigt väldigt dåligt namn på metod nedanför
# Makes sure that given coordinate is within list
def control_coordinate(world: list, k: int, l: int, choice: Actor):
    if k + 1 > len(world) or k < 0:
        return 0
    elif l + 1 > len(world) or l < 0:
        return 0
    # If within the world and att specified location. Add one to the count
    elif world[k][l] == choice:
        return 1
    else:
        return 0


def copy_matrix(matrix_in: list):
    copied_matrix = []

    for i in range(len(matrix_in)):
        for j in range(len(matrix_in[i])):
            copied_matrix = matrix_in[i][j]

    return copied_matrix


# Check if inside world
def is_valid_location(size: int, row: int, col: int):
    return 0 <= row < size and 0 <= col < size


# ------- Testing -------------------------------------

# Here you run your tests i.e. call your logic methods
# to see that they really work
def test():
    # A small hard coded world for testing
    test_world = [
        [Actor.RED, Actor.RED, Actor.NONE],
        [Actor.NONE, Actor.BLUE, Actor.NONE],
        [Actor.RED, Actor.NONE, Actor.BLUE]
    ]
    print(Actor.BLUE)
    th = 0.5  # Simpler threshold used for testing

    size = len(test_world)
    print(is_valid_location(size, 0, 0))
    print(not is_valid_location(size, -1, 0))
    print(not is_valid_location(size, 0, 3))
    print(is_valid_location(size, 2, 2))

    # TODO More tests

    # Counts the "NONE" Actors around position 1,1. Should be 4 as seen above
    print(count_neighbors(test_world, 1, 1, Actor.NONE) == 4)
    # Counts the "RED" Actors around position 1,0. Should be 3 as seen above
    print(count_neighbors(test_world, 1, 0, Actor.RED) == 3)
    # Counts the Red Actors in the list. This value should be 3
    print(count(test_world, Actor.RED) == 3)

    exit(0)


# Helper method for testing
def count(a_list, to_find):
    the_count = 0
    for i in range(len(a_list)):
        for a in a_list[i]:
            if a == to_find:
                the_count += 1
    return the_count
# ###########  NOTHING to do below this row, it's pygame display stuff  ###########
# ... but by all means have a look at it, it's fun!
class NeighboursView:
    # static class variables
    WIDTH = 400  # Size for window
    HEIGHT = 400
    MARGIN = 50

    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

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
    test()
