import math
import random

import geopandas as gpd

import config


# Amsterdam
# Rotterdam
# The Hague
# Utrecht
# Eindhoven
# Groningen
# Tilburg
# Almere
# Breda
# Nijmegen
# Apeldoorn
# Arnhem
# Haarlem
# Haarlemmermeer
# Amersfoort
# Zaanstad
# Enschede
# Den Bosch
# Zwolle
# Leiden
# Zoetermeer
# Leeuwarden
# Ede
# Maastricht
# Dordrecht
# Westland
# Alphen aan den Rijn
# Alkmaar
# Emmen
# Delft
# Venlo
# Deventer


class RandomTrainingTrainingDataSampler:
    def __init__(self) -> None:
        self.index = gpd.read_file(config.env("BAG3D_INDEX_FILENAME"))

    # TODO: Find out why the override decorator cannot be imported from the typing
    #       module.
    def sample(self, size: int = -1):
        cities = _get_citizen_data()

        # Randomly select one of the most populous cities in the Netherlands (i.e.,
        # a municipality with at least 100.000 residents) using a uniform distribution.
        # TODO: Sample with replacement to allow for more samples.
        size = len(cities) if size == -1 else size
        sample = random.sample(cities, size)
        # TODO: Harmonize the name of this variable.
        # TODO: Keep sampling until the sample size is equal to the provided value.
        for i in sample:
            # Randomly select a point within a 10 km radius from the city center using a
            # normal distribution.
            center = cities[i]
            offset = _gen_random_point(*center, radius=10000)
            # Find the corresponding tile.
            # TODO: If the intersection of the point with the 3DBAG sheet index is null,
            #       then the point is out of bounds and the closest tile to it should be
            #       picked.
            # tile = index.overlay(point)
            # TODO: Check whether the tile has been selected before.
            pass


def _get_citizen_data() -> list:
    pass


def _gen_random_point(x, y, radius):
    # Generate two independent normally distributed random variables
    u, v = random.normalvariate(0, 1), random.normalvariate(0, 1)

    # Calculate the magnitude of the vector (u, v)
    magnitude = math.sqrt(u**2 + v**2)

    # Scale the point to lie within the specified radius
    u, v = (u / magnitude) * radius, (v / magnitude) * radius

    # Add the offset of the original point (x, y)
    random_point = (x + u, y + v)

    return random_point
