import geopandas as gpd
import geopy.geocoders
import numpy as np
import pyproj

cities = [
    "Amsterdam",
    "Rotterdam",
    "The Hague",
    "Utrecht",
    "Eindhoven",
    "Groningen",
    "Tilburg",
    "Almere",
    "Breda",
    "Nijmegen",
    "Apeldoorn",
    "Arnhem",
    "Haarlem",
    "Haarlemmermeer",
    "Amersfoort",
    "Zaanstad",
    "Enschede",
    "Den Bosch",
    "Zwolle",
    "Leiden",
    "Zoetermeer",
    "Leeuwarden",
    "Ede",
    "Maastricht",
    "Dordrecht",
    "Westland",
    "Alphen aan den Rijn",
    "Alkmaar",
    "Emmen",
    "Delft",
    "Venlo",
    "Deventer",
]


def get_seeds():
    geocoder = geopy.geocoders.GoogleV3(api_key="", user_agent="RoofSense")
    transform = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:28992")

    coords = []
    for city in cities:
        loc = geocoder.geocode(city + ", The Netherlands")
        coords.append(transform.transform(loc.latitude, loc.longitude))

    a = np.array(coords)

    pts = gpd.points_from_xy(a[:, 0], a[:, 1], crs="EPSG:28992")
    gpd.GeoDataFrame(data={"id": cities}, geometry=pts).to_file("cities.gpkg")
