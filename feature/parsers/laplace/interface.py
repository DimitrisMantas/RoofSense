# -- interface.py
# -- geo1015.2023.hw01
# -- Hugo Ledoux <h.ledoux@tudelft.nl>
# -- 2023-11-07

# ------------------------------------------------------------------------------
# DO NOT MODIFY THIS FILE!!!
# ------------------------------------------------------------------------------


from tkinter import *

import numpy as np
import rasterio
from PIL import Image, ImageTk
from matplotlib import cm

from my_code_hw01 import Tin


class MyInterface(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("geo1015.2023.hw01")
        self.resizable(0, 0)
        self.bind("q", self.exit)
        self.bind("r", self.reset)
        self.bind("i", self.toggle_interpolate)
        self.bind("d", self.toggle_drawing_dtvd)
        self.output_info()
        self.drawdt = True
        self.interpolate = False
        self.dt = Tin()

        self.ds = rasterio.open("./data/dem_01.tif", "r")
        self.band1 = self.ds.read(1)
        im = Image.fromarray(
            np.uint8(cm.gist_earth(self.band1 / self.band1.max()) * 255)
        )
        im.save("./tmp.png")

        img = Image.open("./tmp.png")
        img.putalpha(127)
        self.img = ImageTk.PhotoImage(img)

        self.canvas = Canvas(
            self, bg="white", width=self.ds.width, height=self.ds.height
        )
        self.canvas.pack()
        self.set_display()

        self.draw()

    def set_display(self):
        self.bind("<Motion>", self.display_coords_text)
        self.bind("<ButtonRelease>", self.mouse_click)
        self.coordstext = self.canvas.create_text(
            self.ds.width, self.ds.height, fill="white", anchor="se", text=""
        )

    def toggle_interpolate(self, event):
        if self.interpolate == True:
            self.interpolate = False
        else:
            self.interpolate = True
        self.draw()

    def toggle_drawing_dtvd(self, event):
        if self.drawdt == True:
            self.drawdt = False
        else:
            self.drawdt = True
        self.draw()

    def mouse_click(self, event):
        wx, wy = self.ds.xy(event.x, event.y)
        if self.interpolate:
            wz = self.dt.interpolate_tin(wx, wy)
            print("Estimation z={}".format(wz))
        else:
            wz = self.band1[event.y][event.x]
            self.dt.insert_one_pt(wx, wy, wz)
            s = "+ (%.1f, %.1f, %.1f)" % (wx, wy, wz)
            print(s)
            vi_last = self.dt.number_of_vertices()
            area = self.dt.get_area_voronoi_cell(vi_last)
            if area == np.inf:
                print("Area Voronoi cell is infinite")
            else:
                print("Area Voronoi cell is {}".format(area))
        self.draw()

    def output_info(self):
        print("===== USAGE =====")
        print("keyboard 'd' to toggle between DT and VD.")
        print("keyboard 'i' to toggle between insertion and interpolation.")
        print("keyboard 'q' to quit the program.")
        print("keyboard 'r' to reset the DT/VD to an empty one.")
        print("=================")

    def draw_point(self, x, y, colour):
        radius = 3
        row, col = self.ds.index(x, y)
        self.canvas.create_oval(
            row - radius, col - radius, row + radius, col + radius, fill=colour
        )

    def draw_voronoi(self):
        edges = self.dt.get_voronoi_edges()
        for i in range(0, len(edges), 2):
            row, col = self.ds.index(edges[i][0], edges[i][1])
            row2, col2 = self.ds.index(edges[i + 1][0], edges[i + 1][1])
            self.draw_edge(
                row, self.ds.height - col, row2, self.ds.height - col2, "red"
            )

    def draw_delaunay(self):
        edges = self.dt.get_delaunay_edges()
        for i in range(0, len(edges), 2):
            row, col = self.ds.index(edges[i][0], edges[i][1])
            row2, col2 = self.ds.index(edges[i + 1][0], edges[i + 1][1])
            self.draw_edge(
                row, self.ds.height - col, row2, self.ds.height - col2, "black"
            )

    def display_coords_text(self, event):
        if (
            (event.x > 0)
            and (event.x < self.ds.width)
            and (event.y > 0)
            and (event.y < self.ds.height)
        ):
            wx, wy = self.ds.xy(event.x, event.y)
            wz = self.band1[event.y][event.x]
            s = "(%.1f, %.1f, %.1f)" % (wx, wy, wz)
            self.canvas.itemconfig(self.coordstext, text=s)

    def draw_edge(self, x1, y1, x2, y2, colour):
        # colour = "black"
        self.canvas.create_line(
            x1, self.ds.height - y1, x2, self.ds.height - y2, fill=colour
        )

    def draw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.img, anchor=NW)
        self.set_display()
        pts = self.dt.get_delaunay_vertices()
        if self.drawdt == True:
            self.draw_delaunay()
        else:
            self.draw_voronoi()
        for pt in pts[1:]:
            self.draw_point(pt[0], pt[1], "lightblue")
        self.canvas.create_text(
            5,
            self.ds.height,
            anchor="sw",
            fill="white",
            text="[pts={}/trs={}]".format(
                self.dt.number_of_vertices(), self.dt.number_of_triangles()
            ),
        )
        if self.interpolate:
            self.canvas.create_text(
                self.ds.width / 2,
                self.ds.height,
                anchor="s",
                fill="white",
                text="interpolate",
            )
        else:
            self.canvas.create_text(
                self.ds.width / 2,
                self.ds.height,
                anchor="s",
                fill="white",
                text="insert",
            )

    def reset(self, event):
        print("Reset DT to empty one")
        self.dt = Tin()
        self.draw()

    def exit(self, event):
        print("bye bye.")
        self.destroy()
