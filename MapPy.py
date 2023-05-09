import os
import copy

import rasterio

import pickle

import tkinter
import tkinter.messagebox
import customtkinter

from tkinter import *
from tkinter import filedialog as fd

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from svgpathtools import svg2paths
from svgpathtools import real, imag
from svgpathtools import wsvg

from shapely.geometry import MultiLineString
from shapely.geometry import LineString
from shapely import affinity

import geopandas as gpd

# Modes: "System" (standard), "Dark", "Light"
customtkinter.set_appearance_mode("System")
# Themes: "blue" (standard), "green", "dark-blue"
customtkinter.set_default_color_theme("blue")


def save_lines(linestrings, self):

    data = []

    lx = float(self.llx_entry.get())
    ly = float(self.lly_entry.get())
    cellsize = float(self.cellsize_entry.get())
    epsg = self.epsg_entry.get()

    multiline = MultiLineString([line.coords for line in linestrings])

    data.append(lx)
    data.append(ly)
    data.append(cellsize)
    data.append(epsg)
    data.append(self.width)
    data.append(self.height)
    data.append(multiline)

    # Save data
    filename = self.filename_svg.replace('.svg', '_lines.dat')
    with open(filename, "wb") as lines_file:
        pickle.dump(data, lines_file, pickle.HIGHEST_PROTOCOL)


def save_shp(linestrings, elev, filename_svg, lx, ly, cellsize, epsg):

    path_geom = gpd.GeoDataFrame({
        'ELEV': elev,
        'geometry': linestrings
    },
                                 geometry='geometry',
                                 crs='epsg:' + epsg)
    path_geom["geometry"] = path_geom["geometry"].apply(affinity.scale,
                                                        xfact=cellsize,
                                                        yfact=cellsize,
                                                        origin=(0, 0))
    path_geom["geometry"] = path_geom["geometry"].apply(affinity.translate,
                                                        xoff=lx,
                                                        yoff=ly)
    filename_shp = filename_svg.replace('svg', 'shp')
    path_geom.to_file(filename_shp)

def plot_svg(self,paths):

    for path in paths:

        for count, elem in enumerate(path):

            # print(elem)

            if 'CubicBezier' in str(type(elem)):

                p = elem.poly()
                p_start = elem.start
                p_end = elem.end
                p = elem.poly()

                xp = real(p)
                yp = imag(p)

                s = np.linspace(0, 1, 10)

                x = np.polyval(xp, s)
                y = np.polyval(yp, s)

                self.ax.plot(x, self.height - y, '-g')

            elif 'Line' in str(type(elem)):

                p_start = elem.start
                x_start = real(p_start)
                y_start = imag(p_start)

                p_end = elem.end
                x_end = real(p_end)
                y_end = imag(p_end)

                x = np.linspace(x_start, x_end, 2)
                y = np.linspace(y_start, y_end, 2)

                self.ax.plot(x, self.height - y, '-g')

    self.ax.set_xlim([0, self.width])
    self.ax.set_ylim([0, self.height])

    self.canvas.draw()


def draw_paths(self, paths):

    path_points = 50
    min_length = 30

    text_kwargs = dict(ha='center', va='center', fontsize=12, color='r')

    xx = []
    yy = []

    xx = np.array(xx)
    yy = np.array(yy)

    self.linestrings = []

    p_endOLD = paths[0][0].start
    p_startOLD = paths[0][0].end
    pathOLD = paths[0][0]

    self.lines = []
    self.labels = []

    for path in paths:

        for count, elem in enumerate(path):

            # print(elem)

            if 'CubicBezier' in str(type(elem)):

                p = elem.poly()
                p_start = elem.start
                p_end = elem.end
                p = elem.poly()

                xp = real(p)
                yp = imag(p)

                s = np.linspace(0, 1, path_points)

                x = np.polyval(xp, s)
                y = np.polyval(yp, s)

                # self.ax.plot(x, self.height - y, '-g')
                """
                s0 = 0.5
                s1 = 0.51
                x0 = np.polyval(xp, s0)
                y0 = np.polyval(yp, s0)
                x1 = np.polyval(xp, s1)
                y1 = np.polyval(yp, s1)
                self.ax.arrow(x0,
                          y0,
                          x1 - x0,
                          y1 - y0,
                        shape='full',
                        color='k',
                          lw=0,
                          length_includes_head=True,
                          head_width=.25)
                """

            elif 'Line' in str(type(elem)):

                p_start = elem.start
                x_start = real(p_start)
                y_start = imag(p_start)

                p_end = elem.end
                x_end = real(p_end)
                y_end = imag(p_end)

                x = np.linspace(x_start, x_end, path_points)
                y = np.linspace(y_start, y_end, path_points)

                # self.ax.plot(x, self.height - y, '-g')
                """
                s0 = 0.5
                s1 = 0.51
                x0 = x_start + s0 * (x_end - x_start)
                y0 = y_start + s0 * (y_end - y_start)
                x1 = x_start + s1 * (x_end - x_start)
                y1 = y_start + s1 * (y_end - y_start)
                self.ax.arrow(x0,
                          y0,
                          x1 - x0,
                          y1 - y0,
                          shape='full',
                          color='k',
                          lw=0,
                          length_includes_head=True,
                          head_width=.25)
                """

            # else:

            # print('OTHER')

            angle_check = FALSE

            if p_start == p_endOLD:

                side1 = 'start'
                side2 = 'end'

                vect1_x, vect1_y = tangent_path(elem, side1)
                vect2_x, vect2_y = tangent_path(pathOLD, side2)

                cos12 = vect1_x * vect2_x + vect1_y * vect2_y
                angle_12 = np.degrees(np.arccos(-cos12))

                # print('CHECK 1')
                # print('count',count)
                # print(p_start)
                # print(p_end)
                # print(p_startOLD)
                # print(p_endOLD)
                # print(angle_12)

                if angle_12 < 20.0:

                    angle_check = TRUE

                    xx = np.append(xx, x)
                    yy = np.append(yy, y)

                    p_endOLD = p_end
                    pathOLD = elem

            elif p_end == p_startOLD:

                side1 = 'end'
                side2 = 'start'

                vect1_x, vect1_y = tangent_path(elem, side1)
                vect2_x, vect2_y = tangent_path(pathOLD, side2)

                cos12 = vect1_x * vect2_x + vect1_y * vect2_y
                angle_12 = np.degrees(np.arccos(-cos12))

                # print('CHECK 2')
                # print('count',count)
                # print(p_start)
                # print(p_end)
                # print(p_startOLD)
                # print(p_endOLD)
                # print(angle_12)


                if angle_12 < 20.0:

                    angle_check = TRUE

                    xx = np.append(x, xx)
                    yy = np.append(y, yy)

                    p_startOLD = p_start
                    pathOLD = elem

            if not angle_check:

                # print('NEW PATH', count)

                xy = zip(xx, self.height - yy)
                line = LineString(xy)
                length = line.length

                [minx, miny, maxx, maxy] = line.bounds

                area = (maxx - minx) * (maxy - miny)

                if (length > min_length):

                    ln, = self.ax.plot(xx, self.height - yy, '-k', picker=True)

                    self.lines.append(ln)
                    self.linestrings.append(line)
                    idx = int(np.floor(0.5 * len(xx)))
                    label = self.ax.text(xx[idx], self.height - yy[idx], '',
                                         **text_kwargs)
                    self.labels.append(label)

                xx = np.array(x)
                yy = np.array(y)

                p_startOLD = p_start
                p_endOLD = p_end
                pathOLD = elem

    self.ax.set_xlim([0, self.width])
    self.ax.set_ylim([0, self.height])

    self.linevalues = np.zeros(len(self.lines))
    self.lines_set = np.zeros(len(self.lines), dtype=bool)

    self.canvas.draw()

def change_paths_order(self, paths):

    npaths = len(paths[0])

    connected_list = [[] for _ in range(npaths)]

    for count1, path1 in enumerate(paths[0]):

        for count2, path2 in enumerate(paths[0][count1 + 1:]):

            if (path1.start == path2.end) or (path1.start == path2.start) or (
                path1.end == path2.end) or (path1.end == path2.start):

                if (path1.start == path2.end):

                    side1 = 'start'
                    side2 = 'end'

                elif (path1.start == path2.start):

                    side1 = 'start'
                    side2 = 'start'

                elif (path1.end == path2.end):

                    side1 = 'end'
                    side2 = 'end'

                elif (path1.end == path2.start):

                    side1 = 'end'
                    side2 = 'start'

                vect1_x, vect1_y = tangent_path(path1, side1)
                vect2_x, vect2_y = tangent_path(path2, side2)

                cos12 = vect1_x * vect2_x + vect1_y * vect2_y
                angle_12 = np.degrees(np.arccos(-cos12))

                if angle_12 < 30.0:

                    connected_list[count1].append(count1 + count2 + 1)
                    connected_list[count1 + count2 + 1].append(count1)

    new_path = []
    idx_new_path = -1

    self.bar.start()

    progress_old = 0

    for count1, path1 in enumerate(paths[0]):

        progress_step = count1 / (len(paths[0]) - 1)

        if (progress_step >= progress_old + 0.01):

            self.bar.set(progress_step)
            self.update_idletasks()
            progress_old += 0.01

        if len(connected_list[count1]) == 1:

            # print('')
            # print('New path', count1, end='')

            path_length = 1
            new_path.append(path1)
            idx_new_path += 1

            actual = count1
            next = connected_list[actual][0]

            connected_list[actual].remove(next)

            while len(connected_list[next]) >= 1:

                # print(',', next, end='')

                connected_list[next].remove(actual)

                previous = actual
                actual = next

                path_previous = paths[0][previous]
                path = paths[0][actual]

                if len(connected_list[actual]) > 0:

                    next = connected_list[actual][0]
                    connected_list[actual].remove(next)

                if not (path_previous.start == path.end
                        or path_previous.end == path.start):

                    # reverse path
                    # print('Rev')
                    path_temp = copy.copy(path)
                    path.start = path_temp.end
                    path.end = path_temp.start
                    if 'CubicBezier' in str(type(path_temp)):
                        path.control1 = path_temp.control2
                        path.control2 = path_temp.control1

                    paths[0][actual] = copy.copy(path)

                path_length += 1
                new_path.append(path)
                idx_new_path += 1

            # print('')
            # print('Length', path_length)

    # print('len paths[0]', len(paths[0]))
    paths[0] = new_path
    # print('len paths[0]', len(paths[0]))

    self.bar.stop()

    return paths


def tangent_path(elem, side):

    p_start = elem.start
    p_end = elem.end

    if 'CubicBezier' in str(type(elem)):

        p = elem.poly()
        p = elem.poly()

        xp = real(p)
        yp = imag(p)

        if (side == 'end'):

            x1 = real(p_end)
            y1 = imag(p_end)

            x2 = np.polyval(xp, 0.99)
            y2 = np.polyval(yp, 0.99)

        else:

            x1 = real(p_start)
            y1 = imag(p_start)

            x2 = np.polyval(xp, 0.01)
            y2 = np.polyval(yp, 0.01)

    elif 'Line' in str(type(elem)):

        if (side == 'end'):

            x1 = real(p_end)
            y1 = imag(p_end)

            x2 = real(p_start)
            y2 = imag(p_start)

        else:

            x1 = real(p_start)
            y1 = imag(p_start)

            x2 = real(p_end)
            y2 = imag(p_end)

    den = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    tangent_x = (x1 - x2) / den
    tangent_y = (y1 - y2) / den

    return tangent_x, tangent_y


def modify_paths(self, paths, side):

    dist_max = 2.0
    angle_max = 25.0

    new_paths = copy.copy(paths)

    self.bar.start()

    progress_old = 0

    npaths = len(paths[0])

    for count, path1 in enumerate(new_paths[0]):

        progress_step = count / (npaths - 1)
        # print('count',count,progress_step)

        if (progress_step >= progress_old + 0.01):

            if side == 'start':

                self.bar.set(0.5 * progress_step)
                self.update_idletasks()

            else:

                self.bar.set(0.5 + 0.5 * progress_step)
                self.update_idletasks()

            progress_old += 0.01

        p_start1 = path1.start
        p_end1 = path1.end

        if (p_start1 == p_end1):

            continue

        xp_start1 = real(p_start1)
        yp_start1 = imag(p_start1)

        xp_end1 = real(p_end1)
        yp_end1 = imag(p_end1)

        if side == 'start':

            x1 = xp_start1
            y1 = yp_start1
            side1 = 'start'

        elif side == 'end':

            x1 = xp_end1
            y1 = yp_end1
            side1 = 'end'

        distMin = 1.e10

        for count2, path2 in enumerate(new_paths[0]):

            if count <= count2:

                continue

            p_start2 = path2.start
            p_end2 = path2.end

            if (p_start2 == p_end2):

                continue

            # vect1_x, vect1_y = tangent_path(path1, side1)

            if ( side1 == 'start'):

                if (p_start1 == p_start2) or (p_start1 == p_end2):

                    distMin = 0.0
                    break

                    """
                    if (p_start1 == p_start2):

                        vect2_x, vect2_y = tangent_path(path2, 'start')

                    else:

                        vect2_x, vect2_y = tangent_path(path2, 'end')

                    cos12 = vect1_x * vect2_x + vect1_y * vect2_y
                    angle_12 = np.degrees(np.arccos(-cos12))

                    if ( angle_12 < 20.0):

                        distMin = 0.0
                        break

                    else:

                        continue
                    """

            elif ( side1 == 'end'):

                if (p_end1 == p_start2) or (p_end1 == p_end2):

                    distMin = 0.0
                    break

                    """
                    if (p_end1 == p_start2):

                        vect2_x, vect2_y = tangent_path(path2, 'start')

                    else:

                        vect2_x, vect2_y = tangent_path(path2, 'end')

                    cos12 = vect1_x * vect2_x + vect1_y * vect2_y
                    angle_12 = np.degrees(np.arccos(-cos12))

                    if ( angle_12 < 20.0):

                        distMin = 0.0
                        break

                    else:

                        continue
                    """

            xp_start2 = real(p_start2)
            yp_start2 = imag(p_start2)

            xp_end2 = real(p_end2)
            yp_end2 = imag(p_end2)

            # distance with start-point2
            dist1 = np.sqrt((x1 - xp_start2)**2 + (y1 - yp_start2)**2)
            # distance with end-point2
            dist2 = np.sqrt((x1 - xp_end2)**2 + (y1 - yp_end2)**2)

            if np.amin([dist1, dist2]) < distMin:

                distMin = np.amin([dist1, dist2])
                sideMin = np.argmin([dist1, dist2])
                idxMin = count2

        # for path with index ixMin, if one edge is close enough
        # to path1, check if it is close to be aligned with path1
        if distMin > 0 and distMin <= dist_max:

            path2 = new_paths[0][idxMin]

            p_start2 = path2.start
            p_end2 = path2.end

            xp_start2 = real(p_start2)
            yp_start2 = imag(p_start2)

            xp_end2 = real(p_end2)
            yp_end2 = imag(p_end2)

            if sideMin == 0:

                x2 = xp_start2
                y2 = yp_start2
                side2 = 'start'

            elif sideMin == 1:

                x2 = xp_end2
                y2 = yp_end2
                side2 = 'end'

            vect1_x, vect1_y = tangent_path(path1, side1)
            vect2_x, vect2_y = tangent_path(path2, side2)

            cos12 = vect1_x * vect2_x + vect1_y * vect2_y
            angle_12 = np.degrees(np.arccos(-cos12))

            if angle_12 > angle_max:

                continue

            vect12_x = (x2 - x1) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            vect12_y = (y2 - y1) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            cosA = vect12_x * vect2_x + vect12_y * vect2_y
            angle_A = np.degrees(np.arccos(cosA))

            cosB = vect1_x * vect12_x + vect1_y * vect12_y
            angle_B = np.degrees(np.arccos(cosB))

            angle_AB = np.amin(
                [angle_A, 180 - angle_A, angle_B, 180 - angle_B])

            if (np.maximum(angle_12, angle_AB) < angle_max):

                # print('New connected paths:', count, idxMin)

                if side1 == 'start':

                    if side2 == 'start':

                        new_paths[0][count].start = new_paths[0][idxMin].start

                    elif side2 == 'end':

                        new_paths[0][count].start = new_paths[0][idxMin].end

                elif side1 == 'end':

                    if side2 == 'start':

                        new_paths[0][count].end = new_paths[0][idxMin].start

                    elif side2 == 'end':

                        new_paths[0][count].end = new_paths[0][idxMin].end

    self.bar.stop()

    return new_paths


def get_svg_size(filename):

    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    width_str, height_str = tree.getroot().attrib["width"], tree.getroot(
    ).attrib["height"]

    print("Width:", width_str, "Height:", height_str)

    i = 0
    for c in width_str:
        if c.isdigit():

            i += 1

        else:

            break

    width = float(width_str[:i])

    i = 0
    for c in height_str:
        if c.isdigit():

            i += 1

        else:

            break

    height = float(height_str[:i])

    print("Width:", width, "Height:", height)

    return width, height


class App(customtkinter.CTk):

    def __init__(self):
        super().__init__()

        self.val = 0
        self.dz = 5
        self.idx = 0

        # configure window
        self.title("MapPy")
        self.geometry(f"{1100}x{600}")

        # configure grid layout (2x3)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=30, minsize=100)
        self.grid_rowconfigure(1, weight=0, minsize=30)
        self.grid_rowconfigure(2, weight=0, minsize=50)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self,
                                                    width=140,
                                                    corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")

        # create sidebar frame with widgets
        self.coord_frame = customtkinter.CTkFrame(self.sidebar_frame,
                                                  width=140,
                                                  corner_radius=10)
        self.coord_frame.grid(row=8, column=0, sticky="sew")

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame,
                                                 text="CreateMap",
                                                 font=customtkinter.CTkFont(
                                                     size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(5, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(
            self.sidebar_frame, text="Load image", command=self.select_file)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(
            self.sidebar_frame, text="Load svg", command=self.select_svg)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(
            self.sidebar_frame, text="Load lines", command=self.select_lines)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)

        self.connect_button = customtkinter.CTkButton(self.sidebar_frame,
                                                      text="Analyze",
                                                      state=DISABLED,
                                                      command=self.connect)
        self.connect_button.grid(row=4, column=0, padx=20, pady=10)

        self.change_order_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Sort order",
            state=DISABLED,
            command=self.change_order)
        self.change_order_button.grid(row=5, column=0, padx=20, pady=10)

        self.plot_paths_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Plot paths",
            state=DISABLED,
            command=self.plot_paths)
        self.plot_paths_button.grid(row=6, column=0, padx=20, pady=10)

        self.bar = customtkinter.CTkProgressBar(master=self.sidebar_frame,
                                                orientation='horizontal',
                                                mode='determinate')

        self.bar.grid(row=7, column=0, pady=10, padx=20, sticky="s")

        # Set default starting point to 0
        self.bar.set(0)

        # create figure
        self.fig_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.fig_frame.grid(row=0,
                            column=1,
                            padx=(10, 10),
                            pady=(0, 0),
                            sticky="nsew")
        self.fig_frame.grid_rowconfigure(0, weight=1)
        self.fig_frame.grid_columnconfigure(0, weight=1)

        # add toolbar frame
        self.toolbar_frame = customtkinter.CTkFrame(self,
                                                    fg_color="transparent")
        self.toolbar_frame.grid(row=1,
                                column=1,
                                padx=(10, 10),
                                pady=(0, 0),
                                sticky="new")
        self.toolbar_frame.grid_rowconfigure(0, weight=1)
        # self.toolbar_frame.grid_columnconfigure(0, weight=1)

        # create button frame
        self.button_frame = customtkinter.CTkFrame(self,
                                                   fg_color="transparent")
        self.button_frame.grid(row=2,
                               column=1,
                               padx=(10, 10),
                               pady=(0, 0),
                               sticky="nsew")

        # create figure
        fig = Figure(dpi=100)
        fig.subplots_adjust(left=0,
                            right=1,
                            bottom=0,
                            top=1,
                            wspace=0,
                            hspace=0)
        fig.canvas.mpl_connect('pick_event', self.onpick)
        self.ax = fig.add_subplot()

        fig.set_facecolor("lightgrey")
        self.ax.set_facecolor("none")

        self.ax.axis('equal')

        # A tk.DrawingArea.
        self.canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        self.canvas.get_tk_widget().grid(row=0,
                                         column=0,
                                         padx=(10, 10),
                                         pady=(0, 0),
                                         sticky="nsew")
        self.canvas.draw()

        self.canvas.mpl_connect(
            "key_press_event", lambda event: print(f"you pressed {event.key}"))
        self.canvas.mpl_connect("key_press_event", key_press_handler)

        # create toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas,
                                            self.toolbar_frame,
                                            pack_toolbar=False)
        self.toolbar.grid(row=0,
                          column=1,
                          padx=(10, 10),
                          pady=(0, 0),
                          sticky="new")
        self.toolbar.update()

        # create buttons
        button_ypad = 0

        self.button_1 = customtkinter.CTkButton(self.button_frame,
                                                width=30,
                                                text="--",
                                                command=self.subtract_two)
        self.button_1.grid(row=1,
                           column=0,
                           padx=10,
                           pady=button_ypad,
                           sticky="n")
        self.button_2 = customtkinter.CTkButton(self.button_frame,
                                                width=30,
                                                text="-",
                                                command=self.subtract_one)
        self.button_2.grid(row=1,
                           column=1,
                           padx=10,
                           pady=button_ypad,
                           sticky="n")
        self.button_3 = customtkinter.CTkButton(self.button_frame,
                                                width=30,
                                                text="=",
                                                command=self.equal_prev)
        self.button_3.grid(row=1,
                           column=2,
                           padx=10,
                           pady=button_ypad,
                           sticky="n")
        self.button_4 = customtkinter.CTkButton(self.button_frame,
                                                width=30,
                                                text="+",
                                                command=self.add_one)
        self.button_4.grid(row=1,
                           column=3,
                           padx=10,
                           pady=button_ypad,
                           sticky="n")
        self.button_5 = customtkinter.CTkButton(self.button_frame,
                                                width=30,
                                                text="++",
                                                command=self.add_two)
        self.button_5.grid(row=1,
                           column=4,
                           padx=10,
                           pady=button_ypad,
                           sticky="n")
        # create main entry and button
        self.entry = customtkinter.CTkEntry(self.button_frame,
                                            placeholder_text=str(self.val),
                                            insertontime=0)
        self.entry.grid(row=1,
                        column=5,
                        padx=(10, 10),
                        pady=button_ypad,
                        sticky="sew")
        self.entry.insert(0, str(self.val))

        self.next_button = customtkinter.CTkButton(self.button_frame,
                                                   width=40,
                                                   text="Set",
                                                   state=DISABLED,
                                                   command=self.set_val)
        self.next_button.grid(row=1,
                              column=6,
                              padx=10,
                              pady=button_ypad,
                              sticky="n")

        self.save_button = customtkinter.CTkButton(self.button_frame,
                                                   width=40,
                                                   text="Save",
                                                   state=DISABLED,
                                                   command=self.save)
        self.save_button.grid(row=1,
                              column=7,
                              padx=10,
                              pady=button_ypad,
                              sticky="n")

        self.quit_button = customtkinter.CTkButton(self.button_frame,
                                                   width=40,
                                                   text="Quit",
                                                   state='normal',
                                                   command=self.close)
        self.quit_button.grid(row=1,
                              column=8,
                              padx=10,
                              pady=button_ypad,
                              sticky="n")

        self.coord_text = customtkinter.CTkLabel(self.coord_frame,
                                                 text="Coords",
                                                 font=customtkinter.CTkFont(
                                                     size=20, weight="bold"))

        self.coord_text.grid(row=0, column=0, padx=20, pady=(10, 0))

        self.llx_entry = customtkinter.CTkEntry(master=self.coord_frame,
                                                placeholder_text="llx coords",
                                                height=30,
                                                border_width=2,
                                                corner_radius=10)

        self.llx_entry.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.lly_entry = customtkinter.CTkEntry(master=self.coord_frame,
                                                placeholder_text="lly coords",
                                                height=30,
                                                border_width=2,
                                                corner_radius=10)

        self.lly_entry.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.cellsize_entry = customtkinter.CTkEntry(
            master=self.coord_frame,
            placeholder_text="cellsize [m]",
            height=30,
            border_width=2,
            corner_radius=10)

        self.cellsize_entry.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        self.epsg_entry = customtkinter.CTkEntry(master=self.coord_frame,
                                                 placeholder_text="epsg",
                                                 height=30,
                                                 border_width=2,
                                                 corner_radius=10)

        self.epsg_entry.grid(row=4, column=0, padx=10, pady=5, sticky="w")

        self.dz_entry = customtkinter.CTkEntry(master=self.coord_frame,
                                                 placeholder_text="dz",
                                                 height=30,
                                                 border_width=2,
                                                 corner_radius=10)

        self.dz_entry.grid(row=5, column=0, padx=10, pady=5, sticky="w")

        self.linestring_idx = -1

        # self.appearance_mode_optionemenu.set("Dark")
        # self.scaling_optionemenu.set("100%")

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:",
                                              title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")

    def subtract_two(self):

        dz = self.dz_entry.get()

        try:
            dz_val = float(dz)
        except ValueError:
            print('Check dz')
            return

        self.val -= 2.0 * dz_val
        # print("sidebar_button click",self.val)
        self.entry.delete(0, END)
        self.entry.insert(0, str(self.val))

    def subtract_one(self):

        dz = self.dz_entry.get()

        try:
            dz_val = float(dz)
        except ValueError:
            print('Check dz')
            return

        self.val -= 1.0 * dz_val
        # print("sidebar_button click",self.val)
        self.entry.delete(0, END)
        self.entry.insert(0, str(self.val))

    def equal_prev(self):

        dz = self.dz_entry.get()

        try:
            dz_val = float(dz)
        except ValueError:
            print('Check dz')
            return

        self.val -= 0.0 * dz_val
        # print("sidebar_button click",self.val)
        self.entry.delete(0, END)
        self.entry.insert(0, str(self.val))

    def add_one(self):

        dz = self.dz_entry.get()

        try:
            dz_val = float(dz)
        except ValueError:
            print('Check dz')
            return

        self.val += 1.0 * dz_val
        # print("sidebar_button click",self.val)
        self.entry.delete(0, END)
        self.entry.insert(0, str(self.val))

    def add_two(self):

        dz = self.dz_entry.get()

        if dz.isnumeric():
            dz_val = float(dz)
        else:
            print('Check dz')
            return

        self.val += 2.0 * dz_val
        # print("sidebar_button click",self.val)
        self.entry.delete(0, END)
        self.entry.insert(0, str(self.val))

    def set_val(self):

        self.val = float(self.entry.get())
        self.idx += 1

        f = float(self.val)

        if self.linestring_idx >= 0:

            self.linevalues[self.linestring_idx] = f
            self.lines_set[self.linestring_idx] = True
            self.lines[self.linestring_idx].set_color('g')
            self.labels[self.linestring_idx].set_text(str(f))

        line = self.linestrings[self.linestring_idx]

        dist = np.zeros(len(self.linestrings))

        if self.linestring_idx > 0:

            dist[0:self.linestring_idx] = np.array([
                line.distance(Sline)
                for Sline in self.linestrings[0:self.linestring_idx]
            ])

        dist[self.linestring_idx] = 1.e10

        dist[self.linestring_idx:] = np.array([
            line.distance(Sline)
            for Sline in self.linestrings[self.linestring_idx:]
        ])

        dist[self.lines_set > 0] = 1.e10

        idxMin = np.argmin(dist)

        self.linestring_idx = idxMin

        self.lines[self.linestring_idx].set_color('b')

        x, y = line.coords.xy

        self.save_button.configure(state='normal')

        # print(self.linevalues)

        # required to update canvas and attached toolbar!
        self.canvas.draw()

    def save(self):

        linestrings = []
        elev = []

        rem_linestrings = []
        remaining = False

        for count, line in enumerate(self.linestrings):

            if self.lines_set[count]:

                linestrings.append(line)
                elev.append(self.linevalues[count])

            else:

                rem_linestrings.append(line)
                remaining = True

        lx = self.llx_entry.get()
        ly = self.lly_entry.get()
        cellsize = self.cellsize_entry.get()
        epsg = self.epsg_entry.get()

        print(lx, ly, cellsize, epsg)

        try:
            lx = float(lx)
        except ValueError:
            print('Check lx')
            return

        try:
            ly = float(ly)
        except ValueError:
            print('Check ly')
            return

        try:
            cellsize = float(cellsize)
        except ValueError:
            print('Check cellsize')
            return

        try:
            epsg_int = int(epsg)
        except ValueError:
            print('Check epsg')
            return   

        if len(linestrings)>0:

            save_shp(linestrings, elev, self.filename_svg, lx, ly, cellsize, epsg)

        if remaining:

            save_lines(rem_linestrings, self)

    def select_file(self):
        filetypes = (('jpg or tif files', '*.jpg *.tif'), ('All files',
                                                                   '*.*'))

        self.filename = fd.askopenfilename(title='Open a file',
                                           initialdir='./',
                                           filetypes=filetypes)

        autotrace_cmd = '../autotrace-master/autotrace'
        autotrace_opt = '--centerline --background-color ffffff --color-count 2 --preserve-width'
        autotrace_inp = '-input-format ppm'
        autotrace_out = '-output-format svg -output-file'

        filename_split = self.filename.split('.')

        self.filename_svg = filename_split[0] + '.svg'
        cmd = autotrace_cmd + ' ' + autotrace_opt + ' ' + autotrace_inp + \
            ' ' + self.filename + ' ' + autotrace_out + ' ' + self.filename_svg
        os.system(cmd)
        self.filename_jpg = self.filename
        print(cmd)

        if self.filename[-3:]=='tif':
        
            tiff = rasterio.open(self.filename)
            
            xll = tiff.bounds.left
            yll = tiff.bounds.bottom
            xtr = tiff.bounds.right
            ytr = tiff.bounds.top
            
            lx = tiff.bounds.left
            self.llx_entry.delete(0, END)
            self.llx_entry.insert(0, str(xll))
            
            ly = tiff.bounds.bottom
            self.lly_entry.delete(0, END)
            self.lly_entry.insert(0, str(yll))

            dx = (xtr-xll)/tiff.width
            dy = (ytr-yll)/tiff.height
            cellsize = 0.5*(dx+dy)
            self.cellsize_entry.delete(0, END)
            self.cellsize_entry.insert(0, str(cellsize))

            crs = str(tiff.crs).split(':')[-1]
            self.epsg_entry.delete(0, END)
            self.epsg_entry.insert(0, str(crs))

        img = plt.imread(self.filename)

        self.paths, attributes = svg2paths(self.filename_svg)
        self.width, self.height = get_svg_size(self.filename_svg)

        self.ax.imshow(img,
                       extent=[0, self.width, 0, self.height],
                       origin='upper')

        self.canvas.draw()

        self.npaths = len(self.paths[0])

        plot_svg(self, self.paths)

        print('npaths', self.npaths)
        print('')
        self.connect_button.configure(state='normal')
        self.sidebar_button_1.configure(state=DISABLED)
        self.sidebar_button_2.configure(state=DISABLED)
        self.sidebar_button_3.configure(state=DISABLED)

    def select_svg(self):
        filetypes = (('svg file', '*.svg'), ('All files', '*.*'))

        self.filename = fd.askopenfilename(title='Open a file',
                                           initialdir='./',
                                           filetypes=filetypes)

        filename_split = self.filename.split('.')

        self.filename_svg = filename_split[0] + '.svg'

        self.paths, attributes = svg2paths(self.filename_svg)
        self.width, self.height = get_svg_size(self.filename_svg)

        self.npaths = len(self.paths[0])

        plot_svg(self, self.paths)

        print('npaths', self.npaths)
        print('')
        self.connect_button.configure(state='normal')
        self.sidebar_button_1.configure(state=DISABLED)
        self.sidebar_button_2.configure(state=DISABLED)
        self.sidebar_button_3.configure(state=DISABLED)

    def select_lines(self):

        filetypes = (('lines file', '*.dat'), ('All files', '*.*'))

        self.filename = fd.askopenfilename(title='Open a file',
                                           initialdir='./',
                                           filetypes=filetypes)

        # Load data
        with open(self.filename, "rb") as lines_file:
            data = pickle.load(lines_file)

        lx = data[0]
        self.llx_entry.delete(0, END)
        self.llx_entry.insert(0, str(lx))

        ly = data[1]
        self.lly_entry.delete(0, END)
        self.lly_entry.insert(0, str(ly))

        cellsize = data[2]
        self.cellsize_entry.delete(0, END)
        self.cellsize_entry.insert(0, str(cellsize))

        epsg = data[3]
        self.epsg_entry.delete(0, END)
        self.epsg_entry.insert(0, str(epsg))

        self.width = float(data[4])
        self.height = float(data[5])

        print(self.width, self.height)

        multiline = data[6]

        self.lines = []
        self.labels = []
        self.linestrings = []

        text_kwargs = dict(ha='center', va='center', fontsize=10, color='r')

        for line in multiline.geoms:

            xx, yy = line.coords.xy
            xx = np.array(xx)
            yy = np.array(yy)

            ln, = self.ax.plot(xx, yy, '-k', picker=True)

            self.lines.append(ln)
            self.linestrings.append(line)
            idx = int(np.floor(0.5 * len(xx)))
            label = self.ax.text(xx[idx], yy[idx], '', **text_kwargs)
            self.labels.append(label)

        self.linevalues = np.zeros(len(self.lines))
        self.lines_set = np.zeros(len(self.lines), dtype=bool)

        self.ax.set_xlim([0, self.width])
        self.ax.set_ylim([0, self.height])
        self.ax.axis('equal')

        self.canvas.draw()

        self.sidebar_button_1.configure(state=DISABLED)
        self.sidebar_button_2.configure(state=DISABLED)
        self.sidebar_button_3.configure(state=DISABLED)
        self.save_button.configure(state='normal')


    def connect(self):

        # FIRST LOOP TO SEARCH THE FOR PATHS TO CONNECT TO
        # THE START POINTS OF PATHS

        print('CHECK START')
        new_paths = modify_paths(self, self.paths, 'start')

        # SECOND LOOP TO SEARCH THE FOR PATHS TO CONNECT TO
        # THE END POINTS OF PATHS

        print('CHECK END')
        new_paths2 = modify_paths(self, new_paths, 'end')

        self.paths = copy.copy(new_paths2)

        self.change_order_button.configure(state='normal')
        self.connect_button.configure(state=DISABLED)

    def change_order(self):

        self.paths = change_paths_order(self, self.paths)
        self.plot_paths_button.configure(state='normal')
        self.change_order_button.configure(state=DISABLED)

    def plot_paths(self):

        draw_paths(self, self.paths)
        self.plot_paths_button.configure(state=DISABLED)
        self.save_button.configure(state='normal')

    def onpick(self, event):

        if self.linestring_idx >= 0:

            if self.lines[self.linestring_idx].get_color() == ('b'):

                self.lines[self.linestring_idx].set_color('k')

        self.linestring_idx = self.lines.index(event.artist)

        self.lines[self.linestring_idx].set_color('b')

        self.next_button.configure(state='normal')

        self.canvas.draw()

        # Define a function to close the window
    def close(self):

        self.quit()


if __name__ == "__main__":
    app = App()
    app.mainloop()
