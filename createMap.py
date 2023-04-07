from scipy.interpolate import griddata
from svgpathtools import svg2paths
from svgpathtools import real, imag
from shapely.geometry import LineString
from shapely import hausdorff_distance
from shapely import frechet_distance

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import numpy as np
import copy

filename = 'crop.svg'
dz = 5
min_length = 1

dist_max = 2.0
angle_max = 25.0

distance = 'cartesian'
# distance = 'frechet'
# distance = 'hausdorff'
# distance = 'centroid'

debug = False

xtest = 18.23
ytest = 92.35
disttest = -0.5
test = False


def tangent_path(elem, side):

    p_start = elem.start
    p_end = elem.end

    if 'CubicBezier' in str(type(elem)):

        p = elem.poly()
        p = elem.poly()

        xp = real(p)
        yp = imag(p)

        if (side == 'end'):

            x1 = np.polyval(xp, 1.0)
            y1 = np.polyval(yp, 1.0)

            x2 = np.polyval(xp, 0.99)
            y2 = np.polyval(yp, 0.99)

        else:

            x1 = np.polyval(xp, 0.0)
            y1 = np.polyval(yp, 0.0)

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

    tangent_x = (x1 - x2) / np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    tangent_y = (y1 - y2) / np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    return tangent_x, tangent_y


paths, attributes = svg2paths(filename)

figure, ax1 = plt.subplots()

npaths = len(paths[0])
print('npaths', npaths)

# FIRST LOOP TO SEARCH THE FOR PATHS TO CONNECT TO
# THE START POINTS OF PATHS

for count, path1 in enumerate(paths[0]):

    # print('')
    if debug:

        print('path1', count)
    # print(path1)

    p_start1 = path1.start
    p_end1 = path1.end

    if (p_start1 == p_end1):

        if debug:
            print('point')
        continue

    xp_start1 = real(p_start1)
    yp_start1 = imag(p_start1)

    xp_end1 = real(p_end1)
    yp_end1 = imag(p_end1)

    if (np.sqrt((xp_start1-xtest)**2 + (yp_start1-ytest)**2) < disttest) or \
       (np.sqrt((xp_end1-xtest)**2 + (yp_end1-ytest)**2) < disttest):

        test1 = True

    else:

        test1 = False

    distMin = 1.e10

    for count2, path2 in enumerate(paths[0]):

        if count <= count2:

            continue

        p_start2 = path2.start
        p_end2 = path2.end

        if (p_start2 == p_end2):

            continue

        if (p_start1 == p_start2) or (p_start1 == p_end2):

            if debug:
                print('Connected paths:', count, count2)
            distMin = 0.0
            break

        # if (p_start1 == p_start2) or (p_start1 == p_end2 ) or (p_end1 == p_start2) or (p_end1 == p_end2 ):

        #    distMin = 0.0
        #    continue

        xp_start2 = real(p_start2)
        yp_start2 = imag(p_start2)

        xp_end2 = real(p_end2)
        yp_end2 = imag(p_end2)

        # distance between start-points
        dist1 = np.sqrt((xp_start1 - xp_start2)**2 +
                        (yp_start1 - yp_start2)**2)
        # distance between start-point1 and end-point2
        dist2 = np.sqrt((xp_start1 - xp_end2)**2 + (yp_start1 - yp_end2)**2)

        if np.amin([dist1, dist2]) < distMin:

            distMin = np.amin([dist1, dist2])
            sideMin = np.argmin([dist1, dist2])
            idxMin = count2
            pathMin = path2

        if ((np.sqrt((xp_start2 - xtest)**2 +
                     (yp_start2 - ytest)**2) < disttest) or
            (np.sqrt((xp_end2 - xtest)**2 +
                     (yp_end2 - ytest)**2) < disttest)) and test1:

            print('')
            print('test1', count)
            test1Idx = count
            print(path1)
            print('test2', count2)
            test2Idx = count2
            print(path2)
            print('dist', np.amin([dist1, dist2]))
            print('distMin', distMin)
            test = True
            a = input('PAUSE')

    if distMin > 0 and distMin <= dist_max:

        path2 = paths[0][idxMin]

        p_start2 = path2.start
        p_end2 = path2.end

        xp_start2 = real(p_start2)
        yp_start2 = imag(p_start2)

        xp_end2 = real(p_end2)
        yp_end2 = imag(p_end2)

        x1 = xp_start1
        y1 = yp_start1
        side1 = 'start'

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

        angle_AB = np.amin([angle_A, 180 - angle_A, angle_B, 180 - angle_B])

        if test and test1Idx == count and test2Idx == idxMin:

            print('angle_12', angle_12)
            print('angle_AB', angle_AB)
            a = input('PAUSE')

        if (np.maximum(angle_12, angle_AB) < angle_max):

            print('New connected paths:', count, idxMin)
            # print('dist',distMin)
            # print('angle',angle_12)
            # print('angle_AB',angle_AB)

            if side2 == 'start':

                paths[0][count].start = paths[0][idxMin].start

            elif side2 == 'end':

                paths[0][count].start = paths[0][idxMin].end

                # ax1.plot(x2,y2,'or')
                # ax1.plot(x1,y1,'^g')

# SECOND LOOP TO CHECK THE END POINTS OF PATHS

for count, path1 in enumerate(paths[0]):

    # print('')
    if debug:

        print('path1', count)
    # print(path1)

    p_start1 = path1.start
    p_end1 = path1.end

    if (p_start1 == p_end1):

        if debug:
            print('point')
        continue

    xp_start1 = real(p_start1)
    yp_start1 = imag(p_start1)

    xp_end1 = real(p_end1)
    yp_end1 = imag(p_end1)

    if (np.sqrt((xp_start1-xtest)**2 + (yp_start1-ytest)**2) < disttest) or \
       (np.sqrt((xp_end1-xtest)**2 + (yp_end1-ytest)**2) < disttest):

        test1 = True

    else:

        test1 = False

    distMin = 1.e10

    for count2, path2 in enumerate(paths[0]):

        if count <= count2:

            continue

        p_start2 = path2.start
        p_end2 = path2.end

        if (p_start2 == p_end2):

            continue

        if (p_end1 == p_start2) or (p_end1 == p_end2):

            if debug:
                print('Connected paths:', count, count2)
            distMin = 0.0
            break

        # if (p_end1 == p_start2) or (p_end1 == p_end2 ) or (p_start1 == p_start2) or (p_start1 == p_end2 ):

        #    distMin = 0.0
        #
        #     continue

        xp_start2 = real(p_start2)
        yp_start2 = imag(p_start2)

        xp_end2 = real(p_end2)
        yp_end2 = imag(p_end2)

        # distance between end-point1 and start-point2
        dist3 = np.sqrt((xp_end1 - xp_start2)**2 + (yp_end1 - yp_start2)**2)
        # distance between end-points
        dist4 = np.sqrt((xp_end1 - xp_end2)**2 + (yp_end1 - yp_end2)**2)

        if np.amin([dist3, dist4]) < distMin:

            distMin = np.amin([dist3, dist4])
            sideMin = np.argmin([dist3, dist4])
            idxMin = count2
            pathMin = path2

        if ((np.sqrt((xp_start2 - xtest)**2 +
                     (yp_start2 - ytest)**2) < disttest) or
            (np.sqrt((xp_end2 - xtest)**2 +
                     (yp_end2 - ytest)**2) < disttest)) and test1:

            print('')
            print('test1', count)
            test1Idx = count
            print(path1)
            print('test2', count2)
            test2Idx = count2
            print(path2)
            print('dist', np.amin([dist3, dist4]))
            test = True
            a = input('PAUSE')

    if distMin > 0 and distMin <= dist_max:

        # print('distMin',distMin)
        # print('idxMin',idxMin)
        # print('pathMin',pathMin)
        # print('path2',count+idxMin)

        path2 = paths[0][idxMin]

        # print(path1)
        # print(path2)

        p_start2 = path2.start
        p_end2 = path2.end

        xp_start2 = real(p_start2)
        yp_start2 = imag(p_start2)

        xp_end2 = real(p_end2)
        yp_end2 = imag(p_end2)

        x1 = xp_end1
        y1 = yp_end1
        side1 = 'end'

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

        angle_AB = np.amin([angle_A, 180 - angle_A, angle_B, 180 - angle_B])

        if test and test1Idx == count and test2Idx == idxMin:

            print('angle_12', angle_12)
            print('angle_AB', angle_AB)
            a = input('PAUSE')

        if (np.maximum(angle_12, angle_AB) < angle_max):

            print('New connected paths:', count, idxMin)
            # print('dist',distMin)
            # print('angle',angle_12)
            # print('angle_AB',angle_AB)

            if side2 == 'start':

                paths[0][count].end = paths[0][idxMin].start

            elif side2 == 'end':

                paths[0][count].end = paths[0][idxMin].end

                # ax1.plot(x2,y2,'or')
                # ax1.plot(x1,y1,'^g')

# CONTARE I VICINI
# RIORDINARE METTENDO IN CIMA QUELLI CHE HANNO 1 SOLO VICINO
# CREARE LE CATENE DI VICINI

first_connected = np.zeros(npaths, dtype=int)
second_connected = np.zeros(npaths, dtype=int)

connected_list = [[] for _ in range(npaths)]

for count1, path1 in enumerate(paths[0]):

    for count2, path2 in enumerate(paths[0][count1 + 1:]):

        if (path1.start == path2.end) or (path1.start == path2.start) or (
                path1.end == path2.end) or (path1.end == path2.start):

            connected_list[count1].append(count1 + count2 + 1)
            connected_list[count1 + count2 + 1].append(count1)

if test:

    print('TEST')
    print('test1Idx', test1Idx)
    print(connected_list[test1Idx])
    print(paths[0][test1Idx])
    print('test2Idx', test2Idx)
    print(connected_list[test2Idx])
    print(paths[0][test2Idx])
    print('')
    a = input('PAUSE')

new_path = []
idx_new_path = -1

for count1, path1 in enumerate(paths[0]):

    if len(connected_list[count1]) == 1:

        print('')
        print('New path', count1, end='')

        path_length = 1
        new_path.append(path1)
        idx_new_path += 1

        actual = count1
        next = connected_list[actual][0]

        if test:

            if actual == test1Idx:

                print('TEST RENUMBERING')
                print(actual, idx_new_path)
                test1Idx = idx_new_path
                a = input('PAUSE')

            if actual == test2Idx:

                print('TEST RENUMBERING')
                print(actual, idx_new_path)
                test2Idx = idx_new_path
                a = input('PAUSE')

        connected_list[actual].remove(next)

        while len(connected_list[next]) >= 1:

            if test:
                print('')
                print(',', next)

            else:

                print(',', next, end='')

            if test:

                print('number_connected', len(connected_list[next]))
                print('connected_list', connected_list[next])

            connected_list[next].remove(actual)

            previous = actual
            actual = next

            if test:

                print('connected_list', connected_list[actual])
                a = input('PAUSE')

            path_previous = paths[0][previous]
            path = paths[0][actual]

            if len(connected_list[actual]) > 0:

                next = connected_list[actual][0]
                connected_list[actual].remove(next)

            if not (path_previous.start == path.end
                    or path_previous.end == path.start):

                # reverse path
                print('Rev')
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

            if test:

                if actual == test1Idx:

                    print('TEST RENUMBERING')
                    print(actual, idx_new_path)
                    test1Idx = idx_new_path
                    a = input('PAUSE')

                if actual == test2Idx:

                    print('TEST RENUMBERING')
                    print(actual, idx_new_path)
                    test2Idx = idx_new_path
                    a = input('PAUSE')

        print('')
        print('Length', path_length)

print('len paths[0]', len(paths[0]))
paths[0] = new_path
print('len paths[0]', len(paths[0]))

# print(ciao)

# NEW TO HERE

xx = []
yy = []

linestrings = []

p_endOLD = paths[0][0].start
p_startOLD = paths[0][0].end

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

            ax1.plot(x, y, '-g')

            s0 = 0.5
            s1 = 0.51
            x0 = np.polyval(xp, s0)
            y0 = np.polyval(yp, s0)
            x1 = np.polyval(xp, s1)
            y1 = np.polyval(yp, s1)
            ax1.arrow(x0,
                      y0,
                      x1 - x0,
                      y1 - y0,
                      shape='full',
                      color='k',
                      lw=0,
                      length_includes_head=True,
                      head_width=.25)

        elif 'Line' in str(type(elem)):

            p_start = elem.start
            x_start = real(p_start)
            y_start = imag(p_start)

            p_end = elem.end
            x_end = real(p_end)
            y_end = imag(p_end)

            x = [x_start, x_end]
            y = [y_start, y_end]

            ax1.plot(x, y, '-g')

            s0 = 0.5
            s1 = 0.51
            x0 = x_start + s0 * (x_end - x_start)
            y0 = y_start + s0 * (y_end - y_start)
            x1 = x_start + s1 * (x_end - x_start)
            y1 = y_start + s1 * (y_end - y_start)
            ax1.arrow(x0,
                      y0,
                      x1 - x0,
                      y1 - y0,
                      shape='full',
                      color='k',
                      lw=0,
                      length_includes_head=True,
                      head_width=.25)

        else:

            print('OTHER')

        if test and (count == test1Idx or count == test2Idx):

            print('count', count)
            print('')
            print('p_startOLD', p_startOLD)
            print('p_endOLD', p_endOLD)
            print('')
            print('p_start', p_start)
            print('p_end', p_end)
            print('CONNECTED:', p_start == p_endOLD or p_end == p_startOLD)
            a = input('PAUSE')

        if p_start == p_endOLD:

            if debug:

                print('CONNECTED PATH AT', p_start)

            xx = np.append(xx, x)
            yy = np.append(yy, y)

            if test and (count == test1Idx or count == test2Idx):

                print('xx', xx)
                print('yy', yy)
                a = input('PAUSE')

            p_endOLD = p_end

        elif p_end == p_startOLD:

            if debug:

                print('CONNECTED PATH AT', p_end)

            xx = np.append(x, xx)
            yy = np.append(y, yy)

            if test and (count == test1Idx or count == test2Idx):

                print('xx', xx)
                print('yy', yy)
                a = input('PAUSE')

            p_startOLD = p_start

        else:

            print('NEW PATH', count)

            xy = zip(xx, yy)
            line = LineString(xy)
            length = line.length

            [minx, miny, maxx, maxy] = line.bounds

            area = (maxx - minx) * (maxy - miny)

            if (length > min_length):
                ax1.plot(xx, yy, '-k')
                # ax1.plot(xx[0],yy[0],'ok')
                # ax1.plot(xx[-1],yy[-1],'ok')

                linestrings.append(line)

            xx = x
            yy = y

            p_startOLD = p_start
            p_endOLD = p_end

ax1.axis('equal')
ax1.invert_yaxis()

if debug:

    plt.show()

plt.show(block=False)

nl = len(linestrings)

print('Number of linestrings', nl)

elevation = []
line1, = ax1.plot([], [], 'r--', linewidth=2)

xx = []
yy = []
zz = []

for idx in range(nl - 1):

    line = linestrings[idx]

    if distance == 'cartesian':

        dist = [line.distance(linestrings[j]) for j in range(idx + 1, nl)]

    elif distance == 'hausdorff':

        dist = [
            hausdorff_distance(line, linestrings[j])
            for j in range(idx + 1, nl)
        ]

    elif distance == 'frechet':

        dist = [
            frechet_distance(line, linestrings[j]) for j in range(idx + 1, nl)
        ]

    # print(dist)

    idxMin = np.argmin(dist)

    linestrings.insert(idx + 1, linestrings.pop(idx + idxMin + 1))

text_kwargs = dict(ha='center', va='center', fontsize=8, color='r')

el = 0.0

for count, line in enumerate(linestrings):

    print('Line ', count)
    x, y = line.coords.xy
    line1.set_xdata(x)
    line1.set_ydata(y)
    str_inp = input('elevation?')

    if str_inp == '+':

        el += dz

    elif str_inp == '++':

        el += 2 * dz

    elif str_inp == '-':

        el -= dz

    elif str_inp == '--':

        el -= 2 * dz

    elif str_inp == '=':

        el += 0.0

    elif str_inp == 'd':

        x = []
        y = []

    elif str_inp == 'end':

        break

    else:

        el = float(str_inp)

    # el = 50*np.random.rand()

    xx = np.append(xx, x)
    yy = np.append(yy, y)
    zz = np.append(zz, el * np.ones_like(x))

    if len(x) > 0:
        print('Elevation: ', el)
        idx = int(np.floor(0.5 * len(x)))
        ax1.text(x[idx], y[idx], str(el), **text_kwargs)

    plt.draw()

xmin = np.amin(xx)
xmax = np.amax(xx)

dx = (xmax - xmin) / 200

ymin = np.amin(yy)
ymax = np.amax(yy)

xlin = np.arange(xmin, xmax + dx, dx)
ylin = np.arange(ymin, ymax + dx, dx)

Xlin, Ylin = np.meshgrid(xlin, ylin, indexing='ij')

points = np.zeros((len(xx), 2))
points[:, 0] = xx
points[:, 1] = yy

grid_z1 = griddata(points, zz, (Xlin, Ylin), method='cubic')

ls = LightSource(azdeg=135, altdeg=45)

extent = (xmin, xmax, ymin, ymax)

figure, ax2 = plt.subplots()

# ax2.imshow(grid_z1.T, extent=extent, origin='lower')
ax2.imshow(ls.hillshade(grid_z1.T, vert_exag=1.0, dx=dx, dy=dx),
           cmap='gray',
           extent=extent,
           origin='lower')

ax2.invert_yaxis()

plt.draw()

plt.show()
