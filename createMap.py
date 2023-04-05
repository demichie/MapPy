from svgpathtools import svg2paths, wsvg
from svgpathtools import real, imag, rational_limit
from shapely.geometry import Point, LineString

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import numpy as np
import copy

filename = 'crop.svg'
dz = 5
min_length = 5

dist_max = 1.0
angle_max = 20.0

def tangent_path(elem,side):

    p_start = elem.start
    p_end = elem.end
        
    if 'CubicBezier' in str(type(elem)):
    
        p = elem.poly()
        p = elem.poly()

        xp = real(p)
        yp = imag(p)

        if ( side == 'end'):

            x1 = np.polyval(xp,1.0)
            y1 = np.polyval(yp,1.0)
            
            x2 = np.polyval(xp,0.99)
            y2 = np.polyval(yp,0.99)

        else:

            x1 = np.polyval(xp,0.0)
            y1 = np.polyval(yp,0.0)
            
            x2 = np.polyval(xp,0.01)
            y2 = np.polyval(yp,0.01)

    elif 'Line' in str(type(elem)):

        if ( side == 'end'):

            x1 = real(p_end)
            y1 = imag(p_end)
            
            x2 = real(p_start)
            y2 = imag(p_start)

        else:

            x1 = real(p_start)
            y1 = imag(p_start)

            x2 = real(p_end)
            y2 = imag(p_end)
             
    tangent_x =  (x1 - x2) / np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 ) 
    tangent_y =  (y1 - y2) / np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 ) 

    return tangent_x,tangent_y


paths, attributes = svg2paths(filename)

xx = []
yy = []

linestrings = []

figure, (ax1,ax2) = plt.subplots(1,2)

### NEW FROM HERE

npaths = len(paths[0])

print('npaths',npaths)

for count in range(npaths):

    path1 = paths[0][count]

    # print('path1',count)
    # print(path1)

    p_start1 = path1.start
    p_end1 = path1.end

    if ( p_start1 == p_end1 ):
        
        continue

    xp_start1 = real(p_start1)
    yp_start1 = imag(p_start1)

    xp_end1 = real(p_end1)
    yp_end1 = imag(p_end1)

    distMin = 1.e10
    
    idx = 0

    for path2 in paths[0][count+1:]:
        
        idx +=1
        p_start2 = path2.start
        p_end2 = path2.end
        
        if ( p_start2 == p_end2 ):
        
            continue

        xp_start2 = real(p_start2)
        yp_start2 = imag(p_start2)

        xp_end2 = real(p_end2)
        yp_end2 = imag(p_end2)

        # distance between start-points
        dist1 = np.sqrt( (xp_start1-xp_start2)**2 + (yp_start1-yp_start2)**2 ) 
        # distance between start-point1 and end-point2
        dist2 = np.sqrt( (xp_start1-xp_end2)**2 + (yp_start1-yp_end2)**2 ) 
        # distance between end-point1 and start-point2
        dist3 = np.sqrt( (xp_end1-xp_start2)**2 + (yp_end1-yp_start2)**2 ) 
        # distance between end-points
        dist4 = np.sqrt( (xp_end1-xp_end2)**2 + (yp_end1-yp_end2)**2 ) 

        if np.amin([dist1,dist2,dist3,dist4]) < distMin:
        
            distMin = np.amin([dist1,dist2,dist3,dist4])
            sideMin = np.argmin([dist1,dist2,dist3,dist4])
            idxMin = idx
            pathMin = path2
              
    if distMin <= dist_max:
    
        # print('distMin',distMin)
        # print('idxMin',idxMin)
        # print('pathMin',pathMin)
        # print('path2',count+idxMin)
        
        # print(path1)

        path2 = paths[0][count+idxMin]

        # print(path2)
    
        p_start2 = path2.start
        p_end2 = path2.end

        xp_start2 = real(p_start2)
        yp_start2 = imag(p_start2)

        xp_end2 = real(p_end2)
        yp_end2 = imag(p_end2)

        if sideMin == 0:
        
            x1 = xp_start1
            y1 = yp_start1
            x2 = xp_start2
            y2 = yp_start2

            side1 = 'start'

            print('Reverse path start-start',count+idxMin)
            
            # reverse path
            path_temp = copy.copy(paths[0][count+idxMin])
            paths[0][count+idxMin].start = path_temp.end
            paths[0][count+idxMin].end = path_temp.start
            if 'CubicBezier' in str(type(path_temp)):
                paths[0][count+idxMin].control1 = path_temp.control2
                paths[0][count+idxMin].control2 = path_temp.control1
            
            path2 = paths[0][count+idxMin]             
            # set to end because of reversal
            side2 = 'end'

        elif sideMin == 1:
        
            x1 = xp_start1
            y1 = yp_start1
            side1 = 'start'

            x2 = xp_end2
            y2 = yp_end2
            side2 = 'end'

        elif sideMin == 2:
        
            x1 = xp_end1
            y1 = yp_end1
            side1 = 'end'

            x2 = xp_start2
            y2 = yp_start2
            side2 = 'start'

        elif sideMin == 3:
        
            x1 = xp_end1
            y1 = yp_end1
            x2 = xp_end2
            y2 = yp_end2

            side1 = 'end'
            side2 = 'end'

            print('Reverse path end-end',count+idxMin)
            
            # reverse path
            path_temp = copy.copy(paths[0][count+idxMin])
            paths[0][count+idxMin].start = path_temp.end
            paths[0][count+idxMin].end = path_temp.start
            
            if 'CubicBezier' in str(type(path_temp)):
                paths[0][count+idxMin].control1 = path_temp.control2
                paths[0][count+idxMin].control2 = path_temp.control1
            side2 = 'start'
            
            
            path2 = paths[0][count+idxMin]             


        if distMin > 0:
        
           vect1_x,vect1_y = tangent_path(path1,side1)       
           vect2_x,vect2_y = tangent_path(path2,side2)       

           cos12 = vect1_x * vect2_x + vect1_y * vect2_y 
           angle_12 = np.degrees(np.arccos(-cos12))

           vect12_x = (x2 - x1) / np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
           vect12_y = (y2 - y1) / np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
      
           cosA = vect12_x * vect2_x + vect12_y * vect2_y 
           angle_A = np.degrees(np.arccos(cosA))
 
           cosB = vect1_x * vect12_x + vect1_y * vect12_y 
           angle_B = np.degrees(np.arccos(cosB))

           angle_AB = np.amin([angle_A,180-angle_A,angle_B,180-angle_B])
           
           if ( np.maximum(angle_12,angle_AB) < angle_max ):
           
               print('CONSECUTIVE PATHS:',count,count+idxMin)
               # print('dist',distMin)
               # print('angle',angle_12)
               # print('angle_AB',angle_AB)

               if side1 == 'start' and side2 == 'start':

                   paths[0][count+idxMin].start = paths[0][count].start

               elif side1 == 'start' and side2 == 'end':

                   paths[0][count+idxMin].end = paths[0][count].start

               elif side1 == 'end' and side2 == 'start':

                   paths[0][count+idxMin].start = paths[0][count].end

               elif side1 == 'end' and side2 == 'end':

                   paths[0][count+idxMin].end = paths[0][count].end

               ax1.plot(x2,y2,'or')
               ax1.plot(x1,y1,'^g')

        paths[0].insert(count+1, paths[0].pop(count+idxMin))


### NEW TO HERE

p_endOLD = paths[0][0].start
p_startOLD = paths[0][0].end

for path in paths:

    for elem in path:
    
        # print(elem)
   
        if 'CubicBezier' in str(type(elem)):
        
            p = elem.poly()
            p_start = elem.start
            p_end = elem.end
            p = elem.poly()

            xp = real(p)
            yp = imag(p)

            s = np.linspace(0,1,10)

            x = np.polyval(xp,s)
            y = np.polyval(yp,s)
            
            ax1.plot(x,y,'-g')   
            
            s0 = 0.5
            s1 = 0.51
            x0 = np.polyval(xp,s0)
            y0 = np.polyval(yp,s0)
            x1 = np.polyval(xp,s1)
            y1 = np.polyval(yp,s1)
            ax1.arrow(x0, y0, x1-x0, y1-y0, shape='full', color='k', lw=0, length_includes_head=True, head_width=.15)
            
        elif 'Line' in str(type(elem)):
        
            p_start = elem.start
            x_start = real(p_start)
            y_start = imag(p_start)

            p_end = elem.end
            x_end = real(p_end)
            y_end = imag(p_end)
            
            x = [x_start,x_end]
            y = [y_start,y_end]

            ax1.plot(x,y,'-g')   

            s0 = 0.5
            s1 = 0.51
            x0 = x_start + s0 * ( x_end - x_start )
            y0 = y_start + s0 * ( y_end - y_start )
            x1 = x_start + s1 * ( x_end - x_start )
            y1 = y_start + s1 * ( y_end - y_start )
            ax1.arrow(x0, y0, x1-x0, y1-y0, shape='full', color='k', lw=0, length_includes_head=True, head_width=.15)

            
        else:
        
            print('OTHER')    

        if p_start == p_endOLD:
        
            # print('CONNECTED PATH',p_start,p_endOLD)
        
            xx = np.append(xx,x)
            yy = np.append(yy,y)

            p_endOLD = p_end
            
        elif p_end == p_startOLD:
        
            # print('CONNECTED PATH',p_start,p_endOLD)
        
            xx = np.append(x,xx)
            yy = np.append(y,yy)

            pstartOLD = p_start 
            
        else:
        
            # print('NEW PATH',p_start,p_endOLD)
            
            xy = zip(xx,yy)
            line = LineString(xy)
            length = line.length
            if ( length > min_length ):
                ax1.plot(xx,yy,'-k')
                
                linestrings.append(line)
            
            xx = []
            yy = []

            p_endOLD = p_end
    
ax1.axis('equal')   
ax1.invert_yaxis()

plt.show()        
plt.show(block=False)

nl = len(linestrings)

print('Number of linestrings',nl)

elevation = []
line1, = ax1.plot([],[],'r--',linewidth=2)

xx = []
yy = []
zz = []

for idx in range(nl):

    line = linestrings[idx]

    dist = [ line.distance(linestrings[j]) for j in range(idx+1,nl)]
    
    linestrings[idx+1:nl] = [x for _, x in sorted(zip(dist, linestrings[idx+1:nl]))]
    
text_kwargs = dict(ha='center', va='center', fontsize=8, color='C1')

for count,line in enumerate(linestrings):
    
    print('Line ',count)
    x,y = line.coords.xy 
    line1.set_xdata(x)
    line1.set_ydata(y)
    str_inp = input('elevation?')
    if str_inp == '+':
    
        el += dz
        
    elif str_inp == '++':
    
        el += 2*dz

    elif str_inp == '-':
    
        el -= dz
        
    elif str_inp == '--':
    
        el -= 2*dz
        
    elif str_inp == 'd':
    
        x = []
        y = []    
        
    else:    

        el = float(str_inp)
    
    # el = 50*np.random.rand()
    
    xx = np.append(xx,x)
    yy = np.append(yy,y)
    zz = np.append(zz,el*np.ones_like(x))        

    if len(x) > 0:
        print('Elevation: ',el)
        idx = int(np.floor(0.5*len(x)))    
        ax1.text(x[idx],y[idx],str(el), **text_kwargs)
        
    plt.draw()

xmin = np.amin(xx)
xmax = np.amax(xx)

dx = ( xmax - xmin ) / 200

ymin = np.amin(yy)
ymax = np.amax(yy)

xlin = np.arange(xmin,xmax+dx,dx)
ylin = np.arange(ymin,ymax+dx,dx)
 
Xlin,Ylin = np.meshgrid(xlin,ylin, indexing='ij')

from scipy.interpolate import griddata

points = np.zeros((len(xx),2))
points[:,0] = xx
points[:,1] = yy

grid_z1 = griddata(points, zz, (Xlin, Ylin), method='cubic')

ls = LightSource(azdeg=135, altdeg=45)

extent=(xmin,xmax,ymin,ymax)

# ax2.imshow(grid_z1.T, extent=extent, origin='lower')    
ax2.imshow(ls.hillshade(grid_z1.T, vert_exag=1.0, dx=dx, dy=dx),
           cmap='gray',
           extent=extent,
           origin='lower')

ax2.invert_yaxis()


plt.draw()            
            
plt.show()        
    
    
