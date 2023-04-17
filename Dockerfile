# Download base image ubuntu 22.04
FROM ubuntu:22.04

# Update Ubuntu Software repository
RUN apt update

RUN apt install -y libgraphicsmagick1-dev libpng-dev libexiv2-dev libtiff-dev libjpeg-dev libxml2-dev libbz2-dev libfreetype6-dev libpstoedit-dev autoconf automake libtool intltool autopoint wget make unzip libstdc++-11-dev python3-pip gdal-bin libgdal-dev

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

RUN apt install -y python3.10-tk

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install numpy pandas shapely gdal fiona pyproj six rtree geopandas matplotlib darkdetect svgpathtools

# add user and create group
RUN adduser --disabled-password user_sw

# install the code
USER user_sw
WORKDIR /home/user_sw

RUN wget https://github.com/autotrace/autotrace/archive/refs/heads/master.zip \
    && mv master.zip autotrace.zip \
    && wget https://github.com/TomSchimansky/CustomTkinter/archive/refs/heads/master.zip \
    && wget https://github.com/demichie/MapPy/archive/refs/heads/main.zip \
    && unzip autotrace.zip \
    && cd autotrace-master/ \
    && ./autogen.sh \
    && ./configure --enable-magick-readers \
    && make \
    && make check \
    && cd .. \
    && unzip master.zip \
    && unzip main.zip \
    && cp -r CustomTkinter-master/customtkinter MapPy-main \
    && rm -rf CustomTkinter-master \
    && mkdir TMP

WORKDIR /home/user_sw/TMP

CMD ["/home/user_sw/MapPy-main/MapPy.py"]
ENTRYPOINT ["python3"]


        



