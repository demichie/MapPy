#!/bin/bash
IP=$(/usr/sbin/ipconfig getifaddr en0)
/opt/X11/bin/xhost + "$IP"
docker run -it -e DISPLAY="${IP}:0" -v /tmp/.X11-unix:/tmp/.X11-unix -v $PWD:/home/user_sw/TMP demichie/mappy


