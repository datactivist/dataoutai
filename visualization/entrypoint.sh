#!/bin/bash

if [ -z "${WEBSOCKER_ORIGIN}" ]; then
  bokeh serve visualization_bokeh_server.py
else
  bokeh serve visualization_bokeh_server.py --allow-websocket-origin="${WEBSOCKER_ORIGIN}"
fi
