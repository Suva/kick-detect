#!/bin/sh
# Serve the project on localhost (browsers only allow audio input on http://localhost or https) and open the live view.
cd "$(dirname "$0")/.." || exit 1
PORT="${1:-8377}"
( sleep 1; open "http://localhost:$PORT/live/" ) &
exec python3 -m http.server "$PORT" --bind 127.0.0.1
