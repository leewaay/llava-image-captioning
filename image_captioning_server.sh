#!/bin/bash

CURRENT_DIR=$(pwd)
SCRIPT_PATH="$CURRENT_DIR/main:app"
LOG_PATH="/dev/null"
PID_PATH="$CURRENT_DIR/server.pid"
UVICORN="uvicorn"
HOST="0.0.0.0"
PORT=${2:-5010}

start() {
    if [ -f "$PID_PATH" ]; then
        echo "Server is already running."
    else
        echo "Starting server..."
        nohup $UVICORN_PATH $SCRIPT_PATH --host $HOST --port $PORT --reload > $LOG_PATH 2>&1 &
        echo $! > $PID_PATH
        echo "Server started."
    fi
}

stop() {
    if [ -f "$PID_PATH" ]; then
        PID=$(cat $PID_PATH)
        echo "Stopping server..."
        kill $PID
        rm $PID_PATH
        echo "Server stopped."
    else
        echo "Server is not running."
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        ;;
esac