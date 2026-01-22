#!/usr/bin/env bash 


### Base Parameters

pathadd() {
    if [ -d "$1" ] && [[ ":$PYTHONPATH:" != *":$1:"* ]]; then
        PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$1"
    fi
}

proj_dir="/path/to/project/path"
pathadd $proj_dir
export PYTHONPATH


## DB Parameters 

PYMSSQL_ELIXR_SERVER="elixr.dbmi.columbia.edu"
export PYMSSQL_ELIXR_SERVER

PYMSSQL_USERNAME="dbmi\<your uni>"
export PYMSSQL_USERNAME

PYMSSQL_PASSWORD="<password>"
export PYMSSQL_PASSWORD 

### Project Parameters 

TP_PROJ_DIR=$proj_dir
export TP_PROJ_DIR

TP_CONFIG_DIR="$TP_PROJ_DIR/config"
export TP_CONFIG_DIR

TP_BIN_DIR="$TP_PROJ_DIR/bin"
export TP_BIN_DIR

export PATH="$TP_BIN_DIR:$PATH"

### Set up work enviornment
dirs="data data-exp test"

# do not double-quote $dirs (o.w. it's treated as a single value)
for dir in $dirs
do
    target_dir="$proj_dir/$dir"
    [[ -d "$target_dir" ]] || mkdir "$target_dir"
done
