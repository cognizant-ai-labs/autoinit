#!/bin/bash

# Copyright (C) 2021 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# AutoInit Software in commercial settings.
#
# END COPYRIGHT

RCFILE=build_scripts/pylintrc
UP_TO_SNUFF_DIRS="autoinit"

dirs=$1
if [ "x${dirs}" == "x" ]
then
    dirs=${UP_TO_SNUFF_DIRS}
fi

retval=0
for dir in ${dirs}
do
    echo "Running pylint on directory '${dir}':"
    find "${dir}" -iname "*.py" | \
        xargs pylint --load-plugins=pylint_protobuf --rcfile=${RCFILE}
    current_retval=$?
    if [ ${current_retval} -ne 0 ]
    then
        retval=${current_retval} 
    fi
done

exit ${retval}
