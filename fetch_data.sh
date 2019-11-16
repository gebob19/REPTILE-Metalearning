#!/bin/bash
#
# Fetch Omniglot and Mini-ImageNet.
# Orig. Ref: https://github.com/openai/supervised-reptile/blob/master/fetch_data.sh
#

OMNIGLOT_URL=https://raw.githubusercontent.com/brendenlake/omniglot/master/python

set -e

mkdir tmp
trap 'rm -r tmp' EXIT

if [ ! -d data ]; then
    mkdir data
fi

if [ ! -d data/omniglot ]; then
    mkdir tmp/omniglot
    mkdir data/omniglot

    for name in images_background images_evaluation; do
        echo "Fetching omniglot/$name ..."
        curl -s "$OMNIGLOT_URL/$name.zip" > "tmp/$name.zip"
        echo "Extracting omniglot/$name ..."
        unzip -q "tmp/$name.zip" -d tmp
        rm "tmp/$name.zip"
        mv tmp/$name/* tmp/omniglot
    done
    
    mkdir data/omniglot/data
    mkdir data/omniglot/splits

    mv tmp/omniglot/* data/omniglot/data


    # download data split
    export fileid=1xSUHZur5Q7iZ_K9w9fEPmAG_wW_o5_vY
    export filename=vinyals.zip

    curl -L -o $filename 'https://docs.google.com/uc?export=download&id='$fileid
    unzip $filename
    rm $filename

    mv vinyals data/omniglot/splits/
fi

