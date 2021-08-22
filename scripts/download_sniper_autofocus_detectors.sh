#-----------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Script to download the trained detectors with SNIPER
#-----------------------------------------------------
#!/bin/bash
file_name="sniper_models.tar.gz"
cur_dir=${PWD##*/}
target_dir="./"

if [ $cur_dir = "scripts" ]; then
   target_dir="../"
fi

if [ ! -f "${target_dir}${file_name}" ]; then
   echo "Downloading ${file_name}..."
   curl -LA "github `date`" https://bit.ly/3j6mAXJ --output "${target_dir}${file_name}"
   echo "Done!"
else
   echo "File already exists!"
fi

echo "Unzipping the file..."
tar -xvzf "${target_dir}${file_name}" -C ${target_dir}

echo "Cleaning up..."
rm "${target_dir}${file_name}"
echo "All done!"
