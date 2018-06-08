#---------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Script to download the imagenet pre-trained models
#---------------------------------------------------
#!/bin/bash
url="legacydirs.umiacs.umd.edu/~najibi/download.php"
method_name="sniper_imgnet"
file_name="pretrained_model.tar.gz"
cur_dir=${PWD##*/}
target_dir="./data/"

if [ $cur_dir = "scripts" ]; then
   target_dir="../data/"
fi

if [ ! -f ${file_name} ]; then
   echo "Downloading ${file_name}..."
   wget "${url}?name=${method_name}" -O "${target_dir}${file_name}"
   echo "Done!"
else
   echo "File already exists!"
fi

echo "Unzipping the file..."
tar -xvzf "${target_dir}${file_name}" -C ${target_dir}

echo "Cleaning up..."
rm "${target_dir}${file_name}"
echo "All done!"