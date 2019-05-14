git submodule update --init --recursive
cd third_party/opencv
mkdir -p build
cd build
cmake ..
make -j8
sudo make install

cd ../../../../
