# C/C++ Implementation of FCNN

## Step 1 - Local Weight Files and Test Images
- Download a zip folder containing a set of trained weights from [here](https://drive.google.com/file/d/1GiogYM4WuHz6Cc0HfD2L0zdYIHJHj__D/view?usp=drive_link).
- Extract the trianed weights into `C_Implementation/trained_weights`
- Run `generate_weights.py`
- Download zip folder containg sets of real and imaginary feature maps of images from [here](https://drive.google.com/file/d/1MtgtFIYsTgAmGbDUBn1uI6lN8lu9O1Ij/view?usp=drive_link)
- Extract the test images into `C_Implementation/test_images`

## Step 2 - Compile and run `fcnn.cpp`
> g++ *.cpp -o fcnn

> ./fcnn


# Profiling on Linux

### 1. Install gprof:
> sudo apt-get install binutils

### 2. Compile with profiling flags:
> /usr/bin/g++-12 -pg *.cpp -o fcnn

### 3. Run:
> ./fcnn

### 4. Generate profiling data:
> gprof fcnn gmon.out > analysis.txt


## Profiling Power
> sudo powerstat -R 0.5 120 > pwr.txt

## Using Different Conpiler Optimisations
> /usr/bin/g++-12 -pg *.cpp -o fcnn

> /usr/bin/g++-12 -pg -O1 *.cpp -o fcnn

> /usr/bin/g++-12 -pg -O2 *.cpp -o fcnn

> /usr/bin/g++-12 -pg -O3 *.cpp -o fcnn

## Combined Profiling Commands (EXAMPLE)
Sleep for 10 seconds, then run fcnn. Start powerstat at the start of sleep and export to text file. After powerstat has run, export the profiling measuremet to a text file.

> TEST_NAME="test1" && ((sleep 10 && ./fcnn) & sudo powerstat -R 0.5 120 > runs/O0/pwr_${TEST_NAME}.txt; wait) && gprof fcnn gmon.out > runs/O0/gprof_${TEST_NAME}.txt
### Ten consecutive runs:
No Optimisation:

> for RUN_NAME in run{1..10}; do ((sleep 10 && ./fcnn) & sudo powerstat -R 0.5 120 > runs/O0/pwr_${RUN_NAME}.txt; wait) && gprof fcnn gmon.out > runs/O0/gprof_${RUN_NAME}.txt; done

O2 Optimisation:

> for RUN_NAME in run{1..10}; do ((sleep 10 && ./fcnn) & sudo powerstat -R 0.5 120 > runs/O2/pwr_${RUN_NAME}.txt; wait) && gprof fcnn gmon.out > runs/O2/gprof_${RUN_NAME}.txt; done


Python (done in upper directory):

> for RUN_NAME in run{1..10}; do ((sleep 10 && /bin/python3 /home/ben/Repos/fourier_rg_cnn/fourier_model.py > tf_runs/out_${RUN_NAME}.txt) & sudo powerstat -R 0.5 120 > tf_runs/pwr_${RUN_NAME}.txt; wait) && echo "done ${RUN_NAME}"; done