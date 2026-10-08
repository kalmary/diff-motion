1. program has multiple entrypoints:
  1.  cli
  2. importable as rust crate
  3. importable as python package

1. cli:
  2. config how algorithm should work
  3. another option is running with config.yaml file - argument is just path then

2/3 importable packages rust and python get just video stream

4. two output modes: display from camera (should work on macos, linux, windows (webcam/ usb camera: on of args should be camera index), json camera: step for later), headless