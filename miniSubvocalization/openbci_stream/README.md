# OpenBCI data prediction stream

We are using OpenBCI Ganglion to collect data with its GUI networking widget and uses OSC to stream data, catching the EMG signals from the vocal cords and translating it to words and sentences.

- Uses the python-osc library to communicate with the OpenBCI device.
- Since I'm using synchronous streaming, which is not the most optimal way to stream data, but what I want to do is to get it to work first without losing any data from the device.

- While using the python-osc library, had a problem where I was getting duplicate data from the device, and after I restart the OpenBCI GUI, the data would stream properly, so reminder is to restart the GUI whenever you want to stream or record data.

- Device used: [OpenBCI Ganglion](http://docs.openbci.com/Tutorials/02-Ganglion_Getting%20Started_Guide)


#### setup for data collection
- Install required modules:
    - `pip install -r requirements.txt`
- Create required directories:
    - `python setup.py`

#### `osc_stream.py`
- Reads in the OpenBCI from the OSC server, takes 20 data points and converts them to csv files continuously up to 30 files,
and then rewrites them.

#### Additional parameters
- `--option` 
  - `print` : prints the data streaming from the device. Mainly used for checking the connection between the GUI and OSC.
  - `predict` : records a fixed interval and runs the data through a pre-trained model and predicts what the data corresponds to. (ex: 'yes' or 'no')

#### `osc_collect_data.py`
- This scripts works by outputting the label it is currently recording onto the terminal, and records for a fixed duration.
- Once the user sees the output on the terminal, he/she should make the gesture according to the lable.

#### Additional parameters
- `--option`
    - `fname` : default output directory will be the current time, if you want to change it you can name it with this option.