# Crazyflie PC client for esp-drone

The esp-drone PC client is a fork of the [Crazyflie PC client](./ORIGIN_README.md). The communication with esp-drone and the implementation of the CRTP protocol to control the esp-drone is handled by the modified [cflib](https://github.com/leeebo/crazyflie-lib-python)

## cflib
The client requires the folked [cflib](https://github.com/leeebo/crazyflie-lib-python).
If you want to develop with the lib too, follow the cflib readme to install it.

## Installation

Clone repository:
```bash
git clone https://github.com/jobitjoseph/crazyflie-clients-python.git
cd crazyflie-clients-python
```
Install the client from source:
```bash
pip3 install -e .
```
## Run

```bash
cfclient
```

## Documentation

Check out https://espressif-docs.readthedocs-hosted.com/projects/espressif-esp-drone/en/latest/gettingstarted.html

then Check out the [Bitcraze crazyflie-client-python documentation](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/) on website.
