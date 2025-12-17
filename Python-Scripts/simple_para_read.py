import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

# URI for your LiteWing drone
DRONE_URI = "udp://192.168.43.42"

# Initialize CRTP drivers
cflib.crtp.init_drivers()

def debug_toc(cf):
    print("Debugging TOC structure...")
    toc = cf.log.toc
    print("TOC groups:", list(toc.toc.keys())[:10], "...")  # Print first 10 groups
    # Print a sample group with its parameters
    sample_group = list(toc.toc.keys())[0]
    print(f"Sample group '{sample_group}':", toc.toc[sample_group])
    # Print a sample LogTocElement
    sample_param = list(toc.toc[sample_group].values())[0]
    print(f"Sample LogTocElement attributes - group: {sample_param.group}, name: {sample_param.name}")

def list_controller_parameters(cf):
    print("\nListing all parameters in the 'controller' group...")
    toc = cf.log.toc
    controller_params = []
    for group, params in toc.toc.items():
        if group == 'controller':
            for param_name, elem in params.items():
                full_name = f"{elem.group}.{elem.name}"
                controller_params.append(full_name)
    for param in sorted(controller_params):
        print(f" - {param}")

    print("\nListing all parameters in the 'stabilizer' group (for reference)...")
    stabilizer_params = []
    for group, params in toc.toc.items():
        if group == 'stabilizer':
            for param_name, elem in params.items():
                full_name = f"{elem.group}.{elem.name}"
                stabilizer_params.append(full_name)
    for param in sorted(stabilizer_params):
        print(f" - {param}")

def main():
    with SyncCrazyflie(DRONE_URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        print("Connecting to drone...")
        time.sleep(2.0)  # Wait for TOC download

        # Debug TOC structure
        debug_toc(cf)

        # List controller and stabilizer parameters
        list_controller_parameters(cf)

        print("Disconnecting from drone...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    print("Done")