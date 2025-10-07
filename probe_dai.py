import os, time
import depthai as dai

os.environ["DEPTHAI_BOOTUP_TIMEOUT"] = "20000"
os.environ["DEPTHAI_DEVICE_TIMEOUT"] = "20000"
os.environ["DEPTHAI_USB2"] = "1"

print("DepthAI version:", dai.__version__)
print("Scanning devices...")
for d in dai.Device.getAllAvailableDevices():
    print(" -", d.getMxId(), d.state, d.protocol)

print("Building empty pipeline...")
p = dai.Pipeline()
# No nodes: just boot the device

def cfg():
    c = dai.Device.Config()
    c.board.usb.maxUsbSpeed = dai.UsbSpeed.HIGH
    c.logLevel = dai.LogLevel.DEBUG
    return c

for i in range(2):
    try:
        print(f"Attempt {i+1}/2: opening device...")
        with dai.Device(p, usb2Mode=True, config=cfg()) as dev:
            print("Opened. Waiting 1s, then closing...")
            time.sleep(1)
        print("Closed cleanly.")
        break
    except Exception as e:
        print("Open failed:", e)
        time.sleep(1.5)

print("Probe done.")
input("Press Enter to exit...")
