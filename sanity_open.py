import os, time, depthai as dai
os.environ["DEPTHAI_BOOTUP_TIMEOUT"]="20000"
os.environ["DEPTHAI_DEVICE_TIMEOUT"]="20000"
os.environ["DEPTHAI_USB2"]="1"

print("DepthAI python:", dai.__version__)
try:
    devs = dai.Device.getAllAvailableDevices()
    print("Devices:", [(getattr(d, "mxid", getattr(d, "getMxId", lambda:None)()), getattr(d, "state", None), getattr(d, "protocol", None)) for d in devs])
except Exception as e:
    print("List devices failed:", e)

p = dai.Pipeline()
for i in (1,2):
    try:
        print(f"Open try {i}...")
        dev = dai.Device(p, usb2Mode=True)
        print("Opened:", dev.getDeviceName(), dev.getUsbSpeed())
        dev.close()
        print("Closed OK")
        break
    except Exception as e:
        print("Open failed:", e)
        time.sleep(1.5)
print("Done.")
