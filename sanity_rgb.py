import os, depthai as dai

os.environ["DEPTHAI_USB2"] = "1"
os.environ["DEPTHAI_BOOTUP_TIMEOUT"] = "20000"
os.environ["DEPTHAI_DEVICE_TIMEOUT"] = "20000"

p = dai.Pipeline()
cam = p.createColorCamera()
cam.setPreviewSize(640, 400)
cam.setFps(20)
xout = p.createXLinkOut(); xout.setStreamName("preview")
cam.preview.link(xout.input)

dev = dai.Device(p, usb2Mode=True)
q = dev.getOutputQueue("preview", 2, False)

print("RGB preview running — press Ctrl+C to quit")
while True:
    _ = q.get()  # if it blocks forever or throws, we’ll see it here
