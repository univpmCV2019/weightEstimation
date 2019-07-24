import os
import sys
import cv2
from openni import openni2
import argparse
import numpy as np

def openDevice(video_path):
    try:
        if sys.platform == "win32":
            libpath = "lib/Windows"
        else:
            libpath = "lib/Linux"
        openni2.initialize(libpath)
        dev = openni2.Device.open_file(video_path)
        pbs = openni2.PlaybackSupport(dev)

        pbs.set_repeat_enabled(False)
        pbs.set_speed(-1.0)

        return dev,pbs
    except Exception as ex:
        print(ex)
        raise Exception("Initialization Error")

def processDepth(dev,pbs,interval,dst):
    try:
        depth_stream = dev.create_depth_stream()
        depth_stream.start()
        depth_scale_factor = 255.0 / (650.0-520.0)
        depth_scale_beta_factor = -520.0*255.0/(650.0-520.0)
        print("Depth frames: " + str(depth_stream.get_number_of_frames()))
        for i in range(1,depth_stream.get_number_of_frames()+1,interval):
            pbs.seek(depth_stream,i)
            frame_depth = depth_stream.read_frame()
            frame_depth_data = frame_depth.get_buffer_as_uint16()
            depth_array = np.ndarray((frame_depth.height, frame_depth.width),dtype=np.uint16,buffer=frame_depth_data)
            depth_uint8 = depth_array*depth_scale_factor+depth_scale_beta_factor
            depth_uint8[depth_uint8>255] = 255
            depth_uint8[depth_uint8<0] = 0
            depth_uint8 = depth_uint8.astype('uint8')
            cv2.imwrite(dst + "/" + str(frame_depth.frameIndex) + "_16bit.png",depth_array)
            cv2.imwrite(dst + "/" + str(frame_depth.frameIndex) + "_8bit.png",depth_uint8)
        depth_stream.close()
        print("All depth frames extracted")
    except Exception as ex:
        print(ex)

def processColor(dev,pbs,interval,dst):
    try:
        color_stream = dev.create_color_stream()
        color_stream.start()
        print("Color frames: " + str(color_stream.get_number_of_frames()))
        for i in range(1,color_stream.get_number_of_frames()+1,interval):
            pbs.seek(color_stream,i)
            frame_color = color_stream.read_frame()
            frame_color_data = frame_color.get_buffer_as_uint8()
            color_array = np.ndarray((frame_color.height, frame_color.width, 3),dtype=np.uint8,buffer=frame_color_data)
            color_array = cv2.cvtColor(color_array, cv2.COLOR_BGR2RGB)
            cv2.imwrite(dst + "/" + str(frame_color.frameIndex) + "_color.png",color_array)
        color_stream.close()
        print("All color frames extracted")
    except Exception as ex:
        print(ex)

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    p.add_argument('--v', dest='video_path', action='store', required=True, help='path Video')
    p.add_argument('--d', dest='dst', action='store', default='img', help='Destination Folder')
    p.add_argument('--i', dest='interval', action='store', default=1, help='Interval')
    args = p.parse_args()
    interval = int(args.interval)
    dst = args.dst
    if not os.path.exists(dst):
        os.mkdir(dst)
        print("Directory ",dst ," Created ")
    try:
        dev,pbs = openDevice(args.video_path.encode('utf-8'))
        pbs.set_repeat_enabled(True)
        if dev.has_sensor(openni2.SENSOR_COLOR):
            print("Color Stream found")
            processColor(dev,pbs,interval,dst)
        if dev.has_sensor(openni2.SENSOR_DEPTH):
            print("Depth Stream found")
            processDepth(dev,pbs,interval,dst)
        print("Done!")
    except Exception as ex:
        print(ex)
    openni2.unload()

if __name__ == '__main__':
    main()
