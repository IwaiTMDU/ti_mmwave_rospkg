#!/usr/bin/env python
import rospy
from ti_mmwave_rospkg.msg import RadarRaw
import numpy as np

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

range_azi_spec_pub = rospy.Publisher("/ti_mmwave/range_azimuth_spectrum", Image, queue_size=1)
range_azi_hmap_pub = rospy.Publisher("/ti_mmwave/range_azimuth_heatmap", Image, queue_size=1)

def range_azimuth_callback(data):
	layout = data.data.layout
	range_azimuth_data = np.array(data.data.data).reshape([layout.dim[0].size, layout.dim[1].size, layout.dim[2].size])
	range_azimuth_data = np.where(range_azimuth_data < 32767, range_azimuth_data, range_azimuth_data - 65536)
	range_azimuth_data = range_azimuth_data[:,:,-1::-1] #Imag, Real to Real Imag
	range_azimuth_data = np.apply_along_axis(lambda args: [complex(*args)], 2, range_azimuth_data) # convert to complex
	range_azimuth_fft = np.abs(np.fft.fft(range_azimuth_data, axis=2)) #power spectrum
	range_azimuth_heatmap = (120.0/(np.min(range_azimuth_fft)-np.max(range_azimuth_fft)) * (range_azimuth_fft -np.max(range_azimuth_fft))).astype(np.uint8) #H of HSV
	range_azimuth_heatmap = np.concatenate([range_azimuth_heatmap, 255*np.ones((layout.dim[0].size, layout.dim[1].size, 2) ,np.uint8)], axis = 2)
	range_azimuth_heatmap_cv = cv2.cvtColor(range_azimuth_heatmap, cv2.COLOR_HSV2BGR)

	bridge = CvBridge()
	msg = bridge.cv2_to_imgmsg(range_azimuth_fft.astype("u2"), encoding="mono16")
	range_azi_spec_pub.publish(msg)
	msg = bridge.cv2_to_imgmsg(range_azimuth_heatmap_cv, encoding="bgr8")
	range_azi_hmap_pub.publish(msg)

if __name__ == '__main__':
	rospy.init_node('ti_data_processor_node', anonymous=True)
	rospy.Subscriber("/ti_mmwave/range_azimuth", RadarRaw, range_azimuth_callback, queue_size=1)
	rospy.spin()