#!/usr/bin/env python
import rospy
from ti_mmwave_rospkg.msg import RadarRaw
import numpy as np

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

range_dop_hmap_pub = rospy.Publisher("/ti_mmwave/range_doppler_heatmap", Image, queue_size=1)
range_azi_spec_pub = rospy.Publisher("/ti_mmwave/range_azimuth_spectrum", Image, queue_size=1)
range_azi_hmap_pub = rospy.Publisher("/ti_mmwave/range_azimuth_heatmap", Image, queue_size=1)

def HeatmapMaker(data):
	if len(data.shape) == 2:
		data = data.reshape(data.shape[0], data.shape[1], 1)
	h_img = (120.0/(np.min(data)-np.max(data)) * (data -np.max(data))).astype(np.uint8) #H of HSV
	hsv_img = np.concatenate([h_img, 255*np.ones((h_img.shape[0], h_img.shape[1], 2) ,np.uint8)], axis = 2)
	return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

def range_doppler_callback(data):
	layout = data.data.layout
	range_doppler_data = np.array(data.data.data).reshape([layout.dim[0].size, layout.dim[1].size])
	range_doppler_heatmap = HeatmapMaker(range_doppler_data)

	bridge = CvBridge()
	msg = bridge.cv2_to_imgmsg(range_doppler_heatmap, encoding="bgr8")
	range_dop_hmap_pub.publish(msg)

def range_azimuth_callback(data):
	layout = data.data.layout
	range_azimuth_data = np.array(data.data.data).reshape([layout.dim[0].size, layout.dim[1].size, layout.dim[2].size])
	range_azimuth_data = np.where(range_azimuth_data < 32767, range_azimuth_data, range_azimuth_data - 65536)
	range_azimuth_data = range_azimuth_data[:,:,-1::-1] #Imag, Real to Real Imag
	range_azimuth_data = np.apply_along_axis(lambda args: [complex(*args)], 2, range_azimuth_data) # convert to complex
	range_azimuth_fft = np.abs(np.fft.fft(range_azimuth_data, axis=2)) #power spectrum
	range_azimuth_heatmap = HeatmapMaker(range_azimuth_fft)

	bridge = CvBridge()
	msg = bridge.cv2_to_imgmsg(range_azimuth_fft.astype("u2"), encoding="mono16")
	range_azi_spec_pub.publish(msg)
	msg = bridge.cv2_to_imgmsg(range_azimuth_heatmap, encoding="bgr8")
	range_azi_hmap_pub.publish(msg)

if __name__ == '__main__':
	rospy.init_node('ti_data_processor_node', anonymous=True)
	rospy.Subscriber("/ti_mmwave/range_azimuth", RadarRaw, range_azimuth_callback, queue_size=1)
	rospy.Subscriber("/ti_mmwave/range_doppler", RadarRaw, range_doppler_callback, queue_size=1)
	rospy.spin()