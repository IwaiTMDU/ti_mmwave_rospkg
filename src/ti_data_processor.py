#!/usr/bin/env python
import rospy
from ti_mmwave_rospkg.msg import RadarRaw
import numpy as np
import math
from scipy.spatial import Delaunay, delaunay_plot_2d

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

range_dop_spec_pub = rospy.Publisher("/ti_mmwave/range_doppler_spectrum", Image, queue_size=1)
range_dop_hmap_pub = rospy.Publisher("/ti_mmwave/range_doppler_heatmap", Image, queue_size=1)
range_azi_spec_pub = rospy.Publisher("/ti_mmwave/range_azimuth_spectrum", Image, queue_size=1)
range_azi_hmap_pub = rospy.Publisher("/ti_mmwave/range_azimuth_heatmap", Image, queue_size=1)

def HeatmapMaker(data):
	if len(data.shape) == 2:
		data = data.reshape(data.shape[0], data.shape[1], 1)
	h_img = (120.0/(np.min(data)-np.max(data)) * (data -np.max(data))).astype(np.uint8) #H of HSV
	hsv_img = np.concatenate([h_img, 255*np.ones((h_img.shape[0], h_img.shape[1], 2) ,np.uint8)], axis = 2)
	return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

def SmoothRangeAzimuthHeatmap(data):
	pass

def sign (p1, p2, p3):
	return (p1 - p3)[0] * (p2 - p3)[1] - (p2 - p3)[0] * (p1 - p3)[1]

 
def PointInTriangle (pt, tri_index, posxy):
	b1 = sign(pt, posxy[tri_index[0]], posxy[tri_index[1]]) < 0.0
	b2 = sign(pt, posxy[tri_index[1]], posxy[tri_index[2]]) < 0.0
	b3 = sign(pt, posxy[tri_index[2]], posxy[tri_index[0]]) < 0.0
 
	return ((b1 == b2) and (b2 == b3))

def range_doppler_callback(data):
	layout = data.data.layout
	# range doppler spectrum
	range_doppler_data = np.array(data.data.data).reshape([layout.dim[0].size, layout.dim[1].size])
	range_doppler_data = np.concatenate([range_doppler_data[:,layout.dim[1].size/2:], range_doppler_data[:,0:layout.dim[1].size/2]], axis=1)

	# range doppler heatmap image
	range_doppler_heatmap = HeatmapMaker(range_doppler_data.transpose([1,0]))

	bridge = CvBridge()
	msg = bridge.cv2_to_imgmsg(range_doppler_data.astype("u2"), encoding="mono16")
	range_dop_spec_pub.publish(msg)
	msg = bridge.cv2_to_imgmsg(range_doppler_heatmap, encoding="bgr8")
	range_dop_hmap_pub.publish(msg)

def range_azimuth_callback(data):
	layout = data.data.layout
	# digital processing for range azimuth spectrum
	range_azimuth_data = np.array(data.data.data).reshape([layout.dim[0].size, layout.dim[1].size, layout.dim[2].size])
	range_azimuth_data = np.where(range_azimuth_data < 32767, range_azimuth_data, range_azimuth_data - 65536)
	range_azimuth_data = range_azimuth_data[:,:,-1::-1] #Imag, Real to Real Imag
	range_azimuth_data = np.apply_along_axis(lambda args: [complex(*args)], 2, range_azimuth_data) # convert to complex
	range_azimuth_fft = np.abs(np.fft.fft(range_azimuth_data, axis=2)) #power spectrum
	range_azimuth_fft = np.concatenate([range_azimuth_fft[:,layout.dim[1].size/2:], range_azimuth_fft[:,0:layout.dim[1].size/2]], axis=1)
	range_azimuth_fft = range_azimuth_fft[:,1:,:] # pop 0
	range_azimuth_fft = range_azimuth_fft[:,::-1,:] #reverse axis 1
	
	# range doppler heatmap image
	range_resolution = 0.17
	heatmap_shape = 128

	theta = np.arcsin(np.arange(-layout.dim[1].size/2+1, layout.dim[1].size/2, dtype="float32")*2/layout.dim[1].size)
	#theta = np.insert(np.array([-0.5*np.pi, 0.5*np.pi]), 1, theta)
	#print(theta)
	range_bins = range_resolution * np.arange(0, layout.dim[0].size, dtype="float32")
	posXY = range_bins.reshape((range_bins.shape[0],1,1)) * np.array([np.sin(theta), np.cos(theta)]).transpose().reshape(1, theta.shape[0], 2)
	posXY = posXY.reshape((range_bins.shape[0]*theta.shape[0], 2))
	
	delaynay_output = Delaunay(posXY)
	triangles = delaynay_output.simplices

	#fig = delaunay_plot_2d(delaynay_output)
	#fig.savefig('scipy_matplotlib_delaunay.png')

	range_azimuth_fft_reshaped = range_azimuth_fft.flatten()
	img_indexes = np.array(np.where(np.ones((heatmap_shape, heatmap_shape)))).transpose()
	img_ranges = range_resolution * (img_indexes - np.array([heatmap_shape/2-1,0]))
	range_azimuth_heatmap  = np.zeros(shape=(heatmap_shape, heatmap_shape))
	print(triangles.shape[0])
	for i in range(img_indexes.shape[0]):
		points = img_ranges[i]
		img_index = img_indexes[i]
		index = -1
		for j in range(triangles.shape[0]):
			if PointInTriangle(points, triangles[j], posXY):
				index = j
				break
		if index > 0:
			tri1 = np.array([posXY[triangles[index][0], 0], posXY[triangles[index][0], 1], range_azimuth_fft_reshaped[triangles[index][0]]])
			tri2 = np.array([posXY[triangles[index][1], 0], posXY[triangles[index][1], 1], range_azimuth_fft_reshaped[triangles[index][1]]])
			tri3 = np.array([posXY[triangles[index][2], 0], posXY[triangles[index][2], 1], range_azimuth_fft_reshaped[triangles[index][2]]])
			u = tri2 - tri1
			v = tri3 - tri1
			n = np.cross(u,v)
			n = n / np.linalg.norm(n)
			D = -np.dot(tri1, n)
			range_azimuth_heatmap[img_index[0], img_index[1]] = -(n[0]*points[0]+n[1]*points[1]+D)/n[2]
	range_azimuth_heatmap = HeatmapMaker(range_azimuth_heatmap)
	print("pub")
	#zi.reshape(())
	
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