#!/usr/bin/env python
import rospy
from ti_mmwave_rospkg.msg import RadarRaw
import numpy as np

def radar_data_callback(data):
	layout = data.data.layout
	range_azimuth_data = np.array(data.data.data).reshape([layout.dim[0].size, layout.dim[1].size, layout.dim[2].size])
	range_azimuth_data = range_azimuth_data[:,:,-1::-1] #Imag, Real to Real Imag
	range_azimuth_data = np.apply_along_axis(lambda args: [complex(*args)], 2, range_azimuth_data) # to complex
	range_azimuth_fft = np.abs(np.fft.fft(range_azimuth_data, axis=2)) #power spectrum
	print(range_azimuth_fft)

if __name__ == '__main__':
	rospy.init_node('ti_data_processor_node', anonymous=True)
	rospy.Subscriber("/ti_mmwave/range_azimuth", RadarRaw, radar_data_callback, queue_size=1)
	rospy.spin()