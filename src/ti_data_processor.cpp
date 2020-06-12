#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unsupported/Eigen/FFT>
#include <complex>
#include "ros/ros.h"
#include "delaunator.hpp"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/UInt16MultiArray.h"
#include <ti_mmwave_rospkg/RadarScan.h>
#include <ti_mmwave_rospkg/RadarRaw.h>

class TIDataProcessor{
public:
	TIDataProcessor(ros::NodeHandle *nd):
		nd_(nd),
		range_fft_num(128),
		virtual_anntena_num(8)
	{
		ROS_INFO("Initializing...");
		InitCartesian();
		ROS_INFO("Completed");
		range_doppler_h_pub = nd->advertise<sensor_msgs::Image>("/ti_mmwave/range_doppler_heatmap", 100);
		range_azimuth_h_pub = nd->advertise<sensor_msgs::Image>("/ti_mmwave/range_azimuth_heatmap", 100);
		range_azimuth_sub = nd->subscribe("/ti_mmwave/range_azimuth", 100, &TIDataProcessor::RangeAzimuthCallback, this);
		range_doppler_sub = nd->subscribe("/ti_mmwave/range_doppler", 100, &TIDataProcessor::RangeDopplerCallback, this);
	}

	void RangeAzimuthCallback(ti_mmwave_rospkg::RadarRaw radar_raw)
	{
		std_msgs::MultiArrayLayout layout = radar_raw.data.layout;
		int azimuth_num = virtual_anntena_num - 1;
		std::vector<std::complex<float>> raw_data(virtual_anntena_num);
		std::vector<std::complex<float>> fft_data(virtual_anntena_num);
		float range_azimuth_data[range_fft_num * azimuth_num] = {0};
		Eigen::FFT<float> fft;
		const int cp_size = virtual_anntena_num / 2;
		const int cp_bite_size = cp_size * sizeof(float);

		// FFT
		for (int i = 0; i < range_fft_num; i++)
		{
			for (int j = 0; j < virtual_anntena_num; j++)
			{
				int index = (i * virtual_anntena_num + j)*2;
				float comp_buf[2] = {float(radar_raw.data.data[index]), float(radar_raw.data.data[index+1])};
				for(int k=0; k<2; k++){
					if(comp_buf[k] > 32767){
						comp_buf[k] -= 65536;
					}
				}

				raw_data[j].imag(comp_buf[0]);
				raw_data[j].real(comp_buf[1]);
			}

			fft.fwd(fft_data, raw_data);
			float fft_data_abs[virtual_anntena_num];
			for(int j=0; j<virtual_anntena_num; j++){
				fft_data_abs[j] = std::abs(fft_data[j]);
			}

			// concat & slice(1)
			int idx = i * azimuth_num;
			std::memcpy(&range_azimuth_data[idx], &fft_data_abs[cp_size + 1], cp_bite_size - sizeof(float));
			std::memcpy(&range_azimuth_data[idx + cp_size - 1], fft_data_abs, cp_bite_size);
		}

		// To cartesian
		cv::Mat cartesian_data(cv::Size(heatmap_shape, heatmap_shape), CV_16UC1, cv::Scalar(0));
		cv::Mat range_azimuth_heatmap;

		for(int i=0; i < heatmap_shape; i++){
			for(int j=0; j < heatmap_shape; j++){
				Eigen::Vector3d point = points[i * heatmap_shape+j];
				std::array<Eigen::Vector3d, 3> tri_vertex = triangles[i * heatmap_shape + j];
				Eigen::Vector3d tri1 = tri_vertex[0];
				Eigen::Vector3d tri2 = tri_vertex[1];
				Eigen::Vector3d tri3 = tri_vertex[2];

				tri1[2] = GetRangeAzimuthData(tri1, range_azimuth_data);
				tri2[2] = GetRangeAzimuthData(tri2, range_azimuth_data);
				tri3[2] = GetRangeAzimuthData(tri3, range_azimuth_data);
				Eigen::Vector3d vec1 = tri2 - tri1;
				Eigen::Vector3d vec2 = tri3 - tri1;
				Eigen::Vector3d n = (vec1.cross(vec2)).normalized();
				double d = -tri1.dot(n);
				float elem = - (n[0] * point[0] + n[1] * point[1] + d) / n[2];
				
				cartesian_data.at<uint16_t>(heatmap_shape - 1 - j, i) = uint16_t(elem);
			}
		}

		HeatmapMaker(cartesian_data, range_azimuth_heatmap);
		cv::Mat resized_heatmap;
		cv::resize(range_azimuth_heatmap, resized_heatmap, cv::Size(0,0), 1, 1, cv::INTER_NEAREST);
		sensor_msgs::ImagePtr msg = cv_bridge::CvImage(radar_raw.header, "bgr8", resized_heatmap).toImageMsg();
		range_azimuth_h_pub.publish(msg);
	}

	void RangeDopplerCallback(ti_mmwave_rospkg::RadarRaw radar_raw)
	{
		std_msgs::MultiArrayLayout layout = radar_raw.data.layout;
		int doppler_fft_num = layout.dim[1].size;
		int doppler_num = doppler_fft_num - 1;
		cv::Mat range_doppler_data(cv::Size(range_fft_num, doppler_num), CV_16UC1, cv::Scalar(0));
		cv::Mat range_doppler_heatmap;
		const int cp_size = doppler_fft_num / 2;
		const int cp_bite_size = cp_size * sizeof(uint16_t);

		for (int i = 0; i < range_fft_num; i++)
		{
			uint16_t buf[doppler_num];
			std::memcpy(buf, &radar_raw.data.data[i*doppler_fft_num + cp_size + 1], cp_bite_size - sizeof(uint16_t));
			std::memcpy(&buf[cp_size-1], &radar_raw.data.data[i*doppler_fft_num], cp_bite_size);
			
			for (int j = 0; j < doppler_num; j++)
			{
				range_doppler_data.at<uint16_t>(j,i)= buf[j];
			}
		}
		
		HeatmapMaker(range_doppler_data, range_doppler_heatmap);
		cv::Mat resized_heatmap;
		cv::resize(range_doppler_heatmap, resized_heatmap, cv::Size(0,0), 1, 1, cv::INTER_NEAREST);
		sensor_msgs::ImagePtr msg = cv_bridge::CvImage(radar_raw.header, "bgr8", resized_heatmap).toImageMsg();
		range_doppler_h_pub.publish(msg);
	}

private:
	ros::NodeHandle *nd_;
	ros::Publisher range_doppler_h_pub;
	ros::Publisher range_azimuth_h_pub;
	ros::Subscriber range_azimuth_sub;
	ros::Subscriber range_doppler_sub;

	int range_fft_num;
	int virtual_anntena_num;

	double heatmap_range = 15; //[m]
	int heatmap_shape = 128;
	double range_resolution = 0.17;
	std::vector<std::array<Eigen::Vector3d, 3>> triangles;
	std::vector<Eigen::Vector3d> points;

	void InitCartesian(){
		triangles.clear();
		// To cartesian
		int azimuth_num = virtual_anntena_num - 1;
		double theta[azimuth_num+2];
		std::vector<double> posxy;

		for (int i = (-virtual_anntena_num / 2 + 1); i < virtual_anntena_num / 2; i++)
		{
			theta[i + 1 - (-virtual_anntena_num / 2 + 1)] = asin(i * 2.0 / azimuth_num);
		}
		theta[0] = -0.5*M_PI;
		theta[azimuth_num+1] = 0.5*M_PI;
		double sin_theta[azimuth_num+2], cos_theta[azimuth_num+2];
		for(int i=0; i<azimuth_num+2; i++){
			sin_theta[i] = sin(theta[i]);
			cos_theta[i] = cos(theta[i]);
		}
		for (int i = 0; i < range_fft_num; i++)
		{
			for (int j = 0; j < azimuth_num + 2; j++)
			{
				double range = range_resolution * i;
				posxy.push_back(range * sin_theta[j]);
				posxy.push_back(range * cos_theta[j]);
			}
		}
		delaunator::Delaunator d_triangles(posxy);
		int triangles_num = d_triangles.triangles.size() / 3;
		ROS_INFO("Delaunay triangles num = %d\n", triangles_num);
		std::vector<std::vector<std::array<Eigen::Vector3d, 3>>> theta_triangles(azimuth_num);
		for(int i=0; i < triangles_num; i++){
			int idx1 = 2 * d_triangles.triangles[3 * i];
			int idx2 = 2 * d_triangles.triangles[3 * i + 1];
			int idx3 = 2 * d_triangles.triangles[3 * i + 2];
			Eigen::Vector3d tri1({d_triangles.coords[idx1], d_triangles.coords[idx1 + 1], 0});
			Eigen::Vector3d tri2({d_triangles.coords[idx2], d_triangles.coords[idx2 + 1], 0});
			Eigen::Vector3d tri3({d_triangles.coords[idx3], d_triangles.coords[idx3 + 1], 0});
			Eigen::Vector3d center = (tri1 + tri2 + tri3) / 3.0;
			int theta_index = int((center[0] / center.norm() * azimuth_num / 2.0)) + virtual_anntena_num / 2 - 1;
			theta_triangles[theta_index].push_back(std::array<Eigen::Vector3d, 3>({tri1, tri2, tri3}));
		}

		for (int i = 0; i < heatmap_shape; i++)
		{
			Eigen::Vector3d point = Eigen::Vector3d::Zero();
			point[0] = (heatmap_range * (i - heatmap_shape / 2)) / double(heatmap_shape);
			for (int j = 0; j < heatmap_shape; j++)
			{
				point[1] = heatmap_range * j / double(heatmap_shape);
				points.push_back(point);
				bool in_triangles;
				int theta_index = int((sin(atan2(point[0], point[1])) * azimuth_num / 2.0)) + virtual_anntena_num / 2 - 1;
				for (int k = 0; k < theta_triangles[theta_index].size(); k++)
				{
					const std::array<Eigen::Vector3d, 3> &vector_array = theta_triangles[theta_index][k];
					Eigen::Vector3d tri1 = vector_array[0];
					Eigen::Vector3d tri2 = vector_array[1];
					Eigen::Vector3d tri3 = vector_array[2];
					in_triangles = PointInTriangle(point, tri1, tri2, tri3);
					if (in_triangles)
					{
						triangles.push_back(vector_array);
						break;
					}
				}
				if(!in_triangles){
					triangles.push_back(std::array<Eigen::Vector3d, 3>({Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}));
				}
			}
		}
	}

	void HeatmapMaker(const cv::Mat &input, cv::Mat &output)
	{
		output = cv::Mat(cv::Size(input.cols, input.rows), CV_8UC3, cv::Scalar(0, 255, 255));
		double max_elem, min_elem;
		cv::minMaxLoc(input, &min_elem, &max_elem, NULL, NULL);
		//std::cout<<"max = "<<max_elem<<", min = "<<min_elem<<std::endl;

		for (int i = 0; i < input.rows; i++)
		{
			for (int j = 0; j < input.cols; j++)
			{
				output.at<cv::Vec3b>(i,j)[0] = uint8_t(120.0 / (min_elem - max_elem) * (input.at<uint16_t>(i,j) - max_elem));
			}
		}
		cvtColor(output, output, CV_HSV2BGR);
	}

	bool PointInTriangle(const Eigen::Vector3d &pt, const Eigen::Vector3d &tri1, const Eigen::Vector3d &tri2, const Eigen::Vector3d &tri3){
		double minx = std::min(tri3[0], std::min(tri1[0], tri2[0]));
		double maxx = std::max(tri3[0], std::max(tri1[0], tri2[0]));
		double miny = std::min(tri3[1], std::min(tri1[1], tri2[1]));
		double maxy = std::max(tri3[1], std::max(tri1[1], tri2[1]));
		if ((pt[0] >= minx) && (pt[0] <= maxx) && (pt[1] >= miny) && (pt[1] <= maxy))
		{
			bool b1 = Crosssign(pt, tri1, tri2);
			bool b2 = Crosssign(pt, tri2, tri3);
			bool b3 = Crosssign(pt, tri3, tri1);

			if((b1 == b2) && (b2 == b3)){
				return true;
			}else{
				return false;
			}
		}else{
			return false;
		}
	}

	int GetRangeAzimuthData(const Eigen::Vector3d &pt, const float range_azimuth_data[])
	{
		int azimuth_num = virtual_anntena_num-1;
		double range = sqrt(pt[0]*pt[0] + pt[1]*pt[1]);
		double azimuth = atan2(pt[0], pt[1]);
		int range_idx = range / 0.17;
		int azimuth_index = 0;
		if(fabs(azimuth) > (0.5*M_PI-0.0174)){
			return 0.0;
		}else{
			azimuth_index = int((sin(azimuth) * azimuth_num / 2.0)) + virtual_anntena_num / 2 - 1;
			return range_azimuth_data[range_idx * azimuth_num + azimuth_index];
		}
	}

	bool Crosssign(const Eigen::Vector3d &pt, const Eigen::Vector3d &p1, const Eigen::Vector3d &p2){
		Eigen::Vector3d vec1 = pt - p1;
		Eigen::Vector3d vec2 = pt - p2;
		return std::signbit(vec1[0] * vec2[1] - vec2[0] * vec1[1]);
	}
};

int main(int argc, char **argv){
	ros::init(argc, argv, "ti_data_processor");
	ros::NodeHandle nd;
	TIDataProcessor ti_data_processor(&nd);
	ros::spin();
	return 0;
}