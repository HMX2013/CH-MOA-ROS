
#ifndef CONVEX_OPTIMIZE_FITTING_H
#define CONVEX_OPTIMIZE_FITTING_H

#include <ros/ros.h>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/centroid.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#define EIGEN_MPL2_ONLY

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "tf2_ros/static_transform_broadcaster.h"
#include "geometry_msgs/TransformStamped.h"

#include "obsdet_msgs/DetectedObject.h"
#include "obsdet_msgs/DetectedObjectArray.h"
#include "obsdet_msgs/CloudCluster.h"
#include "obsdet_msgs/CloudClusterArray.h"
#include "obsdet_msgs/local_obstacle_info.h"

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/PolygonArray.h>

#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_msgs/Float32.h>
#include <cmath>
#include <algorithm>

#include <std_msgs/String.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define __APP_NAME__ "Convex Optimize Fitting:"

static ros::Publisher pub_jskrviz_time_;
static std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
static std_msgs::Float32 time_spent;
static double exe_time = 0.0;

double box_x;
double box_y;

class ConvexOptimizeFitting
{
private:
  tf2_ros::TransformListener tf2_listener;
  tf2_ros::Buffer tf2_buffer;

  std::vector<double> bias_vector;
  std::vector<int> runtime_vector;
  bool first_appear;

  int error_less_0_5;
  int error_less_1_0;
  int error_less_2_0;
  int error_less_5_0;
  int error_less_10_0;

  std::string input_cluster_topic_;
  std::string output_bbox_topic_;

  std::string bbox_source_frame_;
  std::string bbox_target_frame_;

  ros::NodeHandle node_handle_;
  ros::Subscriber sub_object_array_;
  ros::Publisher pub_object_array_;
  ros::Publisher pub_autoware_bboxs_array_;
  ros::Publisher pub_jsk_bboxs_array_;
  ros::Publisher pub_jsk_bboxs_array_gt_;
  ros::Publisher pub_convex_corner_points_;
  ros::Publisher pub_min_reccorner_points_;

  ros::Publisher pub_rec_corner_points_;
  ros::Publisher pub_local_obstacle_info_;
  ros::Publisher pub_cluster_polygon_;
  ros::Publisher pub_angle_valid_pt2_;
  ros::Publisher pub_line_array_;
  ros::Publisher pub_proj_line_;
  ros::Publisher pub_pt_test_visual_;
  ros::Publisher pub_ch_cluster_visual_;

  ros::Publisher pub_result_info_rviz_;

  void MainLoop(const obsdet_msgs::CloudClusterArray& in_cluster_kitti_ros);

  int index_decrease(const int &index, const int &size);

  double Cal_VectorAngle(Eigen::Vector2f &line_vector, Eigen::Vector2f &lineseg);

  void eval_running_time(int running_time);

  void eval_performance(double &theta_kitti, double &theta_optim, const uint32_t &index, const uint32_t &index_seq);

  jsk_recognition_msgs::BoundingBox jsk_bbox_transform(const obsdet_msgs::DetectedObject &autoware_bbox, const std_msgs::Header &header);

  void calcuBoxbyPolygon(double &theta_star, obsdet_msgs::DetectedObject &output, std::vector<cv::Point2f> &rec_corner_points,
                         const std::vector<cv::Point2f> &polygon_cluster, const pcl::PointCloud<pcl::PointXYZ> &cluster);

  void optim_convex_fitting_improve(const std::vector<cv::Point2f> &ch_cluster, int &vl_idx_l, int &vl_idx_r, double &theta_optim, int &projL_idex_l_o, int &projL_idex_r_o, std::vector<cv::Point2f> &rcr_pts_o, pcl::PointCloud<pcl::PointXYZI>::Ptr &pt_test_visual);

  void visualize_edge(const std::vector<cv::Point2f> &ch_cluster, int &vl_idx, int &vl_idx_l, int &vl_idx_r,
                      double la[], double lb[], double lc[], int &projL_idex_, std::vector<cv::Point2f> &rcr_pts);

public:
  ConvexOptimizeFitting();
};

#endif  // CONVEX_OPTIMIZE_FITTING_H