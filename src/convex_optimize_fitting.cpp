#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_datatypes.h>
#include <chrono>
#include "convex_optimize_fitting.h"

ConvexOptimizeFitting::ConvexOptimizeFitting() : tf2_listener(tf2_buffer)
{
  ros::NodeHandle private_nh_("~");

  bias_vector.clear();
  runtime_vector.clear();
  first_appear = true;

  error_less_0_5 = 0;
  error_less_1_0 = 0;
  error_less_2_0 = 0;
  error_less_5_0 = 0;
  error_less_10_0 = 0;

  private_nh_.param<std::string>("input_cluster_topic", input_cluster_topic_, "/kitti3d/cluster_array");
  ROS_INFO("Input_cluster_topic: %s", input_cluster_topic_.c_str());

  node_handle_.param<std::string>("output_bbox_topic_", output_bbox_topic_, "/convex_optim_fitting/bbox_visual_jsk");
  ROS_INFO("output_bbox_topic_: %s", output_bbox_topic_.c_str());

  private_nh_.param<std::string>("bbox_target_frame", bbox_target_frame_, "map");
  ROS_INFO("[%s] bounding box's target frame is: %s", __APP_NAME__, bbox_target_frame_.c_str());

  sub_object_array_ = node_handle_.subscribe(input_cluster_topic_, 1, &ConvexOptimizeFitting::MainLoop, this);

  pub_autoware_bboxs_array_ = node_handle_.advertise<obsdet_msgs::DetectedObjectArray>("/convex_optim_fitting/autoware_bboxs_array", 1);
  pub_jsk_bboxs_array_ = node_handle_.advertise<jsk_recognition_msgs::BoundingBoxArray>("/convex_optim_fitting/jsk_bboxs_array",1);

  pub_rec_corner_points_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/convex_optim_fitting/rec_corner_points", 1);
  pub_convex_corner_points_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/convex_optim_fitting/convex_corner_points", 1);

  pub_min_reccorner_points_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/convex_optim_fitting/min_reccorner_points", 1);

  pub_local_obstacle_info_ = node_handle_.advertise<obsdet_msgs::local_obstacle_info>("/convex_optim_fitting/local_obstacle_info", 1);
  pub_cluster_polygon_ = node_handle_.advertise<jsk_recognition_msgs::PolygonArray>("/convex_optim_fitting/cluster_polygon", 2);
  pub_angle_valid_pt2_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/convex_optim_fitting/lim_angle_pt", 100, true);
  pub_jskrviz_time_ = node_handle_.advertise<std_msgs::Float32>("/time_convex_fitting", 1);

  pub_line_array_ = node_handle_.advertise<visualization_msgs::MarkerArray>("/convex_optim_fitting/line_array", 10);
  pub_proj_line_ = node_handle_.advertise<visualization_msgs::MarkerArray>("/convex_optim_fitting/proj_line", 10);

  pub_pt_test_visual_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/convex_optim_fitting/pt_test", 10);

  pub_ch_cluster_visual_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/convex_optim_fitting/ch_cluster", 10);

  pub_result_info_rviz_ = private_nh_.advertise<std_msgs::String>("/obstacle_detection/obstacle_info_visualization", 2);
}


void ConvexOptimizeFitting::optim_convex_fitting_improve(const std::vector<cv::Point2f> &ch_cluster, int &vl_idx_l, int &vl_idx_r, double &theta_optim, int &projL_idex_l_o, int &projL_idex_r_o, std::vector<cv::Point2f> &rcr_pts_o, pcl::PointCloud<pcl::PointXYZI>::Ptr &pt_test_visual)
{
  const double sp_min_angle = 0.0;
  const double sp_max_angle = M_PI / 2.0;

  const double angle_reso = 0.2 * M_PI / 180.0;
  cv::Point2f evaluate_pt;

  double trapezoid_h, trapezoid_upper, trapezoid_lower, trapezoid_area;
  double convex_area_l, convex_area_r, convex_area_s, break_condition;

  std::vector<std::pair<double /*theta*/, double /*q*/>> Q;
  std::vector<std::pair<double /*convex_area_l*/, double /*convex_area_r*/>> debug;
  std::vector<std::pair<double /*projL_idex_l*/, double /*projL_idex_r*/>> projL_idex_v;

  std::vector<cv::Point2f> rec_corner_near_v;

  std::vector<cv::Point2f> rcr_pts;
  rcr_pts.resize(4);

  double la[4] = {0};
  double lb[4] = {0};
  double lc[4] = {0};
  int projL_idex_l, projL_idex_r;

  for (double theta_search = sp_min_angle; theta_search < sp_max_angle; theta_search += angle_reso)
  {
    Eigen::Vector2d e_1_star;  
    Eigen::Vector2d e_2_star;
    e_1_star << std::cos(theta_search), std::sin(theta_search);
    e_2_star << -std::sin(theta_search), std::cos(theta_search);
    std::vector<double> C_1_star;
    std::vector<double> C_2_star;

    for (int i = 0; i < ch_cluster.size(); i++)
    {
      C_1_star.push_back(ch_cluster[i].x * e_1_star.x() + ch_cluster[i].y * e_1_star.y());
      C_2_star.push_back(ch_cluster[i].x * e_2_star.x() + ch_cluster[i].y * e_2_star.y());
    }

    la[0] = std::cos(theta_search);
    lb[0] = std::sin(theta_search);
    lc[0] = *std::min_element(C_1_star.begin(), C_1_star.end());

    la[1] = -1.0 * std::sin(theta_search);
    lb[1] = std::cos(theta_search);
    lc[1] = *std::min_element(C_2_star.begin(), C_2_star.end());

    la[2] = std::cos(theta_search);
    lb[2] = std::sin(theta_search);
    lc[2] = *std::max_element(C_1_star.begin(), C_1_star.end());

    la[3] = -1.0 * std::sin(theta_search);
    lb[3] = std::cos(theta_search);
    lc[3] = *std::max_element(C_2_star.begin(), C_2_star.end());

    rcr_pts[0].x = (lb[0] * lc[1] - lb[1] * lc[0]) / (la[1] * lb[0] - la[0] * lb[1]);
    rcr_pts[0].y = (la[1] * lc[0] - la[0] * lc[1]) / (la[1] * lb[0] - la[0] * lb[1]);

    rcr_pts[1].x = (lb[1] * lc[2] - lb[2] * lc[1]) / (la[2] * lb[1] - la[1] * lb[2]);
    rcr_pts[1].y = (la[2] * lc[1] - la[1] * lc[2]) / (la[2] * lb[1] - la[1] * lb[2]);

    rcr_pts[2].x = (lb[2] * lc[3] - lb[3] * lc[2]) / (la[3] * lb[2] - la[2] * lb[3]);
    rcr_pts[2].y = (la[3] * lc[2] - la[2] * lc[3]) / (la[3] * lb[2] - la[2] * lb[3]);

    rcr_pts[3].x = (lb[3] * lc[0] - lb[0] * lc[3]) / (la[0] * lb[3] - la[3] * lb[0]);
    rcr_pts[3].y = (la[0] * lc[3] - la[3] * lc[0]) / (la[0] * lb[3] - la[3] * lb[0]);

    visualize_edge(ch_cluster, vl_idx_l, vl_idx_l, vl_idx_r, la, lb, lc, projL_idex_l, rcr_pts);
    visualize_edge(ch_cluster, vl_idx_r, vl_idx_l, vl_idx_r, la, lb, lc, projL_idex_r, rcr_pts);

    convex_area_l = 0;
    int iter_num = 0;

    double x1, x2, y1, y2;
    double x1_m1, y1_m1, x2_m1, y2_m1;
    double x1_p1, y1_p1, x2_p1, y2_p1;
    int dir_l, dir_r;

    x1 = ch_cluster[vl_idx_l].x;
    y1 = ch_cluster[vl_idx_l].y;
    x2 = ch_cluster[vl_idx_r].x;
    y2 = ch_cluster[vl_idx_r].y;

    x1_m1 = ch_cluster[index_decrease(vl_idx_l, ch_cluster.size())].x;
    y1_m1 = ch_cluster[index_decrease(vl_idx_l, ch_cluster.size())].y;
    x2_m1 = ch_cluster[index_decrease(vl_idx_r, ch_cluster.size())].x;
    y2_m1 = ch_cluster[index_decrease(vl_idx_r, ch_cluster.size())].y;

    double org_sign = (0 - x1) * (y2 - y1) - (0 - y1) * (x2 - x1);
    double vl_idx_l_m1_sign = (x1_m1 - x1) * (y2 - y1) - (y1_m1 - y1) * (x2 - x1);
    double vl_idx_r_m1_sign = (x2_m1 - x1) * (y2 - y1) - (y2_m1 - y1) * (x2 - x1);

    if ((vl_idx_l + 1) % ch_cluster.size() == vl_idx_r){
      dir_l = -1;
    }
    else if (vl_idx_l_m1_sign * org_sign > 0){
      dir_l = -1;
    }
    else{
      dir_l = 1;
    }

    if ((vl_idx_r + 1) % ch_cluster.size() == vl_idx_l){
      dir_r = -1;
    }
    else if (vl_idx_r_m1_sign * org_sign > 0){
      dir_r = -1;
    }
    else{
      dir_r = 1;
    }

    int iter_fl, iter_fr;

    for (int i = vl_idx_l; iter_num < ch_cluster.size(); i += dir_l)
    {
      if (i < 0) {i = ch_cluster.size() - 1;}
      if (i > ch_cluster.size() - 1) {i = 0;}
      Eigen::Vector2f evaluate_pt_c;
      Eigen::Vector2f evaluate_pt_n;
      Eigen::Vector2f evaluate_lineseg;

      Eigen::Vector2f line_vector;
      line_vector << lb[projL_idex_l], -la[projL_idex_l];

      int index_n;
      if (dir_l == -1)
      {
        index_n = index_decrease(i, ch_cluster.size());
      }
      if (dir_l == 1)
      {
        index_n = (i + 1) % ch_cluster.size();
      }

      evaluate_pt_c << ch_cluster[i].x, ch_cluster[i].y;
      evaluate_pt_n << ch_cluster[index_n].x, ch_cluster[index_n].y;
      evaluate_lineseg = evaluate_pt_n - evaluate_pt_c;

      trapezoid_h = evaluate_lineseg.dot(line_vector);

      iter_fl = i;

      if (i != vl_idx_l && trapezoid_h * break_condition < 0) {
        break;
      }

      if (i == vl_idx_r)
        break;

      // calculate the area of trapezoid
      trapezoid_upper = std::fabs(la[projL_idex_l] * evaluate_pt_c(0) + lb[projL_idex_l] * evaluate_pt_c(1) - lc[projL_idex_l]) / sqrt(la[projL_idex_l] * la[projL_idex_l] + lb[projL_idex_l] * lb[projL_idex_l]);
      trapezoid_lower = std::fabs(la[projL_idex_l] * evaluate_pt_n(0) + lb[projL_idex_l] * evaluate_pt_n(1) - lc[projL_idex_l]) / sqrt(la[projL_idex_l] * la[projL_idex_l] + lb[projL_idex_l] * lb[projL_idex_l]);
      trapezoid_area = (trapezoid_upper + trapezoid_lower) * std::fabs(trapezoid_h) / 2.0;

      convex_area_l += trapezoid_area;
      break_condition = trapezoid_h;
      iter_num++;
    }

    int gap_step = 0;
    while (iter_fl != vl_idx_r)
    {
      iter_fl += dir_l;
      if (iter_fl < 0) {iter_fl = ch_cluster.size() - 1;}
      if (iter_fl > ch_cluster.size() - 1) {iter_fl = 0;}
      gap_step++;
    }

    convex_area_r = 0;
    iter_num = 0;
    for (int i = vl_idx_r; iter_num < gap_step; i += dir_r)
    {
      // in case the index over the size of vector
      if (i < 0) {i = ch_cluster.size() - 1;}
      if (i > ch_cluster.size() - 1) {i = 0;}

      Eigen::Vector2f line_vector;
      line_vector << lb[projL_idex_r], -la[projL_idex_r];

      Eigen::Vector2f evaluate_pt_c;
      Eigen::Vector2f evaluate_pt_n;
      Eigen::Vector2f evaluate_lineseg;

      int index_n;
      if (dir_r == -1) {
        index_n = index_decrease(i, ch_cluster.size());
      }
      if (dir_r == 1) {
        index_n = (i + 1) % ch_cluster.size();
      }

      evaluate_pt_c << ch_cluster[i % ch_cluster.size()].x, ch_cluster[i % ch_cluster.size()].y;
      evaluate_pt_n << ch_cluster[index_n].x, ch_cluster[index_n].y;
      evaluate_lineseg = evaluate_pt_n - evaluate_pt_c;

      trapezoid_h = evaluate_lineseg.dot(line_vector);

      iter_fr = i;

      if (i != vl_idx_r && trapezoid_h * break_condition < 0){
        break;
      }

      // calculate the area of trapezoid
      trapezoid_upper = std::fabs(la[projL_idex_r] * evaluate_pt_c(0) + lb[projL_idex_r] * evaluate_pt_c(1) - lc[projL_idex_r]) / sqrt(la[projL_idex_r] * la[projL_idex_r] + lb[projL_idex_r] * lb[projL_idex_r]);
      trapezoid_lower = std::fabs(la[projL_idex_r] * evaluate_pt_n(0) + lb[projL_idex_r] * evaluate_pt_n(1) - lc[projL_idex_r]) / sqrt(la[projL_idex_r] * la[projL_idex_r] + lb[projL_idex_r] * lb[projL_idex_r]);
      trapezoid_area = (trapezoid_upper + trapezoid_lower) * std::fabs(trapezoid_h) / 2;

      convex_area_r += trapezoid_area;
      break_condition = trapezoid_h;
      iter_num++;
    }
    convex_area_s = convex_area_l + convex_area_r;

    Q.push_back(std::make_pair(theta_search, convex_area_s));
    projL_idex_v.push_back(std::make_pair(projL_idex_l, projL_idex_r));
    debug.push_back(std::make_pair(convex_area_l, convex_area_r));
  }

  double min_q;
  double convex_area_l_debug, convex_area_r_debug;

  for (size_t i = 0; i < Q.size(); ++i)
  {
    if (Q.at(i).second < min_q  || i == 0)
    {
      min_q = Q.at(i).second;
      theta_optim = Q.at(i).first;
      convex_area_l_debug = debug.at(i).first;
      convex_area_r_debug = debug.at(i).second;
      projL_idex_l_o = projL_idex_v.at(i).first;
      projL_idex_r_o = projL_idex_v.at(i).second;
    }
  }

  ROS_INFO("DEBUG_trapezoid_area=%f, %f,%f", theta_optim * 180 / M_PI, convex_area_l_debug, convex_area_r_debug);

  double vl_idx_l_a = -ch_cluster[vl_idx_l].y / ch_cluster[vl_idx_l].x;
  double vl_idx_l_b = 1.0;
  double vl_idx_r_a = -ch_cluster[vl_idx_r].y / ch_cluster[vl_idx_r].x;
  double vl_idx_r_b = 1.0;

  double vl_inter_lx_v = -(lc[projL_idex_l] * vl_idx_l_b) / (vl_idx_l_a * lb[projL_idex_l] - la[projL_idex_l] * vl_idx_l_b);
  double vl_inter_ly_v = (vl_idx_l_a * lc[projL_idex_l]) / (vl_idx_l_a * lb[projL_idex_l] - la[projL_idex_l] * vl_idx_l_b);

  pcl::PointXYZI pt_test;
  pt_test.x = vl_inter_lx_v;
  pt_test.y = vl_inter_ly_v;
  pt_test.z = 0;
  pt_test.intensity = 1;
  pt_test_visual->push_back(pt_test);
}


void ConvexOptimizeFitting::visualize_edge(const std::vector<cv::Point2f> &ch_cluster, int &vl_idx, int &vl_idx_l, int &vl_idx_r,
                                           double la[], double lb[], double lc[], int &projL_idex_, std::vector<cv::Point2f> &rcr_pts)
{
  std::vector<int> candi_pL_idx;
  bool in_corner = false;
  bool has_corner = false;

  double vl_idx_a = -ch_cluster[vl_idx].y / ch_cluster[vl_idx].x;
  double vl_idx_b = 1.0;

  int neigh_recc_m1 = 999;
  int neigh_recc_p1 = 999;
  int neigh_recc_checked = 999;

  for (int idx_ = 0; idx_ < 4; idx_++)
  {
    double vl_inter_x = -(lc[idx_] * vl_idx_b) / (vl_idx_a * lb[idx_] - la[idx_] * vl_idx_b);
    double vl_inter_y = (vl_idx_a * lc[idx_]) / (vl_idx_a * lb[idx_] - la[idx_] * vl_idx_b);

    // calculate the minimum distance from the intersection point (projection line to rectangle edge) to the rectangle corner point.
    double vl2rec_d_min = std::numeric_limits<double>::max();
    int vl2rec_d_min_idx;

    if (has_corner) {
      if (idx_ == neigh_recc_m1 || idx_ == neigh_recc_p1)
        continue;
    }

    for (int k = 0; k < 4; k++)
    {
      double vl2rec_di = (vl_inter_x - rcr_pts[k].x) * (vl_inter_x - rcr_pts[k].x) 
                       + (vl_inter_y - rcr_pts[k].y) * (vl_inter_y - rcr_pts[k].y);
      if (vl2rec_di < vl2rec_d_min || k == 0)
      {
        vl2rec_d_min = vl2rec_di;
        vl2rec_d_min_idx = k;
      }
    }

    // case 1: intersection point is one of the rectangle corner points
    if (vl2rec_d_min < 0.0001)
    {
      neigh_recc_p1 = (vl2rec_d_min_idx + 1) % 4;

      neigh_recc_checked = vl2rec_d_min_idx;

      candi_pL_idx.push_back(neigh_recc_checked);
      candi_pL_idx.push_back(neigh_recc_p1);

      has_corner = true;
    }

    int idx_m1;

    if (idx_ - 1 < 0)
      idx_m1 = 3;
    else
      idx_m1 = idx_ - 1;

    double vl2rec_di_r = (vl_inter_x - rcr_pts[idx_m1].x) * (vl_inter_x - rcr_pts[idx_m1].x) 
                       + (vl_inter_y - rcr_pts[idx_m1].y) * (vl_inter_y - rcr_pts[idx_m1].y);

    double vl2rec_di_l = (vl_inter_x - rcr_pts[idx_].x) * (vl_inter_x - rcr_pts[idx_].x) 
                       + (vl_inter_y - rcr_pts[idx_].y) * (vl_inter_y - rcr_pts[idx_].y);

    double edge_len = (rcr_pts[idx_m1].x - rcr_pts[idx_].x) * (rcr_pts[idx_m1].x - rcr_pts[idx_].x) 
                    + (rcr_pts[idx_m1].y - rcr_pts[idx_].y) * (rcr_pts[idx_m1].y - rcr_pts[idx_].y);
    
    bool is_candidate = false;
    if (std::max(vl2rec_di_l, vl2rec_di_r) < edge_len) {
      is_candidate = true;
      candi_pL_idx.push_back(idx_);
    }
  }

  if (candi_pL_idx.size() == 1){
    projL_idex_ = candi_pL_idx[0];
  }
  else if (candi_pL_idx.size() == 2 && has_corner){
    double x1, x2, y1, y2;

    x1 = ch_cluster[vl_idx_l].x;
    y1 = ch_cluster[vl_idx_l].y;
    x2 = ch_cluster[vl_idx_r].x;
    y2 = ch_cluster[vl_idx_r].y;

    double org_sign = (0 - x1) * (y2 - y1) - (0 - y1) * (x2 - x1);
    double recc_m1_sign = (rcr_pts[neigh_recc_m1].x - x1) * (y2 - y1) - (rcr_pts[neigh_recc_m1].y - y1) * (x2 - x1);
    double recc_p1_sign = (rcr_pts[neigh_recc_p1].x - x1) * (y2 - y1) - (rcr_pts[neigh_recc_p1].y - y1) * (x2 - x1);

    if (org_sign * recc_m1_sign > 0)
      projL_idex_ = (neigh_recc_m1 + 1) % 4;
    else
      projL_idex_ = neigh_recc_p1;
  }
  else{
    double can_vl_inter_d_min = std::numeric_limits<double>::max();
    int can_vl_inter_d_min_idx;

    for (int i = 0; i < candi_pL_idx.size(); i++)
    {
      double can_vl_inter_x_ = -(lc[candi_pL_idx[i]] * vl_idx_b) / (vl_idx_a * lb[candi_pL_idx[i]] - la[candi_pL_idx[i]] * vl_idx_b);
      double can_vl_inter_y_ = (vl_idx_a * lc[candi_pL_idx[i]]) / (vl_idx_a * lb[candi_pL_idx[i]] - la[candi_pL_idx[i]] * vl_idx_b);
      double can_vl_inter_d_ = can_vl_inter_x_ * can_vl_inter_x_ + can_vl_inter_y_ * can_vl_inter_y_;
      if (can_vl_inter_d_ < can_vl_inter_d_min || i == 0)
      {
        can_vl_inter_d_min = can_vl_inter_d_;
        can_vl_inter_d_min_idx = candi_pL_idx[i];
      }
    }
    projL_idex_ = can_vl_inter_d_min_idx;
  }
}


int ConvexOptimizeFitting::index_decrease(const int &index, const int &size)
{
  if (index - 1 < 0)
    return size - 1;
  else
    return index - 1;
}

jsk_recognition_msgs::BoundingBox ConvexOptimizeFitting::jsk_bbox_transform(const obsdet_msgs::DetectedObject &autoware_bbox, 
          const std_msgs::Header& header)
{
  jsk_recognition_msgs::BoundingBox jsk_bbox;
  jsk_bbox.header = header;
  jsk_bbox.pose = autoware_bbox.pose;

  jsk_bbox.dimensions = autoware_bbox.dimensions;
  jsk_bbox.label = autoware_bbox.id;
  jsk_bbox.value = 1.0f;

  return std::move(jsk_bbox);
}

double ConvexOptimizeFitting::Cal_VectorAngle(Eigen::Vector2f &line_vector, Eigen::Vector2f &lineseg)
{
  double angle_acos, angle_trans;
  angle_acos = acos(lineseg.dot(line_vector) / (lineseg.norm() * line_vector.norm()));

  if (line_vector(1) * lineseg(0) - line_vector(0) * lineseg(1) > 0)
    angle_trans = -angle_acos;
  else
    angle_trans = angle_acos;

  return angle_trans;
}

void ConvexOptimizeFitting::calcuBoxbyPolygon(double &theta_star, obsdet_msgs::DetectedObject &output, std::vector<cv::Point2f> &rec_corner_points,
                                              const std::vector<cv::Point2f> &polygon_cluster, const pcl::PointCloud<pcl::PointXYZ> &cluster)
{
  // calc min and max z for cylinder length
  double min_z = 0;
  double max_z = 0;
  for (size_t i = 0; i < cluster.size(); ++i)
  {
    if (cluster.at(i).z < min_z || i == 0)
      min_z = cluster.at(i).z;
    if (max_z < cluster.at(i).z || i == 0)
      max_z = cluster.at(i).z;
  }

  Eigen::Vector2d e_1_star;  
  Eigen::Vector2d e_2_star;
  e_1_star << std::cos(theta_star), std::sin(theta_star);
  e_2_star << -std::sin(theta_star), std::cos(theta_star);
  std::vector<double> C_1_star;
  std::vector<double> C_2_star;

  for (size_t i = 0; i < polygon_cluster.size(); i++)
  {
    C_1_star.push_back(polygon_cluster[i].x * e_1_star.x() + polygon_cluster[i].y * e_1_star.y());
    C_2_star.push_back(polygon_cluster[i].x * e_2_star.x() + polygon_cluster[i].y * e_2_star.y());
  }

  const double min_C_1_star = *std::min_element(C_1_star.begin(), C_1_star.end());
  const double max_C_1_star = *std::max_element(C_1_star.begin(), C_1_star.end());
  const double min_C_2_star = *std::min_element(C_2_star.begin(), C_2_star.end());
  const double max_C_2_star = *std::max_element(C_2_star.begin(), C_2_star.end());

  const double a_1 = std::cos(theta_star);
  const double b_1 = std::sin(theta_star);
  const double c_1 = min_C_1_star;

  const double a_2 = -1.0 * std::sin(theta_star);
  const double b_2 = std::cos(theta_star);
  const double c_2 = min_C_2_star;

  const double a_3 = std::cos(theta_star);
  const double b_3 = std::sin(theta_star);
  const double c_3 = max_C_1_star;

  const double a_4 = -1.0 * std::sin(theta_star);
  const double b_4 = std::cos(theta_star);
  const double c_4 = max_C_2_star;

  // calc center of bounding box
  double intersection_x_1 = (b_1 * c_2 - b_2 * c_1) / (a_2 * b_1 - a_1 * b_2);
  double intersection_y_1 = (a_1 * c_2 - a_2 * c_1) / (a_1 * b_2 - a_2 * b_1);
  double intersection_x_2 = (b_3 * c_4 - b_4 * c_3) / (a_4 * b_3 - a_3 * b_4);
  double intersection_y_2 = (a_3 * c_4 - a_4 * c_3) / (a_3 * b_4 - a_4 * b_3);

  cv::Point2f rec_corner_p1, rec_corner_p2, rec_corner_p3, rec_corner_p4;

  rec_corner_p1.x = intersection_x_1;
  rec_corner_p1.y = intersection_y_1;

  rec_corner_p2.x = (b_2 * c_3 - b_3 * c_2) / (a_3 * b_2 - a_2 * b_3);
  rec_corner_p2.y = (a_2 * c_3 - a_3 * c_2) / (a_2 * b_3 - a_3 * b_2);

  rec_corner_p3.x = intersection_x_2;
  rec_corner_p3.y = intersection_y_2;

  rec_corner_p4.x = (b_1 * c_4 - b_4 * c_1) / (a_4 * b_1 - a_1 * b_4);
  rec_corner_p4.y = (a_1 * c_4 - a_4 * c_1) / (a_1 * b_4 - a_4 * b_1);

  rec_corner_points[0] = rec_corner_p1;
  rec_corner_points[1] = rec_corner_p2;
  rec_corner_points[2] = rec_corner_p3;
  rec_corner_points[3] = rec_corner_p4;

  // calc dimention of bounding box
  Eigen::Vector2d e_x;
  Eigen::Vector2d e_y;
  e_x << a_1 / (std::sqrt(a_1 * a_1 + b_1 * b_1)), b_1 / (std::sqrt(a_1 * a_1 + b_1 * b_1));
  e_y << a_2 / (std::sqrt(a_2 * a_2 + b_2 * b_2)), b_2 / (std::sqrt(a_2 * a_2 + b_2 * b_2));
  Eigen::Vector2d diagonal_vec;
  diagonal_vec << intersection_x_1 - intersection_x_2, intersection_y_1 - intersection_y_2;

  // calc yaw
  tf2::Quaternion quat;
  quat.setEuler(/* roll */ 0, /* pitch */ 0, /* yaw */ std::atan2(e_1_star.y(), e_1_star.x()));

  output.pose.position.x = (intersection_x_1 + intersection_x_2) / 2.0;
  output.pose.position.y = (intersection_y_1 + intersection_y_2) / 2.0;
  output.pose.position.z = (max_z + min_z) / 2.0;
  output.pose.orientation = tf2::toMsg(quat);

  double ep = 0.01;
  output.dimensions.x = std::fabs(e_x.dot(diagonal_vec));
  output.dimensions.y = std::fabs(e_y.dot(diagonal_vec));
  output.dimensions.z = std::max((max_z - min_z), ep);
  output.pose_reliable = true;

  output.dimensions.x = std::max(output.dimensions.x, ep);
  output.dimensions.y = std::max(output.dimensions.y, ep);
  output.dimensions.z = std::max(output.dimensions.z, ep);
}

void ConvexOptimizeFitting::eval_running_time(int running_time)
{
  double runtime_std;
  double runtime_sqr_sum = 0.0;
  double runtime_aver;

  runtime_vector.push_back(running_time);

  double runtime_total_v = 0.0;

  for (size_t i = 0; i < runtime_vector.size(); i++)
  {
    runtime_total_v += runtime_vector[i];
  }

  runtime_aver = runtime_total_v / runtime_vector.size();

  for (size_t i = 0; i < runtime_vector.size(); i++)
  {
    runtime_sqr_sum += (runtime_vector[i] - runtime_aver) * (runtime_vector[i] - runtime_aver);
  }

  runtime_std = sqrt(runtime_sqr_sum / runtime_vector.size());

  std::cout << "runtime_vector.size() is = " << runtime_vector.size() << std::endl;
  std::cout << "running_time is = " << running_time / 1000.0 << std::endl;
  std::cout << "runtime_aver is = " << runtime_aver / 1000.0 << std::endl;
  std::cout << "runtime_std is = " << runtime_std / 1000.0 << std::endl;
  std::cout << "---------------------------------" << std::endl;
}

void ConvexOptimizeFitting::eval_performance(double &theta_kitti, double &theta_optim, const uint32_t &index, const uint32_t &index_seq)
{
  double bias_org = std::fabs(theta_kitti - theta_optim);

  double bias = std::min(bias_org, M_PI / 2 - bias_org);

  double bias_std;
  double bias_sqr_sum = 0.0;
  double aver_accu;

  bias_vector.push_back(bias);

  double bias_total_v = 0.0;

  for (size_t i = 0; i < bias_vector.size(); i++)
  {
    bias_total_v += bias_vector[i];
  }
  aver_accu = bias_total_v / bias_vector.size();

  for (size_t i = 0; i < bias_vector.size(); i++)
  {
    bias_sqr_sum += (bias_vector[i] - aver_accu) * (bias_vector[i] - aver_accu);
  }

  bias_std = sqrt(bias_sqr_sum / bias_vector.size());

  // Absolute orientation error distribution
  double bias_deg = bias * 180 / M_PI;
  if (bias_deg < 0.5)
    error_less_0_5++;

  if (bias_deg < 1.0)
    error_less_1_0++;

  if (bias_deg < 2.0)
    error_less_2_0++;

  if (bias_deg < 5.0)
    error_less_5_0++;

  if (bias_deg < 10.0)
    error_less_10_0++;

  std::cout << "index is = " << index << std::endl;
  std::cout << "index_seq is = " << index_seq << std::endl;
  std::cout << "theta_kitti is = " << theta_kitti << std::endl;
  std::cout << "theta_optim is = " << theta_optim << std::endl;

  std::cout << "less than 0.5 is = " << error_less_0_5 / (bias_vector.size() * 1.0) << std::endl;
  std::cout << "less than 1.0 is = " << error_less_1_0 / (bias_vector.size() * 1.0) << std::endl;
  std::cout << "less than 2.0 is = " << error_less_2_0 / (bias_vector.size() * 1.0) << std::endl;
  std::cout << "less than 5.0 is = " << error_less_5_0 / (bias_vector.size() * 1.0) << std::endl;
  std::cout << "less than 10.0 is = " << error_less_10_0 / (bias_vector.size() * 1.0) << std::endl;

  std::cout << "bias_total_v is = " << bias_total_v * 180 / M_PI << std::endl;
  std::cout << "bias_vector[] is = " << bias_vector[bias_vector.size() - 1] << std::endl;
  std::cout << "bias_vector size is = " << bias_vector.size() << std::endl;
  std::cout << "bias_std is = " << bias_std * 180 / M_PI << std::endl;
  std::cout << "aver_accy is = " << aver_accu * 180 / M_PI << std::endl;
  std::cout << "bias is = " << bias * 180 / M_PI << std::endl;
  std::cout << "---------------------------------" << std::endl;

  std::stringstream obstacle_info;
  std_msgs::String obstacle_info_rviz;
  obstacle_info << "index is = " << index << "\n"
                << "index_seq is = " << index_seq << "\n"
                << "theta_kitti is = " << theta_kitti << "\n"
                << "theta_optim is = " << theta_optim << "\n"

                << "bias_std is = " << bias_std * 180 / M_PI << "\n"
                << "aver_accy is = " << aver_accu * 180 / M_PI << "\n"
                << "bias is = " << bias * 180 / M_PI << "\n";

  obstacle_info_rviz.data = obstacle_info.str();
  pub_result_info_rviz_.publish(obstacle_info_rviz);


  std::string filename = "/home/dnn/paper_pose_est/convex_optimize_fitting_improve/src/convex_optimize_fitting/evaluation/bias.txt";
  if (boost::filesystem::exists(filename) && first_appear)
  {
    boost::filesystem::remove(filename);
    first_appear = false;
  }

  std::ofstream out_txt(filename, std::ios::app);
  if (bias * 180 / M_PI > 8.0)
    out_txt << index << " " << index_seq << " " << bias * 180 / M_PI << std::endl;
  out_txt.close();
}

void ConvexOptimizeFitting::MainLoop(const obsdet_msgs::CloudClusterArray& in_cluster_kitti_ros)
{
  start_time = std::chrono::system_clock::now();

  obsdet_msgs::DetectedObjectArray out_object_array;

  pcl::PointCloud<pcl::PointXYZI>::Ptr corner_points_visual(new pcl::PointCloud<pcl::PointXYZI>());

  pcl::PointCloud<pcl::PointXYZI>::Ptr convex_corner_visual(new pcl::PointCloud<pcl::PointXYZI>());

  pcl::PointCloud<pcl::PointXYZI>::Ptr min_reccorner_visual(new pcl::PointCloud<pcl::PointXYZI>());

  out_object_array.header = in_cluster_kitti_ros.header;

  int intensity_mark = 1;


  jsk_recognition_msgs::BoundingBox jsk_bbox;
  jsk_recognition_msgs::BoundingBoxArray jsk_bbox_array;

/*----------------------------------transform the bounding box to target frame.-------------------------------------------*/
  geometry_msgs::TransformStamped transform_stamped;
  geometry_msgs::Pose pose, pose_transformed;
  auto bbox_header = in_cluster_kitti_ros.header;
  bbox_source_frame_ = bbox_header.frame_id;
  bbox_header.frame_id = bbox_target_frame_;
  jsk_bbox_array.header = bbox_header;

  try
  {
    transform_stamped = tf2_buffer.lookupTransform(bbox_target_frame_, bbox_source_frame_, ros::Time());
    // ROS_INFO("target_frame is %s",bbox_target_frame_.c_str());
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
    ROS_WARN("Frame Transform Given Up! Outputing obstacles in the original LiDAR frame %s instead...", bbox_source_frame_.c_str());
    bbox_header.frame_id = bbox_source_frame_;
    try
    {
      transform_stamped = tf2_buffer.lookupTransform(bbox_source_frame_, bbox_source_frame_, ros::Time(0));
    }
    catch (tf2::TransformException& ex2)
    {
      ROS_ERROR("%s", ex2.what());
      return;
    }
  }
/*-----------------------------------------------------------------------------------------------------------------------*/
  obsdet_msgs::local_obstacle_info obstacle_corner_info;

  jsk_recognition_msgs::PolygonArray clusters_polygon_;



/*---------------------------------------------------------------------------------------*/
  for (const auto &in_cluster : in_cluster_kitti_ros.clusters)
  {
    // eval_running_time
    const auto eval_start_time = std::chrono::system_clock::now();
    
    pcl::PointCloud<pcl::PointXYZ> current_cluster;
    pcl::fromROSMsg(in_cluster.cloud, current_cluster);
    double theta_optim;
    std::vector<cv::Point2f> ch_cluster;

    if (current_cluster.size() < 10)
      return;

    // project the 3D cluster pc into 2D pc
    std::vector<cv::Point2f> current_points;
    for (unsigned int i = 0; i < current_cluster.points.size(); i++)
    {
      cv::Point2f pt;
      pt.x = current_cluster.points[i].x;
      pt.y = current_cluster.points[i].y;
      current_points.push_back(pt);
    }

    // extract the hull point from 2D cluster using Graham scan algorithm
    cv::convexHull(current_points, ch_cluster);

    double angle_valid;
    double min_angle = std::numeric_limits<double>::max();
    double max_angle = std::numeric_limits<double>::min();

    int vl_idx_l;
    int vl_idx_r;

    for (int i = 0; i < ch_cluster.size(); i++)
    {
      angle_valid = atan2(ch_cluster[i].y, ch_cluster[i].x);
      if (angle_valid < min_angle || i == 0)
      {
        min_angle = angle_valid;
        vl_idx_l = i;
      }
      if (angle_valid > max_angle || i == 0)
      {
        max_angle = angle_valid;
        vl_idx_r = i;
      }
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr pt_test_visual(new pcl::PointCloud<pcl::PointXYZI>());

    int projL_idex_l_o, projL_idex_r_o;

    std::vector<cv::Point2f> rcr_pts_o;
    rcr_pts_o.resize(4);

    optim_convex_fitting_improve(ch_cluster, vl_idx_l, vl_idx_r, theta_optim, 
                                projL_idex_l_o, projL_idex_r_o, rcr_pts_o, pt_test_visual);

    const auto eval_end_time = std::chrono::system_clock::now();
    const auto eval_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(eval_end_time - eval_start_time);
    eval_running_time(eval_exe_time.count());

    pcl::PointCloud<pcl::PointXYZI>::Ptr ch_cluster_pcl(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointXYZI ch_cluster_pt;
    for (int i = 0; i < ch_cluster.size(); i++)
    {
      ch_cluster_pt.x = ch_cluster[i].x;
      ch_cluster_pt.y = ch_cluster[i].y;
      ch_cluster_pt.z = 0.0;
      ch_cluster_pt.intensity = 1;
      ch_cluster_pcl->points.push_back(ch_cluster_pt);
    }
    sensor_msgs::PointCloud2 ch_cluster_ros;
    pcl::toROSMsg(*ch_cluster_pcl, ch_cluster_ros);
    ch_cluster_ros.header.frame_id = "map";
    ch_cluster_ros.header.stamp = ros::Time::now();
    pub_ch_cluster_visual_.publish(ch_cluster_ros);


    sensor_msgs::PointCloud2 pt_test_visual_ros;
    pcl::toROSMsg(*pt_test_visual, pt_test_visual_ros);
    pt_test_visual_ros.header.frame_id = "map";
    pt_test_visual_ros.header.stamp = ros::Time::now();
    pub_pt_test_visual_.publish(pt_test_visual_ros);

    /*----------------------------------------*/
    int projL_idex_l_m1;
    int projL_idex_r_m1;    
    visualization_msgs::Marker proj_line_l;
    visualization_msgs::Marker proj_line_r;
    visualization_msgs::MarkerArray proj_line_;
    geometry_msgs::Point pt_l_1;
    geometry_msgs::Point pt_r_1;
    pt_l_1.x = rcr_pts_o[projL_idex_l_o].x;
    pt_l_1.y = rcr_pts_o[projL_idex_l_o].y;
    pt_l_1.z = 0.0;
    pt_r_1.x = rcr_pts_o[projL_idex_r_o].x;
    pt_r_1.y = rcr_pts_o[projL_idex_r_o].y;
    pt_r_1.z = 0.0;

    if (projL_idex_l_o - 1 < 0)
      projL_idex_l_m1 = 3;
    else
      projL_idex_l_m1 = projL_idex_l_o - 1;

    if (projL_idex_r_o - 1 < 0)
      projL_idex_r_m1 = 3;
    else
      projL_idex_r_m1 = projL_idex_r_o - 1;

    geometry_msgs::Point pt_l_2;
    geometry_msgs::Point pt_r_2;  
    pt_l_2.x = rcr_pts_o[projL_idex_l_m1].x;
    pt_l_2.y = rcr_pts_o[projL_idex_l_m1].y;
    pt_l_2.z = 0.0;

    pt_r_2.x = rcr_pts_o[projL_idex_r_m1].x;
    pt_r_2.y = rcr_pts_o[projL_idex_r_m1].y;
    pt_r_2.z = 0.0;

    proj_line_l.header.frame_id = "map";
    proj_line_l.header.stamp = ros::Time::now();
    proj_line_l.action = visualization_msgs::Marker::ADD;
    proj_line_l.type = visualization_msgs::Marker::LINE_LIST;
    proj_line_l.id = 0;
    proj_line_l.scale.x = 0.03;
    proj_line_l.color.r = 0.0;
    proj_line_l.color.g = 0.545;
    proj_line_l.color.b = 0.545;
    proj_line_l.color.a = 1.0;
    proj_line_l.pose.orientation.x = 0.0;
    proj_line_l.pose.orientation.y = 0.0;
    proj_line_l.pose.orientation.z = 0.0;
    proj_line_l.pose.orientation.w = 1.0;

    proj_line_l.points.push_back(pt_l_1);
    proj_line_l.points.push_back(pt_l_2);
    proj_line_.markers.push_back(proj_line_l);

    proj_line_r.header.frame_id = "map";
    proj_line_r.header.stamp = ros::Time::now();
    proj_line_r.action = visualization_msgs::Marker::ADD;
    proj_line_r.type = visualization_msgs::Marker::LINE_LIST;
    proj_line_r.id = 1;
    proj_line_r.scale.x = 0.03;
    proj_line_r.color.r = 0.0;
    proj_line_r.color.g = 0.545;
    proj_line_r.color.b = 0.545;
    proj_line_r.color.a = 1.0;
    proj_line_r.pose.orientation.x = 0.0;
    proj_line_r.pose.orientation.y = 0.0;
    proj_line_r.pose.orientation.z = 0.0;
    proj_line_r.pose.orientation.w = 1.0;

    proj_line_r.points.push_back(pt_r_1);
    proj_line_r.points.push_back(pt_r_2);
    proj_line_.markers.push_back(proj_line_r);

    pub_proj_line_.publish(proj_line_);


    obsdet_msgs::DetectedObject output_object;
    std::vector<cv::Point2f> rec_corner_points(4);

    double theta_kitti = in_cluster.orientation;

    eval_performance(theta_kitti, theta_optim, in_cluster.index, in_cluster.index_seq);

    calcuBoxbyPolygon(theta_optim, output_object, rec_corner_points, ch_cluster, current_cluster);

    // transform the bounding box
    pose.position = output_object.pose.position;
    pose.orientation = output_object.pose.orientation;
    tf2::doTransform(pose, pose_transformed, transform_stamped);
    output_object.header = bbox_header;
    output_object.pose = pose_transformed;

    // copy the autoware box to jsk box
    jsk_bbox = jsk_bbox_transform(output_object, bbox_header);
    jsk_bbox_array.boxes.push_back(jsk_bbox);

    // push the autoware bounding box in the array
    out_object_array.objects.push_back(output_object);

    // visulization the rectangle four corner points
    pcl::PointXYZI rec_corner_pt;

    // visulization the convex hull
    geometry_msgs::PolygonStamped cluster_polygon_;

    for (size_t i = 0; i < ch_cluster.size(); i++)
    {
      geometry_msgs::Point32 point;
      point.x = ch_cluster[i % ch_cluster.size()].x;
      point.y = ch_cluster[i % ch_cluster.size()].y;
      point.z = 0;
      cluster_polygon_.polygon.points.push_back(point);
      cluster_polygon_.header.stamp = ros::Time::now();
      cluster_polygon_.header.frame_id = in_cluster_kitti_ros.header.frame_id;
    }
    clusters_polygon_.polygons.push_back(cluster_polygon_);

    for (int i = 0; i < 4; i++)
    {
      rec_corner_pt.x = rec_corner_points[i].x;
      rec_corner_pt.y = rec_corner_points[i].y;
      rec_corner_pt.z = output_object.pose.position.z + 0.5 * output_object.dimensions.z;
      rec_corner_pt.intensity = intensity_mark;
      corner_points_visual->push_back(rec_corner_pt);

      // custom obstacle information
      obstacle_corner_info.x.push_back(rec_corner_points[i].x);
      obstacle_corner_info.y.push_back(rec_corner_points[i].y);
    }
    intensity_mark++;

    // visulize the pt in angle limit
    pcl::PointCloud<pcl::PointXYZI>::Ptr angle_valid_pt2(new pcl::PointCloud<pcl::PointXYZI>());

    pcl::PointXYZI angle_lim_pt;
    angle_lim_pt.x = ch_cluster[vl_idx_l].x;
    angle_lim_pt.y = ch_cluster[vl_idx_l].y;
    angle_lim_pt.z = 0;
    angle_lim_pt.intensity = 1;
    angle_valid_pt2->push_back(angle_lim_pt);

    angle_lim_pt.x = ch_cluster[vl_idx_r].x;
    angle_lim_pt.y = ch_cluster[vl_idx_r].y;
    angle_lim_pt.z = 0;
    angle_lim_pt.intensity = 2;
    angle_valid_pt2->push_back(angle_lim_pt);

    visualization_msgs::Marker line_;
    visualization_msgs::MarkerArray line_array;

    geometry_msgs::Point pt_origin;
    pt_origin.x = 0.0;
    pt_origin.y = 0.0;
    pt_origin.z = 0.0;

    geometry_msgs::Point pt;
    pt.x = ch_cluster[vl_idx_l].x;
    pt.y = ch_cluster[vl_idx_l].y;
    pt.z = 0.0;
    line_.header.frame_id = "map";
    line_.header.stamp = ros::Time::now();
    line_.action = visualization_msgs::Marker::ADD;
    line_.type = visualization_msgs::Marker::LINE_LIST;
    line_.id = 0;
    line_.scale.x = 0.02;
    line_.color.r = 0.0;
    line_.color.g = 0.8078;
    line_.color.b = 0.8196;
    line_.color.a = 1.0;
    line_.pose.orientation.x = 0.0;
    line_.pose.orientation.y = 0.0;
    line_.pose.orientation.z = 0.0;
    line_.pose.orientation.w = 1.0;

    line_.points.push_back(pt_origin);
    line_.points.push_back(pt);
    line_array.markers.push_back(line_);

    pt.x = ch_cluster[vl_idx_r].x;
    pt.y = ch_cluster[vl_idx_r].y;
    pt.z = 0;
    line_.id = 1;

    line_.color.r = 0.0;
    line_.color.g = 0.8078;
    line_.color.b = 0.8196;
    line_.color.a = 1.0;
    line_.points.clear();
    line_.points.push_back(pt_origin);
    line_.points.push_back(pt);

    line_array.markers.push_back(line_);
    pub_line_array_.publish(line_array);

    // visualize the lim convex hull points
    sensor_msgs::PointCloud2 angle_valid_pt2_ros;
    pcl::toROSMsg(*angle_valid_pt2, angle_valid_pt2_ros);
    angle_valid_pt2_ros.header.frame_id = "map";
    angle_valid_pt2_ros.header.stamp = ros::Time::now();
    pub_angle_valid_pt2_.publish(angle_valid_pt2_ros);
  }

/*---------------------------------------------------------------------------------------*/
  obstacle_corner_info.num = intensity_mark - 1;
  pub_local_obstacle_info_.publish(obstacle_corner_info);

  out_object_array.header = bbox_header;
  pub_autoware_bboxs_array_.publish(out_object_array);

  jsk_bbox_array.header = bbox_header;
  pub_jsk_bboxs_array_.publish(jsk_bbox_array);
  
  clusters_polygon_.header.frame_id = "map";
  clusters_polygon_.header.stamp = ros::Time::now();
  pub_cluster_polygon_.publish(clusters_polygon_);

  // rectangle corner points
  sensor_msgs::PointCloud2 corner_points_visual_ros;
  pcl::toROSMsg(*corner_points_visual, corner_points_visual_ros);
  corner_points_visual_ros.header = in_cluster_kitti_ros.header;
  pub_rec_corner_points_.publish(corner_points_visual_ros);

  // rectangle corner points
  sensor_msgs::PointCloud2 convex_corner_visual_ros;
  pcl::toROSMsg(*convex_corner_visual, convex_corner_visual_ros);
  convex_corner_visual_ros.header = in_cluster_kitti_ros.header;
  pub_convex_corner_points_.publish(convex_corner_visual_ros);

  // rectangle corner points
  sensor_msgs::PointCloud2 min_reccorner_visual_ros;
  pcl::toROSMsg(*min_reccorner_visual, min_reccorner_visual_ros);
  min_reccorner_visual_ros.header = in_cluster_kitti_ros.header;
  pub_min_reccorner_points_.publish(min_reccorner_visual_ros);

  end_time = std::chrono::system_clock::now();
  exe_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
  time_spent.data = exe_time;
  pub_jskrviz_time_.publish(time_spent);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "convex_optimize_fitting_node");
  ConvexOptimizeFitting app;
  ros::spin();
  return 0;
}