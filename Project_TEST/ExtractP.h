
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void resize_limit(Mat src, Mat &dist, int limit, bool body_or_not, Rect & body_reduced);
string get_image_name();
string save_image_name();
void reduceColor_Quantization(const Mat& src, Mat& dst);
void reduceColor_Stylization(const Mat& src, Mat& dst);
void reduceColor_EdgePreserving(const Mat& src, Mat& dst);
void reduceColor_kmeans(const Mat& src, Mat& dst, int K);
double dist_3D(double in1, double in2, double in3);
void distanceMatrix(Vec3d* src, int len);
void dist_merge_update(int trial, double(*src)[30], int * weight, Vec3b * colors_all, bool(*mark)[30],int K, int area);
void dist_merge_fake(int trial, double(*src)[30], int * weight, Vec3b * colors_all, bool(*mark)[30],int K, int area);
void dist_gravity_update(int trial, double(*src)[30], int * weight, Vec3b * colors_all, bool(*mark)[30], int K, int area);
void dist_merge_update_lab(int trial, double(*src)[30], int * weight, Vec3d * colors_all, bool(*mark)[30], int K, int area);
void dist_merge_fake_lab(int trial, double(*src)[30], int * weight, Vec3d * colors_all, bool(*mark)[30], int K, int area, bool *valid);
void dist_gravity_update_lab(int trial, double(*src)[30], int * weight, Vec3d * colors_all, bool(*mark)[30], int K, int area);
void dist_gravity_fake_lab(int trial, double(*src)[30], int * weight, Vec3d * colors_all, bool(*mark)[30], int K, int area, bool*valid, bool body_or_not, int prempt);
void bgr2lab(Vec3b * colors_all, Vec3d * colors_lab_all, int k);
void lab2bgr(Vec3b * colors_all, Vec3d * colors_lab_all, int k);
void noise_canceling_3b(double(*src)[30], int * weight, Vec3b * colors_all, bool(*mark)[30], int K, int flag, int type);
void noise_canceling_3d(double(*src)[30], int * weight, Vec3d * colors_all, bool(*mark)[30], int K, int flag, int type, bool*valid);
void rgb2cmyk(int r, int g, int b, int c, double m, double y, double k);
void show(Mat k_mean, Mat drawCircle_multi, Mat drawDomAssAcc);
void bgr2S(Vec3b *colors_all, double *sats, int k);
int eliminate_low_sats(double * sats, int *weight, int k, bool* valid);
void HOG_Human(Mat inputImg);
void dom_ass_acc(Vec3b * colors_all, int numbers, int * weight, Vec3b& dom, Vec3b& ass, Vec3b& acc, int K, int area, bool* valid, bool body_or_not, bool domflag, bool assflag, bool accflag, int prempt);
void bgr2lab_single(Vec3b color, Vec3d& colors_lab);
void lab2bgr_single(Vec3b& color, Vec3d& colors_lab);
void assort(Vec3b* colors_all, int *weight, int trial, Vec3b& ass, bool*valid, bool * isItPoss, int K);
void assort_sup(Vec3b* colors_all, int *weight, int dom_hue, int acc_hue, Vec3b& ass, Vec3b &acc, int*hue, bool*valid, bool * isItPoss, int K, int area, int acc_weight);
void bgr2hue(Vec3b*colors_all, int* hue, int k, bool*valid);
void cascade_full(Mat img);
void cascade_upper(Mat img);
void cascade_face(Mat img, vector<Rect> face_pos);
vector<Rect> body_infer(Mat img);
Rect roi_body(Mat img, Rect& body);
bool main_body(Mat img, Rect &body);
void color_sample(int type, int r1, int g1, int b1, int r2, int g2, int b2, int r3, int g3, int b3);
void color_box_design(Vec3b dom, Vec3b ass, Vec3b acc);
void color_check_design(Vec3b dom, Vec3b ass, Vec3b acc);
void vector_clear(vector<int>& m);
Mat temp_make(Mat img, Vec3b dom, Vec3b ass, Vec3b acc);
void dist_gravity_fake_lab_video(int trial, vector<int>weight, vector <Vec3d>& colors_all, vector <vector<bool>> mark, int K, int area);