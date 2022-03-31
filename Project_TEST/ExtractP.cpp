#include <iostream>
#include <opencv2\opencv.hpp>
#include <string.h>
#include <map>
#include <opencv2\photo.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include "ExtractP.h"
#include <math.h>

#define Low_thresh 30

using namespace std;
using namespace cv;

String body_cascade = "C:/opencv/sources/data/haarcascades/haarcascade_fullbody.xml";
String upper_body_cascade = "C:/opencv/sources/data/haarcascades/haarcascade_upperbody.xml";
String face_cascade = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier body;
CascadeClassifier upper_body;
CascadeClassifier face;


string get_image_name() {
	cout << "Enter the name of file" << endl;
	string name;
	cin >> name;
	return name;
}

string save_image_name() {
	cout << "Enter the name of file to save" << endl;
	string name;
	cin >> name;
	return name;
}

void resize_limit(Mat src, Mat &dist, int limit, bool body_or_not, Rect & body_reduced) {
	int height = src.rows;
	int width= src.cols;
	int size = height*width;

	double temp;
	double scale=1.0;
	if (size <= limit);
	else {
		temp = (double)limit / size;
		scale = floor((temp)* pow(float(10), 2) + 0.5f) / pow(float(10), 2);
		scale = temp;
	}
	resize(src, dist, Size(), scale,scale, CV_INTER_AREA);
	if (body_or_not) {
		body_reduced = Rect(body_reduced.x*scale, body_reduced.y*scale, body_reduced.width*scale, body_reduced.height*scale);
	}
}

void reduceColor_Quantization(const Mat& src, Mat& dst)
{
	uchar N = 64;
	dst = src / N;
	dst *= N;
}

void reduceColor_Stylization(const Mat& src, Mat& dst)
{
	stylization(src, dst);
}

void reduceColor_EdgePreserving(const Mat& src, Mat& dst)
{
	edgePreservingFilter(src, dst);
}

void reduceColor_kmeans(const Mat& src, Mat& dst,int K) {
	Mat points, labels, centers;
	int width, height, x, y, n, nPoints, cIndex, iTemp;

	width = src.cols;
	height = src.rows;
	nPoints = width * height;

	//Initialize
	points.create(nPoints, 1, CV_32FC3);
	centers.create(K, 1, points.type());
	dst.create(height, width, src.type());

	//Vector transformation
	for (y = 0, n = 0; y < height; y++) {
		for (x = 0; x < width; x++, n++) {
			points.at<Vec3f>(n)[0] = src.at<Vec3b>(y, x)[0];
			points.at<Vec3f>(n)[1] = src.at<Vec3b>(y, x)[1];
			points.at<Vec3f>(n)[2] = src.at<Vec3b>(y, x)[2];
		}
	}

	// k-means clustering
	kmeans(points, K, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 3.0), 3, KMEANS_PP_CENTERS, centers);

	for (y = 0, n = 0; y < height; y++) {
		for (x = 0; x < width; x++, n++) {
			cIndex = labels.at<int>(n);
			iTemp = cvRound(centers.at<Vec3f>(cIndex)[0]);
			iTemp = iTemp > 255 ? 255 : iTemp < 0 ? 0 : iTemp;
			dst.at<Vec3b>(y, x)[0] = (uchar)iTemp;

			cIndex = labels.at<int>(n);
			iTemp = cvRound(centers.at<Vec3f>(cIndex)[1]);
			iTemp = iTemp > 255 ? 255 : iTemp < 0 ? 0 : iTemp;
			dst.at<Vec3b>(y, x)[1] = (uchar)iTemp;

			cIndex = labels.at<int>(n);
			iTemp = cvRound(centers.at<Vec3f>(cIndex)[2]);
			iTemp = iTemp > 255 ? 255 : iTemp < 0 ? 0 : iTemp;
			dst.at<Vec3b>(y, x)[2] = (uchar)iTemp;
		}
	}
}

void distanceMatrix(Vec3d* src, int len) {
	int i, j;
	int **distMat = new int*[len];
	for (i = 0; i < len; ++i)
		distMat[i] = new int[len];

	for (i = 0; i < len; ++i) {
		
		cout << (int)src[i].val[0] << " " << (int)src[i].val[1] << " " << (int)src[i].val[2] << endl;
		for (j = 0; j < len; ++j) {
			cout << (int)src[j].val[0] << " " << (int)src[j].val[1] << " " << (int)src[j].val[2] << endl;
			cout << "i = " << i << "j = " << j << endl;
			distMat[i][j] = dist_3D(src[i].val[0]-src[j].val[0], src[i].val[1] - src[j].val[1], src[i].val[2] - src[j].val[2]);
			cout <<"distMat : "<< distMat[i][j] << endl;
		}
	}
}

double dist_3D(double in1, double in2, double in3) {
	return sqrt(in1*in1 + in2*in2 + in3*in3);
}

void dist_merge_update(int trial, double(*src)[30], int * weight, Vec3b * colors_all, bool(*mark)[30], int K, int area) {
	//Condition
	if (trial == 0) return;
	double min = INT_MAX;
	int x, y;
	int i,j;
	for (i = 0; i < K; i++) {
		for (j = i + 1; j < K; j++) {
			if (src[i][j] == 0) continue;
			if (min > src[i][j]) {
				min = src[i][j];
				x = i;
				y = j;
			}
		}
	}
	cout << "this time, " << x << "and" << y << "is merged" << endl;

	src[x][y] = INT_MAX;
	src[y][x] = INT_MAX;
	

	if (weight[x] > weight[y]) {
		mark[x][y] = true;
		for (i = 0; i < K; ++i) {
			if (mark[y][i]) mark[x][i] = true;
			if (mark[x][i]) {
				colors_all[i] = colors_all[x];
			}
		}

		for (i = 0; i < K; ++i) {
			src[i][y] = INT_MAX;
			src[y][i] = INT_MAX;
		}
		weight[x] += weight[y];
	}
	else {
		mark[y][x] = true;
		for (i = 0; i < K; ++i) {
			if (mark[x][i]) mark[y][i] = true;
			if (mark[y][i]) {
				colors_all[i] = colors_all[y];
			}
		}
		for (i = 0; i <K; ++i) {
			src[i][x] = INT_MAX;
			src[x][i] = INT_MAX;
		}
		weight[y] += weight[x];
	}

	//Recursion
	dist_merge_update(trial - 1, src, weight, colors_all,mark,K,area);
}

void dist_merge_update_lab(int trial, double(*src)[30], int * weight, Vec3d * colors_all, bool(*mark)[30], int K, int area) {
	//Condition
	if (trial == 0) return;
	double min = INT_MAX;
	int x, y;
	int i, j;
	for (i = 0; i < K; i++) {
		for (j = i + 1; j < K; j++) {
			if (src[i][j] == 0) continue;
			if (min > src[i][j]) {
				min = src[i][j];
				x = i;
				y = j;
			}
		}
	}
	cout << "this time, " << x << "and" << y << "is merged" << endl;

	src[x][y] = INT_MAX;
	src[y][x] = INT_MAX;


	if (weight[x] > weight[y]) {
		mark[x][y] = true;
		for (i = 0; i < K; ++i) {
			if (mark[y][i]) mark[x][i] = true;
			if (mark[x][i]) {
				colors_all[i] = colors_all[x];
			}
		}

		for (i = 0; i < K; ++i) {
			src[i][y] = INT_MAX;
			src[y][i] = INT_MAX;
		}
		weight[x] += weight[y];
	}
	else {
		mark[y][x] = true;
		for (i = 0; i < K; ++i) {
			if (mark[x][i]) mark[y][i] = true;
			if (mark[y][i]) {
				colors_all[i] = colors_all[y];
			}
		}
		for (i = 0; i <K; ++i) {
			src[i][x] = INT_MAX;
			src[x][i] = INT_MAX;
		}
		weight[y] += weight[x];
	}

	//Recursion
	dist_merge_update_lab(trial - 1, src, weight, colors_all, mark, K,area);
}

void dist_merge_fake(int trial, double(*src)[30], int * weight, Vec3b * colors_all, bool(*mark)[30], int K, int area) {
	if (trial == 0) return;
	double min = INT_MAX;
	int x, y;
	int i, j;
	//New var
	int merged_weight;
	Vec3b merged_color;
	vector<int> x_meme;
	vector<int> y_meme;

	for (i = 0; i < K; i++) {
		for (j = i + 1; j < K; j++) {
			if (mark[i][j]) continue;
			else{
				if (min > src[i][j]) {
					min = src[i][j];
					x = i;
					y = j;
				}
			}
		}
	}
	cout << "this time, " << x << "and" << y << "is merged" << endl;
	
	//Get the marks
	x_meme.push_back(x);
	y_meme.push_back(y);

	for (i = 0; i < K; ++i) {
		if (mark[x][i]) x_meme.push_back(i);
		if (mark[y][i]) y_meme.push_back(i);
	}

	//Merge the mark
	for (i = 0; i<x_meme.size(); i++) { 
		for (j = 0; j < y_meme.size(); j++) {
			mark[x_meme[i]][y_meme[j]] = true;
			mark[y_meme[j]][x_meme[i]] = true;
		}
	}
	//Merge the color
	if (weight[x] > weight[y]) {
		merged_color = colors_all[x];
	}
	else {
		merged_color = colors_all[y];
	}
	colors_all[x] = merged_color;
	for (i = 0; i < K; ++i) {
		if (mark[x][i]) colors_all[i] = merged_color;
	}

	//Merge the weight
	merged_weight = weight[x] + weight[y];
	weight[x] = merged_weight;
	for (i = 0; i < K; ++i) {
		if (mark[x][i]) weight[i] = merged_weight;
	}

	//Recursion
	dist_merge_fake(trial - 1, src, weight, colors_all, mark,K, area);
}

void dist_merge_fake_lab(int trial, double(*src)[30], int * weight, Vec3d * colors_all, bool(*mark)[30], int K, int area, bool*valid) {
	if (trial == 0) return;
	double min = INT_MAX;
	int x, y;
	int i, j;
	//New var
	int merged_weight;
	Vec3d merged_color;
	vector<int> x_meme;
	vector<int> y_meme;

	//Noise update
	if (trial < 5) {
		double threshold = 0.5;
		double noise = INT_MAX;
		double curr;
		int flag;
		for (int i = 0; i < K; ++i) {
			curr = (double)100*(double)weight[i] / (double)area;
			
			if (curr < noise) {
				noise = curr;
				flag = i;
			}
		}

		if (noise < threshold) {
			noise_canceling_3d(src, weight, colors_all, mark, K, flag,0,valid); 
			dist_merge_fake_lab(trial - 1, src, weight, colors_all, mark, K, area,valid);
			return;
		}
	}
	//Noise update ends;

	for (i = 0; i < K; i++) {
		for (j = i + 1; j < K; j++) {
			if (mark[i][j]) continue;
			else {
				if (min > src[i][j]) {
					min = src[i][j];
					x = i;
					y = j;
				}
			}
		}
	}
	cout << "this time, " << x << "and" << y << "is merged" << endl;

	//Get the marks
	x_meme.push_back(x);
	y_meme.push_back(y);

	for (i = 0; i < K; ++i) {
		if (mark[x][i]) x_meme.push_back(i);
		if (mark[y][i]) y_meme.push_back(i);
	}

	//Merge the mark
	for (i = 0; i<x_meme.size(); i++) {
		for (j = 0; j < y_meme.size(); j++) {
			mark[x_meme[i]][y_meme[j]] = true;
			mark[y_meme[j]][x_meme[i]] = true;
		}
	}
	//Merge the color
	if (weight[x] > weight[y]) {
		merged_color = colors_all[x];
	}
	else {
		merged_color = colors_all[y];
	}
	colors_all[x] = merged_color;
	for (i = 0; i < K; ++i) {
		if (mark[x][i]) colors_all[i] = merged_color;
	}

	//Merge the weight
	merged_weight = weight[x] + weight[y];
	weight[x] = merged_weight;
	for (i = 0; i < K; ++i) {
		if (mark[x][i]) weight[i] = merged_weight;
	}
	//Recursion
	dist_merge_fake_lab(trial - 1, src, weight, colors_all, mark, K, area, valid);
}

void dist_gravity_update(int trial, double(*src)[30], int * weight, Vec3b * colors_all, bool(*mark)[30], int K, int area) {
	//Stop Condition
	if (trial == 0) return;
	double max = 0.0;
	int x, y;
	int i, j;
	double inertial_momentum;
	double massi, massj;

	for (i = 0; i < K; i++) {
		massi = (double)100 * (double)weight[i] / (double)area;
		for (j = 0; j < K; j++) {
			if (i == j) continue;
			if (src[i][j] == 0)continue;
			else {
				massj = (double)100 * (double)weight[j] / (double)area;
				inertial_momentum = massi / (pow(src[i][j], 3)* (pow(massj, 0.5)));
				if (max < inertial_momentum) {
					max = inertial_momentum;
					x = i;
					y = j;
				}
			}
		}
	}

	cout << "this time, " << x << "and" << y << "is merged" << endl;

	src[x][y] = 0;
	src[y][x] = 0;


	if (weight[x] > weight[y]) {
		mark[x][y] = true;
		for (i = 0; i < K; ++i) {
			if (mark[y][i]) mark[x][i] = true;
			if (mark[x][i]) {
				colors_all[i] = colors_all[x];
			}
		}

		for (i = 0; i < K; ++i) {
			src[i][y] = INT_MAX;
			src[y][i] = INT_MAX;
		}
		weight[x] += weight[y];
	}
	else {
		mark[y][x] = true;
		for (i = 0; i < K; ++i) {
			if (mark[x][i]) mark[y][i] = true;
			if (mark[y][i]) {
				colors_all[i] = colors_all[y];
			}
		}
		for (i = 0; i <K; ++i) {
			src[i][x] = INT_MAX;
			src[x][i] = INT_MAX;
		}
		weight[y] += weight[x];
	}


	//Recursion
	dist_gravity_update(trial - 1, src, weight, colors_all, mark, K,area);
}

void dist_gravity_update_lab(int trial, double(*src)[30], int * weight, Vec3d * colors_all, bool(*mark)[30], int K, int area) {
	//Stop Condition
	if (trial == 0) return;
	double max = 0.0;
	int x, y;
	int i, j;
	double inertial_momentum;
	double massi, massj;

	for (i = 0; i < K; i++) {
		massi = (double)100 * (double)weight[i] / (double)area;
		for (j = 0; j < K; j++) {
			if (i == j) continue;
			if (src[i][j] == 0)continue;
			else {
				massj = (double)100 * (double)weight[j] / (double)area;
				inertial_momentum = massi / (pow(src[i][j], 3)* (pow(massj, 0.5)));
				if (max < inertial_momentum) {
					max = inertial_momentum;
					x = i;
					y = j;
				}
			}
		}
	}

	cout << "this time, " << x << "and" << y << "is merged" << endl;

	src[x][y] = 0;
	src[y][x] = 0;


	if (weight[x] > weight[y]) {
		mark[x][y] = true;
		for (i = 0; i < K; ++i) {
			if (mark[y][i]) mark[x][i] = true;
			if (mark[x][i]) {
				colors_all[i] = colors_all[x];
			}
		}

		for (i = 0; i < K; ++i) {
			src[i][y] = INT_MAX;
			src[y][i] = INT_MAX;
		}
		weight[x] += weight[y];
	}
	else {
		mark[y][x] = true;
		for (i = 0; i < K; ++i) {
			if (mark[x][i]) mark[y][i] = true;
			if (mark[y][i]) {
				colors_all[i] = colors_all[y];
			}
		}
		for (i = 0; i <K; ++i) {
			src[i][x] = INT_MAX;
			src[x][i] = INT_MAX;
		}
		weight[y] += weight[x];
	}


	//Recursion
	dist_gravity_update_lab(trial - 1, src, weight, colors_all, mark, K, area);
}

void dist_gravity_fake_lab(int trial, double(*src)[30], int * weight, Vec3d * colors_all, bool(*mark)[30], int K, int area, bool*valid, bool body_or_not,int prempt) {
	//Stop Condition
	if (trial == 0) return;
	double max = 0.0;
	int x, y;
	int i, j;
	double inertial_momentum;
	double massi, massj;

	int merged_weight;
	Vec3d merged_color;
	vector<int> x_meme;
	vector<int> y_meme;

	//Noise update
	if (trial < 5) {
		double threshold = 0.5;
		double noise = INT_MAX;
		double curr;
		int flag;
		for (int i = 0; i < K; ++i) {
			if (body_or_not) {
				if (i == prempt) continue;
			}
			curr = (double)100 * (double)weight[i] / (double)area;
			if (valid[i]) {
				if (curr < noise) {
					noise = curr;
					flag = i;
				}
			}
		}

		if (noise < threshold) {
			noise_canceling_3d(src, weight, colors_all, mark, K, flag,0, valid);
			dist_gravity_fake_lab(trial - 1, src, weight, colors_all, mark, K, area,valid,body_or_not,prempt);
			return;
		}
	}
	//Noise update ends;


	for (i = 0; i < K; i++) {
		massi = (double)weight[i] / (double)area;
		for (j = 0; j < K; j++) {
			if (body_or_not && (i == prempt||j==prempt))  continue;
			if (mark[i][j]) continue;
			if (i == j) continue;
			else {
				if (valid[i] && valid[j]) {
					massj = (double)100 * (double)weight[j] / (double)area;
					inertial_momentum = massi / (pow(src[i][j], 3)* (pow(massj, 0.5)));
					//inertial_momentum = 1.0 / (pow(src[i][j], 3)* 1.0);
					if (max < inertial_momentum) {
						max = inertial_momentum;
						x = i;
						y = j;
					}
				}
			}
		}
	}
	//cout << "this time, " << x << "and" << y << "is merged" << endl;

	//Get the marks
	x_meme.push_back(x);
	y_meme.push_back(y);

	for (i = 0; i < K; ++i) {
		if (mark[x][i]) x_meme.push_back(i);
		if (mark[y][i]) y_meme.push_back(i);
	}

	//Merge the mark
	for (i = 0; i<x_meme.size(); i++) {
		for (j = 0; j < y_meme.size(); j++) {
			mark[x_meme[i]][y_meme[j]] = true;
			mark[y_meme[j]][x_meme[i]] = true;
		}
	}
	//Merge the color
	if (weight[x] > weight[y]) {
		merged_color = colors_all[x];
	}
	else {
		merged_color = colors_all[y];
	}
	colors_all[x] = merged_color;
	for (i = 0; i < K; ++i) {
		if (mark[x][i]) colors_all[i] = merged_color;
	}

	//Merge the weight
	merged_weight = weight[x] + weight[y];
	weight[x] = merged_weight;
	for (i = 0; i < K; ++i) {
		if (mark[x][i]) weight[i] = merged_weight;
	}

	vector_clear(x_meme);
	vector_clear(y_meme);
	//Recursion
	dist_gravity_fake_lab(trial - 1, src, weight, colors_all, mark, K,area,valid,body_or_not,prempt);
}

void dist_gravity_fake_lab_video(int trial, vector<int> weight, vector <Vec3d>& colors_all, vector <vector<bool>>mark, int K, int area) {
	//Stop Condition
	if (trial == 0) return;
	double max = 0.0;
	int x, y;
	int i, j;
	double inertial_momentum;
	double massi, massj;

	int merged_weight;
	Vec3d merged_color;
	vector<int> x_meme;
	vector<int> y_meme;

	double dist;

	for (i = 0; i < K; i++) {
		massi = (double)weight[i] / (double)area;
		for (j = 0; j < K; j++) {
			if (mark[i][j]) continue;
			if (i == j) continue;
			else {
				massj = (double)100 * (double)weight[j] / (double)area;
				dist = dist_3D(colors_all[i].val[0] - colors_all[j].val[0], colors_all[i].val[1] - colors_all[j].val[1], colors_all[i].val[2] - colors_all[j].val[2]);
				inertial_momentum = massi / (pow(dist, 3)* (pow(massj, 0.5)));
				//inertial_momentum = 1.0 / (pow(src[i][j], 3)* 1.0);
				if (max < inertial_momentum) {
					max = inertial_momentum;
					x = i;
					y = j;
				}
			}
		}
	}
	cout << "this time, " << x << "and" << y << "is merged" << endl;

	//Get the marks
	x_meme.push_back(x);
	y_meme.push_back(y);

	for (i = 0; i < K; ++i) {
		if (mark[x][i]) x_meme.push_back(i);
		if (mark[y][i]) y_meme.push_back(i);
	}

	//Merge the mark
	for (i = 0; i<x_meme.size(); i++) {
		for (j = 0; j < y_meme.size(); j++) {
			mark[x_meme[i]][y_meme[j]] = true;
			mark[y_meme[j]][x_meme[i]] = true;
		}
	}
	//Merge the color
	if (weight[x] > weight[y]) {
		merged_color = colors_all[x];
	}
	else {
		merged_color = colors_all[y];
	}
	colors_all[x] = merged_color;
	for (i = 0; i < K; ++i) {
		if (mark[x][i]) colors_all[i] = merged_color;
	}

	//Merge the weight
	merged_weight = weight[x] + weight[y];
	weight[x] = merged_weight;
	for (i = 0; i < K; ++i) {
		if (mark[x][i]) weight[i] = merged_weight;
	}

	vector_clear(x_meme);
	vector_clear(y_meme);
	//Recursion
	dist_gravity_fake_lab_video(trial - 1, weight, colors_all, mark, K, area);
}

void vector_clear(vector<int>& m) {
	vector <int> xclear;
	m.swap(xclear);
}

void noise_canceling_3b(double(*src)[30], int * weight, Vec3b * colors_all, bool(*mark)[30], int K, int flag, int type) {
	int i, x;
	double min = INT_MAX;
	int merged_weight;
	Vec3d merged_color;
	vector<int> x_meme;
	vector<int> flag_meme;
	int j;
	if (type == 0) {
		for (i = 0; i < K; ++i) {
			if (src[flag][i] == 0) continue;
			if (mark[flag][i]) continue;
			if (min > src[flag][i]) {
				min = src[flag][i];
				x = i;
			}
		}

		cout << "this time, " << flag << "and" << x << "is merged" << endl;

		//Get the marks
		x_meme.push_back(x);
		flag_meme.push_back(flag);

		for (i = 0; i < K; ++i) {
			if (mark[x][i]) x_meme.push_back(i);
			if (mark[flag][i]) flag_meme.push_back(i);
		}

		//Merge the mark
		for (i = 0; i < x_meme.size(); i++) {
			for (j = 0; j < flag_meme.size(); j++) {
				mark[x_meme[i]][flag_meme[j]] = true;
				mark[flag_meme[j]][x_meme[i]] = true;
			}
		}
		//Merge the color
		merged_color = colors_all[x];
		for (i = 0; i < K; ++i) {
			if (mark[x][i]) colors_all[i] = merged_color;
		}
		//Merge the weight
		merged_weight = weight[x] + weight[flag];
		weight[x] = merged_weight;
		for (i = 0; i < K; ++i) {
			if (mark[x][i]) weight[i] = merged_weight;
		}
	}
	if (type == 1) {
		for (i = 0; i < K; ++i) {
			if (src[flag][i] == 0) continue;
			if (mark[flag][i]) continue;
			if (min > src[flag][i]) {
				min = src[flag][i];
				x = i;
			}
		}
		cout << "this time, " << flag << "and" << x << "is merged" << endl;

		src[x][flag] = INT_MAX;
		src[flag][x] = INT_MAX;



		mark[x][flag] = true;
		for (i = 0; i < K; ++i) {
			if (mark[flag][i]) mark[x][i] = true;
			if (mark[x][i]) {
				colors_all[i] = colors_all[x];
			}
		}

		for (i = 0; i < K; ++i) {
			src[i][flag] = INT_MAX;
			src[flag][i] = INT_MAX;
		}
		weight[x] += weight[flag];
	}

}

void noise_canceling_3d(double(*src)[30], int * weight, Vec3d * colors_all, bool(*mark)[30], int K, int flag, int type, bool*valid) {
	int i, x;
	double min = INT_MAX;
	int merged_weight;
	Vec3d merged_color;
	vector<int> x_meme;
	vector<int> flag_meme;
	int j;

	if (type == 0) {
		for (i = 0; i < K; ++i) {
			if (src[flag][i] == 0) continue;
			if (mark[flag][i]) continue;
			if (min > src[flag][i] &&valid[i]) {
				min = src[flag][i];
				x = i;
			}
		}

		//cout << "this time, " << flag << "and" << x << "is merged as noise" << endl;

		//Get the marks
		x_meme.push_back(x);
		flag_meme.push_back(flag);

		for (i = 0; i < K; ++i) {
			if (mark[x][i]) x_meme.push_back(i);
			if (mark[flag][i]) flag_meme.push_back(i);
		}

		//Merge the mark
		for (i = 0; i < x_meme.size(); i++) {
			for (j = 0; j < flag_meme.size(); j++) {
				mark[x_meme[i]][flag_meme[j]] = true;
				mark[flag_meme[j]][x_meme[i]] = true;
			}
		}
		//Merge the color
		merged_color = colors_all[x];
		for (i = 0; i < K; ++i) {
			if (mark[x][i]) colors_all[i] = merged_color;
		}
		//Merge the weight
		merged_weight = weight[x] + weight[flag];
		weight[x] = merged_weight;
		for (i = 0; i < K; ++i) {
			if (mark[x][i]) weight[i] = merged_weight;
		}

	}
	if (type == 1) {
		for (i = 0; i < K; ++i) {
			if (src[flag][i] == 0) continue;
			if (mark[flag][i]) continue;
			if (min > src[flag][i]) {
				min = src[flag][i];
				x = i;
			}
		}
		cout << "this time, " << flag << "and" << x << "is merged by noise" << endl;

		src[x][flag] = INT_MAX;
		src[flag][x] = INT_MAX;



		mark[x][flag] = true;
		for (i = 0; i < K; ++i) {
			if (mark[flag][i]) mark[x][i] = true;
			if (mark[x][i]) {
				colors_all[i] = colors_all[x];
			}
		}

		for (i = 0; i < K; ++i) {
			src[i][flag] = INT_MAX;
			src[flag][i] = INT_MAX;
		}
		weight[x] += weight[flag];
	}
}

void rgb2cmyk(int r, int g, int b, int c, double m, double y, double k) {
	double r_, g_, b_;
	r_ = (double)r / (double)255;
	g_ = (double)g / (double)255;
	b_ = (double)b / (double)255;
	k = 1.0 - max(max(r_, g_), b_);
	c = (double)(1.00 - r_ - (double)k);
	m = (1.0 - g_ - k) / (double)(1.0 - k);
	y = (1.0 - b_ - k) / (double)(1.0 - k);
	cout << c << " " << m << " " << y << " " << k << endl;
}

void bgr2lab(Vec3b * colors_all, Vec3d * colors_lab_all, int k) {
	int i;
	double var_R, var_G, var_B, X, Y, Z, var_X, var_Y, var_Z, L, a, b;
	double ref_X = 95.047;
	double ref_Y = 100.000;
	double ref_Z = 108.883;
	cout.precision(7);
	for (i = 0; i < k; i++) {
		//cout << "B is " << (int)colors_all[i].val[0] << " G is " << (int)colors_all[i].val[1] << " R is " << (int)colors_all[i].val[2] << endl;

		var_B = (double)colors_all[i].val[0] / 255;
		var_G = (double)colors_all[i].val[1] / 255;
		var_R = (double)colors_all[i].val[2] / 255;

		//cout << "varB is " << var_B << " G is " << var_G << " R is " << var_R << endl;

		if (var_R > 0.04045) var_R = pow(((var_R + 0.055) / 1.055), 2.4);
		else var_R = var_R / 12.92;
		if (var_G > 0.04045) var_G = pow(((var_G + 0.055) / 1.055), 2.4);
		else var_G = var_G / 12.92;
		if (var_B > 0.04045) var_B = pow(((var_B + 0.055) / 1.055), 2.4);
		else var_B = var_B / 12.92;

		var_R = var_R * 100.0;
		var_G = var_G * 100.0;
		var_B = var_B * 100.0;

		//cout << "varB2 is " << var_B << " G2 is " << var_G << " R2 is " << var_R << endl;

		X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
		Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
		Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

		//cout << "X is " << X << " Y is " << Y << " Z is " << Z << endl;

		var_X = (double)X / ref_X;
		var_Y = (double)Y / ref_Y;
		var_Z = (double)Z / ref_Z;
		//cout << "varX is " << var_X << "varY is " << var_Y << " Z2 is " << var_Z << endl;

		if (var_X > 0.008856) var_X = pow(var_X, (double)(1.0 / 3));
		else  var_X = (7.787 * var_X) + (double)(16 / 116);
		if (var_Y > 0.008856) var_Y = pow(var_Y, (double)(1.0 / 3));
		else  var_Y = (7.787 * var_Y) + (double)(16 / 116);
		if (var_Z > 0.008856) var_Z = pow(var_Z, (double)(1.0 / 3));
		else  var_Z = (7.787 * var_Z) + (double)(16 / 116);

		//cout << "X2 is " << var_X << " Y2 is " << var_Y << " Z2 is " << var_Z << endl;

		L = (double)(116 * var_Y) - 16.0;
		a = (double)500 * (var_X - var_Y);
		b = (double)200 * (var_Y - var_Z);

		//cout << "L is " << L << " a is " << a << " b is " << b << endl;
		colors_lab_all[i].val[0] = L;
		colors_lab_all[i].val[1] = a;
		colors_lab_all[i].val[2] = b;
		//cout << "color "<< i<< "is " << (double) colors_lab_all[i].val[0] << ", " << (double)colors_lab_all[i].val[1] << ", " << (double)colors_lab_all[i].val[2] << endl;
	}
}

void bgr2lab_single(Vec3b color, Vec3d& colors_lab) {
	int i;
	double var_R, var_G, var_B, X, Y, Z, var_X, var_Y, var_Z, L, a, b;
	double ref_X = 95.047;
	double ref_Y = 100.000;
	double ref_Z = 108.883;
	
	var_B = (double)color.val[0] / 255;
	var_G = (double)color.val[1] / 255;
	var_R = (double)color.val[2] / 255;
	
	if (var_R > 0.04045) var_R = pow(((var_R + 0.055) / 1.055), 2.4);
	else var_R = var_R / 12.92;
	if (var_G > 0.04045) var_G = pow(((var_G + 0.055) / 1.055), 2.4);
	else var_G = var_G / 12.92;
	if (var_B > 0.04045) var_B = pow(((var_B + 0.055) / 1.055), 2.4);
	else var_B = var_B / 12.92;

	var_R = var_R * 100.0;
	var_G = var_G * 100.0;
	var_B = var_B * 100.0;

	X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
	Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
	Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

	var_X = (double)X / ref_X;
	var_Y = (double)Y / ref_Y;
	var_Z = (double)Z / ref_Z;

	if (var_X > 0.008856) var_X = pow(var_X, (double)(1.0 / 3));
	else  var_X = (7.787 * var_X) + (double)(16 / 116);
	if (var_Y > 0.008856) var_Y = pow(var_Y, (double)(1.0 / 3));
	else  var_Y = (7.787 * var_Y) + (double)(16 / 116);
	if (var_Z > 0.008856) var_Z = pow(var_Z, (double)(1.0 / 3));
	else  var_Z = (7.787 * var_Z) + (double)(16 / 116);

	L = (double)(116 * var_Y) - 16.0;
	a = (double)500 * (var_X - var_Y);
	b = (double)200 * (var_Y - var_Z);

	colors_lab.val[0] = L;
	colors_lab.val[1] = a;
	colors_lab.val[2] = b;
}

void lab2bgr(Vec3b * colors_all, Vec3d * colors_lab_all, int k) {
	int i;
	int R, G, B;
	double var_R, var_G, var_B, X, Y, Z, var_X, var_Y, var_Z, L, a, b;
	double ref_X = 95.047;
	double ref_Y = 100.000;
	double ref_Z = 108.883;
	cout.precision(7);

	for (i = 0; i < k; i++) {
		L = colors_lab_all[i].val[0];
		a = colors_lab_all[i].val[1];
		b = colors_lab_all[i].val[2];

		var_Y = (double)(L + 16) / 116;
		var_X = (double)a / 500 + var_Y;
		var_Z = (double)var_Y - b / 200;

		if (pow(var_Y, 3) > 0.008856) var_Y = pow(var_Y, 3);
		else    var_Y = (var_Y - 16 / 116) / 7.787;
		if (pow(var_X, 3) > 0.008856) var_X = pow(var_X, 3);
		else    var_X = (var_X - 16 / 116) / 7.787;
		if (pow(var_Z, 3) > 0.008856) var_Z = pow(var_Z, 3);
		else   var_Z = (var_Z - 16 / 116) / 7.787;

		X = ref_X*var_X;
		Y = ref_Y*var_Y;
		Z = ref_Z*var_Z;

		var_X = (double)X / 100;
		var_Y = (double)Y / 100;
		var_Z = (double)Z / 100;

		var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986;
		var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415;
		var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570;

		if (var_R > 0.0031308) var_R = 1.055 * pow(var_R, (double)(1 / 2.4)) - 0.055;
		else   var_R = 12.92 * var_R;
		if (var_G > 0.0031308) var_G = 1.055 * pow(var_G, (double)(1 / 2.4)) - 0.055;
		else  var_G = 12.92 * var_G;
		if (var_B > 0.0031308) var_B = 1.055 * pow(var_B, (double)(1 / 2.4)) - 0.055;
		else  var_B = 12.92 * var_B;;

		R = var_R * 255;
		G = var_G * 255;
		B = var_B * 255;
		//cout << "B is " << B << " G is " << G << " R is " << R << endl;

		colors_all[i].val[0] = B;
		colors_all[i].val[1] = G;
		colors_all[i].val[2] = R;
		//cout << "colorB is " << (int)colors_all[i].val[0] << " G is " << (int)colors_all[i].val[1] << " R is " << (int)colors_all[i].val[2] << endl;
	}
}

void lab2bgr_single(Vec3b& color, Vec3d& colors_lab) {
	int i;
	int R, G, B;
	double var_R, var_G, var_B, X, Y, Z, var_X, var_Y, var_Z, L, a, b;
	double ref_X = 95.047;
	double ref_Y = 100.000;
	double ref_Z = 108.883;
	cout.precision(7);

	L = colors_lab.val[0];
	a = colors_lab.val[1];
	b = colors_lab.val[2];

	var_Y = (double)(L + 16) / 116;
	var_X = (double)a / 500 + var_Y;
	var_Z = (double)var_Y - b / 200;

	if (pow(var_Y, 3) > 0.008856) var_Y = pow(var_Y, 3);
	else    var_Y = (var_Y - 16 / 116) / 7.787;
	if (pow(var_X, 3) > 0.008856) var_X = pow(var_X, 3);
	else    var_X = (var_X - 16 / 116) / 7.787;
	if (pow(var_Z, 3) > 0.008856) var_Z = pow(var_Z, 3);
	else   var_Z = (var_Z - 16 / 116) / 7.787;
	X = ref_X*var_X;
	Y = ref_Y*var_Y;
	Z = ref_Z*var_Z;

	var_X = (double)X / 100;
	var_Y = (double)Y / 100;
	var_Z = (double)Z / 100;
	var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986;
	var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415;
	var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570;

	if (var_R > 0.0031308) var_R = 1.055 * pow(var_R, (double)(1 / 2.4)) - 0.055;
	else   var_R = 12.92 * var_R;
	if (var_G > 0.0031308) var_G = 1.055 * pow(var_G, (double)(1 / 2.4)) - 0.055;
	else  var_G = 12.92 * var_G;
	if (var_B > 0.0031308) var_B = 1.055 * pow(var_B, (double)(1 / 2.4)) - 0.055;
	else  var_B = 12.92 * var_B;;
	R = var_R * 255;
	G = var_G * 255;
	B = var_B * 255;
		//cout << "B is " << B << " G is " << G << " R is " << R << endl;

	color.val[0] = B;
	color.val[1] = G;
	color.val[2] = R;
		//cout << "colorB is " << (int)colors_all[i].val[0] << " G is " << (int)colors_all[i].val[1] << " R is " << (int)colors_all[i].val[2] << endl;
}

void bgr2S(Vec3b *colors_all, double *sats, int k) {
	int i;
	double var_R, var_G, var_B;
	double Cmax, Cmin, delta;
	cout.precision(7);
	for (i = 0; i < k; i++) {
		//cout << "color was " << colors_all[i] << endl;
		var_B = (double)colors_all[i].val[0] / 255;
		var_G = (double)colors_all[i].val[1] / 255;
		var_R = (double)colors_all[i].val[2] / 255;

		Cmax = max(max(var_B, var_G), var_R);
		Cmin = min(min(var_B, var_G), var_R);
		delta = (double)(Cmax - Cmin);
		
		if (Cmax == 0.0) {
			sats[i] = 0;
		}
		else {
			sats[i] = 100*(double)delta/(1.0-abs((Cmax+Cmin)-1));
		}
		//cout << "sats[i] was " << sats[i] << endl;
	}
}

void dom_ass_acc(Vec3b * colors_all, int numbers, int * weight, Vec3b& dom, Vec3b& ass, Vec3b& acc, int K, int area, bool *valid, bool body_or_not ,bool domflag, bool assflag, bool accflag, int prempt) {
	//Intitializing the matrix	
	int i, j;
	int domin = 0, asso = 0, acce = 0;
	double percent;
	bool * isItPoss = new bool[K];
	for (i = 0; i < K; ++i) isItPoss[i] = false;
	for (i = 0; i < K; ++i) {
		for (j = 0; j < i; ++j) {
			if (colors_all[i] == colors_all[j]) break;
		}
		if (j == i & valid[i]) {
			isItPoss[i] = true;
		}
	}
	//Case No -body
	if (!body_or_not) {
		//Case numb==3
		if (numbers == 3) {
			for (i = 0; i < K; ++i) {
				if (isItPoss[i] && valid[i]) {
					if (domin < weight[i]) {
						acce = asso;
						asso = domin;
						domin = weight[i];

						acc = ass;
						ass = dom;
						dom = colors_all[i];

					}
					else if (asso < weight[i]) {
						acce = asso;
						asso = weight[i];
						acc = ass;
						ass = colors_all[i];
					}
					else {
						acce = weight[i];
						acc = colors_all[i];
					}
				}
			}
		}
		//Case numb==4
		else if (numbers == 4) {
			for (i = 0; i < K; ++i) {
				if (isItPoss[i] && valid[i]) {
					if (domin < weight[i]) {
						domin = weight[i];
						dom = colors_all[i];
					}
				}
			}
		}
		//Case numb >=5
		else {
			Vec3b* temps = new Vec3b[10];
			int cnt = 0;
			int temp = 0;
			int*savor = new int[10];
			int *hue = new int[K];
			int dom_hue;
			int acc_hue;

			bgr2hue(colors_all, hue, K, valid);

			//Dominant Color
			for (i = 0; i < K; ++i) {
				if (isItPoss[i] && valid[i]) {
					if (domin < weight[i]) {
						domin = weight[i];
						dom = colors_all[i];
						isItPoss[temp] = true;
						isItPoss[i] = false;
						dom_hue = hue[i];
						temp = i;
					}
				}
			}
			double thresh = 0.1;
			//cout << "DOM HUE =" << dom_hue;
			//Accent Color
			for (i = 0; i < K; ++i) {
				if (isItPoss[i] && valid[i]) {
					percent = (double)weight[i] / (double)area;
					if (percent < thresh) {
						temps[cnt] = colors_all[i];
						savor[cnt] = i;
						cnt++;
					}
				}
			}
			int acc_weight, lock = 0;
			if (cnt == 0) {
				acce = INT_MAX;
				for (i = 0; i < K; ++i)
				{
					if (acce > weight[i])
					{
						if (lock == 0) {
							acce = weight[i];
							acc = colors_all[i];
							isItPoss[i] = false;
							acc_hue = hue[i];
							acc_weight = weight[i];
							temp = i;
							lock = 1;
						}
						else {
							acce = weight[i];
							acc = colors_all[i];
							isItPoss[temp] = true;
							isItPoss[i] = false;
							acc_weight = weight[i];
							acc_hue = hue[i];
							temp = i;
						}
					}
				}
			}
			//MAX of Hue distance
			else {
				int col_dist = 0;
				int temp_dist;
				for (i = 0; i < cnt; ++i) {
					if (abs(dom_hue - hue[savor[i]]) <= 180) temp_dist = abs(dom_hue - hue[savor[i]]);
					else temp_dist = 360 - abs(dom_hue - hue[savor[i]]);
					if (col_dist < temp_dist) {
						if (lock == 0) {
							acc = temps[i];
							col_dist = temp_dist;
							isItPoss[savor[i]] = false;
							acc_hue = hue[savor[i]];
							acc_weight = weight[savor[i]];
							temp = savor[i];
							lock = 1;
						}
						else {
							acc = temps[i];
							col_dist = temp_dist;
							isItPoss[temp] = true;
							isItPoss[savor[i]] = false;
							acc_weight = weight[savor[i]];
							acc_hue = hue[savor[i]];
							temp = savor[i];
						}
					}
				}
			}
			//cout << "ACC HUE =" << acc_hue << endl;

			free(isItPoss);
			free(temps);
			free(savor);
			//Assort Color
			//assort(colors_all, weight, numbers-3, ass, valid, isItPoss, K);
			assort_sup(colors_all, weight, dom_hue, acc_hue, ass, acc, hue, valid, isItPoss, K, area, acc_weight);
		}
	}
	else {
		numbers++;
		//Case numb==3 with DOM/ASS/ACC FLAG
		if (numbers == 3) {
			if (domflag) {
				for (i = 0; i < K; ++i) {
					if (i == prempt)continue;
					if (isItPoss[i] && valid[i]) {
						if (asso < weight[i]) {
							acce = asso;
							asso = weight[i];
							acc = ass;
							ass = colors_all[i];
						}
						else {
							acce = weight[i];
							acc = colors_all[i];
						}
					}
				}
			}
			else if (assflag) {
				for (i = 0; i < K; ++i) {
					if (i == prempt)continue;
					if (isItPoss[i] && valid[i]) {
						if (domin < weight[i]) {
							acce = asso;
							asso = domin;
							domin = weight[i];

							acc = ass;
							ass = dom;
							dom = colors_all[i];

						}
						else {
							acce = weight[i];
							acc = colors_all[i];
						}
					}
				}
			}
			else if (accflag) {
				for (i = 0; i < K; ++i) {
					if (i == prempt) continue;
					if (isItPoss[i] && valid[i]) {
						if (domin < weight[i]) {
							acce = asso;
							asso = domin;
							domin = weight[i];

							acc = ass;
							ass = dom;
							dom = colors_all[i];

						}
						else if (asso < weight[i]) {
							acce = asso;
							asso = weight[i];
							acc = ass;
							ass = colors_all[i];
						}
					}
				}
			}
			else {
				cout << "FUCK";
				while (1);
			}
		}
		//Case numb==4 여기부터 코딩해야 함 족같네
		else if (numbers == 4) {
			for (i = 0; i < K; ++i) {
				if (isItPoss[i] && valid[i]) {
					if (domin < weight[i]) {
						domin = weight[i];
						dom = colors_all[i];
					}
				}
			}
		}
		//Case numb >=5
		else {
			Vec3b* temps = new Vec3b[10];
			int cnt = 0;
			int temp = 0;
			int*savor = new int[10];
			int *hue = new int[K];
			int dom_hue;
			int acc_hue;

			bgr2hue(colors_all, hue, K, valid);

			if (domflag){				

				dom_hue = hue[prempt];
				cout << "DOM HUE = " << dom_hue << endl;
				//Accent Color
				double thresh = 0.1;
				for (i = 0; i < K; ++i) {
					if (isItPoss[i] && valid[i]) {
						percent = (double)weight[i] / (double)area;
						if (percent < thresh) {
							temps[cnt] = colors_all[i];
							savor[cnt] = i;
							cnt++;
						}
					}
				}
				int acc_weight, lock = 0;
				if (cnt == 0) {
					acce = INT_MAX;
					for (i = 0; i < K; ++i)
					{
						if (acce > weight[i])
						{
							if (lock == 0) {
								acce = weight[i];
								acc = colors_all[i];
								isItPoss[i] = false;
								acc_hue = hue[i];
								acc_weight = weight[i];
								temp = i;
								lock = 1;
							}
							else {
								acce = weight[i];
								acc = colors_all[i];
								isItPoss[temp] = true;
								isItPoss[i] = false;
								acc_weight = weight[i];
								acc_hue = hue[i];
								temp = i;
							}
						}
					}
				}
				//MAX of Hue distance
				else {
					int col_dist = 0;
					int temp_dist;
					for (i = 0; i < cnt; ++i) {
						if (abs(dom_hue - hue[savor[i]]) <= 180) temp_dist = abs(dom_hue - hue[savor[i]]);
						else temp_dist = 360 - abs(dom_hue - hue[savor[i]]);
						if (col_dist < temp_dist) {
							if (lock == 0) {
								acc = temps[i];
								col_dist = temp_dist;
								isItPoss[savor[i]] = false;
								acc_hue = hue[savor[i]];
								acc_weight = weight[savor[i]];
								temp = savor[i];
								lock = 1;
							}
							else {
								acc = temps[i];
								col_dist = temp_dist;
								isItPoss[temp] = true;
								isItPoss[savor[i]] = false;
								acc_weight = weight[savor[i]];
								acc_hue = hue[savor[i]];
								temp = savor[i];
							}
						}
					}
				}
				cout << "ACC HUE =" << acc_hue << endl;

				free(isItPoss);
				free(temps);
				free(savor);
				//Assort Color
				//assort(colors_all, weight, numbers-3, ass, valid, isItPoss, K);
				assort_sup(colors_all, weight, dom_hue, acc_hue, ass, acc, hue, valid, isItPoss, K, area, acc_weight);
			}
			else if (accflag) {
				int acc_weight = weight[prempt];
				acc_hue = hue[prempt];
				//Dominant Color
				for (i = 0; i < K; ++i) {
					if (isItPoss[i] && valid[i]) {
						if (domin < weight[i]) {
							domin = weight[i];
							dom = colors_all[i];
							isItPoss[temp] = true;
							isItPoss[i] = false;
							dom_hue = hue[i];
							temp = i;
						}
					}
				}
				double thresh = 0.1;
				//cout << "ACC HUE = " << acc_hue << endl;
				//cout << "DOM HUE =" << dom_hue << endl;
				free(isItPoss);
				free(temps);
				free(savor);
				//Assort Color
				//assort(colors_all, weight, numbers-3, ass, valid, isItPoss, K);
				assort_sup(colors_all, weight, dom_hue, acc_hue, ass, acc, hue, valid, isItPoss, K, area, acc_weight);
			}
			else if(assflag) {
				//Dominant Color
				for (i = 0; i < K; ++i) {
					if (isItPoss[i] && valid[i]) {
						if (domin < weight[i]) {
							domin = weight[i];
							dom = colors_all[i];
							isItPoss[temp] = true;
							isItPoss[i] = false;
							dom_hue = hue[i];
							temp = i;
						}
					}
				}
				double thresh = 0.1;
				cout << "DOM HUE =" << dom_hue;
				//Accent Color
				for (i = 0; i < K; ++i) {
					if (isItPoss[i] && valid[i]) {
						percent = (double)weight[i] / (double)area;
						if (percent < thresh) {
							temps[cnt] = colors_all[i];
							savor[cnt] = i;
							cnt++;
						}
					}
				}
				int acc_weight, lock = 0;
				if (cnt == 0) {
					acce = INT_MAX;
					for (i = 0; i < K; ++i)
					{
						if (acce > weight[i])
						{
							if (lock == 0) {
								acce = weight[i];
								acc = colors_all[i];
								isItPoss[i] = false;
								acc_hue = hue[i];
								acc_weight = weight[i];
								temp = i;
								lock = 1;
							}
							else {
								acce = weight[i];
								acc = colors_all[i];
								isItPoss[temp] = true;
								isItPoss[i] = false;
								acc_weight = weight[i];
								acc_hue = hue[i];
								temp = i;
							}
						}
					}
				}
				//MAX of Hue distance
				else {
					int col_dist = 0;
					int temp_dist;
					for (i = 0; i < cnt; ++i) {
						if (abs(dom_hue - hue[savor[i]]) <= 180) temp_dist = abs(dom_hue - hue[savor[i]]);
						else temp_dist = 360 - abs(dom_hue - hue[savor[i]]);
						if (col_dist < temp_dist) {
							if (lock == 0) {
								acc = temps[i];
								col_dist = temp_dist;
								isItPoss[savor[i]] = false;
								acc_hue = hue[savor[i]];
								acc_weight = weight[savor[i]];
								temp = savor[i];
								lock = 1;
							}
							else {
								acc = temps[i];
								col_dist = temp_dist;
								isItPoss[temp] = true;
								isItPoss[savor[i]] = false;
								acc_weight = weight[savor[i]];
								acc_hue = hue[savor[i]];
								temp = savor[i];
							}
						}
					}
				}
				cout << "ACC HUE =" << acc_hue << endl;

				free(isItPoss);
				free(temps);
				free(savor);
			}
		}
	}
}
// Heavy weight
void assort(Vec3b* colors_all, int *weight, int trial, Vec3b& ass, bool*valid, bool * isItPoss, int K) {
	if (trial == 0) return;
	double min = INT_MAX;
	int x, y;
	int i, j;
	//New var
	int merged_weight;
	Vec3b merged_color;
	vector<int> x_meme;
	vector<int> y_meme;
	double dist;
	
	for (i = 0; i < K; i++) {
		if(isItPoss[i]&&valid[i]){
			for (j = i + 1; j < K; j++) {
				if (isItPoss[j] && valid[j]) {
					dist = dist_3D(colors_all[i].val[0] - colors_all[j].val[0], colors_all[i].val[1] - colors_all[j].val[1], colors_all[i].val[2] - colors_all[j].val[2]);
					if (min > dist) {
						min = dist;
						x = i;
						y = j;
					}
				}
			}
		}
	}
	
	//Assort Color
	if (weight[x] > weight[y]) {
		ass = colors_all[x];
		merged_weight = weight[x] + weight[y];
		weight[x] = merged_weight;
		isItPoss[y] = false;
	}
	else {
		ass = colors_all[y];
		merged_weight = weight[x] + weight[y];
		weight[y] = merged_weight;
		isItPoss[x] = false;
	}
	assort(colors_all, weight, trial-1, ass, valid, isItPoss, K);
}
// Farrest Distance
void assort_sup(Vec3b* colors_all, int *weight, int dom_hue, int acc_hue, Vec3b& ass,Vec3b &acc, int*hue, bool*valid, bool * isItPoss, int K, int area, int acc_weight) {
	int i;
	int cnt = 0;
	double percent, thresh = 0;
	Vec3b *temp = new Vec3b[10];
	int *savor = new int[10];
	int ass_hue, ass_weight;
	for (i = 0; i < K; ++i) {
		if (isItPoss[i] && valid[i]) {
			percent = (double)weight[i] / (double)area;
			if (percent > thresh && colors_all[i]!=acc) {
				temp[cnt] = colors_all[i];
				savor[cnt] = i;
				//cout << "temp" << i << "is " << temp[cnt] << "and" << hue[savor[cnt]] << endl;
				cnt++;
			}
		}
	}
	double max = 0;
	//If cnt == 0 , Return Most Weighted color
	if (cnt == 0) {
		for (i = 0; i < K; ++i) {
			if (isItPoss[i] && valid[i]) {
				if (max < weight[i]) {
					max = weight[i];
					ass = colors_all[i];
				}
			}
		}
	}
	
	//Return Assort Color that is Triadic Harmony
	else {
		int temp1, temp2;
		double value;
		//cout << "dom is" << dom_hue << "acc is " << acc_hue << endl;
		for (i = 0; i < cnt; ++i) {
			if (abs(dom_hue - hue[savor[i]]) <= 180) temp1 = abs(dom_hue - hue[savor[i]]);
			else temp1 = 360 - abs(dom_hue - hue[savor[i]]);
			if (abs(acc_hue - hue[savor[i]]) <= 180) temp2 = abs(acc_hue - hue[savor[i]]);
			else temp2 = 360 - abs(acc_hue - hue[savor[i]]);
			value = temp1*temp2;
			//cout << "value =" << value << endl;
			if (value > max) { 
				ass = temp[i];
				ass_hue = hue[savor[i]];
				max = value;
				ass_weight = weight[savor[i]];
			}
		}
	}
	if (ass_weight < acc_weight) {
		Vec3b temper;
		temper = ass;
		ass = acc;
		acc = temper;
	}
	

	//cout << "ASS HUE =" << ass_hue << endl;
	free(temp);
	free(savor);
}

void show(Mat k_mean, Mat drawCircle_multi, Mat drawDomAssAcc) {
	namedWindow("Result");
	namedWindow("Dom,Ass,Acc");
	namedWindow("ALL");

	moveWindow("Frequent", 150, 150);

	imshow("Result", k_mean);
	imshow("Dom,Ass,Acc", drawDomAssAcc);
	imshow("ALL", drawCircle_multi);

	waitKey(0);
}

int eliminate_low_sats(double * sats, int *weight,int k, bool*valid) {
	int i, cnt=0;
	for (i = 0; i < k; ++i) {
		if (sats[i] < Low_thresh) {
			valid[i] = false;
			//cout << "color" << i << "th is deleted" << endl;
			cnt++;
		}
	}
	//if (cnt == 0) cout << "No color deleted" << endl;
	return cnt;
}

void HOG_Human(Mat inputImg) {
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	vector <Rect> detected;

	//double time = double(getTickCount());

	hog.detectMultiScale(inputImg, detected, 0, Size(6, 6), Size(24,24), 1.05, 2);
	//time = double(getTickCount()) - time;

	int detectedCNT = detected.size();

	for (int i = 0; i<detectedCNT; i++)
	{
		Rect people = detected[i];
		rectangle(inputImg, people, Scalar(0, 0, 255), 2);
	}

	string fileName = save_image_name();
	imwrite(fileName + ".png", inputImg);
	namedWindow("Result", CV_WINDOW_AUTOSIZE);
	imshow("Result", inputImg);
	waitKey(0);
	while (1);
}

void cascade_full(Mat img) {
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	vector<Rect> body_pos;

	if (!body.load(body_cascade)) {
		cout << "FAIL" << endl;
		while(1){}
	}
	body.detectMultiScale(gray, body_pos, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));
	int i, size = body_pos.size();
	for (i = 0; i < size; ++i) {
		rectangle(img, body_pos[i], Scalar(0, 255, 0), 2);
	}
	namedWindow("Casc", CV_WINDOW_AUTOSIZE);
	imshow("Casc", img);
	waitKey();
	while (1);
}

void cascade_upper(Mat img) {
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	vector<Rect> body_pos;

	if (!upper_body.load(upper_body_cascade)) {
		cout << "FAIL" << endl;
		while (1) {}
	}
	upper_body.detectMultiScale(gray, body_pos, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));
	int i, size = body_pos.size();
	for (i = 0; i < size; ++i) {
		rectangle(img, body_pos[i], Scalar(0, 255, 0), 2);
	}
	namedWindow("Casc", CV_WINDOW_NORMAL);
	imshow("Casc", img);
	waitKey();
	while (1);
}

void cascade_face(Mat img, vector<Rect>face_pos) {
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	if (!face.load(face_cascade)) {
		cout << "FAIL" << endl;
		while (1) {}
	}
	face.detectMultiScale(gray, face_pos, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));
	int i, size = face_pos.size();
	/*for (i = 0; i < size; ++i) {
		rectangle(img, face_pos[i], Scalar(0, 255, 0), 2);
	}
	namedWindow("Casc", CV_WINDOW_AUTOSIZE);
	imshow("Casc", img);
	waitKey();
	while (1);
	*/
}

vector<Rect> body_infer(Mat img) {
	vector<Rect> face_info;
	vector<Rect> body_ass;
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	if (!face.load(face_cascade)) {
		cout << "FAIL" << endl;
		while (1) {}
	}
	face.detectMultiScale(gray, face_info, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));
	body_ass = face_info;
	int i, size = face_info.size();
	for (i = 0; i < size; ++i) {
		body_ass[i] = Rect((face_info[i].x - face_info[i].width),(face_info[i].y + face_info[i].height), face_info[i].width * 3, face_info[i].height *4);
	}
	
	/*
	for (i = 0; i < size; ++i) {
		rectangle(img,body_ass[i], Scalar(0, 255, 0),6);
	}
	
	string fileName = save_image_name();
	imwrite(fileName + ".png", img);
	
	namedWindow("LL", CV_WINDOW_NORMAL);
	imshow("LL", img);
	waitKey();
	while (1);	
	*/
	return body_ass;
	
}

bool main_body(Mat img, Rect &body) {
	vector<Rect> bodies = body_infer(img);
	int size = bodies.size();
	if (size == 0) return false;
	int i,area = 0;
	for (i = 0; i < size; ++i) {
		if (area < bodies[i].area()) {
			area = bodies[i].area();
			body = bodies[i];
		}
	}
	return true;
}

Rect roi_body(Mat img,Rect &body) {
	int img_x, img_y, body_x, body_y, body_endx, body_endy;
	img_x = img.cols;
	img_y = img.rows;
	body_x = body.x;
	body_y = body.y;
	//Initialize
	if (body_x < 0) body_x = 0;
	if (body_y < 0) body_y = 0;
	if (body_x + body.width > img_x) body_endx = img_x;
	else body_endx = body_x + body.width;
	if (body_y + body.height > img_y) body_endy = img_y;
	else body_endy = body_y + body.height;
	//Set ROI
	return (Rect(body_x,body_y,body_endx - body_x,body_endy-body_y));
	//rectangle(img, body, Scalar(0, 255, 0), 6);

}

void bgr2hue(Vec3b*colors_all, int* hue, int k, bool*valid) {
	int i;
	double var_R, var_G, var_B;
	double Cmax, Cmin, delta;
	cout.precision(7);
	for (i = 0; i < k; i++) {
		//cout << "color was " << colors_all[i] << endl;
		var_B = (double)colors_all[i].val[0] / 255;
		var_G = (double)colors_all[i].val[1] / 255;
		var_R = (double)colors_all[i].val[2] / 255;

		Cmax = max(max(var_B, var_G), var_R);
		Cmin = min(min(var_B, var_G), var_R);
		delta = (double)(Cmax - Cmin);

		if (delta == 0.0) {
			hue[i] = 0;
		}

		if (Cmax == var_R) {
			hue[i] = 60 * ((var_G-var_B)/delta);

		}
		else if (Cmax == var_G) {
			hue[i] = 60 * ((var_B - var_R)/delta + 2);
		}
		else if (Cmax == var_B) {
			hue[i] = 60 * ((var_R - var_G)/delta + 4);
		}
		else {
			cout << "Panic" << endl;
			while (1);
		}

		if (hue[i] < 0) hue[i] += 360;
		if (hue[i] > 360)hue[i] -= 360;
		//if(valid[i]) cout << "color " << colors_all[i] << "is" << hue[i] << endl;
	}
}

void color_sample(int type, int r1, int g1, int b1, int r2, int g2, int b2, int r3, int g3, int b3) {
	Mat result;
	result.create(300, 600, CV_8UC3);
	Rect rc1, rc2, rc3;
	//Equal
	if (type == 0) {
		rc1 = Rect(0, 0, 200, 300);
		rc2 = Rect(200, 0, 200, 300);
		rc3 = Rect(400, 0, 200, 300);
	}
	//Dom,Ass,Acc
	else if (type == 1) {
		rc1 = Rect(0, 0, 420, 300);
		rc2 = Rect(420, 0, 30, 300);
		rc3 = Rect(450, 0, 150, 300);
	}

	rectangle(result, rc1, Scalar(b1, g1, r1), CV_FILLED);
	rectangle(result, rc2, Scalar(b3, g3, r3), CV_FILLED);
	rectangle(result, rc3, Scalar(b2, g2, r2), CV_FILLED);
	/*
	string fileName = save_image_name();
	imwrite(fileName + ".png", result);

	
	namedWindow("Result");
	moveWindow("Frequent", 150, 150);
	imshow("Result", result);
	waitKey(0);
	*/
}

void color_box_design(Vec3b dom, Vec3b ass, Vec3b acc) {
	Mat squ;
	squ.create(1123, 794, CV_8UC3);
	Rect rc1, rc2, rc3;
	rc1 = Rect(0, 0, 794, 1123);
	rc2 = Rect(180, 259, 435, 615);
	rc3 = Rect(308, 436, 178, 252);
	rectangle(squ, rc1, Scalar(dom), CV_FILLED);
	rectangle(squ, rc2, Scalar(ass), CV_FILLED);
	rectangle(squ, rc3, Scalar(acc), CV_FILLED);

	string fileName = save_image_name();
	imwrite(fileName + ".png", squ);
	namedWindow("BLAH");
	imshow("BALH",squ);
}

void color_check_design(Vec3b dom, Vec3b ass, Vec3b acc) {
	Mat squ;
	squ.create(1123, 794, CV_8UC3);
	Rect rc1, rc2, rc3;
	rc1 = Rect(0, 0, 794, 1123);
	rectangle(squ, rc1, Scalar(dom), CV_FILLED);
	rc2 = Rect(153, 0, 61, 1123);

	for (int i = 0; i < 3; ++i) {
		rectangle(squ, rc2, Scalar(ass), CV_FILLED);
		rc2 = Rect(153+214*(i+1), 0, 61, 1123);
	}
	rc2 = Rect(0, 176, 794, 61);
	for (int i = 0; i < 4; ++i) {
		rectangle(squ, rc2, Scalar(ass), CV_FILLED);
		rc2 = Rect(0,176+237*(i+1),794, 61);
	}
	
	rc3 = Rect(153, 176, 61, 61);
	int current = 153;
	int curr_j = 176;
	for (int i = 0; i < 3; ++i) {
		curr_j = 176;
		for (int j = 0; j < 4; ++j) {
			rectangle(squ, rc3, Scalar(acc), CV_FILLED);		
			rc3 = Rect(current, curr_j, 61, 61);
			curr_j = 176 + 237 * (j + 1);
		}
		current = 153 + 214 * (i + 1);
	}
	rc3 = Rect(580,886, 61, 61);
	rectangle(squ, rc3, Scalar(acc), CV_FILLED);

	string fileName = save_image_name();
	imwrite(fileName + ".png", squ);
	namedWindow("BLAH");
	imshow("BALH", squ);
}

Mat temp_make(Mat img, Vec3b dom, Vec3b ass, Vec3b acc) {
	Mat squ;
	int flag;
	//가로방향 --> 세로로 배치
	if (img.cols> img.rows) flag = 1;
	//세로방향 --> 가로로 배치
	else flag = 2;
	Rect body_reduced;
	Rect rc1, rc2;
	string name = "HEADLINE";
	string subname = "Subtitle";
	string bottom = "Bottom";
	if (flag == 1)
	{
		squ.create(1123, 794, CV_8UC3);

		resize(img, img, Size(), (double)794 / img.cols, (double)794 / img.cols);


		rc1 = Rect(0, 0, 794, 1123);
		rc2 = Rect(0, 1000, 794, 123);
		rectangle(squ, rc1, Scalar(dom), CV_FILLED);
		rectangle(squ, rc2, Scalar(ass), CV_FILLED);

		putText(squ, name, Point(250, (img.rows+1000)/2), 5, 2.5, Scalar(acc),3);
		putText(squ, subname, Point(300, (img.rows+1000)/2 + 50), 5, 1.5, Scalar(acc),3);
		putText(squ, bottom, Point(300, 1080), 5, 1.5, Scalar(acc), 3);

		img.copyTo(squ(Rect(0, 0, img.cols, img.rows)));
	}
	else
	{
		squ.create(794, 1123, CV_8UC3);
		
		resize(img, img, Size(), (double)794 / img.rows, (double)794 / img.rows);

		rc1 = Rect(0, 0, 1123, 794);
		rc2 = Rect(1000, 0, 123, 794);
		rectangle(squ, rc1, Scalar(dom), CV_FILLED);
		rectangle(squ, rc2, Scalar(ass), CV_FILLED);

		putText(squ, name, Point((img.cols + 1000)/2-100, 300), 5,2.5, Scalar(acc),3);
		putText(squ, subname, Point((img.cols + 1000) / 2-50, 350), 5, 1.5, Scalar(acc),3);

		img.copyTo(squ(Rect(0, 0, img.cols, img.rows)));
	}

	return squ;
}