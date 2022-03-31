#include <iostream>
#include <map>
#include <opencv2\opencv.hpp>
#include <opencv2\photo.hpp>
#include "ExtractP.h"
#include <math.h>
#include <fstream>


using namespace std;
using namespace cv;


#define SCALE 0.2
#define Factor 100000
#define resize_factor 600000
#define K 30
#define Nobody false

struct lessVec3b
{
	bool operator()(const Vec3b& lhs, const Vec3b& rhs) const {
		return (lhs[0] != rhs[0]) ? (lhs[0] < rhs[0]) : ((lhs[1] != rhs[1]) ? (lhs[1] < rhs[1]) : (lhs[2] < rhs[2]));
	}
};
const map<Vec3b, int, lessVec3b> getPalette(const Mat3b& src)
{
	map<Vec3b, int, lessVec3b> palette;
	for (int r = 0; r < src.rows; ++r)
	{
		for (int c = 0; c < src.cols; ++c)
		{
			Vec3b color = src(r, c);
			if (palette.count(color) == 0)
			{
				palette[color] = 1;
			}
			else
			{
				palette[color] += 1;
			}
		}
	}
	return palette;
}

void picture_palette() {
	// Declaration // 
	Mat inputImg = imread(get_image_name(), CV_LOAD_IMAGE_COLOR);
	while (inputImg.rows == 0) {
		cout << "Must put valid name" << endl;
		inputImg = imread(get_image_name(), CV_LOAD_IMAGE_COLOR);
	}
	//HOG_Human(inputImg);
	//body_infer(inputImg);
	Mat roi;
	Rect body_part;
	Rect body_reduced;
	bool body_or_not = main_body(inputImg, body_part);
	if (Nobody) body_or_not = false;
	if (body_or_not) body_reduced = roi_body(inputImg, body_part);
	//body_infer(inputImg);
	//cascade_face(inputImg);
	//cascade_upper(inputImg);

	Mat resizeImg;
	Mat k_mean;
	Mat3b pal;
	string fileName;
	Vec3b most_frequent_color;
	Vec3b * colors_all = new Vec3b[K];
	Vec3d * colors_lab_all = new Vec3d[K];
	double * colors_sats = new double[K];
	map<Vec3b, int, lessVec3b> palette;
	map<Vec3b, int, lessVec3b> palette_mini;
	Vec3b dom, ass, acc;
	int freq = 0;
	int * weight = new int[K];
	int i = 0, j = 0;
	int area;
	int cnt = 0;
	int lowSat;
	double c = 0.0, m = 0.0, y = 0.0, k = 0.0;
	double r_, g_, b_;
	double dist_Mat[30][30];
	bool mark[30][30];
	bool *valid = new bool[K];
	bool domflag=false, assflag=false, accflag=false;
	int prempt=0;
	for (j = 0; j < K; ++j) {
		valid[j] = true;
		for (int k = 0; k < K; ++k) {
			mark[j][k] = false;
		}
	}
	bool low_sat_flag = false;
	string save_or_not;


	cout << "How much color you want? , Number should be between 3~10" << endl;
	int numb_color;
	cin >> numb_color;
	while (numb_color < 3 || numb_color>10) {
		cout << "Must put valid number" << endl;
		cin >> numb_color;
	}


	/* Resizing */
	resizeImg = inputImg.clone();
	resize_limit(inputImg, resizeImg, resize_factor,body_or_not,body_reduced);
	cvtColor(resizeImg, resizeImg, COLOR_BGR2Lab);

	/*Reducing Algorithm*/
	reduceColor_kmeans(resizeImg, k_mean, K);
	//inputImg.release();
	resizeImg.release();
	cvtColor(k_mean, k_mean, COLOR_Lab2BGR);
	if (body_or_not) {
		roi = k_mean(Rect(body_reduced.x, body_reduced.y, body_reduced.width, body_reduced.height)).clone();
		/*
		string fileName = save_image_name();
		imwrite(fileName + ".png", roi);
		namedWindow("LL", CV_WINDOW_NORMAL);
		imshow("LL", roi);
		waitKey();
		while (1);
		*/
		
	}
	pal = k_mean;
	palette = getPalette(pal);
	if (body_or_not) {
		palette_mini = getPalette(roi);
	}
	/* 1st Color Palette */
	for (auto color : palette)
	{
		colors_all[i] = color.first;
		weight[i] = color.second;
		i++;
	}
	//Even if it is low sat, Get it if it has huge salicity
	if (body_or_not) {
		for (auto color : palette_mini) {
			if (color.second > freq)
			{
				most_frequent_color = color.first;
				freq = color.second;
			}
		}
	}
	palette.clear();
	palette_mini.clear();

	/*Eliminate Low Sats*/
	bgr2S(colors_all, colors_sats, K);
	lowSat = eliminate_low_sats(colors_sats, weight, K, valid);
	//cout << lowSat << "is eliminated" << endl;

	//Low Sat color Palette Case
	if (body_or_not) {
		for (i = 0; i < K; ++i) {
			if (most_frequent_color == colors_all[i]) {
				if (!valid[i])low_sat_flag = true;
				valid[i] = true;
				prempt = i;
			}
		}
	}


	/*Color Palette RGB to Lab*/
	bgr2lab(colors_all, colors_lab_all, K);
	//Color Matrix for rgb
	/*for (i = 0; i < K; ++i) {
	//cout << (int)colors_all[i].val[0] << " " << (int)colors_all[i].val[1] << " " << (int)colors_all[i].val[2] << endl;
	for (j = 0; j < K; ++j) {
	//cout << (int)colors_all[j].val[0] << " " << (int)colors_all[j].val[1] << " " << (int)colors_all[j].val[2] << endl;
	//cout << "i = " << i << "j = " << j << endl;
	dist_Mat[i][j] = dist_3D(colors_all[i].val[0] - colors_all[j].val[0], colors_all[i].val[1] - colors_all[j].val[1], colors_all[i].val[2] - colors_all[j].val[2]);
	//cout << "distMat : " << dist_Mat[i][j] << endl;
	}
	}*/


	//Color Matrix for lab
	for (i = 0; i < K; ++i) {
		//cout << (int)colors_all[i].val[0] << " " << (int)colors_all[i].val[1] << " " << (int)colors_all[i].val[2] << endl;
		for (j = 0; j < K; ++j) {
			//cout << (int)colors_all[j].val[0] << " " << (int)colors_all[j].val[1] << " " << (int)colors_all[j].val[2] << endl;
			//cout << "i = " << i << "j = " << j << endl;
			dist_Mat[i][j] = dist_3D(colors_lab_all[i].val[0] - colors_lab_all[j].val[0], colors_lab_all[i].val[1] - colors_lab_all[j].val[1], colors_lab_all[i].val[2] - colors_lab_all[j].val[2]);
			//cout << "distMat : " << dist_Mat[i][j] << endl;
		}
	}

	/*Area calc*/
	area = 0;
	for (i = 0; i < K; ++i) {
		if (valid[i]) {
			area += weight[i];
		}
	}
	
	/*Prempt*/
	if (body_or_not) {
		float acc_line = 15, ass_line = 50;
		if (100.f*float(weight[prempt]) / float(area) < acc_line) {
			accflag = true;
			acc = most_frequent_color;
			cout << "body is accent for "<<acc<< endl;
		}
		else if (100.f*float(weight[prempt]) / float(area) >=acc_line&&100.f*float(weight[prempt]) / float(area) < ass_line) {
			assflag = true;
			ass = most_frequent_color;
			cout << "body is assort for" << ass << endl;
		}
		else {
			domflag = true;
			dom = most_frequent_color;
			cout << "body is dominant for"<<dom << endl;
		}
	}

	/*Distance_Merge*/
	//dist_merge_update(K - numb_color, dist_Mat, weight, colors_all,mark,K, area);
	//dist_merge_fake(K - numb_color, dist_Mat, weight, colors_all, mark, K, area);
	//dist_gravity_update(K - numb_color, dist_Mat, weight, colors_all, mark, K,  area);
	//dist_merge_update_lab(K - numb_color, dist_Mat, weight, colors_lab_all,mark,K, area);
	//dist_merge_fake_lab(K - numb_color, dist_Mat, weight, colors_lab_all, mark, K, area);	
	//dist_gravity_update_lab(K - numb_color, dist_Mat, weight, colors_lab_all, mark, K, area);
	if (low_sat_flag) {
		int n = numb_color - 1;
		dist_gravity_fake_lab(K - n - lowSat, dist_Mat, weight, colors_lab_all, mark, K, area, valid, body_or_not, prempt);
	}
	else dist_gravity_fake_lab(K - numb_color - lowSat, dist_Mat, weight, colors_lab_all, mark, K, area, valid ,body_or_not,prempt);


	/*Lab to BGR*/
	lab2bgr(colors_all, colors_lab_all, K);
	
	free(colors_lab_all);

	/*Area show algorithm*/
	for (i = 0; i < K; ++i) {
		for (j = 0; j < i; ++j) {
			if (colors_all[i] == colors_all[j]) break;
		}
		if ((j == i && valid[i])) {
			cout << "Color: " << colors_all[i] << " for" << 100.f * float(weight[i]) / float(area) << "%" << endl;
		}
	}

	/* Color Palette Visualization - Circle */
	Mat drawCircle_multi;
	drawCircle_multi.create(200, 130 * numb_color + 100, CV_8UC3);
	drawCircle_multi = Scalar(0, 0, 0);
	for (i = 0; i < K; ++i) {
		for (j = 0; j < i; ++j) {
			if (colors_all[i] == colors_all[j]) break;
		}
		if ((j == i && valid[i])) {
			circle(drawCircle_multi, Point(cnt * 130 + 100, 100), 50.0, colors_all[i], -1, 8);
			circle(drawCircle_multi, Point(cnt * 130 + 100, 100), 50.0, Scalar(150,150,150), 2, 8);
			cnt++;
		}
	}

	/*Dominant, Assort, Accent Colors*/
	dom_ass_acc(colors_all, numb_color, weight, dom, ass, acc, K, area, valid, body_or_not,domflag,assflag,accflag,prempt);
	cout << "dom is " << dom << endl;
	cout << "ass is " << ass << endl;
	cout << "acc is" << acc << endl;
	
	free(weight);

	

	Mat drawDomAssAcc;
	drawDomAssAcc.create(250, 450, CV_8UC3);
	drawDomAssAcc = Scalar(0, 0, 0);
	circle(drawDomAssAcc, Point(100, 150), 50.0, dom, -1, 8);
	circle(drawDomAssAcc, Point(230, 150), 50.0, ass, -1, 8);
	circle(drawDomAssAcc, Point(360, 150), 50.0, acc, -1, 8);
	circle(drawDomAssAcc, Point(100, 150), 50.0, Scalar(150, 150, 150), 2, 8);
	circle(drawDomAssAcc, Point(230, 150), 50.0, Scalar(150, 150, 150),2, 8);
	circle(drawDomAssAcc, Point(360, 150), 50.0, Scalar(150, 150, 150), 2, 8);

	putText(drawDomAssAcc, "Dominant", Point(50, 50), 6, 1, Scalar::all(255));
	putText(drawDomAssAcc, "Assort", Point(200, 50),6, 1, Scalar::all(255));
	putText(drawDomAssAcc, "Accent", Point(300, 50), 6,1, Scalar::all(255));

	Mat squ= temp_make(inputImg, dom, ass, acc);
	//Save the image
	cout << "Save the file?: Y/N" << endl;
	cin >> save_or_not;
	if (save_or_not == "y" || save_or_not == "Y")
	{
		fileName = save_image_name();
		//imwrite(fileName + ".png", k_mean);
		imwrite(fileName + ".png", drawCircle_multi);
		imwrite(fileName + "_DomAssAcc.png", drawDomAssAcc);
		imwrite(fileName + "_template.png", squ);
		ofstream outFile(fileName+"_colors.txt");

		outFile << "Color Dom - R: " << (int)dom.val[2] << ", G: " << (int)dom.val[1] << ", B: " << (int)dom.val[0]<<endl;
		outFile << "Color Ass - R: " << (int)ass.val[2] << ", G: " << (int)ass.val[1] << ", B: " << (int)ass.val[0] << endl;
		outFile << "Color Acc - R: " << (int)acc.val[2] << ", G: " << (int)acc.val[1] << ", B: " << (int)acc.val[0]<<endl;
		outFile << endl;
		outFile << "Below we got all colors" << endl;
		
		
		for (i = 0; i < K; ++i) {
		for (j = 0; j < i; ++j) {
		if (colors_all[i] == colors_all[j]) break;
		}
		if (j == i&& valid[i]) {
		outFile << "Color " << i << "is R: " << (int)colors_all[i].val[2] << ", G: " << (int)colors_all[i].val[1] << ", B: " << (int)colors_all[i].val[0];
		r_ = (double)colors_all[i].val[2] / (double)255;
		g_ = (double)colors_all[i].val[1] / (double)255;
		b_ = (double)colors_all[i].val[0] / (double)255;
		k = 1.0 - max(max(r_, g_), b_);
		c = (double)(1.00 - r_ - k) / (double)(1.0 - k);
		m = (1.0 - g_ - k) / (double)(1.0 - k);
		y = (1.0 - b_ - k) / (double)(1.0 - k);
		outFile << " and for CMYK, " << i << "is C: " << setprecision(3)<< c << ", M: " << m << ", y: " << y << " and k: " << k;
		outFile << " Area: " << 100.f * float(weight[i]) / float(area) << "%" << endl;
		}
		}
	}

	color_sample(1, dom.val[2], dom.val[1], dom.val[0], ass.val[2], ass.val[1], ass.val[0], acc.val[2], acc.val[1], acc.val[0]);
	//color_box_design(dom, ass, acc);
	//color_check_design(dom, ass, acc);
	/* Draw and Show */
	if(body_or_not)rectangle(k_mean, Rect(body_reduced.x, body_reduced.y, body_reduced.width, body_reduced.height), Scalar(0, 255, 0), 6);
	
	namedWindow("BLAAA", WINDOW_FULLSCREEN);
	imshow("BLAAA", squ);
	show(k_mean, drawCircle_multi, drawDomAssAcc);
}

void handle_color() {
	int p;
	cout << "TYPE??? 0 for Equal, 1 for dom/ass/acc" << endl;
	cin >> p;
	if (p != 0 && p != 1) {
		cout << "Wrong type" << endl;
		while (1);
	}
	int r1, g1, b1, r2, g2, b2, r3, g3, b3;
	cout << "Write Dominant R, G, B" << endl;
	cin >> r1;
	cin >> g1;
	cin >> b1;
	if (r1 > 255 || r1 < 0 || g1>255 || g1 < 0 || b1>255 || b1 < 0) {
		cout << "Write appropriate RGB" << endl;
		while (1);
	}
	cout << "Write Assort R, G, B" << endl;
	cin >> r2;
	cin >> g2;
	cin >> b2;
	if (r2 > 255 || r2 < 0 || g2>255 || g2 < 0 || b2>255 || b2 < 0) {
		cout << "Write appropriate RGB" << endl;
		while (1);
	}
	cout << "Write Accent R, G, B" << endl;
	cin >> r3;
	cin >> g3;
	cin >> b3;
	if (r3 > 255 || r3 < 0 || g3>255 || g3 < 0 || b3>255 || b3 < 0) {
		cout << "Write appropriate RGB" << endl;
		while (1);
	}
	color_sample(p, r1, g1, b1, r2, g2, b2, r3, g3,b3);
}

void video_palette() {
	string name = get_image_name();
	VideoCapture vc(name);
	if (!vc.isOpened()){
		cout << "Must put valid name" << endl;
		while (1);
	}
	Mat img;
	cout << "How much color you want? , Number should be between 3~10" << endl;
	int numb_color;
	cin >> numb_color;
	if (numb_color < 3 || numb_color>10) {
		cout << "Must put valid number" << endl;
		while (1);
	}
	//Declaration
	Mat resizeImg;
	Mat k_mean;
	bool body_or_not = false;
	Rect body_reduced;
	Mat3b pal;
	map<Vec3b, int, lessVec3b> palette;
	int i;
	int lowSat, area;

	vector <Vec3b> dom_all, ass_all, acc_all;
	vector <int> weight_dom, weight_ass, weight_acc;

	Vec3b * colors_temp = new Vec3b[K];
	Vec3d * colors_lab = new Vec3d[K];
	int * weight = new int[K];
	bool *valid = new bool[K];
	double * colors_sats = new double[K];
	double dist_Mat[30][30];
	bool mark[30][30];

	int cnt = 1;
	int p = 0;
	//Video Part
	while (1) {
		vc >> img;
		if (img.empty())break;
		p++;
		if (p % 150 != 0) continue;
		for (int j = 0; j < K; ++j) valid[j]= true;
		for (int k = 0; k < 30; ++k) {
			for (int j = 0; j < K; ++j) {
				mark[j][k] = false;
			}
		}
		Vec3b dom, ass, acc;
		int dom_temp, ass_temp, acc_temp;

		//Use img as a picture
		resizeImg = img.clone();
		resize_limit(img, resizeImg, resize_factor, body_or_not, body_reduced);
		cvtColor(resizeImg, resizeImg, COLOR_BGR2Lab);

		/*Reducing Algorithm*/
		reduceColor_kmeans(resizeImg, k_mean, K);
		img.release();
		resizeImg.release();
		cvtColor(k_mean, k_mean, COLOR_Lab2BGR);

		pal = k_mean;
		palette = getPalette(pal);

		i = 0;
		/* 1st Color Palette */
		for (auto color : palette)
		{
			colors_temp[i] = color.first;
			weight[i] = color.second;
			i++;
		}
		palette.clear();
		/*Eliminate Low Sats*/
		bgr2S(colors_temp, colors_sats, K);
		lowSat = eliminate_low_sats(colors_sats, weight, K, valid);
		if (K - lowSat < numb_color)continue;
		//Color Palette RGB to Lab
		bgr2lab(colors_temp, colors_lab, K);
		area = 0;
		for (i = 0; i < K; ++i) {
			if (valid[i]) {
				area += weight[i];
			}
		}
		for (i = 0; i < K; ++i) {
			//cout << (int)colors_all[i].val[0] << " " << (int)colors_all[i].val[1] << " " << (int)colors_all[i].val[2] << endl;
			for (int j = 0; j < K; ++j) {
				//cout << (int)colors_all[j].val[0] << " " << (int)colors_all[j].val[1] << " " << (int)colors_all[j].val[2] << endl;
				//cout << "i = " << i << "j = " << j << endl;
				dist_Mat[i][j] = dist_3D(colors_lab[i].val[0] - colors_lab[j].val[0], colors_lab[i].val[1] - colors_lab[j].val[1], colors_lab[i].val[2] - colors_lab[j].val[2]);
				//cout << "distMat : " << dist_Mat[i][j] << endl;
			}
		}
		dist_gravity_fake_lab(K - numb_color - lowSat, dist_Mat, weight, colors_lab, mark, K, area, valid, body_or_not, false);

		lab2bgr(colors_temp, colors_lab, K);
		dom_ass_acc(colors_temp, numb_color, weight, dom, ass, acc, K, area, valid, body_or_not, false, false,false,false);

		int j;
		for (j = 0; j < K; ++j) {
			if (dom == colors_temp[j]) {
				dom_temp = weight[j];
				break;
			}
		}
		for (j = 0; j < K; ++j) {
			if (ass == colors_temp[j]) {
				ass_temp = weight[j];
				break;
			}
		}
		for (j = 0; j < K; ++j) {
			if (acc == colors_temp[j]) {
				acc_temp = weight[j];
				break;
			}
		}
		for (i = 0; i < dom_all.size(); ++i) {
			if (dom_all[i] == dom) {
				weight_dom[i] += dom_temp;
				break;
			}
		}
		if (i == dom_all.size()||dom_all.size()==0) {
			dom_all.push_back(dom);
			weight_dom.push_back(dom_temp);
		}
		for (i = 0; i < ass_all.size(); ++i) {
			if (ass_all[i] == ass)
			{
				weight_ass[i] += ass_temp;
				break;
			}
		}
		if (i == ass_all.size() || ass_all.size() == 0) {
			ass_all.push_back(ass);
			weight_ass.push_back(ass_temp);
		}
		for (i = 0; i < acc_all.size(); ++i) {
			if (acc_all[i] == acc) {
				weight_acc[i] += acc_temp;
				break;
			}
		}
		if (i == acc_all.size() || acc_all.size() == 0) {
			acc_all.push_back(acc);
			weight_acc.push_back(acc_temp);
		}

		cout << "processing "<<cnt << endl;
		cout << "p = " << p << endl;
		cnt++;
		//10 frame per sec
		waitKey(100);
		if (waitKey(10) == 27)break;
	}
	
	cout << "Video done" << endl;
	int area1=0, area2=0, area3=0;
	vector <Vec3d> dom_lab, ass_lab, acc_lab;
	vector <vector<bool>> mark1(dom_all.size(),vector<bool>(dom_all.size(),false)), mark2(ass_all.size(), vector<bool>(ass_all.size(), false)), mark3(acc_all.size(), vector<bool>(acc_all.size(), false));
	Vec3d temp;
	Vec3b pp;
	for (i = 0; i < dom_all.size(); ++i) {
		bgr2lab_single(dom_all[i], temp);
		dom_lab.push_back(temp);
		area1 += weight_dom[i];
	}
	for (i = 0; i < ass_all.size(); ++i) {
		bgr2lab_single(ass_all[i], temp);
		ass_lab.push_back(temp);
		area2 += weight_ass[i];

	}
	for (i = 0; i < acc_all.size(); ++i) {
		bgr2lab_single(acc_all[i], temp);
		acc_lab.push_back(temp);
		area3 += weight_acc[i];

	}
	vector<Vec3b> dd, aa, ac;

	dom_all.clear();
	ass_all.clear();
	acc_all.clear();

	int tr1, tr2, tr3;
	tr3 = numb_color / 3;
	tr2 = (numb_color - tr3) / 2;
	tr1 = (numb_color - tr2 - tr3);

	dist_gravity_fake_lab_video(dom_lab.size() - tr1, weight_dom, dom_lab, mark1, dom_lab.size(), area1);
	dist_gravity_fake_lab_video(ass_lab.size() - tr2, weight_ass, ass_lab, mark2, ass_lab.size(), area2);
	dist_gravity_fake_lab_video(acc_lab.size() - tr3, weight_acc, acc_lab, mark3, acc_lab.size(), area3);

	cnt = 0;
	int *weight_aft = new int[numb_color];
	int new_area = 0;
	int t;
	for (i = 0; i < dom_lab.size(); ++i) {
		for (t = 0; t < i; ++t) {
			if (dom_lab[i] == dom_lab[t]) break;
		}
		if (t == i) {
			lab2bgr_single(pp, dom_lab[i]);
			dd.push_back(pp);
			weight_aft[cnt] = weight_dom[i];
			new_area += weight_aft[cnt];
			cnt++;
		}
	}
	for (i = 0; i < ass_lab.size(); ++i) {
		for (t = 0; t < i; ++t) {
			if (ass_lab[i] == ass_lab[t]) break;
		}
		if (t == i) {
			lab2bgr_single(pp, ass_lab[i]);
			aa.push_back(pp);
			weight_aft[cnt] = weight_ass[i];
			new_area += weight_aft[cnt];
			cnt++;
		}
	}
	for (i = 0; i < acc_lab.size(); ++i) {
		for (t = 0; t < i; ++t) {
			if (acc_lab[i] == acc_lab[t]) break;
		}
		if (t == i) {
			lab2bgr_single(pp, acc_lab[i]);
			ac.push_back(pp);
			weight_aft[cnt] = weight_acc[i];
			new_area += weight_aft[cnt];
			cnt++;
		}
	}

	Vec3b * allcol = new Vec3b[numb_color];
	bool *valid_aft = new bool[numb_color];
	Vec3b dom, ass, acc;
	Mat drawCircle_multi;
	drawCircle_multi.create(200, 130 * numb_color + 100, CV_8UC3);
	drawCircle_multi = Scalar(0, 0, 0);
	for (i = 0; i < numb_color; ++i) {
		if (i < tr1) {
			circle(drawCircle_multi, Point(i * 130 + 100, 100), 50.0, dd[i], -1, 8);
			circle(drawCircle_multi, Point(i * 130 + 100, 100), 50.0, Scalar(150, 150, 150), 2, 8);
			allcol[i] = dd[i];
		}
		else if (i <tr1+tr2) {
			circle(drawCircle_multi, Point(i * 130 + 100, 100), 50.0, aa[i-tr1], -1, 8);
			circle(drawCircle_multi, Point(i * 130 + 100, 100), 50.0, Scalar(150, 150, 150), 2, 8);
			allcol[i] =aa[i -tr1];
		}
		else {
			circle(drawCircle_multi, Point(i * 130 + 100, 100), 50.0, ac[i - dom_all.size()-ass_all.size()], -1, 8);
			circle(drawCircle_multi, Point(i * 130 + 100, 100), 50.0, Scalar(150, 150, 150), 2, 8);
			allcol[i] = ac[i - tr1-tr2];
		}
		valid_aft[i] = true;
	}


	dom_ass_acc(allcol, numb_color, weight_aft, dom, ass, acc, numb_color, new_area, valid_aft, false,false, false, false, false);

	cout << "dom is " << dom << endl;
	cout << "ass is " << ass << endl;
	cout << "acc is" << acc << endl;

	Mat drawDomAssAcc;
	drawDomAssAcc.create(250, 450, CV_8UC3);
	drawDomAssAcc = Scalar(0, 0, 0);
	circle(drawDomAssAcc, Point(100, 150), 50.0, dom, -1, 8);
	circle(drawDomAssAcc, Point(230, 150), 50.0, ass, -1, 8);
	circle(drawDomAssAcc, Point(360, 150), 50.0, acc, -1, 8);
	circle(drawDomAssAcc, Point(100, 150), 50.0, Scalar(150, 150, 150), 2, 8);
	circle(drawDomAssAcc, Point(230, 150), 50.0, Scalar(150, 150, 150), 2, 8);
	circle(drawDomAssAcc, Point(360, 150), 50.0, Scalar(150, 150, 150), 2, 8);

	putText(drawDomAssAcc, "Dominant", Point(50, 50), 6, 1, Scalar::all(255));
	putText(drawDomAssAcc, "Assort", Point(200, 50), 6, 1, Scalar::all(255));
	putText(drawDomAssAcc, "Accent", Point(300, 50), 6, 1, Scalar::all(255));


	string fileName = save_image_name();
	imwrite(fileName+".png",drawCircle_multi);
	imwrite(fileName + "_colors.png", drawDomAssAcc);
	namedWindow("ALL");
	namedWindow("DOMASSACC");
	imshow("ALL", drawCircle_multi);
	imshow("DOMASSACC", drawDomAssAcc);
	waitKey(0);

}

int main()
{
	//handle_color();
	//video_palette();
	picture_palette();
	return 0;
}