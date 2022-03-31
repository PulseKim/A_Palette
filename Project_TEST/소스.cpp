/*cout << "Enter the name of file" << endl;
string out;
cin >> out;

Mat inputImg;
Mat resizeImg;
inputImg = imread(out, CV_LOAD_IMAGE_COLOR);
//	resize(inputImg, inputImg, Size(), SCALE, SCALE, CV_INTER_AREA);
cout << inputImg.rows << endl;
cout << inputImg.cols << endl;

resizeImg = resize_limit(inputImg, 100000);

int(*color_hist)[256][256] = histo(resizeImg);

int max_r = 0;
int max_g = 0;
int max_b = 0;
int maxVal = 0;

int i, j, k;

for (i = 0; i < 256; ++i) {
	for (j = 0; j < 256; ++j) {
		for (k = 0; k < 256; ++k) {
			if (color_hist[i][j][k] > maxVal) {
				maxVal = color_hist[i][j][k];
				max_r = i;
				max_g = j;
				max_b = k;
			}
		}
	}
}

Mat drawCircle = Mat::zeros(400, 400, CV_8UC3);
circle(drawCircle, Point(200, 200), 32.0, CV_RGB(max_r, max_g, max_b), -1, 8);

MatND histogramB, histogramG, histogramR;
const int channel_numbersB[] = { 0 };  // Blue
const int channel_numbersG[] = { 1 };  // Green
const int channel_numbersR[] = { 2 };  // Red
float channel_range[] = { 0.0, 255.0 };
const float* channel_ranges = channel_range;
int number_bins = 255;

Mat dominant;

int dom_Blue;
int dom_Red;
int dom_Green;

// R, G, B별로 각각 히스토그램을 계산한다.
calcHist(&resizeImg, 1, channel_numbersB, Mat(), histogramB, 1, &number_bins, &channel_ranges);
calcHist(&resizeImg, 1, channel_numbersG, Mat(), histogramG, 1, &number_bins, &channel_ranges);
calcHist(&resizeImg, 1, channel_numbersR, Mat(), histogramR, 1, &number_bins, &channel_ranges);

// Plot the histogram
int hist_w = 512; int hist_h = 400;
int bin_w = cvRound((double)hist_w / number_bins);

Mat histImageB(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
normalize(histogramB, histogramB, 0, histImageB.rows, NORM_MINMAX, -1, Mat());

Mat histImageG(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
normalize(histogramG, histogramG, 0, histImageG.rows, NORM_MINMAX, -1, Mat());

Mat histImageR(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
normalize(histogramR, histogramR, 0, histImageR.rows, NORM_MINMAX, -1, Mat());

for (int i = 1; i < number_bins; i++)
{

	line(histImageB, Point(bin_w*(i - 1), hist_h - cvRound(histogramB.at<float>(i - 1))),
		Point(bin_w*(i), hist_h - cvRound(histogramB.at<float>(i))),
		Scalar(255, 0, 0), 2, 8, 0);
	line(histImageG, Point(bin_w*(i - 1), hist_h - cvRound(histogramG.at<float>(i - 1))),
		Point(bin_w*(i), hist_h - cvRound(histogramG.at<float>(i))),
		Scalar(0, 255, 0), 2, 8, 0);

	line(histImageR, Point(bin_w*(i - 1), hist_h - cvRound(histogramR.at<float>(i - 1))),
		Point(bin_w*(i), hist_h - cvRound(histogramR.at<float>(i))),
		Scalar(0, 0, 255), 2, 8, 0);

}




namedWindow("Original", CV_WINDOW_AUTOSIZE);
namedWindow("HistogramB", CV_WINDOW_AUTOSIZE);
namedWindow("HistogramG", CV_WINDOW_AUTOSIZE);
namedWindow("HistogramR", CV_WINDOW_AUTOSIZE);
namedWindow("MainColor", CV_WINDOW_AUTOSIZE);

moveWindow("Original", 100, 100);
moveWindow("HistogramB", 110, 110);
moveWindow("HistogramG", 120, 120);
moveWindow("HistogramR", 130, 130);
moveWindow("MainColor", 150, 150);


imshow("Original", resizeImg);
imshow("HistogramB", histImageB);
imshow("HistogramG", histImageG);
imshow("HistogramR", histImageR);
imshow("MainColor", drawCircle);

waitKey(0);
return 0;
*/