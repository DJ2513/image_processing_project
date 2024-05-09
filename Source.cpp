#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

const double PI = 3.14926;

int tx = 0; 
int ty = 0;
int degrees = 0;  
int sx = 0; 
int sy = 0;
int skh = 0;
int skv = 0;

int w1 = 0;
int w2 = 0; 

int wr1 = 0;
int wr2 = 0;

Mat img = imread("./foto.jpg", IMREAD_GRAYSCALE);


Mat Translation = (Mat_<double>(3,3)<<1,0,tx,0,1,ty,0,0,1);
Mat Rotation = (Mat_<double>(3, 3)<<cos(0), sin(0),0,-sin(0),cos(0),0,0,0,1);
Mat Scale = (Mat_<double>(3,3) <<1, 0, 0, 0, 1, 0, 0, 0, 1);
Mat SkewH = (Mat_<double>(3,3) <<1, skh, 0, 0, 1, 0, 0, 0, 1);
Mat SkewV = (Mat_<double>(3,3) <<1, 0, 0, skv, 1, 0, 0, 0, 1);

int interpolate(Mat img, float u, float v) {
	int x1 = (int)floor(u);
	int x2 = (int)ceil(u);
	int y1 = (int)ceil(v);
	int y2 = (int)floor(v);

	if (u < 0 || v < 0 || u >= img.cols || v >= img.rows) return 0;
	if (x2 >= img.cols || x1 >= img.cols || y1 >= img.rows || y2 >= img.rows) return 0;
	if (x2 < 0 || x1 < 0 || y1 < 0 || y2 < 0) return 0;
	if (x1 == x2 && y1 == y2) return img.at<uchar>(y1, x1);

	float Q11 = img.at<uchar>(y1, x1);
	float Q12 = img.at<uchar>(y1, x2);
	float Q21 = img.at<uchar>(y2, x1);
	float Q22 = img.at<uchar>(y2, x2);

	if (x2 == x1) {
		return ((Q22 - Q11) / (y2 - y1))* (v - y1) + Q11;
	}
	if (y1 == y2) {
		return ((Q21 - Q11) / (x2 - x1)) * (u - x1) + Q11;
	}
	float R1 = ((Q21 - Q11) / (x2 - x1)) * (u - x1) + Q11;
	float R2 = ((Q22 - Q12) / (x2 - x1)) * (u - x1) + Q12;
	int P = ((R2 - R1) / (y2 - y1)) * (v - y1) + R1;
	return P;
}


Mat Wave1(Mat img){
	Mat res(img.rows, img.cols, CV_8UC1);
	for (int row = 0; row < res.rows; row++){
		for(int col = 0; col < res.cols; col++){
			float u = col + (w1 * sin(2 * 3.1416 * row/128));
			float v = row;
			res.at<uchar>(row, col) = interpolate(img, u, v);
		}
	}
	return res; 
}

Mat Wave2(Mat img) {
	Mat res(img.rows, img.cols, CV_8UC1);
	for (int row = 0; row < res.rows; row++) {
		for (int col = 0; col < res.cols; col++) {
			float u = col + (w2 * sin(2 * 3.1416 * col / 30));
			float v = row;
			res.at<uchar>(row, col) = interpolate(img, u, v);
		}
	}
	return res;
}

Mat Warp1(Mat img){
	Mat res(img.rows, img.cols, CV_8UC1);
	
	int y0 = (res.rows/2); 
	int x0 = (res.cols/2);
	for (int y = 0; y < res.rows; y++) {
		float sign = y - y0 >= 0 ? 1 : -1; 
		for (int x = 0; x < res.cols; x++) {
			float v = (((100 - wr1) / (float)100) * sign * (pow(y - y0, 2) / y0)) + y0;
			if(wr1 == 0) v = y;
			float u = x; 
			res.at<uchar>(y, x) = interpolate(img, u, v);
		}
	}
	return res;
}


Mat Warp2(Mat img) {
	Mat res(img.rows, img.cols, CV_8UC1);
	Mat toCenter = (Mat_<double>(3,3)<<1,0,img.cols/2,0,1,img.rows/2,0,0,1);
	toCenter = toCenter.inv();
	int y0 = (img.rows / 2);
	int x0 = (img.cols / 2);
	for (int y = 0; y < res.rows; y++) {
		for (int x = 0; x < res.cols; x++) {
			float radius = pow(pow(x - x0, 2) + pow(y - y0, 2), 0.5);
			float theta = (PI * radius * wr2 / (512 * 100));
			float c = cos(theta);
			float s = sin(theta);

			float u = ((x - x0)*c) + ((y - y0)*s) + x0;
			float v = ((-x + x0)*s) + ((y - y0)*c) + y0;

			res.at<uchar>(y,x) = interpolate(img, u, v);
		}
	}
	return res;
}

Mat transformImage(Mat img){
	
	Mat toCenter = (Mat_<double>(3,3)<<1,0,img.cols/2,0,1,img.rows/2,0,0,1);
	Mat toOrigin = (Mat_<double>(3,3)<<1,0,-img.cols/2,0,1,-img.rows/2,0,0,1);
	Mat res(img.rows * (1 + (sx/(float)100)), img.cols * (1 + (sx/(float)100)), CV_8UC1);
	Mat Transformation = (toCenter * Translation * Rotation * toOrigin) * Scale * SkewH * SkewV;
	Transformation = Transformation.inv();

	for (int row = 0; row < res.rows; row++){
		for(int col = 0; col < res.cols; col++){
			Mat Coords = (Mat_<double>(3,1)<<col, row, 1); 
			Mat UV = Transformation * Coords; 

			double u = UV.at<double>(0,0); 
			double v = UV.at<double>(1,0); 
			res.at<uchar>(row, col) = interpolate(img, u, v);
		}
	}

	

	return res; 
}

static void on_trackbar_tx(int, void*){
	Translation = (Mat_<double>(3,3)<<1,0,tx,0,1,ty,0,0,1);
}

static void on_trackbar_ty(int, void*){
	Translation = (Mat_<double>(3,3)<<1,0,tx,0,1,ty,0,0,1);
}

static void on_trackbar_rad(int, void*){
	double radians = degrees * 3.1416 / (float)180;
	Rotation = (Mat_<double>(3, 3)<<cos(radians), sin(radians),0,-sin(radians),cos(radians),0,0,0,1); 
}

static void on_trackbar_sx(int, void*){
	Scale = (Mat_<double>(3,3) <<1 + (sx/(float)100), 0, 0, 0, 1 + (sy/(float)100), 0, 0, 0, 1); 
}

static void on_trackbar_sy(int, void*){
	Scale = (Mat_<double>(3,3) <<1 + (sx/(float)100), 0, 0, 0, 1 + (sy/(float)100), 0, 0, 0, 1); 
}

static void on_trackbar_skh(int, void*){
	SkewH = (Mat_<double>(3,3) <<1, skh / (float)100, 0, 0, 1, 0, 0, 0, 1);
}

static void on_trackbar_skv(int, void*){
	SkewV = (Mat_<double>(3,3) <<1, 0, 0, skv / (float)100, 1, 0, 0, 0, 1);
}


int main() {
	namedWindow("lineal options", WINDOW_AUTOSIZE);
	namedWindow("non lineal options", WINDOW_AUTOSIZE);

	namedWindow("Result", WINDOW_AUTOSIZE);
	createTrackbar("Translate X", "lineal options", &tx, 100, on_trackbar_tx);
	createTrackbar("Translate Y", "lineal options", &ty, 100, on_trackbar_ty);
	createTrackbar("Rotation (rad)", "lineal options", &degrees, 360, on_trackbar_rad);
	createTrackbar("Scale X", "lineal options", &sx, 200, on_trackbar_sx);
	createTrackbar("Scale Y", "lineal options", &sy, 200, on_trackbar_sy);
	createTrackbar("Skew H", "lineal options", &skh, 100, on_trackbar_skh);
	createTrackbar("Skew V", "lineal options", &skv, 100, on_trackbar_skv);

	createTrackbar("Wave1", "non lineal options", &w1, 100);
	createTrackbar("Wave2", "non lineal options", &w2, 100);
	createTrackbar("Warp1", "non lineal options", &wr1, 100);
	createTrackbar("Warp2", "non lineal options", &wr2, 500);

	VideoCapture cap(0);  // Abre la camara por default

	cap.set(CAP_PROP_FRAME_WIDTH, 100);
	cap.set(CAP_PROP_FRAME_HEIGHT, 100);

	if (!cap.isOpened()) { //Revisa si la camara abrio 
		return -1;
	}
	Mat res = img;
	cap >> res;

	bool withTrasnformation = false;
	bool warping = false;
	bool waving = false;
	while (true) {
		cap >> img; // obtiene un nuevo cuadro de la cámara
		int key = waitKey(10);
		cvtColor(img, img, COLOR_RGB2GRAY); //change to b&w
		if(key == 'q') break;

		if (key == ' ') withTrasnformation = !withTrasnformation;
		if (key == 'w') warping = !warping;
		if (key == 'e') waving = !waving;

		if(withTrasnformation){
			res = transformImage(img);
		}
		if (waving) {
			res = Wave1(img);
			res = Wave2(res);
			
		}
		if (warping) {
			res = Warp1(img);
			res = Warp2(res);
		}
		if(!withTrasnformation && !warping && !waving){
			res = img;
		}
		imshow("Result", res);
	}

	return 0;
}

