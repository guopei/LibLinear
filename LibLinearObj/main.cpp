// Train and test liblinear model in C++ format.
//
#include <iostream>
#include "linear.h"
#include "LibLinear.hpp"
using namespace std;

int main(int argc, char* argv[])
{
	// Set up training data
	float labels[4] = { 1.0, -1.0, -1.0, -1.0 };
	Mat labelsMat(4, 1, CV_32FC1, labels);

	float trainingData[4][2] = { { 501, 10 }, { 255, 10 }, { 501, 255 }, { 10, 501 } };
	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

	float testData[2] = { 100, 100 };
	Mat testDataMat(1, 2, CV_32FC1, testData);
    int i = 100;
    while (i>0) {
        LibLinear *linear = new LibLinear;
        Mat outputMat;
        parameter param = LinearParam::construct_param(0);
        linear->train(trainingDataMat, labelsMat, param);
        linear->predict(testDataMat, outputMat);
        delete linear;
        cout<<outputMat;
        i--;
    }

	return 0;
}