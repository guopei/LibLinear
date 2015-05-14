// Train and test liblinear model in C++ format.
//
#include <iostream>
#include "LibLinear.hpp"
using namespace std;

int main(int argc, char* argv[])
{
    // Set up training data
    float labels[6] = { 1.0, 1.0, 1.0, -1.0, -1.0, -1.0 };
    Mat labelsMat(6, 1, CV_32FC1, labels);
    
    float trainingData[6][2] = { { 3, 1 }, { 3, -1 }, { 6, 1 },
        { -3, 1 }, { -3, -1 }, { -6, 0 } };
    Mat trainingDataMat(6, 2, CV_32FC1, trainingData);
    
    float testData[2][2] = { { 4, 0 }, {-1, 0} };
    Mat testDataMat(2, 2, CV_32FC1, testData);
    
    LibLinear *linear = new LibLinear;
    parameter param = LinearParam::construct_param(0);
    linear->train(trainingDataMat, labelsMat, param);
    
    // predict one sample each call
    for (int i = 0; i < testDataMat.rows; i++){
        Mat sampleMat = testDataMat.row(i);
        double out = linear->predict(sampleMat);
        cout<<out<<" ";
    }
    cout<<endl;
    
    // predict all sample in one call
    Mat outputMat;
    linear->predict(testDataMat, outputMat);
    cout<<outputMat<<endl;
    
    // predict value of one sample.
    for (int i = 0; i < testDataMat.rows; i++){
        Mat sampleMat = testDataMat.row(i);
        Mat valueMat;
        double out = linear->predict_values(sampleMat, valueMat);
        cout<<out<<" "<<endl;
        cout<<valueMat<<endl;
    }
    
    // predict probability of one sample
    for (int i = 0; i < testDataMat.rows; i++){
        Mat sampleMat = testDataMat.row(i);
        Mat probMat;
        double out = linear->predict_probabilities(sampleMat, probMat);
        cout<<out<<" "<<endl;
        cout<<probMat<<endl;
    }
    
    delete linear;
    
    return 0;
}