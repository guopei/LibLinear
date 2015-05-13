#pragma once
#include "linear.h"
#include <opencv.hpp>
using namespace std;
using namespace cv;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class LinearParam{
public:
	static parameter construct_param(){
		return parameter{ L2R_L2LOSS_SVC_DUAL, 0.1, 1, 0, NULL, NULL, 0.1 };
	}
	static parameter construct_param(int solver_type,
		double eps,
		double C,
		int nr_weight,
		int *weight_label,
		double* weight,
		double p){
		return parameter{ solver_type, eps, C, nr_weight, weight_label, weight, p };
	}
};


class LibLinear{
private:
	struct parameter _param;
	struct problem _prob;
	struct model *_model;
	feature_node *_sample;

	/**
	 * prob: l->number of samples(labels), n->number of features, bias->-1 by default,
	 * x->pointer to feature data, y->pointer to label data.
	 */
	void convert_train_data(const Mat &FeatureMat, const Mat &LabelMat, problem &prob){
		if (FeatureMat.type() != CV_32FC1 || LabelMat.type() != CV_32FC1)
			cerr << "sorry, feature mat and label mat should be float\n";
		prob.bias = -1; // in this version, no bias term is considered.
		prob.l = FeatureMat.rows;
		prob.n = FeatureMat.cols;
		prob.x = Malloc(feature_node *, prob.l);
		prob.y = Malloc(double, prob.l);
		int feature_num = prob.n + 1;
		_sample = Malloc(feature_node, prob.l*feature_num);
		for (int i = 0; i < prob.l; i++)
		{
			for (int j = 0; j < prob.n; j++)
			{
				_sample[i*feature_num+j].index = j+1;
				_sample[i*feature_num+j].value = FeatureMat.at<float>(i, j);
			}
			_sample[(i+1)*feature_num-1].index = -1;
			prob.x[i] = &_sample[i*feature_num];
		}
		for (int i = 0; i < LabelMat.rows; i++)
			prob.y[i] = LabelMat.at<float>(i);
	}
	/**
	 * one sample each call.
	 */
	void convert_test_data(const Mat &SampleMat, feature_node **x){
		int feature_num = SampleMat.cols + 1;
		*x = Malloc(feature_node, feature_num);
		for (int i = 0; i < SampleMat.cols; i++){
			(*x)[i].index = i+1;
			(*x)[i].value = SampleMat.at<float>(i);
		}
		(*x)[feature_num - 1].index = -1;
	}

public:
	~LibLinear(){
		destroy_param(&_param);
        if(_model)
            free_and_destroy_model(&_model);
		if(_prob.y)
            free(_prob.y);
		if(_prob.y)
            free(_prob.x);
		if(_sample)
            free(_sample);
	}
	LibLinear(){
		_param = parameter();
		_prob = problem();
		_model = nullptr;
		_sample = nullptr;
	}
	void save_model(string filename){
        ::save_model(filename.c_str(), _model);
	}
	/**
	 * FeatureMat: M samples * N features.
	 * LabelMat: N lables * 1.
	 * param: liblinear parameters.
	 */
	void train(Mat &FeatureMat,
		Mat &LabelMat,
		parameter &param){
		_param = param;
		convert_train_data(FeatureMat, LabelMat, _prob);
        if(_model){
            free_and_destroy_model(&_model);
        }
        const int64 s1 = cv::getTickCount();
        _model = ::train(&_prob, &_param);
        const int64 s2 = cv::getTickCount();
        fprintf(stdout, "train finished! Use %8d s\n", static_cast<int>((s2 - s1) / cv::getTickFrequency()));
        // clear internal data right after model is trained.
        // as they are useless now.
        if(_sample){
            free(_prob.x);
            free(_prob.y);
            free(_sample);
            _prob.x = nullptr;
            _prob.y = nullptr;
            _sample = nullptr;
        }
	}
    
    /**
     * SampleMat contains only one sample.
     */
    double predict(Mat &SampleMat){
        assert(SampleMat.rows == 1);
        feature_node *x = nullptr;
        convert_test_data(SampleMat, &x);
        double out = ::predict(_model, x);
        free(x);
        return out;
    }
    
    /**
     * SamplesMat contains multiple samples in multiple rows.
     * OutputMat is the predicted labels for each sample.
     */
	void predict(Mat &SamplesMat, Mat &OutputMat){
		feature_node *x = nullptr;
		OutputMat.create(SamplesMat.rows, 1, CV_32FC1);
		for (int i = 0; i < SamplesMat.rows; i++){
			convert_test_data(SamplesMat.row(i), &x);
            double out = ::predict(_model, x);
			OutputMat.at<float>(i) = out;
			free(x);
		}
	}
    void load_model(string model_file_name){
        if(_model){
            free_and_destroy_model(&_model);
        }
        _model = ::load_model(model_file_name.c_str());
    }
};