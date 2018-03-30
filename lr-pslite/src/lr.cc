//#include "ps/ps.h"
#include "cmath"
#include "lr.h"
#include "util.h"
#include "sample.h"
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace distlr {

LR::LR(int num_feature_dim, float learning_rate, float C, int random_state): num_feature_dim_(num_feature_dim), learning_rate_(learning_rate), C_(C), random_state_(random_state) {
	InitWeight_();
}

//void LR::SetKVWorker(ps::KVWorker<float>* kv) {
//	kv_ = kv;
//}

void LR::Train(DataIter& iter, int batch_size = 100) {
	//while(iter.HashNext) {
	//	std::vector<Sample> batch = iter.NextBatch(batch_size);

	//	PullWeight_();

	//	std::vector<float> grad(weight_.size());

	//	for (size_t j = 0; j < weight_.size(); ++j) {
	//		grad[j] = 0;
	//		for (size_t i = 0; i < batch_size(); ++i) {
	//			auto& sample = batch[i];
	//			grad[j] += (Sigmoid_(sample.GetFeature()) - sample.GetLabel()) * sample.GetFeature(j);
	//		}
	//		grad[j] = 1. * grad[j] / batch.size() + C_ * weight_[j] / batch.size();
	//	}
	//	PushGradient_(grad);
	//}
}

std::vector<float> LR::GetWeight() {
	return weight_;
}

//ps::KVWorker<float>* LR::GetKVWorker() {
//	return kv_;
//}	

bool LR::SaveModel(std::string& filename) {
	std::ofstream fout(filename.c_str());
	fout << num_feature_dim_ << std::endl;
	for(int i = 0; i < num_feature_dim_; ++i) {
		fout << weight_[i] << ' ';
	}
	fout << std::endl;
	fout.close();
	return true;
}

void LR::InitWeight_() {
	srand(random_state_);
	weight_.resize(num_feature_dim_);
	for (size_t i = 0; i < weight_.size(); ++i) {
		weight_[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}
}

float LR::Sigmoid_(std::vector<float> feature) {
	float z = 0;
	for (size_t j = 0; j < weight_.size(); ++j) {
		z += weight_[j] * feature[j];
	}
	return 1. / (1. + exp(-z));
}

//void LR::PullWeight_() {
//	std::vector<ps::Key> keys(num_feature_dim_);
//	std::vector<float> vals;
//	for (int i = 0; i < num_feature_dim_; ++i) {
//		keys[i] = i;
//	}
//	kv_->Wait(kv_->Pull(keys, &vals));
//	weight_ = vals;
//}
//
//void LR::PushGradient_(const std::vector<float>& grad) {
//	std::vector<ps::Key> keys(num_feature_dim_);
//	for (int i = 0; i < num_feature_dim_; ++i) {
//		keys[i] = i;
//	}
//	kv_->Wait(kv_->Push(keys, grad));
//}
}

