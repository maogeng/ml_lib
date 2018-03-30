#ifndef DISTLR_LR_H_
#define DISTLR_LR_H_

#include "data_iter.h"

namespace distlr {

class LR {
public:
	explicit LR(int num_feature_dim, float learning_rate=0.001, float C_=1, int random_state=0);
	virtual ~LR() {
	}

	void Train(DataIter& iter, int num_iter);

	void Test(DataIter& iter, int num_iter);

	std::vector<float> GetWeight();

	bool SaveModel(std::string& filename);

private:
	void InitWeight_();

	int Predict_(std::vector<float> feature);

	float Sigmoid_(std::vector<float> feature);

	void PullWeight_();

	int num_feature_dim_;

	float learning_rate_;

	float C_;

	int random_state_;

	std::vector<float> weight_;
};

} // namespace distlr

#endif // LR_H_
