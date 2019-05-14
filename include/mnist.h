#include <torch/torch.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

struct Options {
	std::string data_root{ "./data/mnist" };
	int32_t batch_size{ 64 };
	int32_t epochs{ 10 };
	double lr{ 0.01 };
	double momentum{ 0.5 };
	bool no_cuda{ false };
	int32_t seed{ 1 };
	int32_t test_batch_size{ 1000 };
	int32_t log_interval{ 10 };
};

struct Normalize : public torch::data::transforms::TensorTransform<> {
	Normalize(float mean, float stddev)
		: mean_(torch::tensor(mean)), stddev_(torch::tensor(stddev)) {}
	torch::Tensor operator()(torch::Tensor input) {
		return input.sub_(mean_).div_(stddev_);
	}
	torch::Tensor mean_, stddev_;
};

struct Net : torch::nn::Module {
	Net()
		: conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
		conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
		fc1(320, 50),
		fc2(50, 10) {
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv2_drop", conv2_drop);
		register_module("fc1", fc1);
		register_module("fc2", fc2);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
		x = torch::relu(
			torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
		x = x.view({ -1, 320 });
		x = torch::relu(fc1->forward(x));
		x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
		x = fc2->forward(x);
		return torch::log_softmax(x, /*dim=*/1);
	}

	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::FeatureDropout conv2_drop;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
};
