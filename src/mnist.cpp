#include "mnist.h"
#include "model.h"

#include <torch/torch.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>


template <typename DataLoader>
void train(
	int32_t epoch,
	const Options& options,
	Net& model,
	torch::Device device,
	DataLoader& data_loader,
	torch::optim::SGD& optimizer,
	size_t dataset_size) {
	model.train();
	size_t batch_idx = 0;
	for (auto& batch : data_loader) {
		auto data = batch.data.to(device), targets = batch.target.to(device);
		optimizer.zero_grad();
		auto output = model.forward(data);
		auto loss = torch::nll_loss(output, targets);
		loss.backward();
		optimizer.step();

		if (batch_idx++ % options.log_interval == 0) {
			std::cout << "Train Epoch: " << epoch << " ["
				<< batch_idx * batch.data.size(0) << "/" << dataset_size
				<< "]\tLoss: " << loss.template item<float>() << std::endl;
		}
	}
}

template <typename DataLoader>
void test(
	Net& model,
	torch::Device device,
	DataLoader& data_loader,
	size_t dataset_size) {
	torch::NoGradGuard no_grad;
	model.eval();
	double test_loss = 0;
	int32_t correct = 0;
	for (const auto& batch : data_loader) {
		auto data = batch.data.to(device), targets = batch.target.to(device);
		auto output = model.forward(data);
		test_loss += torch::nll_loss(
			output,
			targets,
			/*weight=*/{},
			Reduction::Sum)
			.template item<float>();
		auto pred = output.argmax(1);
		correct += pred.eq(targets).sum().template item<int64_t>();
	}

	test_loss /= dataset_size;
	std::cout << "Test set: Average loss: " << test_loss
		<< ", Accuracy: " << correct << "/" << dataset_size << std::endl;
}


auto main(int argc, const char* argv[]) -> int {
	torch::manual_seed(0);

	Options options;
	torch::DeviceType device_type;
	if (torch::cuda::is_available() && !options.no_cuda) {
		std::cout << "CUDA available! Training on GPU" << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "Training on CPU" << std::endl;
		device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	Net model;
	model.to(device);

	auto train_dataset =
		torch::data::datasets::MNIST(
			options.data_root, torch::data::datasets::MNIST::Mode::kTrain)
		.map(Normalize(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());
	const auto dataset_size = train_dataset.size();

	auto train_loader = torch::data::make_data_loader(
		std::move(train_dataset), options.batch_size);

	auto test_loader = torch::data::make_data_loader(
		torch::data::datasets::MNIST(
			options.data_root, torch::data::datasets::MNIST::Mode::kTest)
		.map(Normalize(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>()),
		options.batch_size);

	torch::optim::SGD optimizer(
		model.parameters(),
		torch::optim::SGDOptions(options.lr).momentum(options.momentum));

	for (size_t epoch = 1; epoch <= options.epochs; ++epoch) {
		train(
			epoch, options, model, device, *train_loader, optimizer, dataset_size.value());
		test(model, device, *test_loader, dataset_size.value());
	}
}
