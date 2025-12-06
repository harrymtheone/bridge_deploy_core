#ifndef BRIDGE_CORE_TENSOR_HELPERS_HPP
#define BRIDGE_CORE_TENSOR_HELPERS_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <onnxruntime_cxx_api.h>

namespace bridge_core {

/**
 * @brief Lightweight builder for populating vectors element by element
 * 
 * Takes a reference to an existing vector and provides a fluent interface
 * for populating it. The builder does not own the data.
 * 
 * Usage:
 *   std::vector<float> proprio(56);
 *   
 *   TensorBuilder(proprio)
 *       .add(ang_vel)
 *       .add(gravity_proj)
 *       .add(clock_sin)
 *       .add(commands)
 *       .clip(-100.0f, 100.0f);
 */
class TensorBuilder {
public:
    /**
     * @brief Construct builder that writes to an existing vector
     * @param data Reference to vector to populate
     */
    explicit TensorBuilder(std::vector<float>& data) 
        : data_(data), idx_(0) {}
    
    /**
     * @brief Add a single float value
     * @param value The value to add
     * @return Reference to this for chaining
     */
    TensorBuilder& add(float value) {
        if (idx_ >= data_.size()) {
            throw std::out_of_range("TensorBuilder overflow: tried to add beyond capacity");
        }
        data_[idx_++] = value;
        return *this;
    }
    
    /**
     * @brief Add elements from a std::vector
     * @param vec Vector of values to add
     * @return Reference to this for chaining
     */
    TensorBuilder& add(const std::vector<float>& vec) {
        for (float v : vec) {
            add(v);
        }
        return *this;
    }
    
    /**
     * @brief Add elements from a std::array
     * @param arr Array of values to add
     * @return Reference to this for chaining
     */
    template<size_t N>
    TensorBuilder& add(const std::array<float, N>& arr) {
        for (float v : arr) {
            add(v);
        }
        return *this;
    }
    
    /**
     * @brief Add elements from a raw pointer
     * @param ptr Pointer to float array
     * @param count Number of elements to add
     * @return Reference to this for chaining
     */
    TensorBuilder& add(const float* ptr, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            add(ptr[i]);
        }
        return *this;
    }
    
    /**
     * @brief Clip all values to a range
     * @param min_val Minimum value
     * @param max_val Maximum value
     * @return Reference to this for chaining
     */
    TensorBuilder& clip(float min_val, float max_val) {
        for (float& v : data_) {
            v = std::clamp(v, min_val, max_val);
        }
        return *this;
    }

private:
    std::vector<float>& data_;
    size_t idx_;
};

/**
 * @brief Helper class for extracting data from ONNX output tensors
 */
class TensorExtractor {
public:
    /**
     * @brief Construct extractor with output tensors
     * @param tensors Vector of output tensors from ONNX Runtime
     */
    explicit TensorExtractor(std::vector<Ort::Value>& tensors) 
        : tensors_(tensors) {}
    
    /**
     * @brief Extract tensor data to a new vector
     * @param tensor_idx Index of the output tensor to extract
     * @return Vector containing the tensor data
     */
    std::vector<float> extractToVector(size_t tensor_idx) {
        if (tensor_idx >= tensors_.size()) {
            throw std::out_of_range("TensorExtractor: tensor index out of range");
        }

        float* data = tensors_[tensor_idx].GetTensorMutableData<float>();
        auto shape = tensors_[tensor_idx].GetTensorTypeAndShapeInfo().GetShape();
        
        // Calculate total size
        size_t total_size = 1;
        for (auto dim : shape) {
            total_size *= static_cast<size_t>(dim);
        }

        std::vector<float> result(total_size);
        std::copy(data, data + total_size, result.begin());
        return result;
    }

    /**
     * @brief Extract tensor data into an existing vector
     * @param tensor_idx Index of the output tensor to extract
     * @param dest Reference to destination vector
     * @return Reference to this for chaining
     */
    TensorExtractor& extractTo(size_t tensor_idx, std::vector<float>& dest) {
        if (tensor_idx >= tensors_.size()) {
            // Optional outputs might not be present, skip if index is out of bounds
            return *this;
        }

        float* data = tensors_[tensor_idx].GetTensorMutableData<float>();
        auto shape = tensors_[tensor_idx].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t total_size = 1;
        for (auto dim : shape) {
            total_size *= static_cast<size_t>(dim);
        }

        if (dest.size() != total_size) {
            // Resize if needed, but prefer pre-sized vectors for performance
            dest.resize(total_size);
        }

        std::copy(data, data + total_size, dest.begin());
        return *this;
    }

    /**
     * @brief Get raw pointer to tensor data
     * @param tensor_idx Index of the output tensor
     * @return Pointer to float data
     */
    float* getData(size_t tensor_idx) {
        if (tensor_idx >= tensors_.size()) {
            throw std::out_of_range("TensorExtractor: tensor index out of range");
        }
        return tensors_[tensor_idx].GetTensorMutableData<float>();
    }

    /**
     * @brief Get size of a specific tensor
     * @param tensor_idx Index of the output tensor
     * @return Total number of elements
     */
    size_t getSize(size_t tensor_idx) {
        if (tensor_idx >= tensors_.size()) {
            throw std::out_of_range("TensorExtractor: tensor index out of range");
        }
        auto shape = tensors_[tensor_idx].GetTensorTypeAndShapeInfo().GetShape();
        size_t total_size = 1;
        for (auto dim : shape) {
            total_size *= static_cast<size_t>(dim);
        }
        return total_size;
    }

private:
    std::vector<Ort::Value>& tensors_;
};

} // namespace bridge_core

#endif // BRIDGE_CORE_TENSOR_HELPERS_HPP

