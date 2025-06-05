# neuralforge

A complete implementation of a feedforward neural network built from scratch using only NumPy for mathematical operations.

## 🎯 Project Overview

This project demonstrates a deep understanding of neural network fundamentals by implementing:
* Forward and backward propagation algorithms
* Multiple activation functions (ReLU, Sigmoid, Tanh)
* Gradient descent optimization
* Multi-class classification on MNIST dataset

## 🚀 Key Features

* **Pure Python Implementation**: No TensorFlow, PyTorch, or Keras dependencies
* **Configurable Architecture**: Easily adjust layer sizes and activation functions
* **Multiple Activation Functions**: ReLU, Sigmoid, Tanh with proper derivatives
* **Comprehensive Visualization**: Training curves, sample predictions, and accuracy plots
* **Robust Error Handling**: Numerical stability and dimension checking

## 📊 Results

* **Final Training Accuracy**: 86.20%
* **Final Test Accuracy**: 86.17%
* **Architecture**: 784 → 128 → 32 → 10 neurons
* **Training Epochs**: 200
* **Learning Rate**: 0.05

## 🛠️ Technical Implementation

### Core Components
* **Forward Propagation**: Matrix multiplication with activation functions
* **Backward Propagation**: Chain rule implementation for gradient computation
* **Parameter Updates**: Gradient descent with learning rate optimization
* **Cost Function**: Cross-entropy loss with numerical stability

### Mathematical Foundations

```
# Forward pass
Z = W·A + b
A = activation(Z)

# Backward pass  
dW = (1/m) * dZ · A_prev^T
db = (1/m) * sum(dZ)
```

## 🏃‍♂️ Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/neuralforge.git
cd neuralforge

# Install dependencies
pip install -r requirements.txt

# Run the neural network
python neural_network.py
```

## 📈 Performance Analysis

The model achieves solid results for a from-scratch implementation:
- Consistent training and test accuracy (~86%) indicates good generalization
- No significant overfitting observed
- Smooth convergence as shown in training curves

## 🔧 Customization

```python
# Modify architecture in neural_network.py
layers = [784, 256, 128, 10]  # Custom layer sizes
learning_rate = 0.01          # Adjust learning rate
epochs = 300                  # More training epochs
```

## 🎓 Learning Outcomes

This project demonstrates:
- Deep understanding of neural network mathematics
- Ability to implement complex algorithms from scratch
- Proficiency in NumPy and Python for numerical computing
- Data visualization and model evaluation skills

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📚 References

* MNIST Dataset: Yann LeCun's MNIST Database
* Neural Networks and Deep Learning by Michael Nielsen

## 📄 License

MIT License - feel free to use this code for learning and projects!
