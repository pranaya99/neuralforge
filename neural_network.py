import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class NeuralNetwork:
    """
    A neural network implementation from scratch for multiclass classification.
    Supports multiple hidden layers and different activation functions.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, activation='relu', 
                 num_classes=10, architecture=[128, 32]):
        """
        Initialize the neural network.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - X_test: Test features  
        - y_test: Test labels
        - activation: Activation function ('relu', 'sigmoid', 'tanh')
        - num_classes: Number of output classes
        - architecture: List of hidden layer sizes
        """
        # Store data
        self.X_train = self.normalize(X_train)
        self.y_train = y_train
        self.X_test = self.normalize(X_test)
        self.y_test = y_test
        
        # Network configuration
        self.activation = activation
        self.num_classes = num_classes
        
        # Build complete architecture (input -> hidden layers -> output)
        input_size = X_train.shape[1]
        self.architecture = [input_size] + architecture + [num_classes]
        
        # Initialize storage dictionaries
        self.parameters = {}
        self.layers = {}
        self.costs = []
        self.train_accuracies = []
        self.test_accuracies = []
        
        # Assertions for data validation
        assert X_train.shape[0] == y_train.shape[0], "Training data size mismatch"
        assert X_test.shape[0] == y_test.shape[0], "Test data size mismatch"
        assert X_train.shape[1] == X_test.shape[1], "Feature dimension mismatch"
    
    def normalize(self, X):
        """Min-max normalization to scale features to [0,1] range."""
        return (X - X.min()) / (X.max() - X.min() + 1e-8)
    
    def one_hot_encode(self, y):
        """Convert labels to one-hot encoded format."""
        encoded = np.eye(self.num_classes)[y.astype(int)]
        return encoded.T  # Transpose for correct dimensions
    
    # Activation functions
    def relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to prevent overflow
    
    def tanh(self, z):
        """Tanh activation function."""
        return np.tanh(z)
    
    def softmax(self, z):
        """Softmax activation for output layer."""
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    # Activation function derivatives
    def relu_derivative(self, z):
        """Derivative of ReLU."""
        return (z > 0).astype(float)
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid."""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def tanh_derivative(self, z):
        """Derivative of tanh."""
        return 1 - np.tanh(z)**2
    
    def get_activation_function(self):
        """Get the chosen activation function."""
        if self.activation == 'relu':
            return self.relu
        elif self.activation == 'sigmoid':
            return self.sigmoid
        elif self.activation == 'tanh':
            return self.tanh
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def get_activation_derivative(self):
        """Get the derivative of the chosen activation function."""
        if self.activation == 'relu':
            return self.relu_derivative
        elif self.activation == 'sigmoid':
            return self.sigmoid_derivative
        elif self.activation == 'tanh':
            return self.tanh_derivative
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def initialize_parameters(self):
        """Initialize weights and biases for all layers."""
        np.random.seed(42)  # For reproducibility
        
        for i in range(1, len(self.architecture)):
            # Weight initialization with proper scaling
            self.parameters[f'W{i}'] = np.random.randn(
                self.architecture[i], self.architecture[i-1]
            ) * np.sqrt(2.0 / self.architecture[i-1])  # He initialization for ReLU
            
            # Bias initialization
            self.parameters[f'b{i}'] = np.zeros((self.architecture[i], 1))
    
    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        
        Parameters:
        - X: Input data
        
        Returns:
        - cost: Current cost value
        """
        # Input layer
        self.layers['A0'] = X.T  # Transpose for correct dimensions (features x samples)
        
        activation_func = self.get_activation_function()
        
        # Forward pass through hidden layers
        for i in range(1, len(self.architecture) - 1):
            # Linear transformation
            self.layers[f'Z{i}'] = (
                np.dot(self.parameters[f'W{i}'], self.layers[f'A{i-1}']) + 
                self.parameters[f'b{i}']
            )
            # Activation
            self.layers[f'A{i}'] = activation_func(self.layers[f'Z{i}'])
        
        # Output layer (softmax)
        i = len(self.architecture) - 1
        self.layers[f'Z{i}'] = (
            np.dot(self.parameters[f'W{i}'], self.layers[f'A{i-1}']) + 
            self.parameters[f'b{i}']
        )
        self.layers[f'A{i}'] = self.softmax(self.layers[f'Z{i}'])
        
        # Compute cost (cross-entropy loss)
        m = X.shape[0]  # Number of examples
        y_encoded = self.one_hot_encode(self.y_train)
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(self.layers[f'A{i}'], epsilon, 1 - epsilon)
        
        cost = -np.sum(y_encoded * np.log(predictions)) / m
        
        return cost
    
    def backward_propagation(self):
        """
        Perform backward propagation to compute gradients.
        
        Returns:
        - derivatives: Dictionary containing gradients
        """
        derivatives = {}
        m = self.X_train.shape[0]  # Number of examples
        L = len(self.architecture) - 1  # Last layer index
        
        # One-hot encode labels
        y_encoded = self.one_hot_encode(self.y_train)
        
        # Output layer gradients
        derivatives[f'dZ{L}'] = self.layers[f'A{L}'] - y_encoded
        derivatives[f'dW{L}'] = np.dot(derivatives[f'dZ{L}'], self.layers[f'A{L-1}'].T) / m
        derivatives[f'db{L}'] = np.sum(derivatives[f'dZ{L}'], axis=1, keepdims=True) / m
        derivatives[f'dA{L-1}'] = np.dot(self.parameters[f'W{L}'].T, derivatives[f'dZ{L}'])
        
        # Hidden layers gradients
        activation_derivative = self.get_activation_derivative()
        
        for i in range(L-1, 0, -1):
            derivatives[f'dZ{i}'] = derivatives[f'dA{i}'] * activation_derivative(self.layers[f'Z{i}'])
            derivatives[f'dW{i}'] = np.dot(derivatives[f'dZ{i}'], self.layers[f'A{i-1}'].T) / m
            derivatives[f'db{i}'] = np.sum(derivatives[f'dZ{i}'], axis=1, keepdims=True) / m
            if i > 1:
                derivatives[f'dA{i-1}'] = np.dot(self.parameters[f'W{i}'].T, derivatives[f'dZ{i}'])
        
        return derivatives
    
    def update_parameters(self, derivatives, learning_rate):
        """Update parameters using gradient descent."""
        for i in range(1, len(self.architecture)):
            self.parameters[f'W{i}'] -= learning_rate * derivatives[f'dW{i}']
            self.parameters[f'b{i}'] -= learning_rate * derivatives[f'db{i}']
    
    def predict(self, X):
        """Make predictions on input data."""
        # Normalize input
        X_norm = self.normalize(X)
        
        # Forward pass
        A = X_norm.T
        activation_func = self.get_activation_function()
        
        for i in range(1, len(self.architecture) - 1):
            Z = np.dot(self.parameters[f'W{i}'], A) + self.parameters[f'b{i}']
            A = activation_func(Z)
        
        # Output layer
        i = len(self.architecture) - 1
        Z = np.dot(self.parameters[f'W{i}'], A) + self.parameters[f'b{i}']
        A = self.softmax(Z)
        
        # Return class with highest probability
        return np.argmax(A, axis=0)
    
    def accuracy(self, X, y):
        """Calculate accuracy on given dataset."""
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100
    
    def fit(self, learning_rate=0.03, epochs=200, print_cost=True):
        """
        Train the neural network.
        
        Parameters:
        - learning_rate: Learning rate for gradient descent
        - epochs: Number of training epochs
        - print_cost: Whether to print cost during training
        """
        # Initialize parameters
        self.initialize_parameters()
        
        print(f"Training neural network with architecture: {self.architecture}")
        print(f"Activation function: {self.activation}")
        print(f"Learning rate: {learning_rate}, Epochs: {epochs}")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Forward propagation
            cost = self.forward_propagation(self.X_train)
            
            # Backward propagation
            derivatives = self.backward_propagation()
            
            # Update parameters
            self.update_parameters(derivatives, learning_rate)
            
            # Store metrics
            self.costs.append(cost)
            
            # Calculate accuracies every 10 epochs
            if epoch % 10 == 0:
                train_acc = self.accuracy(self.X_train, self.y_train)
                test_acc = self.accuracy(self.X_test, self.y_test)
                
                self.train_accuracies.append(train_acc)
                self.test_accuracies.append(test_acc)
                
                if print_cost:
                    print(f"Epoch {epoch:3d}: Cost = {cost:.4f}, "
                          f"Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%")
        
        # Final results
        final_train_acc = self.accuracy(self.X_train, self.y_train)
        final_test_acc = self.accuracy(self.X_test, self.y_test)
        
        print("-" * 50)
        print(f"Training completed!")
        print(f"Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    
    def plot_results(self):
        """Plot training results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot cost
        ax1.plot(self.costs)
        ax1.set_title('Training Cost Over Time')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Cost')
        ax1.grid(True)
        
        # Plot accuracies
        epochs_acc = range(0, len(self.costs), 10)
        ax2.plot(epochs_acc, self.train_accuracies, label='Training Accuracy', marker='o')
        ax2.plot(epochs_acc, self.test_accuracies, label='Test Accuracy', marker='s')
        ax2.set_title('Accuracy Over Time')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    
    # Load MNIST data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X_train, X_test, y_train, y_test


def visualize_sample_digits(X, y, num_samples=5):
    """Visualize sample digits from the dataset."""
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    
    for i in range(num_samples):
        # Random sample
        idx = np.random.randint(0, len(X))
        image = X[idx].reshape(28, 28)
        label = y[idx]
        
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.suptitle('Sample MNIST Digits')
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the neural network training."""
    # Load data
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    # Visualize some sample digits
    visualize_sample_digits(X_train, y_train)
    
    # Create and train neural network
    nn = NeuralNetwork(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        activation='relu',
        num_classes=10,
        architecture=[128, 32]  # Two hidden layers with 128 and 32 neurons
    )
    
    # Train the network
    nn.fit(learning_rate=0.03, epochs=200, print_cost=True)
    
    # Plot results
    nn.plot_results()
    
    # Test predictions on a few samples
    print("\nSample Predictions:")
    print("-" * 30)
    for i in range(5):
        idx = np.random.randint(0, len(X_test))
        prediction = nn.predict(X_test[idx:idx+1])[0]
        actual = y_test[idx]
        print(f"Sample {i+1}: Predicted = {prediction}, Actual = {actual}")


if __name__ == "__main__":
    main()
