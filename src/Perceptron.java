import java.util.Random;

public class Perceptron {
    private double[][] weights;
    private double[] bias;
    private double learningSpeed;

    private int inputSize;
    private int outputSize;
    private Random random;

    public Perceptron(int inputSize, int outputSize, double learningSpeed) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.learningSpeed = learningSpeed;
        this.random = new Random();

        weights = new double[outputSize][inputSize];
        bias = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = random.nextDouble() - 0.5;
            }
            bias[i] = random.nextDouble() - 0.5;
        }
    }

    private double sigmoid(double sum) {
        return 1.0 / (1.0 + Math.exp(-sum));
    }

    private double sigmoidDerivative(double output) {
        return output * (1.0 - output);
    }

    private double activation(double sum) {
        return sigmoid(sum);
    }

    public double[] predict(int[] inputs) {
        double[] outputs = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = bias[i];
            for (int j = 0; j < inputSize; j++) {
                sum += weights[i][j] * inputs[j];
            }
            outputs[i] = activation(sum);
        }
        return outputs;
    }

    public void train(int[][] inputsList, int[][] desiredOutputs, int maxEpochs, double errorThreshold) {
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            double totalError = 0.0;

            for (int i = 0; i < inputsList.length; i++) {
                int[] inputs = inputsList[i];
                int[] desired = desiredOutputs[i];

                for (int neuron = 0; neuron < outputSize; neuron++) {
                    double weightedSum = bias[neuron];
                    for (int j = 0; j < inputSize; j++) {
                        weightedSum += weights[neuron][j] * inputs[j];
                    }

                    double output = activation(weightedSum);

                    double error = desired[neuron] - output;
                    totalError += error * error;

                    double gradient = error * sigmoidDerivative(output);

                    for (int j = 0; j < inputSize; j++) {
                        weights[neuron][j] += learningSpeed * gradient * inputs[j];
                    }
                    bias[neuron] += learningSpeed * gradient;
                }
            }

            if (epoch % 100 == 0 || epoch == 1 || epoch == maxEpochs) {
                System.out.printf("Епоха: %d -> Квадратична помилка: %.4f\n", epoch, totalError);
            }

            if (totalError < errorThreshold) {
                System.out.printf("Навчання зупинено на епосі %d, загальна квадратична помилка: %.4f\n", epoch, totalError);
                break;
            }
        }

        for (int neuron = 0; neuron < outputSize; neuron++) {
            printWeights(neuron);
            System.out.println("-----------------------------------------------");
        }
    }

    private void printWeights(int neuronIndex) {
        System.out.print("Ваги: ");
        for (int j = 0; j < weights[neuronIndex].length; j++) {
            System.out.printf("%.4f ", weights[neuronIndex][j]);
        }
        System.out.printf("bias: %.4f\n", bias[neuronIndex]);
    }
}
