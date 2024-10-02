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

    private int activation(double sum) {
        return sum >= 0 ? 1 : 0;
    }

    public int[] predict(int[] inputs) {
        int[] outputs = new int[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = bias[i];
            for (int j = 0; j < inputSize; j++) {
                sum += weights[i][j] * inputs[j];
            }
            outputs[i] = activation(sum);
        }
        return outputs;
    }

    public void trainNeuron(int[][] inputsList, int neuronIndex, int[] desiredOutput) {
        System.out.println("Тренуємо нейрон для літери: " + (neuronIndex + 1));
        System.out.println("Поточні ваги перед оновленням:");
        printWeights(neuronIndex);

        for (int epoch = 0; epoch < 1000; epoch++) {
            int totalError = 0;

            for (int i = 0; i < inputsList.length; i++) {
                int[] inputs = inputsList[i];
                int desired = desiredOutput[i];

                double weightedSum = bias[neuronIndex];
                for (int j = 0; j < inputs.length; j++) {
                    weightedSum += weights[neuronIndex][j] * inputs[j];
                }

                int output = activation(weightedSum);
                int error = desired - output;
                totalError += Math.abs(error);

                for (int j = 0; j < inputs.length; j++) {
                    weights[neuronIndex][j] += learningSpeed * error * inputs[j];
                }
                bias[neuronIndex] += learningSpeed * error;
            }

            if (totalError == 0) {
                System.out.println("Нейрон для літери " + (neuronIndex + 1) + " навчився за " + (epoch + 1) + " епох.");
                printWeights(neuronIndex);
                System.out.println("-----------------------------------------------");
                break;
            }

            if ((epoch + 1) % 100 == 0) {
                System.out.println("Епоха: " + (epoch + 1) + " -> Помилка: " + totalError);
            }
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
