import java.util.Random;

public class Perceptron {

    static final int NUM_INPUTS = 12;
    double[] weights = new double[NUM_INPUTS];
    double bias;
    double learningRate = 0.1;

    Random random = new Random();

    public Perceptron() {
        for (int i = 0; i < NUM_INPUTS; i++) {
            weights[i] = random.nextDouble() - 0.5;
        }
        bias = random.nextDouble() - 0.5;
    }

    public double weightedSum(int[] inputs) {
        double sum = 0;
        for (int i = 0; i < NUM_INPUTS; i++) {
            sum += weights[i] * inputs[i];
        }
        sum += bias;
        return sum;
    }

    public int predict(int[] inputs) {
        double sum = weightedSum(inputs);
        return sum >= 0 ? 1 : 0;
    }

    public void train(int[] inputs, int expectedOutput) {
        int output = predict(inputs);
        int error = expectedOutput - output;

        if (error != 0) {
            for (int i = 0; i < NUM_INPUTS; i++) {
                weights[i] += learningRate * error * inputs[i];
            }
            bias += learningRate * error * 1;
        }
    }

    public int[] flattenMatrix(int[][] matrix) {
        int[] inputVector = new int[NUM_INPUTS];
        int index = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                inputVector[index++] = matrix[i][j];
            }
        }
        return inputVector;
    }

    public double[] getWeights() {
        double[] allWeights = new double[NUM_INPUTS + 1];
        for (int i = 0; i < NUM_INPUTS; i++) {
            allWeights[i] = weights[i];
        }
        allWeights[NUM_INPUTS] = bias;
        return allWeights;
    }
}
