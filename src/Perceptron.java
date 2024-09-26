import java.util.Random;

public class Perceptron {

    static final int NUM_INPUTS = 12;

    double[] weights = new double[NUM_INPUTS];

    double theta;

    Random random = new Random();

    public Perceptron() {
        for (int i = 0; i < NUM_INPUTS; i++) {
            weights[i] = random.nextDouble() - 0.5;
        }
        theta = random.nextDouble() * 5;
    }

    public double weightedSum(int[] inputs) {
        double sum = 0;
        for (int i = 0; i < NUM_INPUTS; i++) {
            sum += weights[i] * inputs[i];
        }
        return sum;
    }

    public int predict(int[] inputs) {
        double sum = weightedSum(inputs);
        return sum >= theta ? 1 : 0;
    }

    public void train(int[] inputs, int expectedOutput) {
        int output = predict(inputs);

        if (output != expectedOutput) {
            for (int i = 0; i < NUM_INPUTS; i++) {
                if (output == 0) {
                    weights[i] += inputs[i];
                } else {
                    weights[i] -= inputs[i];
                }
            }
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
}
