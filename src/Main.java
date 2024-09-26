public class Main {
    public static void main(String[] args) {
        Perceptron perceptron = new Perceptron();

        int[][][] inputs = {
                {{1, 1, 1}, {1, 0, 1}, {1, 0, 1}, {1, 1, 1}}, // Цифра 0
                {{0, 1, 1}, {1, 0, 1}, {0, 0, 1}, {1, 1, 1}}, // Цифра 1
                {{1, 1, 1}, {0, 0, 1}, {1, 1, 0}, {1, 1, 1}}, // Цифра 2
                {{1, 1, 1}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1}}, // Цифра 3
                {{0, 1, 1}, {0, 1, 0}, {1, 1, 1}, {1, 1, 0}}, // Цифра 4
                {{1, 1, 1}, {1, 1, 0}, {0, 1, 1}, {1, 1, 1}}, // Цифра 5
                {{0, 1, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}}, // Цифра 6
                {{1, 1, 1}, {0, 0, 1}, {1, 1, 1}, {1, 0, 0}}, // Цифра 7
                {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, // Цифра 8
                {{1, 1, 1}, {1, 1, 1}, {0, 1, 1}, {1, 1, 1}}  // Цифра 9
        };

        int[] outputs = {0, 0, 1, 0, 1, 0, 1, 0, 1, 0};

        int epochs = 1000;
        boolean allCorrect = false;
        int completedEpochs = 0;

        for (int epoch = 0; epoch < epochs && !allCorrect; epoch++) {
            allCorrect = true;
            for (int i = 0; i < inputs.length; i++) {
                int[] inputVector = perceptron.flattenMatrix(inputs[i]);
                int expectedOutput = outputs[i];
                int output = perceptron.predict(inputVector);

                if (output != expectedOutput) {
                    perceptron.train(inputVector, expectedOutput);
                    allCorrect = false;
                }
            }

            completedEpochs = epoch + 1;
            System.out.println("Епоха: " + completedEpochs + " -> Всі передбачення правильні: " + allCorrect);
        }

        System.out.println("Задача була коректно виконана за " + completedEpochs + " епох.");

        for (int i = 0; i < inputs.length; i++) {
            int[] inputVector = perceptron.flattenMatrix(inputs[i]);
            int output = perceptron.predict(inputVector);
            System.out.println("Число: " + i + " -> Вихід: " + output + " (Очікуваний: " + outputs[i] + ")");
        }
    }
}
