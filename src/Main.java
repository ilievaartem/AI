public class Main {
    public static void main(String[] args) {
        final int NUM_LETTERS = 33;

        Perceptron[] perceptrons = new Perceptron[NUM_LETTERS];
        for (int i = 0; i < NUM_LETTERS; i++) {
            perceptrons[i] = new Perceptron();
        }

        char[] letters = {
                'А', 'Б', 'В', 'Г', 'Ґ', 'Д', 'Е', 'Є', 'Ж', 'З',
                'И', 'І', 'Ї', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П',
                'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ',
                'Ь', 'Ю', 'Я'
        };

        int[][][] inputs = {
                // А
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 0, 1},
                        {1, 1, 1}
                },
                // Б
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 0, 1}
                },
                // В
                {
                        {1, 1, 1},
                        {0, 1, 0},
                        {0, 1, 0},
                        {1, 1, 1}
                },
                // Г
                {
                        {0, 1, 0},
                        {1, 1, 1},
                        {1, 0, 1},
                        {0, 0, 1}
                },
                // Ґ
                {
                        {1, 1, 1},
                        {0, 0, 1},
                        {1, 1, 1},
                        {1, 0, 1}
                },
                // Д
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 0, 1},
                        {1, 1, 1}
                },
                // Е
                {
                        {1, 1, 1},
                        {1, 0, 0},
                        {1, 0, 0},
                        {1, 1, 1}
                },
                // Є
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 0, 1}
                },
                // Ж
                {
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 1, 1}
                },
                // З
                {
                        {1, 1, 1},
                        {0, 0, 1},
                        {0, 1, 0},
                        {1, 1, 1}
                },
                // И
                {
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 0, 1}
                },
                // І
                {
                        {0, 1, 0},
                        {1, 1, 1},
                        {1, 0, 1},
                        {0, 1, 0}
                },
                // Ї
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 1, 1}
                },
                // Й
                {
                        {0, 1, 0},
                        {1, 1, 1},
                        {0, 1, 0},
                        {0, 1, 0}
                },
                // К
                {
                        {1, 0, 1},
                        {1, 1, 0},
                        {1, 1, 0},
                        {1, 0, 1}
                },
                // Л
                {
                        {1, 0, 0},
                        {1, 1, 0},
                        {1, 0, 1},
                        {1, 1, 1}
                },
                // М
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 0, 1},
                        {1, 1, 1}
                },
                // Н
                {
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 0, 1}
                },
                // О
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 0, 1},
                        {1, 1, 1}
                },
                // П
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 0, 0}
                },
                // Р
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 0, 1}
                },
                // С
                {
                        {1, 1, 1},
                        {1, 0, 0},
                        {1, 0, 0},
                        {1, 1, 1}
                },
                // Т
                {
                        {1, 1, 1},
                        {0, 1, 0},
                        {0, 1, 0},
                        {0, 1, 0}
                },
                // У
                {
                        {1, 0, 1},
                        {1, 0, 1},
                        {1, 0, 1},
                        {1, 1, 1}
                },
                // Ф
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 0, 1}
                },
                // Х
                {
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 1, 1},
                        {1, 0, 1}
                },
                // Ц
                {
                        {1, 1, 1},
                        {0, 1, 1},
                        {1, 1, 0},
                        {1, 1, 1}
                },
                // Ч
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 0, 1},
                        {1, 1, 1}
                },
                // Ш
                {
                        {1, 0, 1},
                        {1, 0, 1},
                        {1, 0, 1},
                        {1, 1, 1}
                },
                // Щ
                {
                        {1, 0, 1},
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 1, 1}
                },
                // Ь
                {
                        {1, 0, 0},
                        {1, 1, 0},
                        {1, 0, 1},
                        {1, 0, 1}
                },
                // Ю
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 1, 1},
                        {1, 0, 1}
                },
                // Я
                {
                        {1, 1, 1},
                        {1, 0, 1},
                        {1, 1, 0},
                        {1, 1, 1}
                }
        };

        if (inputs.length != NUM_LETTERS) {
            System.out.println("Кількість вхідних матриць не відповідає кількості букв!");
            return;
        }

        int epochs = 1000;
        boolean allCorrect = false;
        int completedEpochs = 0;

        for (int epoch = 0; epoch < epochs && !allCorrect; epoch++) {
            allCorrect = true;
            for (int i = 0; i < inputs.length; i++) {
                int[] inputVector = perceptrons[i].flattenMatrix(inputs[i]);

                for (int j = 0; j < perceptrons.length; j++) {
                    int expectedOutput = (j == i) ? 1 : 0;
                    perceptrons[j].train(inputVector, expectedOutput);
                }

                boolean correct = true;
                for (int j = 0; j < perceptrons.length; j++) {
                    int output = perceptrons[j].predict(inputVector);
                    if (j == i && output != 1) {
                        correct = false;
                        break;
                    } else if (j != i && output != 0) {
                        correct = false;
                        break;
                    }
                }

                if (!correct) {
                    allCorrect = false;
                }
            }

            completedEpochs = epoch + 1;
            System.out.println("Епоха: " + completedEpochs + " -> Всі передбачення правильні: " + allCorrect);
        }

        if (allCorrect) {
            System.out.println("Задача була коректно виконана за " + completedEpochs + " епох.");
        } else {
            System.out.println("Максимальна кількість епох досягнута. Навчання може бути неповним.");
        }

        for (int i = 0; i < perceptrons.length; i++) {
            double[] finalWeights = perceptrons[i].getWeights();
            System.out.println("Фінальні ваги та зміщення для букви " + letters[i] + ":");
            for (int j = 0; j < finalWeights.length; j++) {
                if (j < finalWeights.length - 1) {
                    System.out.printf("w%d: %.4f ", j + 1, finalWeights[j]);
                } else {
                    System.out.printf("bias: %.4f\n", finalWeights[j]);
                }
            }
        }

        for (int i = 0; i < inputs.length; i++) {
            int[] inputVector = perceptrons[i].flattenMatrix(inputs[i]);
            int predictedLetter = -1;
            for (int j = 0; j < perceptrons.length; j++) {
                int output = perceptrons[j].predict(inputVector);
                if (output == 1) {
                    predictedLetter = j;
                    break;
                }
            }
            if (predictedLetter != -1) {
                System.out.println("Буква: " + letters[i] + " -> Виявлена буква: " + letters[predictedLetter] + " (Очікувана: " + letters[i] + ")");
            } else {
                System.out.println("Буква: " + letters[i] + " -> Виявлена буква: Невідома (Очікувана: " + letters[i] + ")");
            }
        }
    }
}
