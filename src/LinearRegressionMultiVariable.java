/**
 * Created by Solomon on 7/7/2017.
 */

/*
Linear regression to predict house prices based on size of house and number of bedrooms
 */
public class LinearRegressionMultiVariable {

    double dHypothesis[];

    // Load House attributes
    double[][] dHousingAttributes = new double[][]{{1,2104,3}, {1,1600,3}, {1,2400,3}, {1,1416,2}, {1,3000,4}, {1,1985,4}, {1,1534,3}, {1,1427,3}, {1,1380,3}, {1,1494,3}, {1,1940,4}, {1,2000,3}, {1,1890,3}, {1,4478,5}, {1,1268,3}, {1,2300,4}, {1,1320,2}, {1,1236,3}, {1,2609,4}, {1,3031,4}, {1,1767,3}, {1,1888,2}, {1,1604,3}, {1,1962,4}, {1,3890,3}, {1,1100,3}, {1,1458,3}, {1,2526,3}, {1,2200,3}, {1,2637,3}, {1,1839,2}, {1,1000,1}, {1,2040,4}, {1,3137,3}, {1,1811,4}, {1,1437,3}, {1,1239,3}, {1,2132,4}, {1,4215,4}, {1,2162,4}, {1,1664,2}, {1,2238,3}, {1,2567,4}, {1,1200,3}, {1,852,2}, {1,1852,4}, {1,1203,3}};

    // Load price data
    double[] dPrices = {399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999, 212000, 242500, 239999, 347000, 329999, 699900, 259900, 449900, 299900, 199900, 499998, 599000, 252900, 255000, 242900, 259900, 573900, 249900, 464500, 469000, 475000, 299900, 349900, 169900, 314900, 579900, 285900, 249900, 229900, 345000, 549000, 287000, 368500, 329900, 314000, 299000, 179900, 299900, 239500};

    // Initialize theta array
    double[] dTheta = {0.0, 0.0, 0.0};

    // Number of Iterations
    int iIterations = 100;

    // Learning rate
    double dLearningRate = 0.3;

    // number of training examples
    int iNumberOfTrainingExamples = dPrices.length;

    // For calculating mean and standard deviation on input data
    double dMeanValues[] = new double[dHousingAttributes[0].length];
    double dStandardDeviation[] = new double[dHousingAttributes[0].length];

    public static void main(String args[]){

        LinearRegressionMultiVariable objLinearRegressionMultiVariable = new LinearRegressionMultiVariable();

        // normalize input data
        objLinearRegressionMultiVariable.normalizeInputData();

        // perform gradient descent to find optimum theta values
        objLinearRegressionMultiVariable.gradientDescent();

        // Test sample data to predict house price
        // 1650 sqft, 3 bedroom house
        double dHousingData[][] = {{1, 1650, 3}};
        // Normalize test data
        for(int index1=0; index1<dHousingData.length; index1++){
            for(int index2=1; index2<dHousingData[0].length; index2++) {
                dHousingData[index1][index2] = dHousingData[index1][index2] - objLinearRegressionMultiVariable.dMeanValues[index2];
                dHousingData[index1][index2] = dHousingData[index1][index2] / objLinearRegressionMultiVariable.dStandardDeviation[index2];
            }
        }

        double dPredictedPrice = objLinearRegressionMultiVariable.predictHousePrice(dHousingData);
        System.out.println("Predicted price : " + dPredictedPrice);

    }

    // Function to normalize input data
    void normalizeInputData(){

        // temp variables
        double dSumOfColumnsForMean[] = new double[dHousingAttributes[0].length];
        double dSumOfColumnsForStandardDeviation[] = new double[dHousingAttributes[0].length];

        // calculate the sum of input data
        for(int index1=0; index1<dHousingAttributes.length; index1++){
            for(int index2=1; index2<dHousingAttributes[0].length; index2++) {
                dSumOfColumnsForMean[index2] += dHousingAttributes[index1][index2];
            }
        }

        // calculate the mean of input data
        for(int index1=1; index1<dSumOfColumnsForMean.length; index1++){
            dMeanValues[index1] = dSumOfColumnsForMean[index1] / dHousingAttributes.length;
        }

        // calculate prelim value for std deviation - X[i] - mean
        for(int index1=0; index1<dHousingAttributes.length; index1++){
            for(int index2=1; index2<dHousingAttributes[0].length; index2++) {
                Double dTemp = dHousingAttributes[index1][index2] - dMeanValues[index2];
                dSumOfColumnsForStandardDeviation[index2] += (dTemp * dTemp);
            }
        }

        // calculate the standard deviation of input data
        for(int index1=1; index1<dSumOfColumnsForStandardDeviation.length; index1++){
            dStandardDeviation[index1] = Math.sqrt(dSumOfColumnsForStandardDeviation[index1] / dHousingAttributes.length);
        }

        // Normalize input data
        for(int index1=0; index1<dHousingAttributes.length; index1++){
            for(int index2=1; index2<dHousingAttributes[0].length; index2++) {
                dHousingAttributes[index1][index2] = dHousingAttributes[index1][index2] - dMeanValues[index2];
                dHousingAttributes[index1][index2] = dHousingAttributes[index1][index2] / dStandardDeviation[index2];
            }
        }

    }

    // perform gradient descent
    void gradientDescent(){

        // transpose input data matrix
        double dTransposedInputData[][] = transposeMatrix(dHousingAttributes);

        // temp variables
        double dDeviation[] = new double[iNumberOfTrainingExamples];
        double dDelta[] = new double[dTransposedInputData.length];

        for(int index1=0; index1<iIterations; index1++){

            // calculate hypothesis
            calculateHypothesis();

            // calculate deviations
            // deviation = h(x) - y
            for(int index2=0; index2<iNumberOfTrainingExamples; index2++){
                dDeviation[index2] = dHypothesis[index2] - dPrices[index2];
            }

            // calculate delta
            // delta = alpha * 1/m * X' * deviation
            for(int index2=0; index2<dTransposedInputData.length; index2++){
                for(int index3=0; index3<dTransposedInputData[0].length; index3++){
                    dDelta[index2] = dDelta[index2] + (dTransposedInputData[index2][index3] * dDeviation[index3]);
                }
                dDelta[index2] = dDelta[index2] / iNumberOfTrainingExamples;
                dDelta[index2] = dLearningRate * dDelta[index2];
            }

            // theta = theta - delta;
            for(int index2=0; index2<dTheta.length; index2++){
                dTheta[index2] = dTheta[index2] - dDelta[index2];
            }

            computeCost();

        }

    }

    // generic function to transpose any input matrix
    double[][] transposeMatrix(double dInputMatrix[][]){

        double dTransposedMatrix[][] = new double[dInputMatrix[0].length][dInputMatrix.length];

        for(int index1=0; index1<dInputMatrix[0].length; index1++){
            for(int index2=0; index2<iNumberOfTrainingExamples; index2++){
                dTransposedMatrix[index1][index2] = dInputMatrix[index2][index1];
            }
        }

        return dTransposedMatrix;

    }

    // calculate hypothesis
    // h(x) = theta(0) * X(0) + theta(1) * X(1) ...
    void calculateHypothesis(){

        dHypothesis = new double[iNumberOfTrainingExamples];

        for(int index1=0; index1<iNumberOfTrainingExamples; index1++){
            for(int index2=0; index2<dHousingAttributes[0].length; index2++){
                dHypothesis[index1] = dHypothesis[index1] + (dHousingAttributes[index1][index2] * dTheta[index2]);
            }
        }

        //computeCost();
    }

    // calculate compute cost
    void computeCost(){

        double dSumOfSquareErrors = 0.0;
        double dDeviation = 0.0;

        // SquareErrors = (Hypothesis - y)^2
        for(int index1=0; index1<iNumberOfTrainingExamples; index1++){
            dDeviation = dHypothesis[index1] - dPrices[index1];
            dSumOfSquareErrors = dSumOfSquareErrors + (dDeviation * dDeviation);
        }

        // J = 1/(2*m) * sum(sqrErrors)
        double dComputedCost =  (1.0 / (2.0 * iNumberOfTrainingExamples) ) * dSumOfSquareErrors;

    }

    // Predict any house price based on size and number of rooms
    double predictHousePrice(double[][] dHousingData){

        double dPredictedPrice = 0.0;

        for(int index1=0; index1<dHousingData.length; index1++){

            for(int index2=0; index2<dHousingData[0].length; index2++){
                dPredictedPrice = dPredictedPrice + (dHousingData[index1][index2] * dTheta[index2]);
            }
        }

        return dPredictedPrice;
        
    }

}
