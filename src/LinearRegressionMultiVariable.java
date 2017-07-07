/**
 * Created by Solomon on 7/7/2017.
 */

/*
Linear regression to predict house prices based on size of house and number of bedrooms
 */
public class LinearRegressionMultiVariable {

    double dHypothesis[];

    // Load Population data into Array X
    double[][] dHousingAttributes = new double[][]{{1,2104,3}, {1,1600,3}, {1,2400,3}, {1,1416,2}, {1,3000,4}, {1,1985,4}, {1,1534,3}, {1,1427,3}, {1,1380,3}, {1,1494,3}, {1,1940,4}, {1,2000,3}, {1,1890,3}, {1,4478,5}, {1,1268,3}, {1,2300,4}, {1,1320,2}, {1,1236,3}, {1,2609,4}, {1,3031,4}, {1,1767,3}, {1,1888,2}, {1,1604,3}, {1,1962,4}, {1,3890,3}, {1,1100,3}, {1,1458,3}, {1,2526,3}, {1,2200,3}, {1,2637,3}, {1,1839,2}, {1,1000,1}, {1,2040,4}, {1,3137,3}, {1,1811,4}, {1,1437,3}, {1,1239,3}, {1,2132,4}, {1,4215,4}, {1,2162,4}, {1,1664,2}, {1,2238,3}, {1,2567,4}, {1,1200,3}, {1,852,2}, {1,1852,4}, {1,1203,3}};

    // Load profit data into y
    double[] dPrices = {399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999, 212000, 242500, 239999, 347000, 329999, 699900, 259900, 449900, 299900, 199900, 499998, 599000, 252900, 255000, 242900, 259900, 573900, 249900, 464500, 469000, 475000, 299900, 349900, 169900, 314900, 579900, 285900, 249900, 229900, 345000, 549000, 287000, 368500, 329900, 314000, 299000, 179900, 299900, 239500};

    // Initialize theta array
    double[] dTheta = {0.0, 0.0, 0.0};

    // Number of Iterations
    int iIterations = 100;

    // Learning rate
    double dLearningRate = 0.3;

    // number of training examples
    int iNumberOfTrainingExamples = dPrices.length;

    public static void main(String args[]){

        LinearRegressionMultiVariable objLinearRegressionMultiVariable = new LinearRegressionMultiVariable();

        objLinearRegressionMultiVariable.normalizeInputData();

        //objLinearRegressionMultiVariable.gradientDescent();

    }

    void normalizeInputData(){

        double dMeanValues[] = new double[dHousingAttributes[0].length];
        double dSumOfColumns1[] = new double[dHousingAttributes[0].length];
        double dSumOfColumns2[] = new double[dHousingAttributes[0].length];

        // calculate the sum of input data
        for(int index1=0; index1<dHousingAttributes.length; index1++){
            for(int index2=1; index2<dHousingAttributes[0].length; index2++) {
                dSumOfColumns1[index2] += dHousingAttributes[index1][index2];
            }
        }

        // calculate the mean of input data
        for(int index1=0; index1<dSumOfColumns1.length; index1++){
            dMeanValues[index1] = dSumOfColumns1[index1] / dHousingAttributes.length;
        }

        for(int index1=0; index1<dSumOfColumns1.length; index1++){
            System.out.println(dSumOfColumns1[index1]);
            System.out.println(dMeanValues[index1]);
        }

        // calculate prelim value for std deviation - X[i] - mean
        for(int index1=0; index1<dHousingAttributes.length; index1++){
            for(int index2=1; index2<dHousingAttributes[0].length; index2++) {
                Double dTemp = dHousingAttributes[index1][index2] - dMeanValues[index2];
                System.out.println("hi1");
                System.out.println("dtemp : " +dTemp);
                dSumOfColumns2[index2] += (dTemp * dTemp);
                System.out.println("hi2");
                System.out.println("sumofcols : " +dSumOfColumns2[index2]);
            }
        }

        System.out.println("sumofcols length : " +dSumOfColumns1.length);

        for(int index1=0; index1<dSumOfColumns2.length; index1++){
            System.out.println(dSumOfColumns2[index1]);
        }



    }

    void gradientDescent(){

        double dDeviation[] = new double[iNumberOfTrainingExamples];

        double dTransposedInputData[][] = transposeMatrix(dHousingAttributes);
        double dDelta[] = new double[dTransposedInputData.length];

        /*for(int index2=0; index2<dTransposedInputData.length; index2++){
            dDelta[index2]= 0.0;
        }*/

        for(int index1=0; index1<iIterations; index1++){

            calculateHypothesis();

            for(int index2=0; index2<iNumberOfTrainingExamples; index2++){
                dDeviation[index2] = dHypothesis[index2] - dPrices[index2];
            }

            for(int index2=0; index2<iNumberOfTrainingExamples; index2++){
                System.out.println(dDeviation[index2]);
            }

            for(int index2=0; index2<dTransposedInputData.length; index2++){
                for(int index3=0; index3<dTransposedInputData[0].length; index3++){
                    dDelta[index2] = dDelta[index2] + (dTransposedInputData[index2][index3] * dDeviation[index3]);
                }
                dDelta[index2] = dDelta[index2] / iNumberOfTrainingExamples;
                dDelta[index2] = dLearningRate * dDelta[index2];
            }

            for(int index2=0; index2<dTheta.length; index2++){
                dTheta[index2] = dTheta[index2] - dDelta[index2];
            }

            computeCost();

        }

        for(int index1=0; index1<dTheta.length; index1++){
            System.out.println(dTheta[index1]);
        }
    }

    double[][] transposeMatrix(double dInputMatrix[][]){

        double dTransposedMatrix[][] = new double[dInputMatrix[0].length][dInputMatrix.length];

        for(int index1=0; index1<dInputMatrix[0].length; index1++){
            for(int index2=0; index2<iNumberOfTrainingExamples; index2++){
                dTransposedMatrix[index1][index2] = dInputMatrix[index2][index1];
            }
        }

        for(int index1=0; index1<dTransposedMatrix.length; index1++){
            for(int index2=0; index2<dTransposedMatrix[0].length; index2++){
                System.out.print(dTransposedMatrix[index1][index2] + " ");
            }
            System.out.println();
        }

        return dTransposedMatrix;

    }

    void calculateHypothesis(){

        // dHypothesis = dInputData * dTheta
        dHypothesis = new double[iNumberOfTrainingExamples];

        // Initialize dHypothesis to zero vector
        /*for(int index1=0; index1<iNumberOfTrainingExamples; index1++){
            dHypothesis[index1] = 0.0;
        }*/


        for(int index1=0; index1<iNumberOfTrainingExamples; index1++){
            for(int index2=0; index2<dHousingAttributes[0].length; index2++){
                dHypothesis[index1] = dHypothesis[index1] + (dHousingAttributes[index1][index2] * dTheta[index2]);
            }
        }

        /*for(int index1=0; index1<iNumberOfTrainingExamples; index1++){
            System.out.println(dHypothesis[index1]);
        }*/

        //computeCost();
    }

    void computeCost(){

        double dSumOfSquareErrors = 0.0;
        double dDeviation = 0.0;

        for(int index1=0; index1<iNumberOfTrainingExamples; index1++){
            dDeviation = dHypothesis[index1] - dPrices[index1];
            dSumOfSquareErrors = dSumOfSquareErrors + (dDeviation * dDeviation);
        }

        //System.out.println(dSumOfSquareErrors);

        double dComputedCost =  (1.0 / (2.0 * iNumberOfTrainingExamples) ) * dSumOfSquareErrors;

        //System.out.println(dComputedCost);

    }

}
